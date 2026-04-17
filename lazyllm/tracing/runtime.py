import atexit
import contextvars
import json
import threading
from typing import Any, Dict, Optional

from lazyllm.common import LOG, globals
from lazyllm.configs import config
from lazyllm.thirdparty import opentelemetry
from .backends import get_tracing_backend
from .context import LazyTraceContext
from .span import LazySpan, LazyTrace


_TRACE_SERVICE_NAME = 'lazyllm'
_in_reconstructed_thread = contextvars.ContextVar('_lazyllm_tracing_reconstructed', default=False)
_current_trace: 'contextvars.ContextVar[Optional[LazyTrace]]' = contextvars.ContextVar(
    '_lazyllm_current_trace', default=None)


def current_trace() -> Optional[LazyTrace]:
    """Return the ``LazyTrace`` bound to the current execution context, if any."""
    return _current_trace.get()


class TracingSetupError(RuntimeError):
    pass


def get_trace_context() -> LazyTraceContext:
    return LazyTraceContext.from_dict(globals.get('trace', {}))


def set_trace_context(ctx) -> LazyTraceContext:
    if isinstance(ctx, LazyTraceContext):
        tc = ctx
    else:
        tc = LazyTraceContext.from_dict(ctx if isinstance(ctx, dict) else {})
    globals['trace'] = tc.to_dict()
    return tc


def _capture_payload_enabled(ctx: LazyTraceContext) -> bool:
    if ctx.debug_capture_payload is not None:
        return bool(ctx.debug_capture_payload)
    return bool(config['trace_content_enabled'])


def _stringify_payload(value: Any, *, limit: int = 8192) -> str:
    try:
        if isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = repr(value)
    if len(text) > limit:
        return text[:limit] + '...<truncated>'
    return text


class TracingRuntime:
    def __init__(self):
        self._lock = threading.RLock()
        self._initialized = False
        self._warned = False
        self._provider = None
        self._tracer = None
        self._trace_api = None
        self._status = None
        self._backend = None

    def available(self) -> bool:
        return self._ensure_runtime() and self._tracer is not None

    def _warn_once(self, message: str):
        if not self._warned:
            LOG.warning(message)
            self._warned = True

    def _get_backend(self):
        backend_name = config['trace_backend']
        return get_tracing_backend(backend_name)

    def _ensure_runtime(self) -> bool:
        if self._initialized:
            return self._tracer is not None
        with self._lock:
            if self._initialized:
                return self._tracer is not None
            self._initialized = True
            try:
                trace_api = opentelemetry.trace
                Resource = opentelemetry.sdk.resources.Resource
                TracerProvider = opentelemetry.sdk.trace.TracerProvider
                BatchSpanProcessor = opentelemetry.sdk.trace.export.BatchSpanProcessor
                Status = opentelemetry.trace.status.Status
                StatusCode = opentelemetry.trace.status.StatusCode
            except ImportError as exc:
                self._warn_once(str(exc))
                return False

            try:
                backend = self._get_backend()
            except (ValueError, TracingSetupError) as exc:
                self._warn_once(str(exc))
                return False

            try:
                exporter = backend.build_exporter()
            except Exception as exc:
                self._warn_once(f'LazyLLM {backend.name} tracing initialization failed: {exc}')
                return False

            resource = Resource.create({'service.name': _TRACE_SERVICE_NAME})
            provider = TracerProvider(resource=resource)
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
            trace_api.set_tracer_provider(provider)

            self._provider = provider
            self._trace_api = trace_api
            self._tracer = trace_api.get_tracer('lazyllm.tracing')
            self._status = (Status, StatusCode)
            self._backend = backend
            atexit.register(self.shutdown)
            return True

    def _trace_enabled(self, ctx: LazyTraceContext) -> bool:
        if ctx.enabled is not None:
            return bool(ctx.enabled) and ctx.sampled is not False
        if ctx.sampled is False:
            return False
        return bool(config['trace_enabled'])

    @staticmethod
    def _target_name(target: Any, span_kind: str) -> str:
        if span_kind == 'module':
            return getattr(target, 'name', None) or getattr(target, '_module_name', None) or target.__class__.__name__
        if span_kind == 'callable':
            return getattr(target, '__name__', None) or target.__class__.__name__
        override = getattr(target, '__span_name__', None)
        if override:
            return override
        return target.__class__.__name__

    @staticmethod
    def _target_id(target: Any, span_kind: str) -> Optional[str]:
        if span_kind == 'module':
            return getattr(target, '_module_id', None)
        if span_kind == 'callable':
            return None
        return getattr(target, '_flow_id', None)

    def _populate_identity(self, span: LazySpan, target: Any) -> None:
        """Fill identity and config fields of LazySpan from the target object."""
        span.component_class = target.__class__.__name__
        span.component_id = self._target_id(target, span.span_kind)

        try:
            if hasattr(target, '__trace_kwargs__'):
                trace_kwargs = target.__trace_kwargs__
                if isinstance(trace_kwargs, dict):
                    span.config = dict(trace_kwargs)
        except Exception:
            pass

        semantic_type = getattr(target, '__semantic_type__', None)
        if semantic_type is None and span.span_kind == 'flow':
            semantic_type = 'workflow_control'
        span.semantic_type = semantic_type

    def start_span(
        self,
        *,
        span_kind: str,
        target: Any,
        args: tuple,
        kwargs: Dict[str, Any],
    ) -> Optional[LazySpan]:
        ctx = get_trace_context()
        if not self._trace_enabled(ctx):
            return None
        if not self._ensure_runtime():
            return None

        capture_payload = _capture_payload_enabled(ctx)
        span_name = self._target_name(target, span_kind)
        pre_parent_span_id = ctx.parent_span_id

        current = self._trace_api.get_current_span()
        is_root_span = not current.get_span_context().is_valid

        parent_context = None
        if is_root_span and ctx.trace_id and ctx.parent_span_id:
            try:
                parent_sc = opentelemetry.trace.SpanContext(
                    trace_id=int(ctx.trace_id, 16),
                    span_id=int(ctx.parent_span_id, 16),
                    is_remote=False,
                    trace_flags=opentelemetry.trace.TraceFlags(0x01),
                )
                parent_context = opentelemetry.trace.set_span_in_context(
                    opentelemetry.trace.NonRecordingSpan(parent_sc)
                )
                is_root_span = False
                _in_reconstructed_thread.set(True)
            except Exception:
                pass

        # Create OTel span (timing + nesting only, no attributes yet)
        span_cm = self._tracer.start_as_current_span(span_name, context=parent_context)
        otel_span = span_cm.__enter__()
        span_context = otel_span.get_span_context()
        trace_id_hex = f'{span_context.trace_id:032x}'
        span_id_hex = f'{span_context.span_id:016x}'

        if not _in_reconstructed_thread.get(False):
            ctx.trace_id = trace_id_hex
            ctx.parent_span_id = span_id_hex
            set_trace_context(ctx)

        lazy_span = LazySpan(
            name=span_name,
            span_kind=span_kind,
            is_root_span=is_root_span,
            trace_id=trace_id_hex,
            span_id=span_id_hex,
            parent_span_id=pre_parent_span_id,
            capture_payload=capture_payload,
            _otel_span=otel_span,
            _otel_span_cm=span_cm,
        )

        self._populate_identity(lazy_span, target)

        if capture_payload:
            lazy_span.input = {'args': args, 'kwargs': kwargs}

        # --- LazyTrace lifecycle ---
        # The first span in the current context (or a span that lands on a new
        # trace_id due to e.g. an explicit reset) creates and owns a LazyTrace.
        # Nested spans reuse it; cross-thread reconstructed slices build a local
        # LazyTrace anchored on the propagated trace_id.
        active_trace = _current_trace.get()
        if active_trace is None or active_trace.trace_id != trace_id_hex or not active_trace.is_active:
            new_trace = LazyTrace(
                trace_id=trace_id_hex,
                root_span_id=span_id_hex if is_root_span else None,
                session_id=ctx.session_id,
                user_id=ctx.user_id,
                request_tags=list(ctx.request_tags) if ctx.request_tags else [],
                is_reconstructed=_in_reconstructed_thread.get(False),
            )
            _current_trace.set(new_trace)
            lazy_span._owns_lazy_trace = True
            active_trace = new_trace

        active_trace._record_span_start(lazy_span)

        if is_root_span:
            lazy_span.session_id = active_trace.session_id
            lazy_span.user_id = active_trace.user_id
            lazy_span.request_tags = list(active_trace.request_tags)

        return lazy_span

    def set_output(self, span: Optional[LazySpan], output: Any):
        if span is None:
            return
        span.status = 'ok'
        if span.capture_payload:
            span.output = output

    def set_usage(self, span: Optional[LazySpan], usage: Dict[str, Any]):
        if span is None:
            return
        span.usage = usage

    def set_attributes(self, span: Optional[LazySpan], attrs: Dict[str, Any]):
        if span is None or not attrs:
            return
        span.output_attrs.update(attrs)

    def set_error(self, span: Optional[LazySpan], exc: Exception):
        if span is None:
            return
        span.status = 'error'
        span.error = exc

    def _build_otel_attributes(self, span: LazySpan, trace: Optional[LazyTrace] = None) -> Dict[str, Any]:
        """Build the generic OTel attribute set for a span."""
        attrs: Dict[str, Any] = {
            'lazyllm.span.kind': span.span_kind,
            'lazyllm.span.is_root': span.is_root_span,
            'lazyllm.entity.name': span.name,
            'lazyllm.entity.class': span.component_class,
            'lazyllm.entity.id': span.component_id or '',
            'lazyllm.status': span.status,
        }

        if span.config:
            for k, v in span.config.items():
                attrs[f'lazyllm.entity.config.{k}'] = (
                    _stringify_payload(v) if isinstance(v, (dict, list)) else str(v))

        if span.semantic_type:
            attrs['lazyllm.semantic_type'] = span.semantic_type

        if span.trace_id:
            attrs['lazyllm.request.trace_id'] = span.trace_id
        if span.parent_span_id:
            attrs['lazyllm.request.parent_span_id'] = span.parent_span_id

        if span.capture_payload:
            if span.input is not None:
                attrs['lazyllm.io.input'] = _stringify_payload(span.input)
            if span.output is not None:
                attrs['lazyllm.io.output'] = _stringify_payload(span.output)

        if span.error is not None:
            attrs['lazyllm.error.message'] = str(span.error)

        if span.usage:
            prompt = span.usage.get('prompt_tokens')
            completion = span.usage.get('completion_tokens')
            if prompt is not None and prompt >= 0:
                attrs['gen_ai.usage.input_tokens'] = int(prompt)
            if completion is not None and completion >= 0:
                attrs['gen_ai.usage.output_tokens'] = int(completion)
            if (prompt is not None and prompt >= 0
                    and completion is not None and completion >= 0):
                attrs['gen_ai.usage.total_tokens'] = int(prompt + completion)

        if span.semantic_type == 'llm' and span.config.get('model'):
            attrs['gen_ai.request.model'] = str(span.config['model'])

        if trace is not None and span.is_root_span:
            attrs['lazyllm.trace.name'] = span.name
            if trace.session_id:
                attrs['session.id'] = trace.session_id
            if trace.user_id:
                attrs['user.id'] = trace.user_id
            if trace.request_tags:
                attrs['lazyllm.trace.tags'] = list(trace.request_tags)
            if trace.metadata:
                for k, v in trace.metadata.items():
                    attrs[f'lazyllm.trace.metadata.{k}'] = (
                        _stringify_payload(v) if isinstance(v, (dict, list)) else v)

        if span.output_attrs:
            for k, v in span.output_attrs.items():
                attrs[k] = v

        return attrs

    def finish_span(self, span: Optional[LazySpan]):
        if span is None:
            return
        otel_span = span._otel_span

        active_trace = _current_trace.get()
        trace_matches = active_trace is not None and active_trace.trace_id == span.trace_id
        otel_attrs = self._build_otel_attributes(span, trace=active_trace if trace_matches else None)

        # 1) lazyllm.* attributes
        for k, v in otel_attrs.items():
            otel_span.set_attribute(k, v)

        # 2) Backend-specific attributes
        if self._backend:
            for k, v in self._backend.map_attributes(otel_attrs).items():
                otel_span.set_attribute(k, v)

        # 3) Error handling
        if span.error:
            status_cls, status_code = self._status
            otel_span.set_status(status_cls(status_code.ERROR, str(span.error)))
            otel_span.record_exception(span.error)

        # 4) Close OTel span
        span._otel_span_cm.__exit__(None, None, None)

        # 5) LazyTrace bookkeeping: aggregate outcome; if this span owns the trace,
        #    finalize it and clear the ContextVar so the next request starts fresh.
        if trace_matches:
            active_trace._record_span_end(span)
            if span._owns_lazy_trace:
                active_trace.finish()
                _current_trace.set(None)

    def shutdown(self):
        if self._provider is None:
            return
        try:
            self._provider.force_flush()
            self._provider.shutdown()
        except Exception:
            pass


_runtime = TracingRuntime()


def tracing_available() -> bool:
    return _runtime.available()


def start_span(*, span_kind: str, target: Any, args: tuple, kwargs: Dict[str, Any]):
    return _runtime.start_span(span_kind=span_kind, target=target, args=args, kwargs=kwargs)


def set_span_output(handle, output: Any):
    _runtime.set_output(handle, output)


def set_span_usage(handle, usage: Dict[str, Any]):
    _runtime.set_usage(handle, usage)


def set_span_attributes(handle, attrs: Dict[str, Any]):
    _runtime.set_attributes(handle, attrs)


def set_span_error(handle, exc: Exception):
    _runtime.set_error(handle, exc)


def finish_span(handle):
    _runtime.finish_span(handle)


import functools

def enable_trace(func=None, *args, **kwargs):
    """
    Explicitly enable and configure tracing for a pipeline or function.
    Can be used as a wrapper or a decorator.

    Usage as a wrapper:
        enable_trace(pipeline, input_data, trace_id="123", session_id="abc")

    Usage as a decorator:
        @enable_trace(session_id="abc")
        def my_flow(query):
            ...
    """
    if func is None:
        def decorator(f):
            @functools.wraps(f)
            def wrapper(*inner_args, **inner_kwargs):
                # Trace kwargs are captured in the outer `kwargs` closure
                merged_kwargs = {**inner_kwargs, **kwargs}
                return enable_trace(f, *inner_args, **merged_kwargs)
            return wrapper
        return decorator

    trace_id = kwargs.pop('trace_id', None)
    parent_span_id = kwargs.pop('parent_span_id', None)
    session_id = kwargs.pop('session_id', None)
    user_id = kwargs.pop('user_id', None)
    request_tags = kwargs.pop('request_tags', None)
    module_trace = kwargs.pop('module_trace', None)

    old_ctx = get_trace_context()
    new_ctx_data = old_ctx.to_dict()

    # Clear inherited trace linkage unless explicitly provided, enforcing a new root trace.
    new_ctx_data['trace_id'] = trace_id
    new_ctx_data['parent_span_id'] = parent_span_id
    
    if session_id is not None:
        new_ctx_data['session_id'] = session_id
    if user_id is not None:
        new_ctx_data['user_id'] = user_id
        
    # Request-specific fields should not be inherited implicitly.
    new_ctx_data['request_tags'] = request_tags if request_tags is not None else []
    new_ctx_data['module_trace'] = module_trace if module_trace is not None else None
    
    new_ctx_data['enabled'] = True

    new_ctx = LazyTraceContext.from_dict(new_ctx_data)
    set_trace_context(new_ctx)

    is_lazyllm_component = hasattr(func, '_module_id') or hasattr(func, '_flow_id')

    span = None
    if not is_lazyllm_component:
        span = start_span(span_kind='callable', target=func, args=args, kwargs=kwargs)

    try:
        result = func(*args, **kwargs)
        if span:
            set_span_output(span, result)
        return result
    except Exception as e:
        if span:
            set_span_error(span, e)
        raise
    finally:
        if span:
            finish_span(span)
        set_trace_context(old_ctx)
