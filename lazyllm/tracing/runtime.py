import atexit
import contextvars
import json
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from lazyllm.common import LOG, globals
from lazyllm.configs import config
from lazyllm.thirdparty import opentelemetry
from .backends import get_tracing_backend


_TRACE_SERVICE_NAME = 'lazyllm'
_in_reconstructed_thread = contextvars.ContextVar('_lazyllm_tracing_reconstructed', default=False)
_TRACE_CONTEXT_DEFAULTS = {
    'enabled': None,
    'trace_id': None,
    'session_id': None,
    'user_id': None,
    'request_tags': None,
    'sampled': None,
    'parent_span_id': None,
    'debug_capture_payload': None,
}


class TracingSetupError(RuntimeError):
    pass


def _normalize_tags(tags: Any) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        return [tags]
    if isinstance(tags, Iterable):
        return [str(tag) for tag in tags if tag is not None]
    return [str(tags)]


def _normalize_trace_context(trace: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    data = dict(trace) if isinstance(trace, dict) else {}
    for key, default in _TRACE_CONTEXT_DEFAULTS.items():
        data.setdefault(key, default)
    data['request_tags'] = _normalize_tags(data.get('request_tags'))
    return data


def get_trace_context() -> Dict[str, Any]:
    return _normalize_trace_context(globals.get('trace', {}))


def set_trace_context(trace: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = _normalize_trace_context(trace)
    globals['trace'] = normalized
    return normalized


def _capture_payload_enabled(trace_ctx: Dict[str, Any]) -> bool:
    debug_capture_payload = trace_ctx.get('debug_capture_payload')
    if debug_capture_payload is not None:
        return bool(debug_capture_payload)
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


@dataclass
class TraceSpanHandle:
    span: Any
    span_cm: Any
    is_root_span: bool


class _TracingRuntime:
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

    def _trace_enabled(self, trace_ctx: Dict[str, Any]) -> bool:
        if trace_ctx.get('enabled') is not None:
            return bool(trace_ctx['enabled']) and trace_ctx.get('sampled') is not False
        if trace_ctx.get('sampled') is False:
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

    def _base_attributes(
        self,
        *,
        span_kind: str,
        span_name: str,
        is_root_span: bool,
        trace_ctx: Dict[str, Any],
        target: Any,
        capture_payload: bool,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        attrs = {
            'lazyllm.span.kind': span_kind,
            'lazyllm.entity.name': span_name,
            'lazyllm.entity.class': target.__class__.__name__,
            'lazyllm.entity.id': self._target_id(target, span_kind) or '',
            'lazyllm.status': 'ok',
        }

        trace_kwargs = {}
        try:
            if hasattr(target, '__trace_kwargs__'):
                trace_kwargs = target.__trace_kwargs__
                if isinstance(trace_kwargs, dict):
                    for k, v in trace_kwargs.items():
                        attrs[f'lazyllm.entity.config.{k}'] = _stringify_payload(v) if isinstance(v, (dict, list)) else str(v)
                    attrs.update(self._backend.metadata_attributes(trace_kwargs))
        except Exception:
            pass

        semantic_type = getattr(target, '__semantic_type__', None)
        if semantic_type is None and span_kind == 'flow':
            semantic_type = 'workflow_control'
        if semantic_type:
            attrs['lazyllm.semantic_type'] = semantic_type
        attrs.update(self._backend.observation_type_attributes(
            span_kind=span_kind, semantic_type=semantic_type, trace_kwargs=trace_kwargs,
        ))

        if trace_ctx.get('trace_id'):
            attrs['lazyllm.request.trace_id'] = str(trace_ctx['trace_id'])
        if trace_ctx.get('parent_span_id'):
            attrs['lazyllm.request.parent_span_id'] = str(trace_ctx['parent_span_id'])
        attrs.update(self._backend.context_attributes(trace_ctx, is_root_span=is_root_span))
        for key, value in self._backend.input_attributes(
            args, kwargs, capture_payload=capture_payload, is_root_span=is_root_span
        ).items():
            attrs[key] = _stringify_payload(value) if isinstance(value, dict) else value
        return attrs

    def start_span(
        self,
        *,
        span_kind: str,
        target: Any,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Optional[TraceSpanHandle]:
        trace_ctx = get_trace_context()
        if not self._trace_enabled(trace_ctx):
            return None
        if not self._ensure_runtime():
            return None

        capture_payload = _capture_payload_enabled(trace_ctx)
        span_name = self._target_name(target, span_kind)
        current = self._trace_api.get_current_span()
        is_root_span = not current.get_span_context().is_valid

        parent_context = None
        if is_root_span and trace_ctx.get('trace_id') and trace_ctx.get('parent_span_id'):
            try:
                parent_sc = opentelemetry.trace.SpanContext(
                    trace_id=int(trace_ctx['trace_id'], 16),
                    span_id=int(trace_ctx['parent_span_id'], 16),
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

        attributes = self._base_attributes(
            span_kind=span_kind,
            span_name=span_name,
            is_root_span=is_root_span,
            trace_ctx=trace_ctx,
            target=target,
            capture_payload=capture_payload,
            args=args,
            kwargs=kwargs,
        )

        span_cm = self._tracer.start_as_current_span(
            span_name, attributes=attributes, context=parent_context
        )
        span = span_cm.__enter__()
        span_context = span.get_span_context()
        if not _in_reconstructed_thread.get(False):
            trace_ctx['trace_id'] = f'{span_context.trace_id:032x}'
            trace_ctx['parent_span_id'] = f'{span_context.span_id:016x}'
            set_trace_context(trace_ctx)
        if is_root_span:
            self._backend.set_root_span_name(span, span_name)
        return TraceSpanHandle(span=span, span_cm=span_cm, is_root_span=is_root_span)

    def set_output(self, handle: Optional[TraceSpanHandle], output: Any):
        if handle is None:
            return
        span = handle.span
        trace_ctx = get_trace_context()
        capture_payload = _capture_payload_enabled(trace_ctx)
        span.set_attribute('lazyllm.status', 'ok')
        if not capture_payload:
            return
        text = _stringify_payload(output)
        for key, value in self._backend.output_attributes(text, is_root_span=handle.is_root_span).items():
            span.set_attribute(key, value)

    def set_usage(self, handle: Optional[TraceSpanHandle], usage: Dict[str, Any]):
        if handle is None:
            return
        span = handle.span
        prompt = usage.get('prompt_tokens')
        completion = usage.get('completion_tokens')
        if prompt is not None and prompt >= 0:
            span.set_attribute('gen_ai.usage.input_tokens', int(prompt))
        if completion is not None and completion >= 0:
            span.set_attribute('gen_ai.usage.output_tokens', int(completion))
        if prompt is not None and prompt >= 0 and completion is not None and completion >= 0:
            span.set_attribute('gen_ai.usage.total_tokens', int(prompt + completion))
        if self._backend:
            for key, value in self._backend.usage_attributes(usage).items():
                span.set_attribute(key, value)

    def set_attributes(self, handle: Optional[TraceSpanHandle], attrs: Dict[str, Any]):
        if handle is None or not attrs:
            return
        span = handle.span
        for key, value in attrs.items():
            span.set_attribute(key, value)

    def set_error(self, handle: Optional[TraceSpanHandle], exc: Exception):
        if handle is None:
            return
        span = handle.span
        status_cls, status_code = self._status
        span.set_status(status_cls(status_code.ERROR, str(exc)))
        span.set_attribute('lazyllm.status', 'error')
        for key, value in self._backend.error_attributes(exc).items():
            span.set_attribute(key, value)
        span.record_exception(exc)

    def finish_span(self, handle: Optional[TraceSpanHandle]):
        if handle is None:
            return
        handle.span_cm.__exit__(None, None, None)

    def shutdown(self):
        if self._provider is None:
            return
        try:
            self._provider.force_flush()
            self._provider.shutdown()
        except Exception:
            pass


_runtime = _TracingRuntime()


def tracing_available() -> bool:
    return _runtime.available()


def start_span(*, span_kind: str, target: Any, args: tuple[Any, ...], kwargs: Dict[str, Any]):
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
