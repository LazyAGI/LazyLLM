from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

from ...common import LOG, globals, HandledException
from ...configs import config
from ...hook import LazyLLMHook, prepare_hooks, register_builtin_hook_provider, run_hooks
from .configs import resolve_default_module_trace, resolve_runtime_module_trace_disabled
from .output_attrs import collect_trace_output_attrs, install_post_process_probe, remove_post_process_probe
from .runtime import finish_span, set_span_attributes, set_span_error, set_span_output, set_span_usage, start_span


@contextmanager
def traced_hook_execution(
    obj: Any,
    *hook_args: Any,
    map_exception: Optional[Callable[[Exception], Exception]] = None,
    **hook_kwargs: Any,
):
    '''Wrap a call with LazyLLM hooks (including tracing).

    Usage::

        with traced_hook_execution(self, *args, **kw) as outcome:
            outcome['r'] = ...  # set exactly on success; value is passed to ``post_hook``

    ``map_exception`` is optional (e.g. modules mapping to ``ModuleExecutionError`` before ``on_error``).
    '''
    hook_objs = prepare_hooks(obj, list(getattr(obj, '_hooks', []) or []), *hook_args, **hook_kwargs)
    outcome: Dict[str, Any] = {}
    try:
        yield outcome
    except HandledException as e:
        LOG.error(f'`{type(obj).__name__}` raised {type(e).__name__}: {e}')
        try:
            run_hooks(hook_objs, 'on_error', e)
        except Exception:
            LOG.warning('Hook on_error phase failed', exc_info=True)
        raise
    except Exception as e:
        err = map_exception(e) if map_exception else e
        nm = getattr(obj, 'name', None)
        LOG.error(
            f'Error in `{type(obj).__name__}`' + (f' name={nm!r}' if nm else '') + f': {err}'
        )
        try:
            run_hooks(hook_objs, 'on_error', err)
        except Exception:
            LOG.warning('Hook on_error phase failed', exc_info=True)
        if map_exception:
            raise err from None
        raise
    else:
        if 'r' not in outcome:
            raise RuntimeError('traced_hook_execution: expected outcome["r"] to be set on success')
        run_hooks(hook_objs, 'post_hook', outcome['r'])
    finally:
        try:
            run_hooks(hook_objs, 'finalize')
        except Exception:
            LOG.warning('Hook finalize phase failed', exc_info=True)


def _unwrap_trace_subject(obj: Any) -> Any:
    '''Resolve ``_FuncWrap`` / ``flow._BindPipelineAdapter`` / ``Bind`` to the inner callable for spans and hook policy.'''
    cls = getattr(obj, '__class__', type(obj))
    name, mod = cls.__name__, getattr(cls, '__module__', '')
    if name == '_FuncWrap' and hasattr(obj, '_f'):
        inner = getattr(obj, '_f', None)
        return _unwrap_trace_subject(inner) if inner is not None else obj
    if name == '_BindPipelineAdapter' and hasattr(obj, '_b'):
        return _unwrap_trace_subject(getattr(obj, '_b'))
    if name == 'Bind' and 'bind' in mod and hasattr(obj, '_f'):
        inner = getattr(obj, '_f', None)
        if inner is not None and callable(inner):
            return inner
    return obj


class LazyTracingHook(LazyLLMHook):
    __hook_priority__ = 0
    __hook_error_mode__ = 'raise'

    def __init__(self, obj):
        self._obj = obj
        self._span = None

    def _trace_target(self) -> Any:
        return _unwrap_trace_subject(self._obj)

    @property
    def _span_kind(self):
        t = self._trace_target()
        if hasattr(t, '_flow_id'):
            return 'flow'
        if hasattr(t, '_module_id'):
            return 'module'
        return 'callable'

    def pre_hook(self, *args, **kwargs):
        trace_cfg = globals.get('trace', {})
        trace_enabled = trace_cfg.get('enabled')
        if trace_enabled is None:
            trace_enabled = config['trace_enabled']
        if not trace_enabled or trace_cfg.get('sampled') is False:
            return

        t = self._trace_target()
        # Runtime override is single-directional (disable-only): registry/default decides
        # whether tracing is on; globals['trace']['module_trace'] can only turn it off.
        if hasattr(t, '_module_id') and resolve_runtime_module_trace_disabled(
            trace_cfg.get('module_trace'),
            module_name=getattr(t, 'name', None) or getattr(t, '_module_name', None),
            module_class=t.__class__,
        ):
            return

        self._span = start_span(span_kind=self._span_kind, target=t, args=args, kwargs=kwargs)
        if self._span is not None:
            install_post_process_probe(self._obj)

    def post_hook(self, output):
        if self._span is None:
            return
        set_span_output(self._span, output)
        module_id = getattr(self._trace_target(), '_module_id', None)
        if module_id:
            usage = (globals.get('usage') or {}).get(module_id)
            if usage:
                set_span_usage(self._span, usage)
        try:
            output_attrs = collect_trace_output_attrs(self._obj, output)
            if output_attrs:
                set_span_attributes(self._span, output_attrs)
        except Exception as e:
            LOG.warning(f'collect_trace_output_attrs failed for {self._obj.__class__.__name__}: {e}')

    def on_error(self, exc):
        if self._span is None:
            return
        set_span_error(self._span, exc)

    def finalize(self):
        remove_post_process_probe(self._obj)
        if self._span is None:
            return
        finish_span(self._span)
        self._span = None


def resolve_tracing_hooks(obj):
    if not config['trace_enabled']:
        return []
    subject = _unwrap_trace_subject(obj)
    if hasattr(subject, '_module_id'):
        if not resolve_default_module_trace(
            module_name=getattr(subject, 'name', None) or getattr(subject, '_module_name', None),
            module_class=subject.__class__,
        ):
            return []
    return [LazyTracingHook]


register_builtin_hook_provider(resolve_tracing_hooks)


__all__ = [
    'LazyTracingHook',
    'resolve_tracing_hooks',
    'traced_hook_execution',
]
