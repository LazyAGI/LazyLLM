from typing import Any

from ...common import LOG, globals
from ...configs import config
from ...hook import LazyLLMHook, register_builtin_hook_provider
from .configs import resolve_default_module_trace, resolve_runtime_module_trace_disabled
from .output_attrs import (
    collect_trace_output_attrs,
    discard_pending_switch_ifs_stack,
    install_post_process_probe,
    remove_post_process_probe,
)
from .runtime import finish_span, set_span_attributes, set_span_error, set_span_output, set_span_usage, start_span


def _unwrap_trace_subject(obj: Any) -> Any:
    cls = getattr(obj, '__class__', type(obj))
    name, mod = cls.__name__, getattr(cls, '__module__', '')
    if name == '_FuncWrap' and hasattr(obj, '_f'):
        inner = getattr(obj, '_f', None)
        return _unwrap_trace_subject(inner) if inner is not None else obj
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
        discard_pending_switch_ifs_stack(self._obj)
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
]
