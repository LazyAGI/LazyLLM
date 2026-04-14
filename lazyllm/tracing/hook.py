from ..common import LOG, globals
from ..configs import config
from ..hook import LazyLLMHook, register_builtin_hook_provider
from .configs import resolve_default_module_trace
from .runtime import finish_span, set_span_attributes, set_span_error, set_span_output, set_span_usage, start_span


class LazyTracingHook(LazyLLMHook):
    __hook_priority__ = 0
    __hook_error_mode__ = 'raise'

    def __init__(self, obj):
        self._obj = obj
        self._span_handle = None

    @property
    def _span_kind(self):
        return 'flow' if hasattr(self._obj, '_flow_id') else 'module'

    def pre_hook(self, *args, **kwargs):
        self._span_handle = start_span(span_kind=self._span_kind, target=self._obj, args=args, kwargs=kwargs)

    def post_hook(self, output):
        if self._span_handle is None:
            return
        set_span_output(self._span_handle, output)
        module_id = getattr(self._obj, '_module_id', None)
        if module_id:
            usage = globals['usage'].get(module_id)
            if usage:
                set_span_usage(self._span_handle, usage)
        if hasattr(self._obj, '__trace_output_attrs__'):
            try:
                output_attrs = self._obj.__trace_output_attrs__(output)
                if output_attrs:
                    set_span_attributes(self._span_handle, output_attrs)
            except Exception as e:
                LOG.debug(f'__trace_output_attrs__ failed for {self._obj.__class__.__name__}: {e}')

    def on_error(self, exc):
        if self._span_handle is None:
            return
        set_span_error(self._span_handle, exc)

    def finalize(self):
        if self._span_handle is None:
            return
        finish_span(self._span_handle)
        self._span_handle = None


def resolve_tracing_hooks(obj):
    trace_cfg = globals.get('trace', {})
    trace_enabled = trace_cfg.get('enabled')
    if trace_enabled is None:
        trace_enabled = config['trace_enabled']
    if not trace_enabled or trace_cfg.get('sampled') is False:
        return []
    if hasattr(obj, '_module_id'):
        if not resolve_default_module_trace(
            module_name=getattr(obj, 'name', None) or getattr(obj, '_module_name', None),
            module_class=obj.__class__,
        ):
            return []
    return [LazyTracingHook]


register_builtin_hook_provider(resolve_tracing_hooks)


__all__ = [
    'LazyTracingHook',
    'resolve_tracing_hooks',
]
