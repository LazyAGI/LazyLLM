import os
import lazyllm


def light_reduce(cls):
    def rebuild(mid): return cls()._set_mid(mid)

    def _impl(self):
        if os.getenv('LAZYLLM_ON_CLOUDPICKLE', False) == 'ON':
            assert self._get_deploy_tasks.flag, f'{cls.__name__[1:-4]} shoule be deployed before used'
            return rebuild, (self._module_id,)
        return super(cls, self).__reduce__()
    cls.__reduce__ = _impl
    return cls


def module_tool_light_reduce(cls):
    def rebuild(module_id, orig_func=None, cls_to_use=None):
        if cls_to_use is not None and isinstance(cls_to_use, type):
            return cls_to_use()._set_mid(module_id)
        if orig_func is not None and callable(orig_func) and not isinstance(orig_func, type):
            from lazyllm.tools.agent.toolsManager import register
            register('tool')(orig_func)
            cls_to_use = getattr(lazyllm.tool, orig_func.__name__)
            return cls_to_use()._set_mid(module_id)

    def _get_orig_apply_func(apply_method):
        func = getattr(apply_method, '__func__', apply_method)
        if not callable(func) or not getattr(func, '__closure__', None):
            return None
        for cell in func.__closure__:
            try:
                c = cell.cell_contents
                if callable(c) and not isinstance(c, type) and getattr(c, '__name__', None):
                    return c
            except ValueError:
                pass
        return None

    def _impl(self):
        if os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON':
            orig = _get_orig_apply_func(self.apply)
            cls_to_use = self.__class__ if orig is None else None
            return (rebuild, (self._module_id, orig, cls_to_use))
        return super(cls, self).__reduce__()
    cls.__reduce__ = _impl
    return cls
