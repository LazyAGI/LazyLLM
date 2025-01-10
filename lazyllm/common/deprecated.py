from .logger import LOG
import functools

def deprecated(func_or_msg=None):
    def impl(func):
        msg = f'{func.__name__} is deprecated and will be removed in a future version.'
        if isinstance(func_or_msg, str): msg += f' Use `{func_or_msg}` instead'
        if isinstance(func, type):
            orig_init = func.__init__

            @functools.wraps(orig_init)
            def new_init(self, *args, **kwargs):
                LOG.warning(f'Class {msg}')
                orig_init(self, *args, **kwargs)

            func.__init__ = new_init
            return func
        else:
            @functools.wraps(func)
            def new_func(*args, **kwargs):
                LOG.warning(f'Function {msg}')
                return func(*args, **kwargs)
            return new_func

    if isinstance(func_or_msg, str):
        return impl
    else:
        return impl(func_or_msg)
