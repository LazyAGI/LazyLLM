from .logger import LOG
import functools
from typing import overload, Callable, Any

@overload
def deprecated(msg: str) -> Callable[[Callable], Callable]:
    ...

@overload
def deprecated(func: Callable) -> Callable[[Any], Any]:
    ...

@overload
def deprecated(flag: bool, msg: str) -> Callable[[Any], Any]:
    ...

def deprecated(func_or_msg=None, item_name=''):
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
    elif isinstance(func_or_msg, bool):
        if func_or_msg: LOG.warning(f'{item_name} is deprecated')
    else:
        return impl(func_or_msg)
