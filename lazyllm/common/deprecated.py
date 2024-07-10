from .logger import LOG
import functools

def deprecated(func):
    if isinstance(func, type):
        orig_init = func.__init__

        @functools.wraps(orig_init)
        def new_init(self, *args, **kwargs):
            LOG.warning(f"Class {func.__name__} is deprecated and will be removed in a future version.")
            orig_init(self, *args, **kwargs)

        func.__init__ = new_init
        return func
    else:
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            LOG.warning(f"Function {func.__name__} is deprecated and will be removed in a future version.")
            return func(*args, **kwargs)
        return new_func
