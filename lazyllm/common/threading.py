import threading
from queue import Queue
import functools
from contextvars import copy_context
from .globals import globals, locals
from concurrent.futures import ThreadPoolExecutor as TPE


def _validate_local_scope(local_scope):
    if local_scope not in ('isolated', 'inherit'):
        raise ValueError('local_scope must be isolated or inherit')


def _context_call(fn, local_scope):
    sid = globals._sid
    local_sid = locals._sid if local_scope == 'inherit' else None
    context = copy_context()

    def impl(*args, **kwargs):
        globals._init_sid(sid)
        with locals._scope(local_sid):
            return fn(*args, **kwargs)

    return functools.partial(context.run, impl)


class Thread(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, *, prehook=None, daemon=None, local_scope='isolated'):
        _validate_local_scope(local_scope)
        self.q = Queue()
        prehook = list(prehook) if isinstance(prehook, (tuple, list)) else [prehook] if prehook else []
        self._call = _context_call(self._run_target, local_scope)
        super().__init__(group, self.work, name, (prehook, target, args), kwargs, daemon=daemon)

    @staticmethod
    def _run_target(prehook, target, args, **kw):
        for hook in prehook:
            hook()
        if target is not None:
            return target(*args, **kw)

    def work(self, prehook, target, args, **kw):
        try:
            result = self._call(prehook, target, args, **kw)
        except BaseException as error:  # noqa: B036 - get_result re-raises on the consuming thread.
            self.q.put((False, error))
        else:
            self.q.put((True, result))

    def get_result(self):
        success, result = self.q.get()
        if not success:
            raise result
        return result


class ThreadPoolExecutor(TPE):
    def __init__(self, *args, local_scope='isolated', **kwargs):
        _validate_local_scope(local_scope)
        self._local_scope = local_scope
        super().__init__(*args, **kwargs)

    def submit(self, fn, /, *args, **kwargs):
        return super().submit(_context_call(fn, self._local_scope), *args, **kwargs)
