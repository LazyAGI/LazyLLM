import threading
from queue import Queue
import functools
from .globals import globals
import contextlib

def _sid_setter(sid):
    globals._init_sid(sid)

class Thread(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, *, prehook=None, daemon=None):
        self.q = Queue()
        if not isinstance(prehook, (tuple, list)): prehook = [prehook] if prehook else []
        prehook.insert(0, functools.partial(_sid_setter, sid=globals._sid))
        super().__init__(group, self.work, name, (prehook, target, args), kwargs, daemon=daemon)

    def work(self, prehook, target, args, **kw):
        [p() for p in prehook]
        try:
            r = target(*args, **kw)
        except Exception as e:
            self.q.put(e)
        else:
            self.q.put(r)

    def get_result(self):
        r = self.q.get()
        if isinstance(r, Exception):
            raise r
        return r


def _new(cls, *args, **kwargs):
    if cls is threading.Thread:
        # 返回子类实例
        return super(threading.Thread, cls).__new__(Thread)
    return super(threading.Thread, cls).__new__(cls)

def _dummy_new(cls, *args, **kw):
    return super(threading.Thread, cls).__new__(cls)


@contextlib.contextmanager
def wrap_threading():
    threading.Thread.__new__ = _new
    yield
    threading.Thread.__new__ = _dummy_new
