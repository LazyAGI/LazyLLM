import threading
from queue import Queue
import functools
from .globals import globals
from concurrent.futures import ThreadPoolExecutor as TPE

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


class ThreadPoolExecutor(TPE):
    def submit(self, fn, /, *args, **kwargs):
        def impl(sid, *a, **kw):
            globals._init_sid(sid)
            return fn(*a, **kw)

        return super(__class__, self).submit(functools.partial(impl, globals._sid), *args, **kwargs)
