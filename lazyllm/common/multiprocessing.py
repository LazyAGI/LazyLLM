import multiprocessing
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor as PPE
import functools
import time
import atexit
from .utils import load_obj, dump_obj

@contextmanager
def _ctx(method='spawn'):
    m = multiprocessing.get_start_method()
    if m != method:
        multiprocessing.set_start_method(method, force=True)
    yield
    if m != method:
        multiprocessing.set_start_method(m, force=True)


class SpawnProcess(multiprocessing.Process):
    def start(self):
        with _ctx('spawn'):
            return super().start()


class ForkProcess(multiprocessing.Process):
    def __init__(self, group=None, target=None, name=None, args=(),
                 kwargs=None, *, daemon=None, sync=True):
        super().__init__(group, ForkProcess.work(target, sync), name, args, kwargs or {}, daemon=daemon)

    @staticmethod
    def work(f, sync):
        def impl(*args, **kw):
            try:
                f(*args, **kw)
                if not sync:
                    while True: time.sleep(1)
            finally:
                atexit._run_exitfuncs()
        return impl

    def start(self):
        with _ctx('fork'):
            return super().start()


def _worker(f):
    return load_obj(f)()

class ProcessPoolExecutor(PPE):
    def submit(self, fn, /, *args, **kwargs):
        f = dump_obj(functools.partial(fn, *args, **kwargs))
        return super(__class__, self).submit(_worker, f)
