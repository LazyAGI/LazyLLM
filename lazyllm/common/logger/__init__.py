import sys


class _LOGWrapper():
    def __getattr__(self, __key):
        from .logger import LOG
        __key = __key.lower()
        return getattr(LOG, __key)

    def close(self):
        # Closing an never-initialized logger must stay a no-op: importing the
        # implementation here (e.g. from the atexit cleanup handler during
        # interpreter shutdown) would lazily create log sinks, and in the
        # default 'merge' log file mode an enqueue SimpleQueue whose semaphore
        # spawns a multiprocessing resource_tracker subprocess. When the
        # resource_tracker itself imports lazyllm on exit, this repeats and
        # forms a process loop (see issue #1326).
        logger_mod = sys.modules.get(__package__ + '.logger')
        if logger_mod is not None:
            logger_mod.LOG.close()

LOG = _LOGWrapper()

__all__ = [
    'LOG',
]
