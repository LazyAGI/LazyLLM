class _LOGWrapper():
    def __getattr__(self, __key):
        from .logger import LOG
        __key = __key.lower()
        return getattr(LOG, __key)

LOG = _LOGWrapper()

__all__ = [
    'LOG',
]
