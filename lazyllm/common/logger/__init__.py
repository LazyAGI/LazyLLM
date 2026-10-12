class _LOGWrapper():
    def __getattr__(self, __key):
        from .logger import LOG
        __key = __key.lower()
        return getattr(LOG, __key)

    def close(self):
        import sys
        module = sys.modules.get(f'{__package__}.logger')
        if module is not None:
            module.LOG.close()

LOG = _LOGWrapper()

__all__ = [
    'LOG',
]
