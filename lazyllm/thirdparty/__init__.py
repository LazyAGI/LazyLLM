import importlib

class PackageWrapper(object):
    def __init__(self, key, package=None) -> None:
        self._Wrapper__key = key
        self._Wrapper__package = package

    def __getattribute__(self, __name):
        if __name in ('_Wrapper__key', '_Wrapper__package'):
            return super(__class__, self).__getattribute__(__name)
        try:
            return getattr(importlib.import_module(
                self._Wrapper__key, package=self._Wrapper__package), __name)
        except (ImportError, ModuleNotFoundError):
            raise ImportError(f'Cannot import module {self._Wrapper__key}, please install it first')


modules = ['cloudpickle', 'transformers', 'peft', 'deepspeed', 'gradio', 'llama_index']
for m in modules:
    vars()[m] = PackageWrapper(m)
