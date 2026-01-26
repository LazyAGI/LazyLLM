import importlib
import lazyllm
from .base_data import DataOperatorRegistry
from .operators import demo_ops  # noqa: F401

def __getattr__(name):
    if name == 'pipelines':
        return importlib.import_module('.pipelines', __package__)
    if name in lazyllm.data:
        return lazyllm.data[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

__all__ = ['DataOperatorRegistry']
