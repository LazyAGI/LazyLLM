import importlib
import lazyllm
from .base_data import LazyLLMDataBase, data_register
from .operators import demo_ops  # noqa: F401
from .operators import refine_op  # noqa: F401
from .operators import token_chunker  # noqa: F401
from .operators import filter_op  # noqa: F401

def __getattr__(name):
    if name == 'pipelines':
        return importlib.import_module('.pipelines', __package__)
    if name in lazyllm.data:
        return lazyllm.data[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

__all__ = ['LazyLLMDataBase', 'data_register']
