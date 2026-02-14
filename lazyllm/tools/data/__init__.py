import importlib
import lazyllm
from .base_data import LazyLLMDataBase, data_register
from .operators import demo_ops  # noqa: F401
from .operators import cot_ops  # noqa: F401
from .operators import math_ops  # noqa: F401
from .operators import pdf_ops  # noqa: F401
from .operators import enQa_ops  # noqa: F401

def __getattr__(name):
    if name == 'pipelines':
        return importlib.import_module('.pipelines', __package__)
    if name in lazyllm.data:
        return lazyllm.data[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

__all__ = ['LazyLLMDataBase', 'data_register']
