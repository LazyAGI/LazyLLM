from importlib import import_module
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .engine import Engine
    from .lightengine import LightEngine
    from .node_meta_hook import NodeMetaHook

def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"Module 'engine' has no attribute '{name}'")

    if name == 'Engine':
        mod = import_module('.engine', package=__package__)
        mod.set_default(LightEngine)
    elif name == 'LightEngine':
        mod = import_module('.lightengine', package=__package__)
    elif name == 'NodeMetaHook':
        mod = import_module('.node_meta_hook', package=__package__)

    globals()[name] = value = getattr(mod, name)
    return value

__all__ = [
    'Engine',
    'LightEngine',
    'NodeMetaHook',
]
