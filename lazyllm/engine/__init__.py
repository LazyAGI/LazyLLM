from importlib import import_module
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .engine import Engine
    from .lightengine import LightEngine
    from .node_meta_hook import NodeMetaHook

def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"Module 'engine' has no attribute '{name}'")

    for obj_name, module_name in [
        ('Engine', '.engine'),
        ('LightEngine', '.lightengine'),
        ('NodeMetaHook', '.node_meta_hook'),
    ]:
        globals()[obj_name] = import_module(module_name, package=__package__).__getattr__(obj_name)
    return globals()[name]

__all__ = [
    'Engine',
    'LightEngine',
    'NodeMetaHook',
]
