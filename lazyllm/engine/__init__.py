from .engine import Engine
from .lightengine import LightEngine


Engine.set_default(LightEngine)


__all__ = [
    'Engine',
    'LightEngine',
]
