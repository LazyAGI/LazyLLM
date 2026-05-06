'''Backward-compatible re-export of StreamCallHelper.

StreamCallHelper has been moved to lazyllm.module (servermodule.py) to avoid
circular imports between the tools and module layers. This shim keeps existing
imports working.
'''
from lazyllm.module import StreamCallHelper  # noqa: F401

__all__ = ['StreamCallHelper']
