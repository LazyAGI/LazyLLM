# Copyright (c) 2026 LazyAGI. All rights reserved.
import importlib.util
import sys
import types
from pathlib import Path


_FS_DOCS_LOADED = False


def load_fs_docs_only(documented_method):
    global _FS_DOCS_LOADED
    if _FS_DOCS_LOADED or getattr(documented_method, '__doc__', None):
        return
    docs_tools_pkg = types.ModuleType('lazyllm.docs.tools')
    docs_tools_pkg.__path__ = [
        str(Path(__file__).parents[3] / 'lazyllm' / 'docs' / 'tools')
    ]
    sys.modules.setdefault('lazyllm.docs.tools', docs_tools_pkg)
    spec = importlib.util.spec_from_file_location(
        'lazyllm.docs.tools.tool_fs_test',
        Path(__file__).parents[3] / 'lazyllm' / 'docs' / 'tools' / 'tool_fs.py',
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    _FS_DOCS_LOADED = True
