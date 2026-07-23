"""Thin wrapper around scikit-build-core that bundles pyproject.toml.

lazyllm.thirdparty reads pyproject.toml at runtime to map missing optional
imports to pip-install suggestions.  scikit-build-core does not copy it into
the wheel automatically the way Poetry's ``include`` directive did, so we
copy it here before the wheel is assembled.

Workaround for https://github.com/LazyAGI/LazyLLM/issues/1245
"""

import shutil
from pathlib import Path

from scikit_build_core.build import *  # noqa: F401,F403

_root = Path(__file__).parent
_src = _root / "pyproject.toml"
_dst = _root / "lazyllm" / "pyproject.toml"
if _src.is_file() and not _dst.is_file():
    shutil.copy2(_src, _dst)
