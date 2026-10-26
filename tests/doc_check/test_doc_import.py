import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('language', ['ENGLISH', 'CHINESE'])
def test_docs_import_without_optional_tracing(language):
    script = '''
import sys
from importlib.abc import MetaPathFinder

class NoOptionalTracing(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'opentelemetry' or fullname.startswith('opentelemetry.'):
            raise ModuleNotFoundError("No module named 'opentelemetry'", name='opentelemetry')

sys.meta_path.insert(0, NoOptionalTracing())
import lazyllm
from lazyllm.module.llms.onlinemodule.supplier.glm import GLMTTS, GLMText2Video
assert GLMTTS.__doc__
assert GLMText2Video.__doc__
assert lazyllm.tracing.resolve_tracing_hooks.__doc__
'''
    env = dict(os.environ, LAZYLLM_INIT_DOC='True', LAZYLLM_LANGUAGE=language)
    result = subprocess.run(
        [sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[2],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
