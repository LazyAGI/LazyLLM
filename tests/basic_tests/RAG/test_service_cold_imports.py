import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('statement', [
    'from lazyllm.tools.rag.doc_service import DocServer',
    'from lazyllm.tools.rag.doc_service.base import DOC_NODE_GROUP_STATUS_TABLE_INFO',
    'from lazyllm.tools.rag.parsing_service import DocumentProcessor',
    'from lazyllm.tools.rag.parsing_service import DocumentProcessorWorker',
    'from lazyllm.tools.rag.parsing_service import *',
    'from lazyllm import DocServer, Document',
])
def test_service_import_in_fresh_process(statement, tmp_path):
    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ, PYTHONPATH=str(root), LAZYLLM_HOME=str(tmp_path), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', '-c', statement], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_task_types_do_not_import_service_implementations(tmp_path):
    test_service_import_in_fresh_process(
        'import sys; from lazyllm.tools.rag.parsing_service.base import TaskType; '
        'assert "lazyllm.tools.rag.parsing_service.worker" not in sys.modules; '
        'assert "lazyllm.tools.rag.parsing_service.server" not in sys.modules', tmp_path)
