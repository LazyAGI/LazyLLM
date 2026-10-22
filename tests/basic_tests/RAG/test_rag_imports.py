import subprocess
import sys

import pytest


@pytest.mark.parametrize('statement', [
    'from lazyllm import DocServer',
    'from lazyllm.tools.rag.parsing_service import DocumentProcessor, DocumentProcessorWorker, _Processor',
])
def test_service_entry_points_import_in_fresh_process(statement):
    result = subprocess.run([sys.executable, '-c', statement], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
