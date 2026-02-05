import os
import tempfile
import pytest

from lazyllm import config
from lazyllm.tools.sandbox import LocalSandbox, SandboxFusion
from lazyllm.tools.agent import code_interpreter


class TestLocalSandbox:
    def test_execute_basic(self):
        sandbox = LocalSandbox(timeout=5)
        result = sandbox(code="print('hello')")
        assert isinstance(result, dict)
        assert result['returncode'] == 0
        assert result['stdout'].strip() == 'hello'

    def test_execute_with_input_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, 'input.txt')
            with open(input_path, 'w', encoding='utf-8') as f:
                f.write('line1\nline2\n')
            code = (
                'import os\n'
                'print(os.path.exists("input.txt"))\n'
            )
            sandbox = LocalSandbox(timeout=5)
            result = sandbox(code=code, input_files=[input_path])
            assert result['stdout'].strip() == 'True'


class TestCodeInterpreter:
    def test_local_code_interpreter(self):
        prev_env = os.environ.get('LAZYLLM_SANDBOX_TYPE')
        try:
            os.environ['LAZYLLM_SANDBOX_TYPE'] = 'local'
            config.refresh()
            result = code_interpreter("print('ok')")
            assert isinstance(result, dict)
            assert result['stdout'].strip() == 'ok'
        finally:
            if prev_env is None:
                os.environ.pop('LAZYLLM_SANDBOX_TYPE', None)
            else:
                os.environ['LAZYLLM_SANDBOX_TYPE'] = prev_env
            config.refresh()


class TestSandboxFusion:
    @pytest.mark.skipif(
        not os.getenv('LAZYLLM_SANDBOX_FUSION_BASE_URL'),
        reason='LAZYLLM_SANDBOX_FUSION_BASE_URL is not set'
    )
    def test_sandbox_fusion_execute(self):
        try:
            sandbox = SandboxFusion(run_timeout=5)
            result = sandbox(code="print('fusion')", language='python')
            assert isinstance(result, dict)
            assert result.get('run_result', {}).get('stdout', '').strip() == 'fusion'
        finally:
            os.environ.pop('LAZYLLM_SANDBOX_FUSION_BASE_URL', None)
            config.refresh()
