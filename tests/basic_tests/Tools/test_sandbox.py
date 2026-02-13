import os
import tempfile
import pytest

import lazyllm
from lazyllm import config
from lazyllm.tools.sandbox import DummySandbox, SandboxFusion
from lazyllm.tools.agent import code_interpreter


class TestDummySandbox:
    def test_sandbox_registry(self):
        assert hasattr(lazyllm, 'sandbox')
        assert 'dummy' in lazyllm.sandbox
        assert 'sandbox_fusion' in lazyllm.sandbox
        assert lazyllm.sandbox['dummy'] is DummySandbox

        sandbox = lazyllm.sandbox['dummy'](timeout=5)
        result = sandbox(code="print('registry_ok')")
        assert isinstance(result, dict)
        assert result['stdout'].strip() == 'registry_ok'

    def test_execute_basic(self):
        sandbox = DummySandbox(timeout=5)
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
            sandbox = DummySandbox(timeout=5)
            result = sandbox(code=code, input_files=[input_path])
            assert result['stdout'].strip() == 'True'


class TestCodeInterpreter:
    def test_dummy_code_interpreter(self):
        prev_env = os.environ.get('LAZYLLM_SANDBOX_TYPE')
        try:
            os.environ['LAZYLLM_SANDBOX_TYPE'] = 'dummy'
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
        sandbox = SandboxFusion(run_timeout=5)
        result = sandbox(code="print('fusion')", language='python')
        assert isinstance(result, dict)
        assert result.get('run_result', {}).get('stdout', '').strip() == 'fusion'

    @pytest.mark.skipif(
        not os.getenv('LAZYLLM_SANDBOX_FUSION_BASE_URL'),
        reason='LAZYLLM_SANDBOX_FUSION_BASE_URL is not set'
    )
    def test_sandbox_fusion_upload_and_download_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, 'input.txt')
            with open(input_path, 'w', encoding='utf-8') as f:
                f.write('hello fusion file io')

            code = (
                'import os\n'
                f'input_file = "input.txt" if os.path.exists("input.txt") else r"{input_path}"\n'
                'with open(input_file, "r", encoding="utf-8") as f:\n'
                '    content = f.read().strip()\n'
                'with open("output.txt", "w", encoding="utf-8") as f:\n'
                '    f.write(content + " -> processed")\n'
                'print(content)\n'
            )

            sandbox = SandboxFusion(run_timeout=5)
            result = sandbox(
                code=code,
                language='python',
                input_files=[input_path],
                output_files=['output.txt'],
            )

            assert isinstance(result, dict)
            assert result.get('run_result', {}).get('stdout', '').strip() == 'hello fusion file io'
            assert len(result.get('output_files', [])) == 1
            output_path = result['output_files'][0]
            assert os.path.exists(output_path)
            with open(output_path, 'r', encoding='utf-8') as f:
                assert f.read() == 'hello fusion file io -> processed'
