import os
import tempfile
import pytest
from typing import List

import lazyllm
from lazyllm import config
from lazyllm.tools.sandbox import DummySandbox, SandboxFusion
from lazyllm.tools.agent import code_interpreter
from lazyllm.tools import ToolManager, fc_register


# ---- tool functions for TestToolManagerSandboxIntegration (registered in setup_class) ---- #

def sandbox_add_tool(a: int, b: int):
    '''
    Add two integers.

    Args:
        a (int): first number.
        b (int): second number.
    '''
    return a + b


def direct_tool(x: int):
    '''
    Return double the value.

    Args:
        x (int): input value.
    '''
    return x * 2


def no_sandbox_tool(val: str):
    '''
    Echo back the value.

    Args:
        val (str): input string.
    '''
    return val.upper()


def read_file_tool(file_paths: str):
    '''
    Read content from a file.

    Args:
        file_paths (str): path to the input file.
    '''
    with open(file_paths, 'r') as f:
        return f.read()


def write_file_tool(out_paths: str):
    '''
    Write content to a file.

    Args:
        out_paths (str): path to the output file.
    '''
    with open(out_paths, 'w') as f:
        f.write('generated_output')
    return 'done'


def gen_summary_tool(text: str):
    '''
    Generate a summary and write to summary.txt.

    Args:
        text (str): input text.
    '''
    with open('summary.txt', 'w') as f:
        f.write(text[:5])
    return 'ok'


def multi_read_tool(paths: List[str]):
    '''
    Read multiple files.

    Args:
        paths (List[str]): list of file paths.
    '''
    results = []
    for p in paths:
        with open(p, 'r') as f:
            results.append(f.read())
    return '|'.join(results)


class TestDummySandbox:
    def test_sandbox_registry(self):
        assert hasattr(lazyllm, 'sandbox')
        assert 'dummy' in lazyllm.sandbox
        assert 'sandbox_fusion' in lazyllm.sandbox
        assert lazyllm.sandbox['dummy'] is DummySandbox

        sandbox = lazyllm.sandbox['dummy'](timeout=5, return_sandbox_result=True)
        result = sandbox(code="print('registry_ok')")
        assert result['success']
        assert result['stdout'].strip() == 'registry_ok'

    def test_execute_basic(self):
        sandbox = DummySandbox(timeout=5, return_sandbox_result=True)
        result = sandbox(code="print('hello')")
        assert result['success']
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
            sandbox = DummySandbox(timeout=5, return_sandbox_result=True)
            result = sandbox(code=code, input_files=[input_path])
            assert result['stdout'].strip() == 'True'

    def test_execute_with_output_files(self):
        with tempfile.TemporaryDirectory() as output_dir:
            sandbox = DummySandbox(timeout=5, return_sandbox_result=True)
            sandbox._output_dir_path = output_dir
            code = (
                'from pathlib import Path\n'
                'Path("result.txt").write_text("output_content")\n'
            )
            result = sandbox(code=code, output_files=['result.txt'])
            assert result['success']
            assert len(result['output_files']) == 1
            with open(result['output_files'][0], 'r') as f:
                assert f.read() == 'output_content'

    def test_execute_with_input_and_output_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, 'source.txt')
            with open(input_path, 'w', encoding='utf-8') as f:
                f.write('abc')

            output_dir = os.path.join(temp_dir, 'out')
            os.makedirs(output_dir)

            sandbox = DummySandbox(timeout=5, return_sandbox_result=True)
            sandbox._output_dir_path = output_dir
            code = (
                'from pathlib import Path\n'
                'data = Path("source.txt").read_text()\n'
                'Path("dest.txt").write_text(data[::-1])\n'
            )
            result = sandbox(
                code=code,
                input_files=[input_path],
                output_files=['dest.txt'],
            )
            assert result['success']
            assert len(result['output_files']) == 1
            with open(result['output_files'][0], 'r') as f:
                assert f.read() == 'cba'

    def test_execute_unsupported_language(self):
        sandbox = DummySandbox(timeout=5, return_sandbox_result=True)
        with pytest.raises(Exception, match='not supported'):
            sandbox(code='echo hello', language='bash')

    def test_execute_unsafe_code(self):
        sandbox = DummySandbox(timeout=5, return_sandbox_result=True)
        result = sandbox(code='import subprocess\nsubprocess.call(["ls"])')
        assert not result['success']

    def test_execute_syntax_error(self):
        sandbox = DummySandbox(timeout=5, return_sandbox_result=True)
        result = sandbox(code='def foo(:\n    pass')
        assert not result['success']

    def test_execute_timeout(self):
        sandbox = DummySandbox(timeout=2, return_sandbox_result=True)
        result = sandbox(code='import time\ntime.sleep(10)')
        assert not result['success']
        assert 'timed out' in result['error_message'].lower()


class TestCodeInterpreter:
    def test_dummy_code_interpreter(self):
        prev_env = os.environ.get('LAZYLLM_SANDBOX_TYPE')
        try:
            os.environ['LAZYLLM_SANDBOX_TYPE'] = 'dummy'
            config.refresh()
            result = code_interpreter("print('ok')")
            assert result['success']
            assert result['stdout'].strip() == 'ok'
        finally:
            if prev_env is None:
                os.environ.pop('LAZYLLM_SANDBOX_TYPE', None)
            else:
                os.environ['LAZYLLM_SANDBOX_TYPE'] = prev_env
            config.refresh()


class TestToolManagerSandboxIntegration:
    '''Tests for ToolManager + sandbox integration.'''

    _tool_names = [
        'sandbox_add_tool', 'direct_tool', 'no_sandbox_tool',
        'read_file_tool', 'write_file_tool', 'gen_summary_tool', 'multi_read_tool',
    ]

    @classmethod
    def setup_class(cls):
        fc_register('tool')(sandbox_add_tool)
        fc_register('tool', execute_in_sandbox=False)(direct_tool)
        fc_register('tool')(no_sandbox_tool)
        fc_register('tool', input_files_parm='file_paths')(read_file_tool)
        fc_register('tool', output_files_parm='out_paths')(write_file_tool)
        fc_register('tool', output_files=['summary.txt'])(gen_summary_tool)
        fc_register('tool', input_files_parm='paths')(multi_read_tool)

    @classmethod
    def teardown_class(cls):
        tool_registry = lazyllm.tool
        for name in cls._tool_names:
            try:
                tool_registry.remove(name)
            except (AttributeError, KeyError):
                pass

    def _make_tool_call(self, name, arguments):
        return {'function': {'name': name, 'arguments': arguments}}

    def test_toolmanager_with_sandbox(self):
        '''When sandbox is configured and execute_in_sandbox=True (default),
        ToolManager should delegate execution to the sandbox.'''
        sandbox = DummySandbox(timeout=10, return_sandbox_result=True)
        tm = ToolManager(['sandbox_add_tool'], sandbox=sandbox)
        result = tm([self._make_tool_call('sandbox_add_tool', {'a': 3, 'b': 5})])
        # Sandbox executes tool via cloudpickle in subprocess; result is converted to dict
        assert result[0]['success']
        assert 'LAZYLLM_TOOL_RESULT:8' in result[0]['stdout']

    def test_toolmanager_execute_in_sandbox_false_bypasses_sandbox(self):
        '''When execute_in_sandbox=False, ToolManager should call the tool
        directly even when a sandbox is configured.'''
        sandbox = DummySandbox(timeout=10, return_sandbox_result=True)
        tm = ToolManager(['direct_tool'], sandbox=sandbox)
        result = tm([self._make_tool_call('direct_tool', {'x': 7})])
        assert result[0] == 14

    def test_toolmanager_no_sandbox_calls_directly(self):
        '''When no sandbox is configured, ToolManager calls tools directly.'''
        tm = ToolManager(['no_sandbox_tool'], sandbox=None)
        result = tm([self._make_tool_call('no_sandbox_tool', {'val': 'hello'})])
        assert result[0] == 'HELLO'

    def test_toolmanager_sandbox_with_input_files(self):
        '''ToolManager should upload files specified by input_files_parm.'''
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, 'data.txt')
            with open(input_path, 'w') as f:
                f.write('sandbox_file_content')

            sandbox = DummySandbox(timeout=10, return_sandbox_result=True)
            tm = ToolManager(['read_file_tool'], sandbox=sandbox)
            result = tm([self._make_tool_call('read_file_tool', {'file_paths': input_path})])
            assert result[0]['success']
            assert 'sandbox_file_content' in result[0]['stdout']

    def test_toolmanager_sandbox_with_output_files_parm(self):
        '''ToolManager should download files specified by output_files_parm.'''
        with tempfile.TemporaryDirectory() as output_dir:
            sandbox = DummySandbox(timeout=10, return_sandbox_result=True)
            sandbox._output_dir_path = output_dir
            tm = ToolManager(['write_file_tool'], sandbox=sandbox)
            result = tm([self._make_tool_call('write_file_tool', {'out_paths': 'result.txt'})])
            assert result[0]['success']

    def test_toolmanager_sandbox_with_static_output_files(self):
        '''ToolManager should also download files listed in output_files metadata.'''
        with tempfile.TemporaryDirectory() as output_dir:
            sandbox = DummySandbox(timeout=10, return_sandbox_result=True)
            sandbox._output_dir_path = output_dir
            tm = ToolManager(['gen_summary_tool'], sandbox=sandbox)
            result = tm([self._make_tool_call('gen_summary_tool', {'text': 'hello world'})])
            assert result[0]['success']

    def test_toolmanager_sandbox_input_files_parm_with_list(self):
        '''input_files_parm pointing to a List[str] parameter should work.'''
        with tempfile.TemporaryDirectory() as temp_dir:
            p1 = os.path.join(temp_dir, 'a.txt')
            p2 = os.path.join(temp_dir, 'b.txt')
            with open(p1, 'w') as f:
                f.write('AAA')
            with open(p2, 'w') as f:
                f.write('BBB')

            sandbox = DummySandbox(timeout=10, return_sandbox_result=True)
            tm = ToolManager(['multi_read_tool'], sandbox=sandbox)
            result = tm([self._make_tool_call('multi_read_tool', {'paths': [p1, p2]})])
            assert result[0]['success']


class TestSandboxFusion:
    @pytest.mark.skipif(
        not os.getenv('LAZYLLM_SANDBOX_FUSION_BASE_URL'),
        reason='LAZYLLM_SANDBOX_FUSION_BASE_URL is not set'
    )
    def test_sandbox_fusion_execute(self):
        sandbox = SandboxFusion(run_timeout=5, return_sandbox_result=True)
        result = sandbox(code="print('fusion')", language='python')
        assert result['success']
        assert result['stdout'].strip() == 'fusion'

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

            sandbox = SandboxFusion(run_timeout=5, return_sandbox_result=True)
            result = sandbox(
                code=code,
                language='python',
                input_files=[input_path],
                output_files=['output.txt'],
            )

            assert result['success']
            assert result['stdout'].strip() == 'hello fusion file io'
            assert len(result['output_files']) == 1
            output_path = result['output_files'][0]
            assert os.path.exists(output_path)
            with open(output_path, 'r', encoding='utf-8') as f:
                assert f.read() == 'hello fusion file io -> processed'
