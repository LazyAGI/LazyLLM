from lazyllm.tools.sandbox.sandbox_base import create_sandbox


def test_syntax_failure_returns_error_text():
    sandbox = create_sandbox('dummy')
    assert sandbox('def broken(').startswith('Syntax error:')


def test_timeout_returns_error_text():
    sandbox = create_sandbox('dummy', timeout=1)
    assert sandbox('while True:\n    pass') == 'Execution timed out after 1 seconds'


def test_success_stdout_control():
    sandbox = create_sandbox('dummy')
    assert sandbox("print('ok')") == 'ok\n'


def test_runtime_stderr_control():
    sandbox = create_sandbox('dummy')
    assert 'ValueError: controlled' in sandbox("raise ValueError('controlled')")


def test_structured_result_control():
    sandbox = create_sandbox('dummy', return_sandbox_result=True)
    result = sandbox('def broken(')
    assert result['success'] is False
    assert result['stderr'] == ''
    assert result['error_message'].startswith('Syntax error:')


def test_success_tool_result_prefix_control():
    sandbox = create_sandbox('dummy')
    assert sandbox("print('LAZYLLM_TOOL_RESULT:42')") == '42'
