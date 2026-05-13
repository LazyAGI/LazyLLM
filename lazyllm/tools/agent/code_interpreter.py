from lazyllm import call_once, once_flag
from lazyllm.tools import fc_register
from lazyllm.tools.sandbox.sandbox_base import create_sandbox as _create_sandbox_impl

_sandbox = None
_sandbox_once = once_flag()


def _create_sandbox():
    global _sandbox
    _sandbox = _create_sandbox_impl(return_sandbox_result=True)
    return _sandbox


@fc_register('tool', execute_in_sandbox=False)
def code_interpreter(code: str, language: str = 'python') -> str:
    '''
    Interpret the code and return the code interpreter result (include stdout, stderr, returncode, etc.).

    Args:
        code (str): The code to interpret.
        language (str): The language of the code. Default is 'python'.
    '''
    call_once(_sandbox_once, _create_sandbox)
    return _sandbox(code=code, language=language)
