import lazyllm
from lazyllm import config, call_once, once_flag
from lazyllm.tools import fc_register

_sandbox = None
_sandbox_once = once_flag()


def _create_sandbox():
    global _sandbox
    try:
        sandbox_cls = lazyllm.sandbox[config['sandbox_type']]
    except KeyError as e:
        message = (
            f'Sandbox type {config["sandbox_type"]} not found, '
            'the code interpreter tool only supports the following sandbox types: '
            f'{list(lazyllm.sandbox.keys())}'
        )
        raise ValueError(message) from e
    _sandbox = sandbox_cls()
    return _sandbox


@fc_register('tool', execute_in_sandbox=False)
def code_interpreter(code: str, language: str = 'python') -> str:
    '''
    Interpret the code and return the result.

    Args:
        code (str): The code to interpret.
        language (str): The language of the code. Default is 'python'.
    '''
    call_once(_sandbox_once, _create_sandbox)
    return _sandbox(code=code, language=language)
