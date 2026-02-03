from lazyllm import config, call_once, once_flag
from lazyllm.tools import fc_register
from lazyllm.tools.sandbox.sandbox_fusion import SandboxFusion
from lazyllm.tools.sandbox.local_sandbox import LocalSandbox


config.add('sandbox_type', str, 'local', 'SANDBOX_TYPE')

_sandbox = None
_sandbox_once = once_flag()


def _create_sandbox():
    global _sandbox
    _sandbox = LocalSandbox() if config['sandbox_type'] == "local" else SandboxFusion()
    return _sandbox


@fc_register("tool", execute_in_sandbox=False)
def code_interpreter(code: str, language: str = "python") -> str:
    """
    Interpret the code and return the result.

    Args:
        code (str): The code to interpret.
        language (str): The language of the code.
    """
    call_once(_sandbox_once, _create_sandbox)
    return _sandbox(code=code, language=language)