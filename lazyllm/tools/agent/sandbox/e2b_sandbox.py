from typing import Dict, Any
from typing import Optional

from lazyllm import config
from lazyllm.tools.agent.toolsManager import ModuleTool
from lazyllm.thirdparty import e2b_code_interpreter

config.add('e2b_api_key', str, None, 'E2B_API_KEY', description='The API key for E2B.')


class E2BSandbox(ModuleTool):
    SUPPORTED_LANGUAGES = ['python', 'javascript', 'js', 'bash', 'r']

    def __init__(self, api_key: str = None, verbose: bool = False, return_trace: bool = True):
        super().__init__(verbose=verbose, return_trace=return_trace)
        self.code_interpreter = e2b_code_interpreter.Sandbox(
            api_key=api_key or config.e2b_api_key
        )

    def apply(self,
              code: str,
              language: Optional[str] = 'python',
              timeout: Optional[int] = 30) -> Dict[str, Any]:
        '''
        Execute code in the E2B sandbox.

        Args:
            code (str): The code to execute.
            language (Optional[str]): The language of the code, can be one of:
                python, javascript, js, bash, r.
            timeout (Optional[int]): The timeout for the execution, default is 30 seconds.
        '''
        if language not in self.SUPPORTED_LANGUAGES:
            return f'Error: Language {language} is not supported'

        call_params = {
            'code': code,
            'language': language,
            'timeout': timeout
        }
        execution = self.code_interpreter.run_code(**call_params)
        return {
            'results': execution.results,
            'stdout': execution.logs.stdout,
            'stderr': execution.logs.stderr,
            'error': execution.error,
        }
