from typing import List, Optional, Any

import requests
from lazyllm.tools.agent.toolsManager import ModuleTool
from requests import exceptions as req_exc
from json import JSONDecodeError
from lazyllm import LOG


class SandboxFusion(ModuleTool):
    SUPPORTED_LANGUAGES = [
        'cpp', 'go', 'go_test', 'java', 'junit', 'nodejs', 'js', 'ts', 'typescript',
        'python', 'pytest', 'csharp', 'rust', 'php', 'bash', 'jest', 'lua', 'R',
        'perl', 'D_ut', 'ruby', 'scala', 'julia', 'kotlin_script', 'verilog',
        'lean', 'swift', 'racket', 'cuda', 'python_gpu',
    ]

    def __init__(
        self,
        base_url: str,
        verbose: bool = False,
        return_trace: bool = True,
    ):
        super().__init__(verbose=verbose, return_trace=return_trace)
        self._base_url = base_url

    @property
    def url(self) -> str:
        return f'{self._base_url}/run_code'

    def _api_execute(
        self,
        call_params: dict[str, Any],
        content_type: str = 'application/json',
    ):
        headers = {
            'Content-Type': content_type,
            'Accept': 'application/json',
        }
        try:
            response = requests.post(self.url, headers=headers, json=call_params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if isinstance(e, req_exc.RequestException):
                last_error = f'API Request Error: {e}'
                LOG.error(last_error)
            elif isinstance(e, JSONDecodeError):
                raw = response.text if 'response' in locals() else 'N/A'
                last_error = f'API Response JSON Decode Error: {e}; raw={raw}'
                LOG.error(last_error)
            else:
                last_error = f'Unexpected Error: {e}'
                LOG.exception(last_error)
        return 'Sandbox API Call Failed'

    def apply(
        self,
        code: str,
        language: Optional[str] = 'python',
        stdin: Optional[str] = None,
        compile_timeout: Optional[int] = 10,
        run_timeout: Optional[int] = 10,
        memory_limit_mb: Optional[int] = -1,
        files: Optional[List[str]] = None,
    ) -> str:
        '''
        Use SandboxFusion remote sandbox to execute code.

        Args:
            code (str): The code to execute.
            language (Optional[str]): The language of the code, can be one of:
                python, javascript, bash, go, rust, default is python.
            stdin (Optional[str]): The standard input to the program at runtime.
            compile_timeout (Optional[int]): The compile timeout in seconds, default is 10.
            run_timeout (Optional[int]): The run timeout in seconds, default is 10.
            memory_limit_mb (Optional[int]): The maximum memory limit in MB, default is -1.
            files (Optional[List[str]]): The list of file paths to upload, default is [].
        '''
        if language not in self.SUPPORTED_LANGUAGES:
            return f'Error: Language {language} is not supported'

        if files is None:
            files = []
        call_params = {
            'code': code,
            'stdin': stdin,
            'compile_timeout': compile_timeout,
            'run_timeout': run_timeout,
            'memory_limit_mb': memory_limit_mb,
            'language': language,
        }
        result = self._api_execute(call_params)
        return result
