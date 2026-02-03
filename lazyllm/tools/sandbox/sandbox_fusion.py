import base64
from typing import List, Any, Optional
import os
import requests
from requests import exceptions as req_exc
from json import JSONDecodeError
from lazyllm import LOG, config
from lazyllm.tools.sandbox.sandbox_base import SandboxBase

config.add('sandbox_fusion_base_url', str, '', 'SANDBOX_FUSION_BASE_URL')

class SandboxFusion(SandboxBase):
    SUPPORTED_LANGUAGES: List[str] = ['python', 'bash']

    def __init__(self, base_url: str = config['sandbox_fusion_base_url'], compile_timeout: int = 10,
                 run_timeout: int = 10, memory_limit_mb: int = -1):
        self._base_url = base_url
        super().__init__()
        self._compile_timeout = compile_timeout
        self._run_timeout = run_timeout
        self._memory_limit_mb = memory_limit_mb

    @property
    def url(self) -> str:
        return f'{self._base_url}/run_code'

    def _is_available(self) -> None:
        try:
            response = requests.get(f'{self._base_url}/v1/ping', timeout=2)
            if response.status_code != 200:
                raise ValueError(
                    f'SandboxFusion _is_available ping failed: status={response.status_code}, text={response.text}')
        except Exception as e:
            raise ValueError(f'SandboxFusion _is_available error: {e}')

    def _call_api(
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

    def _execute(
        self,
        code: str,
        language: str = 'python',
        input_files: Optional[List[str]] = None,
        output_files: Optional[List[str]] = None,
    ) -> str:
        call_params = {
            'code': code,
            'compile_timeout': self._compile_timeout,
            'run_timeout': self._run_timeout,
            'memory_limit_mb': self._memory_limit_mb,
            'language': language,
        }
        if input_files:
            call_params['files'] = {
                file: base64.b64encode(open(file, 'rb').read()).decode('utf-8')
                for file in input_files
            }
        if output_files:
            call_params['fetch_files'] = output_files
        result = self._call_api(call_params)
        if files := result.get('files', None):
            result['output_files'] = []
            for file_name, base64_content in files.items():
                file_path = os.path.join(self._output_dir_path, file_name)
                with open(file_path, 'wb') as f:
                    f.write(base64.b64decode(base64_content))
                    result['output_files'].append(file_path)
        return result
