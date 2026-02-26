import base64
import os
from typing import Any, List, Optional
from json import JSONDecodeError

import requests
from requests import exceptions as req_exc

from lazyllm import LOG, config
from lazyllm.components.utils.file_operate import file_to_base64
from lazyllm.tools.sandbox.sandbox_base import LazyLLMSandboxBase, _SandboxResult

config.add('sandbox_fusion_base_url', str, '', 'SANDBOX_FUSION_BASE_URL')


class SandboxFusion(LazyLLMSandboxBase):
    __lazyllm_registry_key__ = 'sandbox_fusion'
    SUPPORTED_LANGUAGES: List[str] = ['python', 'bash']

    def __init__(self, base_url: str = config['sandbox_fusion_base_url'], compile_timeout: int = 10,
                 run_timeout: int = 10, memory_limit_mb: int = -1, project_dir: str = None):
        self._base_url = base_url
        super().__init__(project_dir=project_dir)
        self._compile_timeout = compile_timeout
        self._run_timeout = run_timeout
        self._memory_limit_mb = memory_limit_mb
        self._project_files_cache = None

    @property
    def url(self) -> str:
        return f'{self._base_url}/run_code'

    def _check_available(self) -> None:
        try:
            resp = requests.get(f'{self._base_url}/v1/ping', timeout=2)
            if resp.status_code != 200:
                raise ValueError(f'SandboxFusion ping failed: status={resp.status_code}, text={resp.text}')
        except Exception as e:
            raise ValueError(f'SandboxFusion _check_available error: {e}')

    def _call_api(self, call_params: dict[str, Any]):
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        try:
            resp = requests.post(self.url, headers=headers, json=call_params)
            resp.raise_for_status()
            return resp.json()
        except req_exc.RequestException as e:
            LOG.error(f'API Request Error: {e}')
        except JSONDecodeError as e:
            LOG.error(f'API Response JSON Decode Error: {e}')
        except Exception as e:
            LOG.exception(f'Unexpected Error: {e}')
        return 'Sandbox API Call Failed'

    def _create_context(self) -> dict:
        return {'files': {}}

    def _process_input_files(self, input_files: List[str], context: dict) -> None:
        for f in input_files:
            context['files'][f] = self._encode_file_base64(f)

    def _process_project_dir(self, context: dict) -> None:
        if self._project_files_cache is None:
            self._project_files_cache = {
                rel: self._encode_file_base64(abs_p)
                for abs_p, rel in self._collect_project_py_files()
            }
        context['files'].update(self._project_files_cache)

    def _process_output_files(self, result: _SandboxResult, output_files: List[str], context: dict) -> List[str]:
        self._ensure_output_dir()
        response_files = context.get('response_files') or {}
        collected = []
        for name in output_files:
            b64 = response_files.get(name)
            if b64 is None:
                LOG.warning(f'SandboxFusion: requested output file {name!r} not found in response')
                continue
            path = os.path.join(self._output_dir_path, name)
            with open(path, 'wb') as f:
                f.write(base64.b64decode(b64))
            collected.append(path)
        return collected

    def _execute(self, code: str, language: str, context: dict,
                 output_files: Optional[List[str]] = None) -> _SandboxResult:
        call_params = {
            'code': code,
            'compile_timeout': self._compile_timeout,
            'run_timeout': self._run_timeout,
            'memory_limit_mb': self._memory_limit_mb,
            'language': language,
            'files': context['files'],
        }
        if output_files:
            call_params['fetch_files'] = output_files

        response = self._call_api(call_params)
        if isinstance(response, str):
            return _SandboxResult(success=False, error_message=response)

        context['response_files'] = response.get('files') or {}
        run_result = response.get('run_result') or {}
        returncode = run_result.get('return_code', -1)
        return _SandboxResult(
            success=(response.get('status') == 'Success' and returncode == 0),
            stdout=run_result.get('stdout', ''),
            stderr=run_result.get('stderr', ''),
            returncode=returncode,
        )

    @staticmethod
    def _encode_file_base64(path: str) -> str:
        encoded = file_to_base64(path)
        if encoded is None:
            raise ValueError(f'Failed to encode file to base64: {path}')
        return encoded[0]
