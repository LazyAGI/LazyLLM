import base64
import hashlib
import os
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional
from json import JSONDecodeError

import requests
from requests import exceptions as req_exc

from lazyllm import LOG, config
from lazyllm.components.utils.file_operate import file_to_base64
from lazyllm.tools.sandbox.sandbox_base import LazyLLMSandboxBase, _SandboxResult

config.add('sandbox_fusion_base_url', str, '', 'SANDBOX_FUSION_BASE_URL')
SKILL_SANDBOX_PROTOCOL_VERSION = 1


class SandboxFusion(LazyLLMSandboxBase):
    __lazyllm_registry_key__ = 'sandbox_fusion'
    SUPPORTED_LANGUAGES: List[str] = ['python', 'bash']

    def __init__(self, base_url: str = config['sandbox_fusion_base_url'], compile_timeout: int = 10,
                 run_timeout: int = 10, memory_limit_mb: int = -1, project_dir: str = None,
                 return_sandbox_result: bool = False, return_trace: bool = False):
        super().__init__(
            return_trace=return_trace,
            project_dir=project_dir,
            return_sandbox_result=return_sandbox_result,
        )
        self._base_url = base_url
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
        return self._call_endpoint(self.url, call_params)

    def _call_endpoint(self, url: str, call_params: dict[str, Any]):
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        try:
            resp = requests.post(url, headers=headers, json=call_params)
            resp.raise_for_status()
            return resp.json()
        except req_exc.RequestException as e:
            LOG.error(f'API Request Error: {e}')
        except JSONDecodeError as e:
            LOG.error(f'API Response JSON Decode Error: {e}')
        except Exception as e:
            LOG.exception(f'Unexpected Error: {e}')
        return 'Sandbox API Call Failed'

    def capabilities(self) -> Dict[str, Any]:
        try:
            response = requests.get(f'{self._base_url}/v1/capabilities', timeout=2)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        return {
            'status': 'unavailable',
            'runtimes': {language: True for language in self.SUPPORTED_LANGUAGES},
            'commands': [],
            'profiles': [],
            'supports': {'script': False, 'command': False, 'declared_outputs': False},
        }

    def execute_script(
        self,
        source_dir: str,
        rel_path: str,
        args: Optional[List[str]] = None,
        cwd: str = '.',
        env: Optional[Dict[str, str]] = None,
        policy: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        input_files: Optional[List[str]] = None,
        declared_outputs: Optional[List[str]] = None,
        skill_id: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        package, tree_hash = self._encode_skill_package(source_dir)
        suffix = Path(rel_path).suffix.lower()
        runtime = {
            '.py': 'python', '.sh': 'bash', '.bash': 'bash',
            '.js': 'node', '.cjs': 'node', '.mjs': 'node',
        }.get(suffix, 'unsupported')
        return self._execute_skill_package({
            'runtime': runtime,
            'mode': 'script',
            'entrypoint': self._relative_path(rel_path, 'entrypoint'),
            'args': list(args or []),
            'argv': list(args or []),
            'cwd': self._relative_path(cwd, 'cwd', allow_dot=True),
            'files': package,
            'tree_hash': tree_hash,
            'input_files': self._relative_paths(input_files or [], 'input_files'),
            'declared_outputs': self._relative_paths(declared_outputs or [], 'declared_outputs', globs=True),
            'env': dict(env or {}),
            'policy': dict(policy or {}),
            'session_id': str(session_id or ''),
            'skill_id': str(skill_id or ''),
            'revision': str(revision or 'latest'),
        })

    def execute_command(
        self,
        source_dir: str,
        argv: List[str],
        cwd: str = '.',
        env: Optional[Dict[str, str]] = None,
        policy: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        input_files: Optional[List[str]] = None,
        declared_outputs: Optional[List[str]] = None,
        skill_id: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not isinstance(argv, list) or not argv or any(not isinstance(item, str) for item in argv):
            return {'status': 'error', 'error_type': 'COMMAND_DENIED', 'error': 'argv must be strings.'}
        package, tree_hash = self._encode_skill_package(source_dir)
        return self._execute_skill_package({
            'runtime': 'cli',
            'mode': 'command',
            'argv': list(argv),
            'cwd': self._relative_path(cwd, 'cwd', allow_dot=True),
            'files': package,
            'tree_hash': tree_hash,
            'input_files': self._relative_paths(input_files or [], 'input_files'),
            'declared_outputs': self._relative_paths(declared_outputs or [], 'declared_outputs', globs=True),
            'env': dict(env or {}),
            'policy': dict(policy or {}),
            'session_id': str(session_id or ''),
            'skill_id': str(skill_id or ''),
            'revision': str(revision or 'latest'),
        })

    def _execute_skill_package(self, request: Dict[str, Any]) -> Dict[str, Any]:
        policy = request.get('policy') or {}
        allowed_domains = list(policy.get('allowed_domains') or [])
        network_policy = {
            'mode': 'domain_allowlist' if allowed_domains else 'deny',
        }
        request.update({
            'version': SKILL_SANDBOX_PROTOCOL_VERSION,
            'action': 'execute',
            'request_id': uuid.uuid4().hex,
            'required_bins': list(policy.get('required_bins') or []),
            'required_env': list(policy.get('required_env') or []),
            'network': network_policy,
            'network_policy': network_policy,
            'limits': {
                'timeout_seconds': max(1, int(self._run_timeout)),
                'memory_mb': self._memory_limit_mb if self._memory_limit_mb > 0 else 1024,
                'cpu_seconds': max(1, int(self._run_timeout)),
                'pids': 64,
                'output_bytes': 1024 * 1024,
            },
        })
        response = self._call_endpoint(f'{self._base_url}/v1/skill/execute', request)
        if isinstance(response, dict):
            return response
        return {
            'status': 'error',
            'error_type': 'SANDBOX_UNAVAILABLE',
            'reason_code': 'SANDBOX_UNAVAILABLE',
            'error': str(response),
            'stdout': '',
            'stderr': str(response),
            'exit_code': None,
            'output_files': [],
        }

    @staticmethod
    def _relative_path(value: str, label: str, *, allow_dot: bool = False, globs: bool = False) -> str:
        raw = str(value or '').strip().replace('\\', '/')
        if allow_dot and raw in ('', '.'):
            return '.'
        path = PurePosixPath(raw)
        if not raw or path.is_absolute() or any(part in ('', '.', '..') for part in path.parts):
            raise ValueError(f'{label} must stay inside the Skill package.')
        if not globs and any(char in raw for char in '*?['):
            raise ValueError(f'{label} may not contain glob syntax.')
        return path.as_posix()

    @classmethod
    def _relative_paths(cls, values: List[str], label: str, *, globs: bool = False) -> List[str]:
        if not isinstance(values, list):
            raise ValueError(f'{label} must be a list.')
        return [cls._relative_path(value, label, globs=globs) for value in values]

    @staticmethod
    def _encode_skill_package(source_dir: str) -> tuple[Dict[str, str], str]:
        root = Path(source_dir).resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f'Skill source directory not found: {source_dir}')
        files: Dict[str, str] = {}
        digest = hashlib.sha256()
        total = 0
        for path in sorted(root.rglob('*')):
            rel_path = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ValueError(f'Skill packages may not contain symbolic links: {rel_path}')
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(f'Skill package contains a non-regular file: {rel_path}')
            if len(files) >= 2048:
                raise ValueError('Skill package contains more than 2048 files.')
            content = path.read_bytes()
            total += len(content)
            if total > 64 * 1024 * 1024:
                raise ValueError('Skill package uncompressed content is too large.')
            digest.update(rel_path.encode('utf-8') + b'\0' + content + b'\0')
            files[rel_path] = base64.b64encode(content).decode('ascii')
        return files, digest.hexdigest()

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
