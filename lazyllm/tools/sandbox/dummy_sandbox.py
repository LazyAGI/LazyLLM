import ast
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from lazyllm import LOG
from lazyllm.common.utils import SecurityVisitor
from lazyllm.tools.sandbox.sandbox_base import LazyLLMSandboxBase, _SandboxResult


class DummySandbox(LazyLLMSandboxBase):
    SUPPORTED_LANGUAGES: List[str] = ['python']
    _PASSTHROUGH_ENV = (
        'LANG', 'LC_ALL', 'LC_CTYPE', 'PATH', 'SYSTEMROOT', 'TZ', 'WINDIR',
    )
    _OUTPUT_ARGUMENTS = {'--output', '--output-dir', '--out', '-o'}

    def __init__(self, timeout: int = 30, return_trace: bool = False, project_dir: Optional[str] = None,
                 return_sandbox_result: bool = False):
        super().__init__(return_trace=return_trace, project_dir=project_dir,
                         return_sandbox_result=return_sandbox_result)
        self._timeout = timeout

    def _check_available(self) -> bool:
        return True

    def _check_code_safety(self, code: str) -> Tuple[bool, Optional[str]]:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f'Syntax error: {e}'
        try:
            SecurityVisitor().visit(tree)
        except ValueError as e:
            return False, str(e)
        return True, None

    def _run_in_subprocess(self, script_path: str, cwd: str,
                           env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        proc = subprocess.Popen(
            [sys.executable, '-u', script_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=cwd, env=env or self._subprocess_env(cwd), text=True, bufsize=1,
        )
        try:
            stdout, stderr = proc.communicate(timeout=self._timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
        return {'returncode': proc.returncode, 'stdout': stdout, 'stderr': stderr}

    def execute_script(self, source_dir: str, rel_path: str, args: Optional[List[str]] = None,
                       cwd: str = '.', allow_unsafe: bool = False) -> Dict[str, Any]:
        del allow_unsafe  # DummySandbox currently has no approval boundary.
        context = self._create_context()
        try:
            sandbox_root = context['temp_dir']
            shutil.copytree(source_dir, sandbox_root, dirs_exist_ok=True)
            script_path = self._resolve_child(sandbox_root, rel_path, 'rel_path')
            run_cwd = self._resolve_child(sandbox_root, cwd or '.', 'cwd')
            if not os.path.isfile(script_path):
                return {
                    'status': 'missing',
                    'path': script_path,
                    'rel_path': rel_path,
                    'cwd': run_cwd,
                }
            if not os.path.isdir(run_cwd):
                raise FileNotFoundError(f'cwd not found: {run_cwd}')
            ext = os.path.splitext(script_path)[1].lower()
            command = self._script_command(
                script_path, list(args or []), sandbox_root, run_cwd, ext,
            )
            completed = subprocess.run(
                command,
                cwd=run_cwd,
                env=self._subprocess_env(sandbox_root),
                text=True,
                capture_output=True,
                timeout=self._timeout,
            )
            return {
                'status': 'ok' if completed.returncode == 0 else 'failed',
                'stdout': completed.stdout,
                'stderr': completed.stderr,
                'exit_code': completed.returncode,
                'cwd': run_cwd,
            }
        finally:
            self._cleanup_context(context)

    @classmethod
    def _subprocess_env(cls, sandbox_root: str) -> Dict[str, str]:
        env = {
            key: os.environ[key]
            for key in cls._PASSTHROUGH_ENV
            if key in os.environ
        }
        env.update({
            'HOME': sandbox_root,
            'TMPDIR': sandbox_root,
            'TEMP': sandbox_root,
            'TMP': sandbox_root,
            'PYTHONNOUSERSITE': '1',
        })
        python_path = [path for path in sys.path if path and os.path.isdir(path)]
        if python_path:
            env['PYTHONPATH'] = os.pathsep.join(python_path)
        return env

    def _script_command(self, script_path: str, args: List[str], sandbox_root: str,
                        run_cwd: str, extension: str) -> List[str]:
        if extension != '.py':
            runner = 'bash' if extension in ('.sh', '.bash') else 'sh'
            return [runner, script_path, *args]

        read_paths = {sandbox_root, script_path}
        write_paths = {sandbox_root}
        for index, arg in enumerate(args):
            raw = str(arg or '').strip()
            if not raw or raw.startswith('-'):
                continue
            resolved = os.path.realpath(raw if os.path.isabs(raw) else os.path.join(run_cwd, raw))
            if index > 0 and args[index - 1] in self._OUTPUT_ARGUMENTS:
                write_paths.add(resolved)
            elif os.path.exists(resolved):
                read_paths.add(resolved)
        read_paths.update(path for path in sys.path if path and os.path.exists(path))
        read_paths.update(path for path in mimetypes.knownfiles if os.path.isfile(path))
        if os.path.exists(os.devnull):
            read_paths.add(os.devnull)
            write_paths.add(os.devnull)

        policy = {
            'read': sorted(os.path.realpath(path) for path in read_paths),
            'write': sorted(os.path.realpath(path) for path in write_paths),
        }
        bootstrap_path = os.path.join(sandbox_root, '_lazyllm_skill_bootstrap.py')
        with open(bootstrap_path, 'w', encoding='utf-8') as bootstrap:
            bootstrap.write(self._python_audit_bootstrap(policy, script_path, args, run_cwd))
        return [sys.executable, '-I', '-u', bootstrap_path]

    @staticmethod
    def _python_audit_bootstrap(policy: Dict[str, List[str]], script_path: str,
                                args: List[str], run_cwd: str) -> str:
        return f'''import ctypes
import os
import sys

POLICY = {json.dumps(policy)!r}
POLICY = __import__('json').loads(POLICY)
SCRIPT = {script_path!r}
ARGS = {args!r}
CWD = {run_cwd!r}

def _inside(path, roots):
    if isinstance(path, int):
        return True
    candidate = os.path.realpath(os.path.abspath(os.fspath(path)))
    for root in roots:
        try:
            if os.path.commonpath((root, candidate)) == root:
                return True
        except ValueError:
            pass
    return False

def _audit(event, event_args):
    if event == 'open':
        path = event_args[0]
        mode = event_args[1] if len(event_args) > 1 else 'r'
        flags = event_args[2] if len(event_args) > 2 else 0
        writing = (isinstance(mode, str) and any(char in mode for char in 'wax+')) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        )
        roots = POLICY['write'] if writing else POLICY['read'] + POLICY['write']
        if not _inside(path, roots):
            raise PermissionError(f'sandbox denied file access: {{path}}')
    elif event in ('os.listdir', 'os.scandir', 'os.chdir') and event_args:
        if not _inside(event_args[0], POLICY['read'] + POLICY['write']):
            raise PermissionError(f'sandbox denied filesystem access: {{event_args[0]}}')
    elif event in ('os.remove', 'os.rmdir', 'os.mkdir') and event_args:
        if not _inside(event_args[0], POLICY['write']):
            raise PermissionError(f'sandbox denied filesystem write: {{event_args[0]}}')
    elif event in ('os.rename', 'os.replace') and len(event_args) >= 2:
        if not all(_inside(path, POLICY['write']) for path in event_args[:2]):
            raise PermissionError('sandbox denied filesystem rename')
    elif event.startswith('socket.') or event in (
        'subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.spawn',
        'ctypes.dlopen', 'ctypes.dlsym', 'ctypes.call_function',
    ):
        raise PermissionError(f'sandbox denied process or network access: {{event}}')

sys.addaudithook(_audit)
os.chdir(CWD)
sys.argv = [SCRIPT, *ARGS]
with open(SCRIPT, 'rb') as source_file:
    source = source_file.read()
scope = {{'__name__': '__main__', '__file__': SCRIPT, '__package__': None, '__cached__': None}}
exec(compile(source, SCRIPT, 'exec'), scope, scope)
'''

    @staticmethod
    def _resolve_child(root: str, rel_path: str, label: str) -> str:
        root_real = os.path.realpath(os.path.abspath(root))
        target = os.path.realpath(os.path.abspath(os.path.join(root_real, rel_path)))
        if os.path.commonpath([root_real, target]) != root_real:
            raise ValueError(f'{label} must stay inside the sandbox directory.')
        return target

    def _create_context(self) -> dict:
        return {'temp_dir': tempfile.mkdtemp(prefix='lazyllm_sandbox_')}

    def _cleanup_context(self, context: dict) -> None:
        temp_dir = context.get('temp_dir')
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _process_input_files(self, input_files: List[str], context: dict) -> None:
        for f in input_files:
            try:
                shutil.copy(f, context['temp_dir'])
            except Exception as e:
                LOG.warning(f'DummySandbox: failed to copy input file {f!r}: {e}')

    def _process_project_dir(self, context: dict) -> None:
        temp_dir = context['temp_dir']
        for abs_path, rel_path in self._collect_project_py_files():
            dst = os.path.join(temp_dir, rel_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(abs_path, dst)

    def _process_output_files(self, result: _SandboxResult, output_files: List[str], context: dict) -> List[str]:
        self._ensure_output_dir()
        collected = []
        for name in output_files:
            src = os.path.join(context['temp_dir'], name)
            dst = os.path.join(self._output_dir_path, name)
            try:
                if os.path.exists(src):
                    shutil.move(src, dst)
                    collected.append(dst)
            except Exception as e:
                LOG.warning(f'DummySandbox: failed to move output file {src!r}: {e}')
        return collected

    def _execute(self, code: str, language: str, context: dict,
                 output_files: Optional[List[str]] = None) -> _SandboxResult:
        is_safe, msg = self._check_code_safety(code)
        if not is_safe:
            return _SandboxResult(success=False, error_message=msg)

        temp_dir = context['temp_dir']
        try:
            script_path = os.path.join(temp_dir, '_script.py')
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(code)
            proc_result = self._run_in_subprocess(script_path, cwd=temp_dir)
            return _SandboxResult(
                success=(proc_result['returncode'] == 0),
                stdout=proc_result['stdout'],
                stderr=proc_result['stderr'],
                returncode=proc_result['returncode'],
            )
        except subprocess.TimeoutExpired:
            return _SandboxResult(success=False, error_message=f'Execution timed out after {self._timeout} seconds')
        except Exception as e:
            return _SandboxResult(success=False, error_message=str(e))
