import ast
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
            cwd=cwd, env=env or os.environ.copy(), text=True, bufsize=1,
        )
        try:
            stdout, stderr = proc.communicate(timeout=self._timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
        return {'returncode': proc.returncode, 'stdout': stdout, 'stderr': stderr}

    def execute_script(self, source_dir: str, rel_path: str, args: Optional[List[str]] = None,
                       cwd: str = '.', allow_unsafe: bool = False,
                       env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
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
            runner = sys.executable if ext == '.py' else 'bash' if ext in ('.sh', '.bash') else 'sh'
            completed = subprocess.run(
                [runner, script_path, *(args or [])],
                cwd=run_cwd,
                env={**os.environ.copy(), **(env or {})},
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
