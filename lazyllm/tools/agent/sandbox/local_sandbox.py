import ast
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional, Tuple

from lazyllm import LOG
from lazyllm.common.utils import SecurityVisitor
from lazyllm.tools.agent.toolsManager import ModuleTool


class LocalSandbox(ModuleTool):
    SUPPORTED_LANGUAGES = ['python']

    def __init__(
        self,
        verbose: bool = False,
        return_trace: bool = True,
    ):
        super().__init__(verbose=verbose, return_trace=return_trace)

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

    def _cleanup_temp_dir(self, temp_dir: Optional[str]) -> None:
        if not temp_dir:
            return
        if not os.path.isdir(temp_dir):
            return
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            LOG.warning(f'LocalSandbox: failed to remove temp dir {temp_dir!r}: {e}')

    def _run_in_subprocess(
        self,
        script_path: str,
        cwd: str,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = 30,
    ) -> Tuple[int, str, str]:
        proc = subprocess.Popen(
            [sys.executable, '-u', script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env or os.environ.copy(),
            text=True,
            bufsize=1,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
        return proc.returncode, stdout or '', stderr or ''

    def _execute_python_code(self, code: str, timeout: Optional[int] = None) -> Tuple[int, str, str]:
        is_safe, safety_msg = self._check_code_safety(code)
        if not is_safe:
            return safety_msg

        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp(prefix='lazyllm_sandbox_')
            script_path = os.path.join(temp_dir, '_script.py')
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(code)

            env = os.environ.copy()
            env['HOME'] = temp_dir
            env['PYTHONPATH'] = temp_dir

            returncode, stdout, stderr = self._run_in_subprocess(
                script_path, cwd=temp_dir, env=env, timeout=timeout
            )
            return {
                'returncode': returncode,
                'stdout': stdout,
                'stderr': stderr,
            }
        except subprocess.TimeoutExpired:
            return f'Execution timed out after {timeout} seconds'
        except Exception as e:
            return str(e)
        finally:
            self._cleanup_temp_dir(temp_dir)

    def apply(
        self,
        code: str,
        language: str = 'python',
        timeout: Optional[int] = 30,
    ) -> Dict[str, Any]:
        '''
        Execute code in a local subprocess sandbox with temp-dir isolation and SecurityVisitor check.

        Args:
            code (str): Python source code to execute (valid module: statements and/or expressions).
            language (str): Must be "python"; other languages are not supported.
            timeout (Optional[int]): The timeout for the execution, default is 30 seconds.
        '''
        if language not in self.SUPPORTED_LANGUAGES:
            return f'Unsupported language: {language}'

        return self._execute_python_code(code, timeout)
