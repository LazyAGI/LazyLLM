import ast
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

from lazyllm import LOG
from lazyllm.common.utils import SecurityVisitor
from lazyllm.tools.sandbox.sandbox_base import SandboxBase


class LocalSandbox(SandboxBase):
    def __init__(self, timeout: int = 30, return_trace: bool = True):
        super().__init__(return_trace=return_trace)
        self._timeout = timeout

    def _is_available(self) -> bool:
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
    ) -> Dict[str, Any]:
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
            stdout, stderr = proc.communicate(timeout=self._timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
        return {
            'returncode': proc.returncode,
            'stdout': stdout,
            'stderr': stderr,
        }

    def _execute(self, code: str, input_files: Optional[List[str]] = None, output_files: Optional[List[str]] = None) -> Union[Dict[str, Any], str]:
        is_safe, safety_msg = self._check_code_safety(code)
        if not is_safe:
            return safety_msg

        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp(prefix='lazyllm_sandbox_')
            script_path = os.path.join(temp_dir, '_script.py')
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(code)
            for file_path in input_files:
                try:
                    shutil.copy(file_path, temp_dir)
                except Exception as e:
                    LOG.warning(f"LocalSandbox: failed to copy input file {file_path!r} to temp dir {temp_dir!r}: {e}")

            env = os.environ.copy()
            env['HOME'] = temp_dir
            env['PYTHONPATH'] = temp_dir

            result = self._run_in_subprocess(
                script_path, cwd=temp_dir, env=env
            )
            if output_files:
                result['output_files'] = []
                for out_file in output_files:
                    src_path = os.path.join(temp_dir, os.path.basename(out_file))
                    dst_path = os.path.join(self._output_dir_path, os.path.basename(out_file))
                    try:
                        if os.path.exists(src_path):
                            shutil.move(src_path, dst_path)
                            result['output_files'].append(dst_path)
                    except Exception as e:
                        LOG.warning(f"LocalSandbox: failed to move output file {src_path!r} to {dst_path!r}: {e}")
            return result
        except subprocess.TimeoutExpired:
            return f'Execution timed out after {self._timeout} seconds'
        except Exception as e:
            return str(e)
        finally:
            self._cleanup_temp_dir(temp_dir)
