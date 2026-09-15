import os
import subprocess
from typing import Dict, Optional

from .toolsManager import register
from .toolError import ToolExecutionError
from .tool_runtime import resolve_host_path

@register('builtin_tools', execute_in_sandbox=False, host_file='OPAQUE', exclusive=True)
@register('tool', execute_in_sandbox=False, host_file='OPAQUE', exclusive=True)
def shell(cmd: str, cwd: Optional[str] = None, timeout: int = 30,
          env: Optional[Dict[str, str]] = None) -> dict:
    '''Run a shell command and return stdout/stderr/exit code.

    Args:
        cmd (str): The shell command to execute.
        cwd (str, optional): Working directory for the command.
        timeout (int, optional): Timeout in seconds. Defaults to 30.
        env (dict, optional): Environment variables to pass to the process.

    Returns:
        dict: Execution result including stdout, stderr, exit_code, and cwd.
    '''
    cmd = cmd.strip()
    if not cmd:
        raise ToolExecutionError('cmd cannot be empty.')
    cwd = resolve_host_path(cwd or '.')
    if not os.path.isdir(cwd):
        raise ToolExecutionError(f'cwd not found: {cwd}')

    try:
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolExecutionError(
            f'Shell command {cmd!r} in {cwd or os.getcwd()} timed out after {timeout} seconds.',
        ) from exc
    return {
        'status': 'ok',
        'stdout': completed.stdout,
        'stderr': completed.stderr,
        'exit_code': completed.returncode,
        'cwd': cwd or os.getcwd(),
    }
