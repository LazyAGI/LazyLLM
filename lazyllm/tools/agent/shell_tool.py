import os
import re
import subprocess
from typing import Dict, Optional

from .toolsManager import register
from .toolError import ToolExecutionError

_DANGEROUS_WORD_TOKENS = [
    'rm', 'sudo', 'chmod', 'chown', 'mkfs', 'dd', 'shutdown', 'reboot', 'poweroff',
    'kill', 'killall', 'pkill', 'apt', 'yum', 'dnf', 'brew', 'pip', 'conda', 'curl',
    'wget', 'scp', 'ssh',
]
_DANGEROUS_PHRASES = [
    'git reset --hard',
    'git clean -fd',
]
_DANGEROUS_SUBSTRINGS = [
    '>>',
    '>',
]


def _detect_dangerous_command(cmd: str) -> Optional[str]:
    lowered = cmd.lower()
    if re.search(r'\brm\s+-rf\b', lowered):
        return 'rm -rf'
    for token in _DANGEROUS_PHRASES:
        if token in lowered:
            return token
    for token in _DANGEROUS_WORD_TOKENS:
        if re.search(rf'\b{re.escape(token)}\b', lowered):
            return token
    for token in _DANGEROUS_SUBSTRINGS:
        if token in lowered:
            return token
    return None


@register('builtin_tools', execute_in_sandbox=False, host_file='OPAQUE')
@register('tool', execute_in_sandbox=False, host_file='OPAQUE')
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
    if cwd is not None and not os.path.isdir(cwd):
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
