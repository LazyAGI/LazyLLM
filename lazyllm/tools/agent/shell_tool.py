import os
import re
import subprocess
from typing import Dict, Optional

from .toolsManager import register

_DANGEROUS_TOKENS = [
    'rm', 'sudo', 'chmod', 'chown', 'mkfs', 'dd', 'shutdown', 'reboot', 'poweroff',
    'kill', 'killall', 'pkill', 'apt', 'yum', 'dnf', 'brew', 'pip', 'conda', 'curl',
    'wget', 'scp', 'ssh', 'git reset --hard', 'git clean -fd', '>', '>>',
]


def _detect_dangerous_command(cmd: str) -> Optional[str]:
    lowered = cmd.lower()
    for token in _DANGEROUS_TOKENS:
        if token in lowered:
            return token
    if re.search(r'\brm\s+-rf\b', lowered):
        return 'rm -rf'
    return None


@register('tool')
def shell_tool(cmd: str, cwd: Optional[str] = None, timeout: int = 30,
               env: Optional[Dict[str, str]] = None, allow_unsafe: bool = False) -> dict:
    '''Run a shell command and return stdout/stderr/exit code.

    Args:
        cmd (str): The shell command to execute.
        cwd (str, optional): Working directory for the command.
        timeout (int, optional): Timeout in seconds. Defaults to 30.
        env (dict, optional): Environment variables to pass to the process.
        allow_unsafe (bool, optional): Allow potentially dangerous commands. Defaults to False.

    Returns:
        dict: Execution result including stdout, stderr, exit_code, and cwd.
    '''
    cmd = cmd.strip()
    if not cmd:
        raise ValueError('cmd cannot be empty.')
    if cwd is not None and not os.path.isdir(cwd):
        raise FileNotFoundError(f'cwd not found: {cwd}')

    dangerous = _detect_dangerous_command(cmd)
    if dangerous and not allow_unsafe:
        return {
            'status': 'needs_approval',
            'reason': f'Command contains potentially dangerous token: {dangerous}',
            'command': cmd,
            'cwd': cwd or os.getcwd(),
        }

    completed = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        shell=True,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return {
        'status': 'ok',
        'stdout': completed.stdout,
        'stderr': completed.stderr,
        'exit_code': completed.returncode,
        'cwd': cwd or os.getcwd(),
    }
