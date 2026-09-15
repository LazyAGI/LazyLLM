import os
import subprocess
import locale
import signal
import select
import threading
import time
from typing import Dict, Optional

from .toolsManager import register
from .toolError import ToolExecutionError
from .tool_runtime import resolve_host_path


class _OutputBuffer:
    def __init__(self):
        self.head = bytearray()
        self.tail = bytearray()
        self.size = 0
        self.error = None

    def drain(self, stream, stop):
        try:
            if os.name == 'nt':
                import ctypes
                import msvcrt
                handle = ctypes.c_void_p(msvcrt.get_osfhandle(stream.fileno()))
                available = ctypes.c_ulong()
                peek = ctypes.windll.kernel32.PeekNamedPipe
            while not stop.is_set():
                if os.name == 'posix':
                    if not select.select([stream], [], [], 0.1)[0]:
                        continue
                else:
                    if not peek(handle, None, 0, None, ctypes.byref(available), None):
                        break  # The writer closed the pipe.
                    if not available.value:
                        stop.wait(0.01)
                        continue
                chunk = stream.read(65536)
                if not chunk:
                    break
                self.size += len(chunk)
                take = min(65536 - len(self.head), len(chunk))
                self.head.extend(chunk[:take])
                self.tail.extend(chunk[take:])
                if len(self.tail) > 65536:
                    del self.tail[:-65536]
        except OSError as error:
            self.error = error

    def text(self):
        encoding = locale.getpreferredencoding(False)
        if self.size <= 131072:
            text = bytes(self.head + self.tail).decode(encoding, errors='replace')
        else:
            head = self.head.decode(encoding, errors='replace')
            tail = self.tail.decode(encoding, errors='replace')
            text = head + '\n... [output truncated] ...\n' + tail
        return text.replace('\r\n', '\n').replace('\r', '\n')


def _kill_shell(process):
    if os.name == 'posix':
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        # taskkill includes descendants that may still own our output pipes.
        try:
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
    if process.poll() is None:
        process.kill()
    process.wait()


@register('builtin_tools', execute_in_sandbox=False, host_file='OPAQUE', exclusive=True)
@register('tool', execute_in_sandbox=False, host_file='OPAQUE', exclusive=True)
def shell(cmd: str, cwd: Optional[str] = None, timeout: int = 30,
          env: Optional[Dict[str, str]] = None) -> dict:
    '''Run a shell command and return stdout/stderr/exit code.

    Large output is bounded to the first and last 64 KiB per stream. Redirect full
    output to a file and use grep/read to inspect relevant parts of large logs.

    Args:
        cmd (str): The shell command to execute.
        cwd (str, optional): Working directory for the command.
        timeout (int, optional): Timeout in seconds. Defaults to 30.
        env (dict, optional): Environment variables to pass to the process.

    Returns:
        dict: stdout, stderr, exit_code, cwd, and per-stream byte counts and truncation flags.
    '''
    cmd = cmd.strip()
    if not cmd:
        raise ToolExecutionError('cmd cannot be empty.')
    cwd = resolve_host_path(cwd or '.')
    if not os.path.isdir(cwd):
        raise ToolExecutionError(f'cwd not found: {cwd}')

    deadline = time.monotonic() + timeout
    process = subprocess.Popen(
        cmd, cwd=cwd, env=env, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        start_new_session=os.name == 'posix',
    )
    stop = threading.Event()
    outputs = [_OutputBuffer(), _OutputBuffer()]
    readers = [threading.Thread(target=output.drain, args=(stream, stop), daemon=True)
               for output, stream in zip(outputs, (process.stdout, process.stderr))]
    try:
        for reader in readers:
            reader.start()
        process.wait(timeout=max(0, deadline - time.monotonic()))
        for reader in readers:
            reader.join(timeout=max(0, deadline - time.monotonic()))
        if any(reader.is_alive() for reader in readers):
            raise subprocess.TimeoutExpired(cmd, timeout)
        for output in outputs:
            if output.error is not None:
                raise output.error
    except subprocess.TimeoutExpired as exc:
        raise ToolExecutionError(
            f'Shell command {cmd!r} in {cwd} timed out after {timeout} seconds.',
        ) from exc
    finally:
        stop.set()
        if process.poll() is None or any(reader.is_alive() for reader in readers):
            _kill_shell(process)
        for reader in readers:
            if reader.ident is not None:
                reader.join(timeout=1)
        for stream in (process.stdout, process.stderr):
            stream.close()
    stdout, stderr = outputs
    return {
        'status': 'ok',
        'stdout': stdout.text(),
        'stderr': stderr.text(),
        'stdout_bytes': stdout.size,
        'stderr_bytes': stderr.size,
        'stdout_truncated': stdout.size > 131072,
        'stderr_truncated': stderr.size > 131072,
        'exit_code': process.returncode,
        'cwd': cwd,
    }
