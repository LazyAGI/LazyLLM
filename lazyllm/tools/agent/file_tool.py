import json
import shutil
import subprocess
import tempfile
import threading
import stat as stat_module
from pathlib import PurePath
import os
from typing import Optional

from .toolsManager import fc_register, ToolGroup
from .toolError import ToolExecutionError
from .tool_runtime import HostFileIntent, HostFileResolution, resolve_host_path


MAX_EDIT_BYTES = 64 * 1024


def _host_files(*fields):
    def resolve(arguments):
        intents = []
        if any(operation == 'delete' for _, operation in fields):
            for name, _ in fields:
                _check_entry(arguments.get(name, '.'))
        for name, operation in fields:
            arguments[name] = resolve_host_path(arguments.get(name, '.'))
            intents.append(HostFileIntent(arguments[name], operation))
        if arguments.get('root'):
            arguments['root'] = resolve_host_path(arguments['root'])
        return HostFileResolution(arguments, tuple(intents))
    return resolve


def _resolve_path(path: str) -> str:
    return resolve_host_path(path)


def _check_entry(path):
    path = os.path.expanduser(os.fspath(path))
    entry = os.path.join(resolve_host_path(os.path.dirname(path) or '.'), os.path.basename(path))
    if os.path.islink(entry):
        raise ToolExecutionError('Moving or removing a symbolic link is not supported; use its target explicitly')


def _open_text(path, mode='r', encoding='utf-8', errors='strict', newline=None):
    flags = {'r': os.O_RDONLY, 'r+': os.O_RDWR, 'w': os.O_WRONLY | os.O_CREAT,
             'a': os.O_WRONLY | os.O_CREAT | os.O_APPEND,
             'x': os.O_WRONLY | os.O_CREAT | os.O_EXCL}[mode]
    fd = os.open(path, flags | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0), 0o600)
    try:
        if not stat_module.S_ISREG(os.fstat(fd).st_mode):
            raise ToolExecutionError('Expected a regular file')
        if mode == 'w':
            os.ftruncate(fd, 0)
        if encoding is None:
            return os.fdopen(fd, mode + 'b')
        return os.fdopen(fd, mode, encoding=encoding, errors=errors, newline=newline)
    except BaseException:
        os.close(fd)
        raise


def _check_root(path: str, root: Optional[str]) -> None:
    if not root:
        return
    root_abs = _resolve_path(root)
    path_abs = _resolve_path(path)
    try:
        inside = os.path.commonpath([path_abs, root_abs]) == root_abs
    except ValueError:
        inside = False
    if not inside:
        raise ToolExecutionError(
            f'Path {path_abs} is outside the allowed root {root_abs}.',
        )


def _atomic_write(path, content, encoding, mode='overwrite'):
    data = content.encode(encoding)
    added_bytes = len(data)
    original = None
    try:
        original = os.stat(path, follow_symlinks=False)
        if not stat_module.S_ISREG(original.st_mode):
            raise ToolExecutionError('Expected a regular file')
    except FileNotFoundError:
        pass
    if mode == 'create' and original is not None:
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(prefix='.lazyllm-', dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, 'wb') as stream:
            if original is not None:
                os.chmod(temporary, stat_module.S_IMODE(original.st_mode))
            if mode == 'append' and original is not None:
                with _open_text(path, encoding=None) as source:
                    shutil.copyfileobj(source, stream, length=1024 * 1024)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if mode == 'create':
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return added_bytes


@fc_register('builtin_tools')
@fc_register('tool', execute_in_sandbox=False)
@fc_register(host_file=_host_files(('path', 'read')))
def read(path: str, offset: int = 1, limit: int = 500, max_bytes: int = 65536,
         encoding: str = 'utf-8', errors: str = 'replace', root: Optional[str] = None) -> dict:
    '''Read a bounded text window without loading the whole file.

    Args:
        path (str): File path.
        offset (int): First logical line, starting at one. Long lines split every 1024 characters.
        limit (int): Maximum logical lines, default 500, hard cap 2000.
        max_bytes (int): UTF-8 content budget, default 65536, hard cap 262144, minimum 4096.
        encoding (str): Source encoding, default utf-8.
        errors (str): Decoding error policy, default replace.
        root (str, optional): Restrict reads to this root directory.

    Returns:
        dict: Content, next_offset, eof and truncation status.
    '''
    if offset < 1 or limit < 1 or max_bytes < 4096:
        raise ToolExecutionError('offset/limit must be positive and max_bytes at least 4096')
    limit, max_bytes = min(limit, 2000), min(max_bytes, 262144)
    _check_root(path, root)
    path = _resolve_path(path)
    content, size, count = [], 0, 0
    with _open_text(path, encoding=encoding, errors=errors) as stream:
        for _ in range(offset - 1):
            if not stream.readline(1024):
                break
        line = stream.readline(1024)
        while line and count < limit:
            encoded = len(line.encode('utf-8', errors='replace'))
            if size + encoded > max_bytes:
                break
            content.append(line)
            size += encoded
            count += 1
            line = stream.readline(1024)
    eof = not line
    return {'status': 'ok', 'path': path, 'offset': offset, 'end_line': offset + count - 1,
            'content': ''.join(content), 'eof': eof, 'truncated': not eof,
            'next_offset': None if eof else offset + count}


@fc_register('builtin_tools')
@fc_register('tool', execute_in_sandbox=False)
@fc_register(host_file=_host_files(('path', 'read')))
def ls(path: str = '.', limit: int = 200, root: Optional[str] = None) -> dict:
    '''List one directory level. Use glob for recursive discovery.

    Args:
        path (str): Directory path, defaults to current directory.
        limit (int): Maximum entries, default 200, hard cap 1000.
        root (str, optional): Restrict listing to this root directory.

    Returns:
        dict: Entries and truncation status.
    '''
    if limit < 1:
        raise ToolExecutionError('limit must be positive')
    limit = min(limit, 1000)
    _check_root(path, root)
    path = _resolve_path(path)
    entries = []
    with os.scandir(path) as directory:
        for entry in directory:
            entries.append(entry.name)
            if len(entries) > limit:
                break
    return {'status': 'ok', 'path': path, 'entries': sorted(entries[:limit]),
            'truncated': len(entries) > limit}


@fc_register('builtin_tools')
@fc_register('tool', execute_in_sandbox=False)
@fc_register(host_file=_host_files(('path', 'read')))
def grep(pattern: str, path: str = '.', glob: Optional[str] = None,
         max_results: int = 100, root: Optional[str] = None) -> dict:
    '''Search file contents with ripgrep without following symbolic links.

    Args:
        pattern (str): Ripgrep regular expression.
        path (str): File or directory to search, defaults to current directory.
        glob (str, optional): Filename glob filter.
        max_results (int): Maximum matches, default and hard cap 100.
        root (str, optional): Restrict search to this root directory.

    Returns:
        dict: Matches with path, line number and bounded snippets, plus truncation status.
    '''
    if not pattern or len(pattern) > 4096 or '\0' in pattern or max_results < 1:
        raise ToolExecutionError('A valid pattern and positive result limit are required')
    _check_root(path, root)
    path = _resolve_path(path)
    max_results = min(max_results, 100)
    executable = shutil.which('rg')
    if not executable:
        raise ToolExecutionError('grep requires ripgrep (rg) to be installed')
    command = [executable, '--no-config', '--json', '--max-count', str(max_results + 1)]
    if glob:
        command += ['--glob', glob]
    command += ['--', pattern, path]
    results, truncated = _run_rg(command, lambda stream: _read_grep_matches(stream, path, max_results))
    return {'status': 'ok', 'results': results, 'truncated': truncated}


def _read_grep_matches(stream, path, limit):
    results, size = [], 0
    while line := stream.readline(65537):
        if len(line) > 65536:
            return results, True
        event = json.loads(line)
        if event['type'] != 'match':
            continue
        data = event['data']
        item = {'path': data['path'].get('text', path), 'line': str(data['line_number']),
                'text': data['lines'].get('text', '').rstrip('\n')[:1000]}
        added = len(json.dumps(item, ensure_ascii=False).encode('utf-8')) + 2
        if len(results) == limit or size + added > 65000:
            return results, True
        results.append(item)
        size += added
    return results, False


def _run_rg(command, collect, cwd=None):
    expired = threading.Event()
    with tempfile.TemporaryFile() as errors, subprocess.Popen(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=errors,
    ) as process:
        def expire():
            expired.set()
            process.kill()

        timer = threading.Timer(30, expire)
        timer.daemon = True
        timer.start()
        try:
            results, truncated = collect(process.stdout)
            if truncated:
                process.kill()
            code = process.wait()
            if expired.is_set():
                raise ToolExecutionError('Search timed out after 30 seconds; narrow the search path')
            if not truncated and code not in (0, 1):
                errors.seek(0)
                raise ToolExecutionError('Search failed: ' + errors.read(4096).decode('utf-8', 'replace'))
            return results, truncated
        finally:
            timer.cancel()
            if process.poll() is None:
                process.kill()
                process.wait()


@fc_register('builtin_tools')
@fc_register('tool', execute_in_sandbox=False)
@fc_register(host_file=_host_files(('path', 'write')))
def mkdir(path: str, parents: bool = True, exist_ok: bool = True,
          root: Optional[str] = None) -> dict:
    '''Create a directory.

    Args:
        path (str): Directory path to create.
        parents (bool, optional): Create parent directories. Defaults to True.
        exist_ok (bool, optional): Ignore if already exists. Defaults to True.
        root (str, optional): Restrict to this root directory.

    Returns:
        dict: Status result.
    '''
    _check_root(path, root)
    path_abs = _resolve_path(path)
    if parents:
        os.makedirs(path_abs, exist_ok=exist_ok)
    elif not (exist_ok and os.path.isdir(path_abs)):
        os.mkdir(path_abs)
    return {'status': 'ok', 'path': path_abs}


@fc_register('builtin_tools')
@fc_register('tool', execute_in_sandbox=False)
@fc_register(host_file=_host_files(('path', 'write')))
def write(path: str, content: str, mode: str = 'overwrite', encoding: str = 'utf-8',
          root: Optional[str] = None, create_parents: bool = True) -> dict:
    '''Write content to a file.

    Args:
        path (str): File path.
        content (str): Content to write.
        mode (str, optional): create|overwrite|append. Defaults to overwrite.
        encoding (str, optional): File encoding. Defaults to utf-8.
        root (str, optional): Restrict writes to this root directory.
        create_parents (bool, optional): Create parent directories if needed.

    Returns:
        dict: Status result.
    '''
    _check_root(path, root)
    path_abs = _resolve_path(path)
    if mode not in ('create', 'overwrite', 'append'):
        raise ToolExecutionError(f'Invalid write mode {mode!r}; expected "create", "overwrite" or "append".')
    parent = os.path.dirname(path_abs)

    if parent and create_parents:
        os.makedirs(parent, exist_ok=True)
    size = _atomic_write(path_abs, content, encoding, mode)
    return {'status': 'ok', 'path': path_abs, 'mode': mode, 'bytes': size}


@fc_register('builtin_tools')
@fc_register('tool', execute_in_sandbox=False)
@fc_register(host_file=_host_files(('path', 'delete')))
def remove(path: str, recursive: bool = False, root: Optional[str] = None) -> dict:
    '''Delete a file or directory. Recursive deletion requires explicit opt-in.

    Args:
        path (str): File path.
        recursive (bool): Delete nonempty directories recursively. Defaults to False.
        root (str, optional): Restrict deletion to this root directory.
    Returns:
        dict: Status result.
    '''
    _check_entry(path)
    _check_root(path, root)
    path_abs = _resolve_path(path)
    if not os.path.exists(path_abs):
        raise ToolExecutionError(f'File not found: {path_abs}')
    if os.path.isdir(path_abs):
        shutil.rmtree(path_abs) if recursive else os.rmdir(path_abs)
    else:
        os.unlink(path_abs)
    return {'status': 'ok', 'path': path_abs}


@fc_register('builtin_tools')
@fc_register('tool', execute_in_sandbox=False)
@fc_register(host_file=_host_files(('src', 'read'), ('src', 'delete'), ('dst', 'write')))
def move(src: str, dst: str, root: Optional[str] = None, overwrite: bool = False,
         create_parents: bool = True) -> dict:
    '''Move or rename a file.

    Args:
        src (str): Source path.
        dst (str): Destination path.
        root (str, optional): Restrict operations to this root directory.
        overwrite (bool, optional): Allow overwrite at destination. Defaults to False.
        create_parents (bool, optional): Create destination parents if needed.

    Returns:
        dict: Status result.
    '''
    _check_entry(src)
    _check_entry(dst)
    _check_root(src, root)
    _check_root(dst, root)
    src_abs = _resolve_path(src)
    dst_abs = _resolve_path(dst)
    if not os.path.exists(src_abs):
        raise ToolExecutionError(f'Source file not found: {src_abs}')
    if os.path.exists(dst_abs) and not overwrite:
        raise ToolExecutionError(f'Destination already exists: {dst_abs}')
    parent = os.path.dirname(dst_abs)
    if parent and create_parents:
        os.makedirs(parent, exist_ok=True)
    os.replace(src_abs, dst_abs) if overwrite else os.rename(src_abs, dst_abs)
    return {'status': 'ok', 'src': src_abs, 'dst': dst_abs}


@fc_register('builtin_tools')
@fc_register('tool', execute_in_sandbox=False)
@fc_register(host_file=_host_files(('path', 'write')))
def edit(path: str, old_text: str, new_text: str, expected_replacements: int = 1,
         encoding: str = 'utf-8') -> dict:
    '''Replace exact text in an existing file up to 64 KiB (65536 bytes).

    Larger files require a targeted or streaming workflow.

    Args:
        path (str): File to edit.
        old_text (str): Exact nonempty text to replace.
        new_text (str): Replacement text.
        expected_replacements (int): Required number of matches, defaults to one.
        encoding (str): Text encoding, defaults to utf-8.

    Returns:
        dict: Path and replacement count.
    '''
    if not old_text or expected_replacements < 1:
        raise ToolExecutionError('old_text and a positive replacement count are required')
    path = _resolve_path(path)
    with _open_text(path, encoding=None) as stream:
        size = os.fstat(stream.fileno()).st_size
        data = stream.read(MAX_EDIT_BYTES + 1) if size <= MAX_EDIT_BYTES else b''
        if size > MAX_EDIT_BYTES or len(data) > MAX_EDIT_BYTES:
            raise ToolExecutionError(
                'File is too large for edit (limit 65536 bytes). '
                'Use a targeted or streaming workflow for large files.'
            )
        content = data.decode(encoding)
        count = content.count(old_text)
        if count != expected_replacements:
            raise ToolExecutionError(f'Expected {expected_replacements} matches, found {count}')
    _atomic_write(path, content.replace(old_text, new_text), encoding)
    return {'status': 'ok', 'path': path, 'replacements': count}


def _read_glob_paths(stream, path, limit):
    results = []
    pending = b''
    size = 0
    while len(results) <= limit:
        chunk = stream.read1(65536)
        if not chunk:
            break
        entries = (pending + chunk).split(b'\0')
        pending = entries.pop()
        for entry in entries:
            candidate = os.path.abspath(os.path.join(path, os.fsdecode(entry)))
            if not os.path.islink(candidate):
                size += len(candidate.encode('utf-8', 'replace')) + 4
                if size > 65000:
                    return results, True
                results.append(candidate)
            if len(results) > limit:
                break
    return results[:limit], len(results) > limit


@fc_register('builtin_tools')
@fc_register('tool', execute_in_sandbox=False)
@fc_register(host_file=_host_files(('path', 'read')))
def glob(pattern: str, path: str = '.', max_results: int = 100) -> dict:
    '''Find files using ripgrep glob rules without following symbolic links.

    Args:
        pattern (str): File filter: *.yml matches any depth, /*.yml only the search
            root, and **/*.{yaml,yml} matches both extensions recursively.
        path (str): Directory to search, defaults to the working directory.
        max_results (int): Positive maximum number of results.

    Returns:
        dict: Matching absolute paths and truncation status, not a total count.
    '''
    if max_results < 1 or not pattern or '\x00' in pattern or '..' in PurePath(pattern).parts:
        raise ToolExecutionError('A valid glob pattern and positive result limit are required')
    max_results = min(max_results, 100)
    path = _resolve_path(path)
    if not os.path.isdir(path):
        raise ToolExecutionError(f'Directory not found: {path}')
    executable = shutil.which('rg')
    if not executable:
        raise ToolExecutionError('glob requires ripgrep (rg) to be installed')
    command = [executable, '--no-config', '--files', '--null', '--glob', pattern,
               '--glob', '!**/.git/**', '--', '.']
    results, truncated = _run_rg(command, lambda stream: _read_glob_paths(stream, path, max_results), cwd=path)
    return {'status': 'ok', 'paths': results, 'truncated': truncated}


@fc_register('builtin_tools')
@fc_register('tool', execute_in_sandbox=False)
@fc_register(host_file=_host_files(('path', 'read')))
def stat(path: str) -> dict:
    '''Inspect file or directory metadata.

    Args:
        path (str): File or directory path.

    Returns:
        dict: Canonical path, type, size and modification time.
    '''
    path = _resolve_path(path)
    info = os.stat(path)
    return {'status': 'ok', 'path': path,
            'type': 'directory' if stat_module.S_ISDIR(info.st_mode) else 'file',
            'size': info.st_size, 'mtime_ns': info.st_mtime_ns}


class FileSystemToolkit(ToolGroup):
    '''Common host filesystem tools, available immediately with short names.'''

    def __init__(self):
        super().__init__(tools=[read, write, edit, ls, glob, grep, mkdir, move, remove, stat],
                         name='FileSystemToolkit', desc=self.__doc__, lazy=False, prefix=False)
