import json
import shutil
import subprocess
import tempfile
import threading
import stat as stat_module
from pathlib import PurePath
import os
from typing import BinaryIO, Optional, TextIO

from .toolsManager import fc_register, ToolGroup
from .toolError import ToolExecutionError
from .tool_runtime import HostFileIntent, HostFileResolution, resolve_host_path


MAX_EDIT_BYTES = 64 * 1024
MAX_READ_BYTES = 64 * 1024
MAX_SEARCH_RESULTS = 100


def _host_files(*fields):
    def resolve(arguments):
        intents = []
        if any(operation == 'delete' for _, operation in fields):
            for name, _ in fields:
                _check_entry(arguments.get(name, '.'))
        for name, operation in fields:
            arguments[name] = resolve_host_path(arguments.get(name, '.'))
            intents.append(HostFileIntent(arguments[name], operation))
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


def _physical_line(stream: TextIO) -> Optional[tuple[str, bool, bool]]:
    # Retain only the display prefix, but consume the entire physical line.
    part = stream.readline(4096)
    if not part:
        return None
    prefix = part[:2001]
    while not part.endswith('\n'):
        part = stream.readline(4096)
        if not part:
            break
        if len(prefix) < 2001:
            prefix += part[:2001 - len(prefix)]
    prefix = prefix.rstrip('\n')
    return prefix[:2000], len(prefix) > 2000, bool(part and part.endswith('\n'))


@fc_register('builtin_tools')
@fc_register('tool', execute_in_sandbox=False)
@fc_register(host_file=_host_files(('file_path', 'read')))
def read(file_path: str, offset: int = 1, limit: int = 2000) -> dict:
    '''Read a bounded text window without loading the whole file.

    Args:
        file_path (str): Absolute path, or a path relative to the current working directory.
        offset (int): First physical file line, starting at one (same as grep).
        limit (int): Maximum physical lines, default and hard cap 2000.

    Returns:
        dict: Up to 64 KiB of UTF-8 text, physical line range, clipped line numbers and EOF. Each line
            has at most 2000 characters. next_offset only advances to later lines;
            use targeted Shell extraction for a clipped line's omitted suffix.
    '''
    if offset < 1 or limit < 1:
        raise ToolExecutionError('offset and limit must be positive')
    limit, max_bytes = min(limit, 2000), MAX_READ_BYTES
    path = _resolve_path(file_path)
    content, size, count, clipped = [], 0, 0, []
    eof = False
    with _open_text(path, errors='replace') as stream:
        for _ in range(offset - 1):
            if _physical_line(stream) is None:
                break
        while count < limit:
            line = _physical_line(stream)
            if line is None:
                eof = True
                break
            prefix, truncated_line, newline = line
            # Preserve record boundaries even when a long line's suffix is omitted.
            display = prefix.rstrip('\n') + ('\n' if newline else '')
            encoded = display.encode('utf-8', errors='replace')
            if size + len(encoded) > max_bytes:
                break
            content.append(display)
            size += len(encoded)
            if truncated_line:
                clipped.append(offset + count)
            count += 1
            if count == limit:
                eof = not stream.read(1)
                break
    result = {'status': 'ok', 'path': path, 'line_unit': 'physical',
              'offset': offset, 'end_line': offset + count - 1,
              'content': ''.join(content), 'eof': eof, 'truncated': not eof or bool(clipped),
              'truncated_lines': clipped, 'next_offset': None if eof else offset + count}
    if clipped:
        result['notice'] = ('Some physical lines were clipped. next_offset reads later lines, not omitted suffixes. '
                            'Use authorized, bounded Shell extraction at the listed physical lines if needed.')
    return result


@fc_register('builtin_tools')
@fc_register('tool', execute_in_sandbox=False)
@fc_register(host_file=_host_files(('path', 'read')))
def ls(path: str = '.', limit: int = 200) -> dict:
    '''List one directory level. Use glob for recursive discovery.

    Args:
        path (str): Directory path, defaults to current directory.
        limit (int): Maximum entries, default 200, hard cap 1000.

    Returns:
        dict: Entries and truncation status.
    '''
    if limit < 1:
        raise ToolExecutionError('limit must be positive')
    limit = min(limit, 1000)
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
def grep(pattern: str, path: str = '.', include: Optional[str] = None) -> dict:
    '''Search file contents with ripgrep without following symbolic links.

    Args:
        pattern (str): Ripgrep regular expression.
        path (str): File or directory to search, defaults to current directory.
        include (str, optional): Filename glob filter, for example *.py or **/*.{yaml,yml}.

    Returns:
        dict: Up to 100 matches with path, line number and bounded snippets, plus truncation status.
            Hidden and ignored files follow ripgrep defaults. Narrow the path or pattern if truncated.
    '''
    if not pattern or len(pattern) > 4096 or '\0' in pattern:
        raise ToolExecutionError('A valid pattern is required')
    path = _resolve_path(path)
    max_results = MAX_SEARCH_RESULTS
    executable = shutil.which('rg')
    if not executable:
        raise ToolExecutionError('grep requires ripgrep (rg) to be installed')
    command = [executable, '--no-config', '--json', '--max-count', str(max_results + 1)]
    if include:
        command += ['--glob', include]
    command += ['--', pattern, path]
    result, _ = _run_rg(command, lambda stream: _read_grep_matches(stream, path, max_results))
    return {'status': 'ok', 'line_unit': 'physical', **result}


def _match_snippet(data: dict) -> tuple[str, bool]:
    raw = data['lines'].get('text', '').rstrip('\r\n')
    if len(raw) <= 1000:
        return raw, False
    matches = data.get('submatches') or []
    encoded = raw.encode('utf-8')
    first = matches[0] if matches else {'start': 0, 'end': 0}
    start = len(encoded[:first['start']].decode('utf-8', 'replace'))
    end = len(encoded[:first['end']].decode('utf-8', 'replace'))
    if end - start > 990:
        return '…' + raw[start:start + 490] + '…[match clipped]…' + raw[end - 490:end] + '…', True
    left = max(0, start - max(0, (996 - (end - start)) // 2))
    right = min(len(raw), left + 996)
    return ('…' if left else '') + raw[left:right] + ('…' if right < len(raw) else ''), True


def _read_grep_matches(stream: BinaryIO, path: str, limit: int) -> tuple[dict, bool]:
    results, size, skipped, output_limit = [], 0, 0, False
    record_limit = 16 * 1024 * 1024
    while line := stream.readline(record_limit + 1):
        if len(line) > record_limit:
            while line and not line.endswith(b'\n'):
                line = stream.readline(65536)
            skipped += 1
            continue
        event = json.loads(line)
        if event['type'] != 'match':
            continue
        data = event['data']
        if 'text' not in data['lines'] or 'text' not in data['path']:
            skipped += 1
            continue
        snippet, clipped = _match_snippet(data)
        item = {'path': data['path'].get('text', path), 'line': str(data['line_number']),
                'text': snippet, 'snippet_truncated': clipped}
        added = len(json.dumps(item, ensure_ascii=False).encode('utf-8')) + 2
        if len(results) == limit or size + added > 65000:
            output_limit = True
            break
        results.append(item)
        size += added
    return {'results': results, 'truncated': output_limit or bool(skipped),
            'output_limit_reached': output_limit, 'skipped_records': skipped,
            'search_incomplete': output_limit or bool(skipped)}, output_limit


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
            results, stop_early = collect(process.stdout)
            if stop_early:
                process.kill()
            code = process.wait()
            if expired.is_set():
                raise ToolExecutionError('Search timed out after 30 seconds; narrow the search path')
            if not stop_early and code not in (0, 1):
                errors.seek(0)
                raise ToolExecutionError('Search failed: ' + errors.read(4096).decode('utf-8', 'replace'))
            return results, stop_early
        finally:
            timer.cancel()
            if process.poll() is None:
                process.kill()
                process.wait()


@fc_register('builtin_tools')
@fc_register('tool', execute_in_sandbox=False)
@fc_register(host_file=_host_files(('path', 'write')))
def mkdir(path: str, parents: bool = True, exist_ok: bool = True) -> dict:
    '''Create a directory.

    Args:
        path (str): Directory path to create.
        parents (bool, optional): Create parent directories. Defaults to True.
        exist_ok (bool, optional): Ignore if already exists. Defaults to True.

    Returns:
        dict: Status result.
    '''
    path_abs = _resolve_path(path)
    if parents:
        os.makedirs(path_abs, exist_ok=exist_ok)
    elif not (exist_ok and os.path.isdir(path_abs)):
        os.mkdir(path_abs)
    return {'status': 'ok', 'path': path_abs}


@fc_register('builtin_tools')
@fc_register('tool', execute_in_sandbox=False)
@fc_register(host_file=_host_files(('file_path', 'write')))
def write(file_path: str, content: str) -> dict:
    '''Create or completely overwrite a UTF-8 text file. Use edit for targeted changes.

    Args:
        file_path (str): Absolute path, or a path relative to the current working directory.
        content (str): Content to write.

    Returns:
        dict: Status result.
    '''
    path_abs = _resolve_path(file_path)
    parent = os.path.dirname(path_abs)

    if parent:
        os.makedirs(parent, exist_ok=True)
    size = _atomic_write(path_abs, content, 'utf-8')
    return {'status': 'ok', 'path': path_abs, 'mode': 'overwrite', 'bytes': size}


@fc_register('builtin_tools')
@fc_register('tool', execute_in_sandbox=False)
@fc_register(host_file=_host_files(('path', 'delete')))
def remove(path: str, recursive: bool = False) -> dict:
    '''Delete a file or directory. Recursive deletion requires explicit opt-in.

    Args:
        path (str): File path.
        recursive (bool): Delete nonempty directories recursively. Defaults to False.
    Returns:
        dict: Status result.
    '''
    _check_entry(path)
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
def move(src: str, dst: str, overwrite: bool = False,
         create_parents: bool = True) -> dict:
    '''Move or rename a file.

    Args:
        src (str): Source path.
        dst (str): Destination path.
        overwrite (bool, optional): Allow overwrite at destination. Defaults to False.
        create_parents (bool, optional): Create destination parents if needed.

    Returns:
        dict: Status result.
    '''
    _check_entry(src)
    _check_entry(dst)
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
@fc_register(host_file=_host_files(('file_path', 'write')))
def edit(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> dict:
    '''Replace exact text in an existing file up to 64 KiB (65536 bytes).

    Larger files require a targeted or streaming workflow.

    Args:
        file_path (str): Absolute path, or a path relative to the current working directory.
        old_string (str): Exact nonempty text to replace.
        new_string (str): Replacement text, which must differ from old_string.
        replace_all (bool): Replace all occurrences. Otherwise old_string must match exactly once.

    Returns:
        dict: Path and replacement count.
    '''
    if not old_string or old_string == new_string:
        raise ToolExecutionError('old_string must be nonempty and differ from new_string')
    path = _resolve_path(file_path)
    with _open_text(path, encoding=None) as stream:
        size = os.fstat(stream.fileno()).st_size
        data = stream.read(MAX_EDIT_BYTES + 1) if size <= MAX_EDIT_BYTES else b''
        if size > MAX_EDIT_BYTES or len(data) > MAX_EDIT_BYTES:
            raise ToolExecutionError(
                'File is too large for edit (limit 65536 bytes). '
                'Use a targeted or streaming workflow for large files.'
            )
        content = data.decode('utf-8')
        count = content.count(old_string)
        if not count or (count != 1 and not replace_all):
            raise ToolExecutionError(f'Expected {"at least one" if replace_all else "1"} matches, found {count}')
    _atomic_write(path, content.replace(old_string, new_string), 'utf-8')
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
def glob(pattern: str, path: str = '.') -> dict:
    '''Find files using ripgrep glob rules without following symbolic links.

    Args:
        pattern (str): File filter: *.yml matches any depth, /*.yml only the search
            root, and **/*.{yaml,yml} matches both extensions recursively.
        path (str): Directory to search, defaults to the working directory.

    Returns:
        dict: Up to 100 matching absolute paths and truncation status, not a total count.
            Hidden and ignored files follow ripgrep defaults. Narrow the path or pattern if truncated.
    '''
    if not pattern or '\x00' in pattern or '..' in PurePath(pattern).parts:
        raise ToolExecutionError('A valid glob pattern is required')
    max_results = MAX_SEARCH_RESULTS
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
