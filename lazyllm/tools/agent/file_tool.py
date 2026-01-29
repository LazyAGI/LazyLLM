import fnmatch
import os
import re
import shutil
from typing import Dict, List, Optional

from .toolsManager import register


def _resolve_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _check_root(path: str, root: Optional[str]) -> Optional[Dict[str, str]]:
    if not root:
        return None
    root_abs = _resolve_path(root)
    path_abs = _resolve_path(path)
    if os.path.commonpath([path_abs, root_abs]) != root_abs:
        return {
            'status': 'needs_approval',
            'reason': 'Path is outside the allowed root.',
            'path': path_abs,
            'root': root_abs,
        }
    return None


@register('tool')
def read_file(path: str, start_line: Optional[int] = None, end_line: Optional[int] = None,
              encoding: str = 'utf-8', errors: str = 'replace', root: Optional[str] = None,
              max_chars: int = 200000) -> dict:
    '''Read a text file with optional line range.

    Args:
        path (str): File path.
        start_line (int, optional): 1-based start line (inclusive).
        end_line (int, optional): 1-based end line (inclusive).
        encoding (str, optional): File encoding. Defaults to utf-8.
        errors (str, optional): Error handling for decoding. Defaults to replace.
        root (str, optional): Restrict reads to this root directory.
        max_chars (int, optional): Max chars to return. Defaults to 200000.

    Returns:
        dict: Content and metadata.
    '''
    guard = _check_root(path, root)
    if guard:
        return guard
    path_abs = _resolve_path(path)
    if not os.path.isfile(path_abs):
        return {'status': 'missing', 'path': path_abs}
    with open(path_abs, 'r', encoding=encoding, errors=errors) as f:
        lines = f.readlines()
    total_lines = len(lines)
    s = 1 if start_line is None else max(1, start_line)
    e = total_lines if end_line is None else min(end_line, total_lines)
    content = ''.join(lines[s - 1:e])
    truncated = False
    if max_chars is not None and len(content) > max_chars:
        content = content[:max_chars]
        truncated = True
    return {
        'status': 'ok',
        'path': path_abs,
        'start_line': s,
        'end_line': e,
        'total_lines': total_lines,
        'truncated': truncated,
        'content': content,
    }


@register('tool')
def list_dir(path: str = '.', recursive: bool = False, max_depth: int = 5,
             root: Optional[str] = None) -> dict:
    '''List directory entries.

    Args:
        path (str, optional): Directory path. Defaults to current directory.
        recursive (bool, optional): Whether to walk recursively.
        max_depth (int, optional): Max recursion depth. Defaults to 5.
        root (str, optional): Restrict listing to this root directory.

    Returns:
        dict: List of entries.
    '''
    guard = _check_root(path, root)
    if guard:
        return guard
    path_abs = _resolve_path(path)
    if not os.path.isdir(path_abs):
        return {'status': 'missing', 'path': path_abs}
    entries: List[str] = []
    if not recursive:
        entries = sorted(os.listdir(path_abs))
    else:
        base_depth = path_abs.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(path_abs):
            depth = dirpath.count(os.sep) - base_depth
            if depth > max_depth:
                dirnames[:] = []
                continue
            rel = os.path.relpath(dirpath, path_abs)
            if rel == '.':
                rel = ''
            for d in dirnames:
                entries.append(os.path.join(rel, d) if rel else d)
            for f in filenames:
                entries.append(os.path.join(rel, f) if rel else f)
    return {'status': 'ok', 'path': path_abs, 'entries': entries}


@register('tool')
def search_in_files(pattern: str, path: str = '.', glob: Optional[str] = None,
                    max_results: int = 50, root: Optional[str] = None,
                    encoding: str = 'utf-8', errors: str = 'replace',
                    max_file_size: int = 2_000_000) -> dict:
    '''Search files for a regex pattern.

    Args:
        pattern (str): Regex pattern to search for.
        path (str, optional): Root path to search. Defaults to current directory.
        glob (str, optional): Filename glob filter (e.g., "*.py").
        max_results (int, optional): Max number of matches to return.
        root (str, optional): Restrict search to this root directory.
        encoding (str, optional): File encoding. Defaults to utf-8.
        errors (str, optional): Error handling for decoding. Defaults to replace.
        max_file_size (int, optional): Skip files larger than this size in bytes.

    Returns:
        dict: List of matches with file path and line number.
    '''
    guard = _check_root(path, root)
    if guard:
        return guard
    path_abs = _resolve_path(path)
    if not os.path.isdir(path_abs):
        return {'status': 'missing', 'path': path_abs}
    regex = re.compile(pattern)
    results: List[Dict[str, str]] = []
    for dirpath, _, filenames in os.walk(path_abs):
        for name in filenames:
            if glob and not fnmatch.fnmatch(name, glob):
                continue
            file_path = os.path.join(dirpath, name)
            try:
                if os.path.getsize(file_path) > max_file_size:
                    continue
                with open(file_path, 'r', encoding=encoding, errors=errors) as f:
                    for idx, line in enumerate(f, start=1):
                        if regex.search(line):
                            results.append({
                                'path': file_path,
                                'line': str(idx),
                                'text': line.rstrip('\n'),
                            })
                            if len(results) >= max_results:
                                return {'status': 'ok', 'results': results}
            except (OSError, UnicodeDecodeError):
                continue
    return {'status': 'ok', 'results': results}


@register('tool')
def make_dir(path: str, parents: bool = True, exist_ok: bool = True,
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
    guard = _check_root(path, root)
    if guard:
        return guard
    path_abs = _resolve_path(path)
    os.makedirs(path_abs, exist_ok=exist_ok) if parents else os.mkdir(path_abs)
    return {'status': 'ok', 'path': path_abs}


@register('tool')
def write_file(path: str, content: str, mode: str = 'overwrite', encoding: str = 'utf-8',
               root: Optional[str] = None, create_parents: bool = True,
               allow_unsafe: bool = False) -> dict:
    '''Write content to a file.

    Args:
        path (str): File path.
        content (str): Content to write.
        mode (str, optional): overwrite|append. Defaults to overwrite.
        encoding (str, optional): File encoding. Defaults to utf-8.
        root (str, optional): Restrict writes to this root directory.
        create_parents (bool, optional): Create parent directories if needed.
        allow_unsafe (bool, optional): Allow overwriting existing files. Defaults to False.

    Returns:
        dict: Status result.
    '''
    guard = _check_root(path, root)
    if guard:
        return guard
    path_abs = _resolve_path(path)
    if mode not in ('overwrite', 'append'):
        raise ValueError('mode must be "overwrite" or "append".')
    if os.path.exists(path_abs) and not allow_unsafe:
        return {
            'status': 'needs_approval',
            'reason': 'Writing to an existing file requires approval.',
            'path': path_abs,
            'mode': mode,
        }
    parent = os.path.dirname(path_abs)

    if parent and create_parents:
        os.makedirs(parent, exist_ok=True)
    fmode = 'a' if mode == 'append' else 'w'
    with open(path_abs, fmode, encoding=encoding) as f:
        f.write(content)
    return {'status': 'ok', 'path': path_abs, 'mode': mode, 'bytes': len(content)}


@register('tool')
def delete_file(path: str, root: Optional[str] = None, allow_unsafe: bool = False) -> dict:
    '''Delete a file.

    Args:
        path (str): File path.
        root (str, optional): Restrict deletion to this root directory.
        allow_unsafe (bool, optional): Allow deletion. Defaults to False.
    Returns:
        dict: Status result.
    '''
    guard = _check_root(path, root)
    if guard:
        return guard
    path_abs = _resolve_path(path)
    if not os.path.exists(path_abs):
        return {'status': 'missing', 'path': path_abs}
    if not allow_unsafe:
        return {
            'status': 'needs_approval',
            'reason': 'Deleting files requires approval.',
            'path': path_abs,
        }
    os.remove(path_abs)
    return {'status': 'ok', 'path': path_abs}


@register('tool')
def move_file(src: str, dst: str, root: Optional[str] = None, allow_unsafe: bool = False,
              overwrite: bool = False, create_parents: bool = True) -> dict:
    '''Move or rename a file.

    Args:
        src (str): Source path.
        dst (str): Destination path.
        root (str, optional): Restrict operations to this root directory.
        allow_unsafe (bool, optional): Allow move/rename. Defaults to False.
        overwrite (bool, optional): Allow overwrite at destination. Defaults to False.
        create_parents (bool, optional): Create destination parents if needed.

    Returns:
        dict: Status result.
    '''
    guard = _check_root(src, root) or _check_root(dst, root)
    if guard:
        return guard
    src_abs = _resolve_path(src)
    dst_abs = _resolve_path(dst)
    if not os.path.exists(src_abs):
        return {'status': 'missing', 'path': src_abs}
    if os.path.exists(dst_abs) and not overwrite:
        raise FileExistsError(f'destination exists: {dst_abs}')
    if not allow_unsafe:
        return {
            'status': 'needs_approval',
            'reason': 'Moving/renaming files requires approval.',
            'src': src_abs,
            'dst': dst_abs,
        }
    parent = os.path.dirname(dst_abs)
    if parent and create_parents:
        os.makedirs(parent, exist_ok=True)
    shutil.move(src_abs, dst_abs)
    return {'status': 'ok', 'src': src_abs, 'dst': dst_abs}
