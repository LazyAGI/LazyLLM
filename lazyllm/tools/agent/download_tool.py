from . import host_file_io
import os
import urllib.error
import urllib.request
from typing import Optional

from .toolsManager import register
from .file_tool import _check_root, _resolve_path, _host_files
from .toolError import ToolExecutionError


@register('builtin_tools', execute_in_sandbox=False)
@register('tool', execute_in_sandbox=False)
@register(host_file=_host_files(('dst', 'write')))
def download_file(url: str, dst: str, timeout: int = 30, root: Optional[str] = None) -> dict:
    '''Download a file from a URL to a local path.

    Args:
        url (str): HTTP/HTTPS URL to download.
        dst (str): Destination file path.
        timeout (int, optional): Request timeout in seconds. Defaults to 30.
        root (str, optional): Restrict writes to this root directory.

    Returns:
        dict: Status result.
    '''
    if not url or not url.startswith(('http://', 'https://')):
        raise ToolExecutionError(f'Only http/https URLs are supported, got: {url!r}.')

    _check_root(dst, root)

    dst_abs = _resolve_path(dst)

    parent = os.path.dirname(dst_abs)
    if parent:
        host_file_io.makedirs(parent, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, host_file_io.open_write(dst_abs) as f:
            data = resp.read()
            f.write(data)
        return {'status': 'ok', 'path': dst_abs, 'bytes': len(data)}
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise ToolExecutionError(
                f'Download from {url} to {dst_abs} is not permitted (HTTP {exc.code}): {exc}',
            ) from exc
        if exc.code in (408, 429, 502, 503, 504):
            raise ToolExecutionError(
                f'Download from {url} to {dst_abs} failed temporarily (HTTP {exc.code}): {exc}',
            ) from exc
        raise ToolExecutionError(
            f'Download from {url} to {dst_abs} failed (HTTP {exc.code}): {exc}',
        ) from exc
    except (TimeoutError, ConnectionError, urllib.error.URLError) as exc:
        raise ToolExecutionError(
            f'Download from {url} to {dst_abs} failed temporarily: {exc}',
        ) from exc
    except Exception as exc:
        raise ToolExecutionError(f'Download from {url} to {dst_abs} failed: {exc}') from exc
