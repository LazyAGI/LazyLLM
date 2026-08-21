import os
import urllib.error
import urllib.request
from typing import Optional

from .toolsManager import register
from .file_tool import _check_root, _resolve_path
from .toolError import ToolExecutionError


@register('builtin_tools', execute_in_sandbox=False)
@register('tool', execute_in_sandbox=False)
def download_file(url: str, dst: str, timeout: int = 30, root: Optional[str] = None,
                  allow_unsafe: bool = False) -> dict:
    '''Download a file from a URL to a local path.

    Args:
        url (str): HTTP/HTTPS URL to download.
        dst (str): Destination file path.
        timeout (int, optional): Request timeout in seconds. Defaults to 30.
        root (str, optional): Restrict writes to this root directory.
        allow_unsafe (bool, optional): Allow network download. Defaults to False.

    Returns:
        dict: Status result.
    '''
    if not url or not url.startswith(('http://', 'https://')):
        raise ToolExecutionError(f'Only http/https URLs are supported, got: {url!r}.')

    _check_root(dst, root)

    dst_abs = _resolve_path(dst)
    if not allow_unsafe:
        raise ToolExecutionError.approval_required(
            f'Downloading {url} to {dst_abs} requires approval.'
        )

    parent = os.path.dirname(dst_abs)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(dst_abs, 'wb') as f:
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
