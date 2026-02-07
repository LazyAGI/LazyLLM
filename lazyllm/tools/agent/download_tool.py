import os
import urllib.request
from typing import Optional

from .toolsManager import register
from .file_tool import _check_root, _resolve_path


@register('builtin_tools')
@register('tool')
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
        return {'status': 'error', 'reason': 'Only http/https URLs are supported.', 'url': url}

    guard = _check_root(dst, root)
    if guard:
        return guard

    dst_abs = _resolve_path(dst)
    if not allow_unsafe:
        return {
            'status': 'needs_approval',
            'reason': 'Downloading remote files requires approval.',
            'url': url,
            'path': dst_abs,
        }

    parent = os.path.dirname(dst_abs)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(dst_abs, 'wb') as f:
            data = resp.read()
            f.write(data)
        return {'status': 'ok', 'path': dst_abs, 'bytes': len(data)}
    except Exception as exc:
        return {'status': 'error', 'reason': str(exc), 'url': url, 'path': dst_abs}
