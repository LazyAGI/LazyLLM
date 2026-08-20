import os
import urllib.error
import urllib.request
from typing import Optional

from .toolsManager import register
from .file_tool import _check_root, _resolve_path
from .toolError import (
    ToolDomainError,
    ToolInvalidArgumentsError,
    ToolPermissionError,
    ToolTransientError,
)


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
        raise ToolInvalidArgumentsError(
            'Only http/https URLs are supported.',
            code='UNSUPPORTED_URL_SCHEME',
            details={'url': url},
        )

    _check_root(dst, root)

    dst_abs = _resolve_path(dst)
    if not allow_unsafe:
        raise ToolPermissionError(
            'Downloading remote files requires approval.',
            code='DOWNLOAD_REQUIRES_APPROVAL',
            details={'url': url, 'path': dst_abs, 'authorization_required': True},
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
        details = {'url': url, 'path': dst_abs, 'status_code': exc.code}
        if exc.code in (401, 403):
            raise ToolPermissionError(
                f'Download is not permitted: {exc}',
                code='DOWNLOAD_PERMISSION_DENIED',
                details=details,
            ) from exc
        if exc.code in (408, 429, 502, 503, 504):
            raise ToolTransientError(
                f'Download failed temporarily: {exc}',
                code='DOWNLOAD_TEMPORARY_FAILURE',
                details=details,
            ) from exc
        raise ToolDomainError(
            f'Download failed: {exc}',
            code='DOWNLOAD_FAILED',
            details=details,
        ) from exc
    except (TimeoutError, ConnectionError, urllib.error.URLError) as exc:
        raise ToolTransientError(
            f'Download failed temporarily: {exc}',
            code='DOWNLOAD_TEMPORARY_FAILURE',
            details={'url': url, 'path': dst_abs},
        ) from exc
    except Exception as exc:
        raise ToolDomainError(
            f'Download failed: {exc}',
            code='DOWNLOAD_FAILED',
            details={'url': url, 'path': dst_abs},
        ) from exc
