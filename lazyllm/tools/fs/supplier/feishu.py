# Copyright (c) 2026 LazyAGI. All rights reserved.
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse

import requests

from lazyllm import LOG, config

from ..base import LazyLLMFSBase, CloudFSBufferedFile

config.add('feishu_app_id', str, None, 'FEISHU_APP_ID', description='Feishu App ID for tenant_access_token.')
config.add('feishu_app_secret', str, None, 'FEISHU_APP_SECRET', description='Feishu App Secret for tenant_access_token.')


_API_BASE = 'https://open.feishu.cn/open-apis'
_TOKEN_URL = 'https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal'
_USER_TOKEN_URL = 'https://open.feishu.cn/open-apis/authen/v2/oauth/token'
_OAUTH_AUTHORIZE_URL = 'https://accounts.feishu.cn/open-apis/authen/v1/authorize'
_TOKEN_REFRESH_BUFFER = 300  # refresh if remaining life <= 5 min
_OAUTH_TIMEOUT = 300         # seconds to wait for user to complete browser auth
# Default OAuth scope: offline_access is required for refresh_token; drive scopes for file access;
# wiki scopes for knowledge-base (wiki space) access.
# Only scopes enabled in the app's permission management (user identity) will actually be granted.
_DEFAULT_OAUTH_SCOPE = (
    'offline_access '
    'drive:drive drive:drive:readonly drive:drive.metadata:readonly '
    'wiki:wiki wiki:wiki:readonly wiki:node:retrieve'
)


def _feishu_acquire_access_token(
    session: requests.Session,
    app_id: str,
    app_secret: str,
) -> Tuple[str, Optional[float]]:
    if not app_id or not app_secret:
        return '', None
    now = time.time()
    payload = {'app_id': app_id, 'app_secret': app_secret}
    resp = session.post(
        _TOKEN_URL, json=payload,
        headers={'Content-Type': 'application/json; charset=utf-8'},
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get('code') != 0:
        raise RuntimeError(
            'Feishu tenant_access_token failed: %s' % data.get('msg', data)
        )
    token = data.get('tenant_access_token', '')
    expire_sec = int(data.get('expire', 0))
    if not token or not expire_sec:
        return '', None
    expires_at = now + expire_sec - _TOKEN_REFRESH_BUFFER
    return token, expires_at


def _feishu_oauth_get_code(app_id: str, port: int, scope: str = _DEFAULT_OAUTH_SCOPE) -> Tuple[str, str]:
    redirect_uri = f'http://localhost:{port}/callback'
    result: Dict[str, str] = {}
    done = threading.Event()

    class _CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            qs = parse_qs(urlparse(self.path).query)
            if 'code' in qs:
                result['code'] = qs['code'][0]
                body = b'<html><body>Authorization successful. You can close this tab.</body></html>'
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(body)
            else:
                result['error'] = qs.get('error', ['unknown'])[0]
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'Authorization failed.')
            done.set()

        def log_message(self, *args): pass  # suppress server access logs

    try:
        server = HTTPServer(('localhost', port), _CallbackHandler)
    except OSError as e:
        raise RuntimeError(
            f'Cannot bind to localhost:{port} for Feishu OAuth callback. '
            f'Please free the port or pass a different oauth_port. ({e})'
        )

    threading.Thread(target=server.handle_request, daemon=True).start()

    auth_url = f'{_OAUTH_AUTHORIZE_URL}?' + urlencode({
        'client_id': app_id,
        'redirect_uri': redirect_uri,
        'scope': scope,
        'response_type': 'code',
    })
    LOG.success(
        f'Feishu OAuth: open the link below in a browser to authorize access.\n'
        f'  Prerequisites:\n'
        f'    1. Register {redirect_uri} in your Feishu app Security Settings → Redirect URL.\n'
        f'    2. Enable user-identity drive permissions (e.g. drive:drive) in Permission Management\n'
        f'       and enable offline_access, then publish the app.\n\n'
        f'  {auth_url}\n'
    )

    if not done.wait(timeout=_OAUTH_TIMEOUT):
        server.server_close()
        raise TimeoutError(f'Feishu OAuth timed out after {_OAUTH_TIMEOUT}s waiting for browser authorization.')
    server.server_close()

    if 'error' in result:
        raise RuntimeError(f'Feishu OAuth authorization denied: {result["error"]}')
    return result['code'], redirect_uri


def _feishu_exchange_code(
    session: requests.Session,
    app_id: str,
    app_secret: str,
    code: str,
    redirect_uri: str,
) -> Tuple[str, Optional[float], str]:
    now = time.time()
    payload = {
        'grant_type': 'authorization_code',
        'client_id': app_id,
        'client_secret': app_secret,
        'code': code,
        'redirect_uri': redirect_uri,
    }
    resp = session.post(_USER_TOKEN_URL, json=payload,
                        headers={'Content-Type': 'application/json; charset=utf-8'})
    resp.raise_for_status()
    data = resp.json()
    if data.get('code', 0) != 0:
        raise RuntimeError('Feishu code exchange failed: %s' % data.get('error_description', data))
    token = data.get('access_token', '')
    expire_sec = int(data.get('expires_in', 0))
    refresh = data.get('refresh_token', '')
    expires_at = (now + expire_sec - _TOKEN_REFRESH_BUFFER) if expire_sec else None
    return token, expires_at, refresh


def _feishu_refresh_user_token(
    session: requests.Session,
    app_id: str,
    app_secret: str,
    refresh_token: str,
) -> Tuple[str, Optional[float], str]:
    now = time.time()
    payload = {
        'grant_type': 'refresh_token',
        'client_id': app_id,
        'client_secret': app_secret,
        'refresh_token': refresh_token,
    }
    resp = session.post(_USER_TOKEN_URL, json=payload,
                        headers={'Content-Type': 'application/json; charset=utf-8'})
    resp.raise_for_status()
    data = resp.json()
    if data.get('code', 0) != 0:
        raise RuntimeError('Feishu user_access_token refresh failed: %s' % data.get('error_description', data))
    token = data.get('access_token', '')
    expire_sec = int(data.get('expires_in', 0))
    new_refresh = data.get('refresh_token', refresh_token)
    expires_at = (now + expire_sec - _TOKEN_REFRESH_BUFFER) if expire_sec else None
    return token, expires_at, new_refresh


class FeishuFSBase(LazyLLMFSBase):
    __lazyllm_registry_disable__ = True

    def __init__(self, base_url: Optional[str] = None, app_id: Optional[str] = None, app_secret: Optional[str] = None,
                 space_id: Optional[str] = None, user_refresh_token: Optional[str] = None,
                 oauth_port: int = 9981, oauth_scope: str = _DEFAULT_OAUTH_SCOPE,
                 asynchronous: bool = False, use_listings_cache: bool = False,
                 skip_instance_cache: bool = False, loop: Optional[Any] = None):
        if not app_id or not app_secret:
            app_id, app_secret = config['feishu_app_id'] or app_id, config['feishu_app_secret']
        assert app_id and app_secret, 'feishu_app_id and feishu_app_secret are required'
        # _secret_key stores app credentials; _user_refresh_token / _oauth_port managed separately
        self._user_refresh_token: str = user_refresh_token or ''
        self._oauth_port: int = oauth_port
        self._oauth_scope: str = oauth_scope
        super().__init__(
            token={'app_id': app_id, 'app_secret': app_secret},
            base_url=base_url or _API_BASE,
            asynchronous=asynchronous,
            use_listings_cache=use_listings_cache,
            skip_instance_cache=skip_instance_cache,
            loop=loop,
        )
        self._space_id = (space_id or '').strip() if space_id is not None else ''

    @property
    def _app_id(self) -> str: return self._secret_key.get('app_id', '')

    @property
    def _app_secret(self) -> str: return self._secret_key.get('app_secret', '')

    def _setup_auth(self) -> None:
        self._session.headers.update({
            'Content-Type': 'application/json; charset=utf-8',
        })

    def _acquire_access_token(self) -> Tuple[str, Optional[float]]:
        if self._user_refresh_token == 'auto':
            code, redirect_uri = _feishu_oauth_get_code(self._app_id, self._oauth_port, self._oauth_scope)
            token, expires_at, self._user_refresh_token = _feishu_exchange_code(
                self._session, self._app_id, self._app_secret, code, redirect_uri)
            return token, expires_at
        if self._user_refresh_token:
            token, expires_at, self._user_refresh_token = _feishu_refresh_user_token(
                self._session, self._app_id, self._app_secret, self._user_refresh_token)
            return token, expires_at
        return _feishu_acquire_access_token(self._session, self._app_id, self._app_secret)

    def _apply_access_token(self, token: str) -> None:
        self._session.headers.update({'Authorization': f'Bearer {token}'})

    def get_user_refresh_token(self) -> str:
        return self._user_refresh_token

    @staticmethod
    def _token_from_path(path: str) -> str:
        stripped = path.lstrip('/')
        if not stripped:
            return ''
        return stripped.split('/')[-1]


class FeishuFS(FeishuFSBase):

    def __new__(cls, base_url: Optional[str] = None, app_id: Optional[str] = None, app_secret: Optional[str] = None,
                space_id: Optional[str] = None, user_refresh_token: Optional[str] = None,
                oauth_port: int = 9981, oauth_scope: str = _DEFAULT_OAUTH_SCOPE,
                asynchronous: bool = False, use_listings_cache: bool = False,
                skip_instance_cache: bool = False, loop: Optional[Any] = None) -> LazyLLMFSBase:
        if space_id is not None and str(space_id).strip():
            return FeishuWikiFS(base_url=base_url, app_id=app_id, app_secret=app_secret, space_id=space_id,
                                user_refresh_token=user_refresh_token, oauth_port=oauth_port,
                                oauth_scope=oauth_scope, asynchronous=asynchronous,
                                use_listings_cache=use_listings_cache,
                                skip_instance_cache=skip_instance_cache, loop=loop)
        return super().__new__(cls)

    def ls(self, path: str = '/', detail: bool = True, **kwargs) -> List:
        folder_token = self._token_from_path(path)
        url = f'{self._base_url}/drive/v1/files'
        # Root listing does not support pagination; do not send page_size (API returns 400).
        params: Dict[str, Any] = {}
        if folder_token:
            params['folder_token'] = folder_token
            params['page_size'] = 200

        results = []
        page_token = None
        while True:
            if page_token:
                params['page_token'] = page_token
            data = self._get(url, params=params)
            items = data.get('data', {}).get('files', []) or []
            for item in items:
                entry = self._item_to_entry(item)
                results.append(entry if detail else entry['name'])
            if not folder_token:
                break  # Root: no pagination
            if not (page_token := data.get('data', {}).get('next_page_token')):
                break
        return results

    def info(self, path: str, **kwargs) -> Dict[str, Any]:
        token = self._token_from_path(path)
        if not token:
            return self._entry('/', ftype='directory')
        url = f'{self._base_url}/drive/v1/files/{token}/statistics'
        try:
            data = self._get(url)
            stat = data.get('data', {}).get('stats', {})
            return self._entry(
                name=path, size=0, ftype='file',
                mtime=stat.get('edit_time'), extra_info=stat,
            )
        except Exception:
            return self._entry(name=path, ftype='directory')

    def _open(self, path: str, mode: str = 'rb',
              block_size: Optional[int] = None,
              autocommit: bool = True,
              cache_options: Optional[Dict] = None,
              **kwargs) -> CloudFSBufferedFile:
        return CloudFSBufferedFile(
            self, path, mode=mode,
            block_size=block_size or self.blocksize,
            autocommit=autocommit, cache_options=cache_options,
        )

    def mkdir(self, path: str, create_parents: bool = True, **kwargs) -> None:
        parts = self._parse_path(path)
        if len(parts) >= 2:
            parent_token, name = parts[-2], parts[-1]
        else:
            parent_token = ''
            name = parts[0] if parts else 'New Folder'
        url = f'{self._base_url}/drive/v1/files/create_folder'
        payload: Dict[str, Any] = {'name': name}
        if parent_token:
            payload['folder_token'] = parent_token
        self._post(url, json=payload)

    def rm_file(self, path: str) -> None:
        token = self._token_from_path(path)
        if not token:
            raise FileNotFoundError(f'Cannot determine file token from path: {path!r}')
        url = f'{self._base_url}/drive/v1/files/{token}'
        self._delete(url, params={'type': 'file'})

    def rmdir(self, path: str) -> None:
        token = self._token_from_path(path)
        if not token:
            return
        url = f'{self._base_url}/drive/v1/files/{token}'
        self._delete(url, params={'type': 'folder'})

    def _download_range(self, path: str, start: int, end: int) -> bytes:
        token = self._token_from_path(path)
        url = f'{self._base_url}/drive/v1/files/{token}/download'
        headers = {'Range': f'bytes={start}-{end - 1}'}
        resp = self._request('GET', url, headers=headers)
        return resp.content

    def _upload_data(self, path: str, data: bytes) -> None:
        parent, name = self._split_parent_name(path)
        prepare_url = f'{self._base_url}/drive/v1/files/upload_prepare'
        prepare_payload = {
            'file_name': name,
            'parent_type': 'explorer',
            'parent_node': parent or 'root',
            'size': len(data),
        }
        resp_data = self._post(prepare_url, json=prepare_payload)
        upload_id = resp_data.get('data', {}).get('upload_id', '')
        block_size = resp_data.get('data', {}).get('block_size', len(data))
        num_blocks = resp_data.get('data', {}).get('block_num', 1)

        part_url = f'{self._base_url}/drive/v1/files/upload_part'
        for i in range(num_blocks):
            chunk = data[i * block_size: (i + 1) * block_size]
            self._request('POST', part_url, data={
                'upload_id': upload_id,
                'seq': str(i),
                'size': str(len(chunk)),
                'file': chunk,
            })

        finish_url = f'{self._base_url}/drive/v1/files/upload_finish'
        self._post(finish_url, json={'upload_id': upload_id, 'block_num': num_blocks})

    def _platform_supports_webhook(self) -> bool:
        return True

    def _register_webhook(self, webhook_url: str, events: List[str], path: str) -> Dict[str, Any]:
        token = self._token_from_path(path)
        url = f'{self._base_url}/event/v1/bot/customize/event_callback'
        payload: Dict[str, Any] = {
            'event_type': events or ['drive.file.edit_v1'],
            'callback_url': webhook_url,
        }
        if token:
            payload['file_token'] = token
        return self._post(url, json=payload)

    @staticmethod
    def _split_parent_name(path: str) -> tuple:
        parts = path.strip('/').split('/')
        if len(parts) >= 2:
            return parts[-2], parts[-1]
        return '', parts[-1] if parts else ''

    @staticmethod
    def _item_to_entry(item: Dict[str, Any]) -> Dict[str, Any]:
        ftype = 'directory' if item.get('type') == 'folder' else 'file'
        name = item.get('token') or item.get('name', '')
        mtime = item.get('modified_time') or item.get('edit_time')
        return LazyLLMFSBase._entry(
            name=name, size=item.get('size', 0), ftype=ftype,
            mtime=float(mtime) if mtime else None,
            title=item.get('name', name),
        )


class FeishuWikiFS(FeishuFSBase):
    __lazyllm_registry_disable__ = True

    def _require_space_id(self) -> None:
        if not self._space_id:
            raise ValueError('space_id is required for FeishuWikiFS')

    def ls(self, path: str, detail: bool = True, **kwargs) -> List:
        self._require_space_id()
        node_token = self._token_from_path(path)
        url = f'{self._base_url}/wiki/v2/spaces/{self._space_id}/nodes'
        params: Dict[str, Any] = {'page_size': 50}  # Wiki API max is 50
        if node_token:
            params['parent_node_token'] = node_token

        results: List[Any] = []
        page_token: Optional[str] = None
        while True:
            if page_token:
                params['page_token'] = page_token
            data = self._get(url, params=params)
            items = (
                data.get('data', {}).get('items')
                or data.get('data', {}).get('nodes')
                or []
            )
            for item in items:
                entry = self._node_to_entry(item)
                results.append(entry if detail else entry['name'])
            if not (page_token := data.get('data', {}).get('page_token') or data.get('data', {}).get('next_page_token')):
                break
        return results

    def info(self, path: str, **kwargs) -> Dict[str, Any]:
        token = self._token_from_path(path)
        if not token:
            return self._entry('/', ftype='directory')
        node = self._get_node(token)
        return self._node_to_entry(node, default_name=path)

    def _open(self, path: str, mode: str = 'rb',
              block_size: Optional[int] = None,
              autocommit: bool = True,
              cache_options: Optional[Dict] = None,
              **kwargs) -> CloudFSBufferedFile:
        if 'b' not in mode:
            raise ValueError('FeishuWikiFS only supports binary mode')
        return CloudFSBufferedFile(
            self, path, mode=mode,
            block_size=block_size or self.blocksize,
            autocommit=autocommit, cache_options=cache_options,
        )

    def mkdir(self, path: str, create_parents: bool = True, **kwargs) -> None:
        raise NotImplementedError('FeishuWikiFS does not support mkdir yet')

    def rm_file(self, path: str) -> None:
        raise NotImplementedError('FeishuWikiFS does not support rm_file yet')

    def rmdir(self, path: str) -> None:
        raise NotImplementedError('FeishuWikiFS does not support rmdir yet')

    def _download_range(self, path: str, start: int, end: int) -> bytes:
        token = self._token_from_path(path)
        if not token:
            raise FileNotFoundError(path)
        node = self._get_node(token)
        obj_type = node.get('obj_type')
        obj_token = node.get('obj_token') or ''
        if not obj_type or not obj_token:
            return b''
        if obj_type in {'doc', 'docx'}:
            data = self._download_doc_raw(obj_token, obj_type=obj_type)
        elif obj_type == 'file':
            data = self._download_file_raw(obj_token)
        else:
            data = b''
        return data[start:end]

    def _upload_data(self, path: str, data: bytes) -> None:
        raise NotImplementedError('FeishuWikiFS does not support write yet')

    def _download_doc_raw(self, doc_token: str, obj_type: str = 'docx') -> bytes:
        if obj_type == 'docx':
            url = f'{self._base_url}/docx/v1/documents/{doc_token}/raw_content'
            resp_json = self._get(url)
            if resp_json.get('code', 0) != 0:
                raise RuntimeError(
                    'Feishu docx raw_content failed (ensure docx:document or '
                    'docx:document:readonly user permission is enabled in Permission Management): '
                    '%s' % resp_json.get('msg', resp_json)
                )
        else:
            url = f'{self._base_url}/doc/v2/{doc_token}/raw_content'
            resp_json = self._get(url)
            if resp_json.get('code', 0) != 0:
                raise RuntimeError('Feishu doc raw_content failed: %s' % resp_json.get('msg', resp_json))
        content = resp_json.get('data', {}).get('content', '')
        return content.encode('utf-8') if isinstance(content, str) else b''

    def _download_file_raw(self, file_token: str) -> bytes:
        url = f'{self._base_url}/drive/v1/files/{file_token}/download'
        resp = self._request('GET', url)
        return resp.content

    def _get_node(self, token: str) -> Dict[str, Any]:
        url = f'{self._base_url}/wiki/v2/spaces/get_node'
        data = self._get(url, params={'token': token})
        node = data.get('data', {}).get('node')
        if not node:
            node = data.get('data', {})
        return node or {}

    @staticmethod
    def _node_to_entry(node: Dict[str, Any], default_name: Optional[str] = None) -> Dict[str, Any]:
        obj_type = node.get('obj_type')
        # has_child: true means the node has sub-pages and must be navigable as a directory,
        # even if its obj_type is 'docx' (wiki allows documents to also be folder nodes).
        is_dir = obj_type in {'folder', 'wiki', 'space'} or bool(node.get('has_child'))
        name = node.get('node_token') or node.get('token') or default_name or ''
        title = node.get('title') or node.get('obj_name') or name
        size = int(node.get('size', 0) or 0)
        mtime = None
        ts = node.get('update_time') or node.get('edit_time') or node.get('modified_time')
        if ts:
            try:
                mtime = float(ts)
            except (TypeError, ValueError):
                mtime = None
        return LazyLLMFSBase._entry(
            name=name,
            size=size,
            ftype='directory' if is_dir else 'file',
            mtime=mtime,
            title=title,
            obj_type=obj_type,
            obj_token=node.get('obj_token'),
        )
