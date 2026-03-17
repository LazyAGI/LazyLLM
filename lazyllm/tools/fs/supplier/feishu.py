# Copyright (c) 2026 LazyAGI. All rights reserved.
import os
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


_TOKENS_FILE = os.path.join(config['home'], '.lazyllm/tokens.txt')

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
    'wiki:wiki wiki:wiki:readonly wiki:node:retrieve docx:document'
)


def _load_persisted_token(key: str) -> str:
    try:
        with open(_TOKENS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                k, sep, v = line.strip().partition(': ')
                if sep and k.strip() == key:
                    return v.strip()
    except FileNotFoundError:
        pass
    return ''


def _save_persisted_token(key: str, value: str) -> None:
    os.makedirs(os.path.dirname(_TOKENS_FILE), exist_ok=True)
    lines: List[str] = []
    found = False
    try:
        with open(_TOKENS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                k, sep, _ = line.strip().partition(': ')
                if sep and k.strip() == key:
                    if value:  # empty value → delete the entry (don't write the line)
                        found = True
                        lines.append(f'{key}: {value}\n')
                else:
                    lines.append(line if line.endswith('\n') else line + '\n')
    except FileNotFoundError:
        pass
    if not found and value:
        lines.append(f'{key}: {value}\n')
    with open(_TOKENS_FILE, 'w', encoding='utf-8') as f:
        f.writelines(lines)


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
        self._oauth_auto: bool = (user_refresh_token == 'auto')
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

    def _do_oauth(self, token_key: str) -> Tuple[str, Optional[float]]:
        code, redirect_uri = _feishu_oauth_get_code(self._app_id, self._oauth_port, self._oauth_scope)
        token, expires_at, self._user_refresh_token = _feishu_exchange_code(
            self._session, self._app_id, self._app_secret, code, redirect_uri)
        _save_persisted_token(token_key, self._user_refresh_token)
        return token, expires_at

    def _acquire_access_token(self) -> Tuple[str, Optional[float]]:
        token_key = f'feishu:{self._app_id}'
        if self._user_refresh_token == 'auto':
            persisted = _load_persisted_token(token_key)
            if persisted:
                self._user_refresh_token = persisted
            else:
                return self._do_oauth(token_key)
        if self._user_refresh_token:
            try:
                token, expires_at, self._user_refresh_token = _feishu_refresh_user_token(
                    self._session, self._app_id, self._app_secret, self._user_refresh_token)
                _save_persisted_token(token_key, self._user_refresh_token)
                return token, expires_at
            except Exception:
                if not self._oauth_auto:
                    raise
                LOG.warning(f'Feishu refresh_token for {self._app_id} is invalid, re-authenticating via OAuth.')
                self._user_refresh_token = ''
                _save_persisted_token(token_key, '')
                return self._do_oauth(token_key)
        return _feishu_acquire_access_token(self._session, self._app_id, self._app_secret)

    def _apply_access_token(self, token: str) -> None:
        self._session.headers.update({'Authorization': f'Bearer {token}'})

    def get_user_refresh_token(self) -> str:
        return self._user_refresh_token

    def _drive_root_folder_token(self) -> str:
        if not getattr(self, '_cached_drive_root_token', ''):
            data = self._get(f'{self._base_url}/drive/explorer/v2/root_folder/meta')
            self._cached_drive_root_token: str = data.get('data', {}).get('token', '')
        return self._cached_drive_root_token

    def _upload_file_to_drive(self, name: str, data: bytes, folder_token: str = '') -> str:
        resp_data = self._post(f'{self._base_url}/drive/v1/files/upload_prepare', json={
            'file_name': name, 'parent_type': 'explorer',
            'parent_node': folder_token or self._drive_root_folder_token(), 'size': len(data),
        })
        upload_id = resp_data.get('data', {}).get('upload_id', '')
        block_size = resp_data.get('data', {}).get('block_size', len(data))
        num_blocks = resp_data.get('data', {}).get('block_num', 1)
        for i in range(num_blocks):
            chunk = data[i * block_size: (i + 1) * block_size]
            self._request('POST', f'{self._base_url}/drive/v1/files/upload_part',
                          files={
                              'upload_id': (None, upload_id),
                              'seq': (None, str(i)),
                              'size': (None, str(len(chunk))),
                              'file': (name, chunk, 'application/octet-stream'),
                          },
                          headers={'Content-Type': None})
        result = self._post(f'{self._base_url}/drive/v1/files/upload_finish',
                            json={'upload_id': upload_id, 'block_num': num_blocks})
        return result.get('data', {}).get('file_token', '')


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

    def _list_files_raw(self, folder_token: str = '') -> List[Dict[str, Any]]:
        url = f'{self._base_url}/drive/v1/files'
        # Root listing does not support pagination; do not send page_size (API returns 400).
        params: Dict[str, Any] = {}
        if folder_token:
            params['folder_token'] = folder_token
            params['page_size'] = 200
        results: List[Dict[str, Any]] = []
        page_token: Optional[str] = None
        while True:
            if page_token:
                params['page_token'] = page_token
            data = self._get(url, params=params)
            results.extend(data.get('data', {}).get('files', []) or [])
            if not folder_token:
                break  # Root: no pagination
            if not (page_token := data.get('data', {}).get('next_page_token')):
                break
        return results

    def _resolve_path_to_token(self, path: str) -> str:
        parts = [p for p in path.strip('/').split('/') if p]
        if not parts:
            return ''
        current_token = ''
        for i, part in enumerate(parts):
            items = self._list_files_raw(current_token)
            match = next((it for it in items if it.get('name') == part), None)
            if match is None:
                raise FileNotFoundError(f"'{part}' not found under '/{'/'.join(parts[:i])}'")
            current_token = match.get('token') or ''
        return current_token

    def ls(self, path: str = '/', detail: bool = True, **kwargs) -> List:
        folder_token = self._resolve_path_to_token(path)
        items = self._list_files_raw(folder_token)
        entries = [self._item_to_entry(item) for item in items]
        return entries if detail else [e['name'] for e in entries]

    def info(self, path: str, **kwargs) -> Dict[str, Any]:
        token = self._resolve_path_to_token(path)
        if not token:
            return self._entry('/', ftype='directory')
        name = path.rstrip('/').split('/')[-1]
        url = f'{self._base_url}/drive/v1/files/{token}/statistics'
        try:
            data = self._get(url)
            stat = data.get('data', {}).get('stats', {})
            return self._entry(name=name, size=0, ftype='file', mtime=stat.get('edit_time'), extra_info=stat)
        except Exception:
            return self._entry(name=name, ftype='directory')

    def _open(self, path: str, mode: str = 'rb', block_size: Optional[int] = None,
              autocommit: bool = True, cache_options: Optional[Dict] = None,
              **kwargs) -> CloudFSBufferedFile:
        return CloudFSBufferedFile(self, path, mode=mode, block_size=block_size or self.blocksize,
                                   autocommit=autocommit, cache_options=cache_options)

    def mkdir(self, path: str, create_parents: bool = True, **kwargs) -> None:
        parts = [p for p in path.strip('/').split('/') if p]
        if not parts:
            return
        name = parts[-1]
        parent_token = self._resolve_path_to_token('/' + '/'.join(parts[:-1])) if len(parts) > 1 else ''
        payload: Dict[str, Any] = {'name': name}
        if parent_token:
            payload['folder_token'] = parent_token
        self._post(f'{self._base_url}/drive/v1/files/create_folder', json=payload)

    def rm_file(self, path: str) -> None:
        token = self._resolve_path_to_token(path)
        if not token:
            raise FileNotFoundError(f'Cannot resolve path: {path!r}')
        self._delete(f'{self._base_url}/drive/v1/files/{token}', params={'type': 'file'})

    def rmdir(self, path: str) -> None:
        token = self._resolve_path_to_token(path)
        if not token:
            return
        self._delete(f'{self._base_url}/drive/v1/files/{token}', params={'type': 'folder'})

    def _download_range(self, path: str, start: int, end: int) -> bytes:
        token = self._resolve_path_to_token(path)
        resp = self._request('GET', f'{self._base_url}/drive/v1/files/{token}/download',
                             headers={'Range': f'bytes={start}-{end - 1}'})
        return resp.content

    def _upload_data(self, path: str, data: bytes) -> None:
        parts = [p for p in path.strip('/').split('/') if p]
        name = parts[-1] if parts else 'untitled'
        parent_token = self._resolve_path_to_token('/' + '/'.join(parts[:-1])) if len(parts) > 1 else ''
        self._upload_file_to_drive(name, data, folder_token=parent_token)

    def _platform_supports_webhook(self) -> bool:
        return True

    def _register_webhook(self, webhook_url: str, events: List[str], path: str) -> Dict[str, Any]:
        token = self._resolve_path_to_token(path) if path and path != '/' else ''
        payload: Dict[str, Any] = {'event_type': events or ['drive.file.edit_v1'], 'callback_url': webhook_url}
        if token:
            payload['file_token'] = token
        return self._post(f'{self._base_url}/event/v1/bot/customize/event_callback', json=payload)

    @staticmethod
    def _item_to_entry(item: Dict[str, Any]) -> Dict[str, Any]:
        ftype = 'directory' if item.get('type') == 'folder' else 'file'
        name = item.get('name', '')
        mtime = item.get('modified_time') or item.get('edit_time')
        return LazyLLMFSBase._entry(
            name=name, size=item.get('size', 0), ftype=ftype,
            mtime=float(mtime) if mtime else None,
            title=name, token=item.get('token') or '',
        )


class FeishuWikiFile(CloudFSBufferedFile):

    def __init__(self, fs: 'FeishuWikiFS', path: str, **kwargs) -> None:
        content = fs._fetch_wiki_content(path)
        self._wiki_content: bytes = content
        super().__init__(fs, path, size=len(content), **kwargs)

    def _fetch_range(self, start: int, end: int) -> bytes:
        return self._wiki_content[start:end]


class FeishuWikiFS(FeishuFSBase):
    __lazyllm_registry_disable__ = True

    def _create_docx_node(self, title: str, parent_token: str = '') -> str:
        url = f'{self._base_url}/wiki/v2/spaces/{self._space_id}/nodes'
        payload: Dict[str, Any] = {'obj_type': 'docx', 'node_type': 'origin', 'title': title}
        if parent_token:
            payload['parent_node_token'] = parent_token
        data = self._post(url, json=payload)
        node = data.get('data', {}).get('node') or {}
        return node.get('obj_token') or ''

    def _append_docx_text(self, document_id: str, text: str) -> None:
        if not text:
            return
        max_chunk = 8000
        chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]
        url = f'{self._base_url}/docx/v1/documents/{document_id}/blocks/{document_id}/children'
        for i in range(0, len(chunks), 50):
            children = [{
                'block_type': 2,
                'text': {'elements': [{'text_run': {'content': chunk}}], 'style': {}},
            } for chunk in chunks[i:i + 50]]
            self._post(url, json={'index': 0, 'children': children})

    def _require_space_id(self) -> None:
        if not self._space_id:
            raise ValueError('space_id is required for FeishuWikiFS')

    def _list_nodes_raw(self, parent_token: str = '') -> List[Dict[str, Any]]:
        url = f'{self._base_url}/wiki/v2/spaces/{self._space_id}/nodes'
        params: Dict[str, Any] = {'page_size': 50}
        if parent_token:
            params['parent_node_token'] = parent_token
        results: List[Dict[str, Any]] = []
        page_token: Optional[str] = None
        while True:
            if page_token:
                params['page_token'] = page_token
            data = self._get(url, params=params)
            results.extend(data.get('data', {}).get('items') or data.get('data', {}).get('nodes') or [])
            if not (page_token := data.get('data', {}).get('page_token') or data.get('data', {}).get('next_page_token')):
                break
        return results

    def _resolve_path_to_token(self, path: str) -> str:
        parts = [p for p in path.strip('/').split('/') if p]
        if not parts:
            return ''
        current_token = ''
        for i, part in enumerate(parts):
            nodes = self._list_nodes_raw(current_token)
            match = next((n for n in nodes if (n.get('title') or '') == part), None)
            if match is None:
                raise FileNotFoundError(f"'{part}' not found under '/{'/'.join(parts[:i])}'")
            current_token = match.get('node_token') or ''
        return current_token

    def ls(self, path: str = '/', detail: bool = True, **kwargs) -> List:
        self._require_space_id()
        node_token = self._resolve_path_to_token(path)
        items = self._list_nodes_raw(node_token)
        entries = [self._node_to_entry(item) for item in items]
        return entries if detail else [e['name'] for e in entries]

    def info(self, path: str, **kwargs) -> Dict[str, Any]:
        token = self._resolve_path_to_token(path)
        if not token:
            return self._entry('/', ftype='directory')
        node = self._get_node(token)
        name = path.rstrip('/').split('/')[-1] if path != '/' else '/'
        return self._node_to_entry(node, default_name=name)

    def _fetch_wiki_content(self, path: str) -> bytes:
        token = self._resolve_path_to_token(path)
        if not token:
            return b''
        node = self._get_node(token)
        obj_type = node.get('obj_type')
        obj_token = node.get('obj_token') or ''
        if not obj_type or not obj_token:
            return b''
        if obj_type in {'doc', 'docx'}:
            return self._download_doc_raw(obj_token, obj_type=obj_type)
        if obj_type == 'file':
            return self._download_file_raw(obj_token)
        return b''

    def cat_file(self, path: str, start: Optional[int] = None, end: Optional[int] = None,
                 **kwargs) -> bytes:
        data = self._fetch_wiki_content(path)
        return data[start:end] if (start is not None or end is not None) else data

    def _open(self, path: str, mode: str = 'rb', block_size: Optional[int] = None,
              autocommit: bool = True, cache_options: Optional[Dict] = None,
              **kwargs) -> CloudFSBufferedFile:
        if 'b' not in mode:
            raise ValueError('FeishuWikiFS only supports binary mode')
        if 'r' in mode:
            return FeishuWikiFile(self, path, mode=mode, block_size=block_size or self.blocksize,
                                  autocommit=autocommit, cache_options=cache_options)
        return CloudFSBufferedFile(self, path, mode=mode, block_size=block_size or self.blocksize,
                                   autocommit=autocommit, cache_options=cache_options)

    def mkdir(self, path: str, create_parents: bool = True, **kwargs) -> None:
        parts = [p for p in path.strip('/').split('/') if p]
        if not parts:
            return
        title = parts[-1]
        parent_token = self._resolve_path_to_token('/' + '/'.join(parts[:-1])) if len(parts) > 1 else ''
        url = f'{self._base_url}/wiki/v2/spaces/{self._space_id}/nodes'
        payload: Dict[str, Any] = {'title': title, 'obj_type': 'docx', 'node_type': 'origin'}
        if parent_token:
            payload['parent_node_token'] = parent_token
        self._post(url, json=payload)

    def rm_file(self, path: str) -> None:
        token = self._resolve_path_to_token(path)
        if not token:
            raise FileNotFoundError(path)
        self._delete(f'{self._base_url}/wiki/v2/spaces/{self._space_id}/nodes/{token}')

    def rmdir(self, path: str) -> None:
        self.rm_file(path)

    def _download_range(self, path: str, start: int, end: int) -> bytes:
        return self._fetch_wiki_content(path)[start:end]

    def _upload_data(self, path: str, data: bytes) -> None:
        parts = [p for p in path.strip('/').split('/') if p]
        name = parts[-1] if parts else 'untitled'
        parent_token = self._resolve_path_to_token('/' + '/'.join(parts[:-1])) if len(parts) > 1 else ''
        doc_id = self._create_docx_node(name, parent_token=parent_token)
        if not doc_id:
            raise RuntimeError('Feishu wiki create docx node failed: empty obj_token')
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError('FeishuWikiFS.put_file only supports utf-8 text content for now') from e
        self._append_docx_text(doc_id, text)

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
        resp = self._request('GET', f'{self._base_url}/drive/v1/files/{file_token}/download')
        return resp.content

    def _get_node(self, token: str) -> Dict[str, Any]:
        data = self._get(f'{self._base_url}/wiki/v2/spaces/get_node', params={'token': token})
        node = data.get('data', {}).get('node') or data.get('data', {})
        return node or {}

    @staticmethod
    def _node_to_entry(node: Dict[str, Any], default_name: Optional[str] = None) -> Dict[str, Any]:
        obj_type = node.get('obj_type')
        # has_child: true means the node has sub-pages and must be navigable as a directory,
        # even if its obj_type is 'docx' (wiki allows documents to also be folder nodes).
        is_dir = obj_type in {'folder', 'wiki', 'space'} or bool(node.get('has_child'))
        title = node.get('title') or node.get('obj_name') or default_name or ''
        node_token = node.get('node_token') or node.get('token') or ''
        size = int(node.get('size', 0) or 0)
        mtime = None
        ts = node.get('update_time') or node.get('edit_time') or node.get('modified_time')
        if ts:
            try:
                mtime = float(ts)
            except (TypeError, ValueError):
                mtime = None
        return LazyLLMFSBase._entry(
            name=title, size=size, ftype='directory' if is_dir else 'file',
            mtime=mtime, title=title, obj_type=obj_type,
            obj_token=node.get('obj_token'), node_token=node_token,
        )
