# Copyright (c) 2026 LazyAGI. All rights reserved.
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse

import requests


from lazyllm import LOG, config
from lazyllm.thirdparty import mistune

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
                 skip_instance_cache: bool = False, loop: Optional[Any] = None, dynamic_auth: bool = False):
        if dynamic_auth:
            if app_id:
                raise ValueError('app_id must be None when dynamic_auth=True')
            if app_secret:
                raise ValueError('app_secret must be None when dynamic_auth=True')
            if user_refresh_token:
                raise ValueError('user_refresh_token must be None when dynamic_auth=True')
            self._oauth_auto = False
            self._user_refresh_token = ''
            self._oauth_port = oauth_port
            self._oauth_scope = oauth_scope
            super().__init__(
                token={},
                base_url=base_url or _API_BASE,
                asynchronous=asynchronous,
                use_listings_cache=use_listings_cache,
                skip_instance_cache=skip_instance_cache,
                loop=loop,
                dynamic_auth=True,
            )
            self._space_id = (space_id or '').strip() if space_id is not None else ''
            return
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
        self._access_token = token

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

    @staticmethod
    def _text_from_ast_node(node: Dict[str, Any]) -> str:
        if node.get('raw') is not None:
            return str(node['raw'])
        children = node.get('children') or []
        return ''.join(FeishuFSBase._text_from_ast_node(c) for c in children)

    @staticmethod
    def _elements_from_inline_children(children: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        elements: List[Dict[str, Any]] = []
        for c in children:
            if not isinstance(c, dict):
                continue
            ct = c.get('type')
            if ct == 'text':
                raw = (c.get('raw') or '')
                if raw:
                    elements.append({'text_run': {'content': raw}})
            elif ct == 'link':
                url = (c.get('attrs') or {}).get('url') or ''
                content = FeishuFSBase._text_from_ast_node(c)
                if content or url:
                    el: Dict[str, Any] = {'text_run': {'content': content or url}}
                    if url:
                        el['link'] = {'url': url}
                    elements.append(el)
            elif ct == 'softbreak':
                elements.append({'text_run': {'content': '\n'}})
            elif ct == 'linebreak':
                elements.append({'text_run': {'content': '\n'}})
            else:
                raw = FeishuFSBase._text_from_ast_node(c)
                if raw:
                    elements.append({'text_run': {'content': raw}})
        return elements

    _DOCX_BLOCK_TYPE_KEY: Dict[int, str] = {
        2: 'text', 3: 'heading1', 4: 'heading2', 5: 'heading3', 6: 'heading4',
        7: 'heading5', 8: 'heading6', 9: 'heading7', 10: 'heading8', 11: 'heading9',
        12: 'bullet', 13: 'ordered', 14: 'code', 15: 'quote',
    }

    @staticmethod
    def _docx_block_with_content(block_type: int, content: str) -> Dict[str, Any]:
        key = FeishuFSBase._DOCX_BLOCK_TYPE_KEY.get(block_type, 'text')
        return {'block_type': block_type, key: {'elements': [{'text_run': {'content': content}}]}}

    @staticmethod
    def _docx_block_with_elements(block_type: int, elements: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not elements:
            return None
        key = FeishuFSBase._DOCX_BLOCK_TYPE_KEY.get(block_type, 'text')
        return {'block_type': block_type, key: {'elements': elements}}

    @staticmethod
    def _table_ast_to_markdown(node: Dict[str, Any]) -> str:
        def cell_text(cell_node: Dict[str, Any]) -> str:
            return FeishuFSBase._text_from_ast_node(cell_node).strip().replace('|', '\\|')
        rows: List[str] = []
        for part in (node.get('children') or []):
            if not isinstance(part, dict):
                continue
            if part.get('type') == 'table_head':
                cells = [cell_text(c) for c in (part.get('children') or [])
                         if isinstance(c, dict) and c.get('type') == 'table_cell']
                if cells:
                    rows.append('| ' + ' | '.join(cells) + ' |')
                    rows.append('| ' + ' | '.join(['---'] * len(cells)) + ' |')
            elif part.get('type') == 'table_body':
                for row_node in (part.get('children') or []):
                    if isinstance(row_node, dict) and row_node.get('type') == 'table_row':
                        cells = [cell_text(c) for c in (row_node.get('children') or [])
                                 if isinstance(c, dict) and c.get('type') == 'table_cell']
                        if cells:
                            rows.append('| ' + ' | '.join(cells) + ' |')
        return '\n'.join(rows) if rows else ''

    @staticmethod
    def _table_ast_to_cells(node: Dict[str, Any]) -> Optional[Tuple[int, int, List[List[str]]]]:
        def cell_text(cell_node: Dict[str, Any]) -> str:
            return FeishuFSBase._text_from_ast_node(cell_node).strip()
        grid: List[List[str]] = []
        for part in (node.get('children') or []):
            if not isinstance(part, dict):
                continue
            if part.get('type') == 'table_head':
                cells = [cell_text(c) for c in (part.get('children') or [])
                         if isinstance(c, dict) and c.get('type') == 'table_cell']
                if cells:
                    grid.append(cells)
            elif part.get('type') == 'table_body':
                for row_node in (part.get('children') or []):
                    if isinstance(row_node, dict) and row_node.get('type') == 'table_row':
                        cells = [cell_text(c) for c in (row_node.get('children') or [])
                                 if isinstance(c, dict) and c.get('type') == 'table_cell']
                        if cells:
                            grid.append(cells)
        if not grid:
            return None
        row_size, col_size = len(grid), max(len(r) for r in grid)
        return (row_size, col_size, grid)

    @staticmethod
    def _inline_children_to_block(block_type: int, node: Dict[str, Any]) -> List[Dict[str, Any]]:
        elements = FeishuFSBase._elements_from_inline_children(node.get('children') or [])
        blk = FeishuFSBase._docx_block_with_elements(block_type, elements)
        return [blk] if blk else []

    @staticmethod
    def _parse_details_summary(raw: str) -> Optional[Tuple[str, str]]:
        raw = raw.strip()
        if not raw.lower().startswith('<details'):
            return None
        m = re.search(r'<summary[^>]*>(.*?)</summary>', raw, re.DOTALL | re.IGNORECASE)
        if not m:
            return None
        summary_html = m.group(1).strip()
        summary_text = re.sub(r'<[^>]+>', '', summary_html).strip()
        rest = raw[m.end():]
        m2 = re.search(r'^(.*?)</details>', rest, re.DOTALL | re.IGNORECASE)
        body = m2.group(1).strip() if m2 else rest.strip()
        return (summary_text or 'Details', body)

    @staticmethod
    def _ast_table_to_block(node: Dict[str, Any]) -> List[Dict[str, Any]]:
        cells_data = FeishuFSBase._table_ast_to_cells(node)
        if not cells_data:
            return []
        row_size, col_size, grid = cells_data
        return [{
            'block_type': 31,
            'table': {'property': {'row_size': row_size, 'column_size': col_size}},
            '_table_cells': grid,
        }]

    @staticmethod
    def _ast_block_html_to_blocks(node: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw = (node.get('raw') or '').rstrip()
        if not raw:
            return []
        if re.match(r'^\s*</(?:details|summary|div|p)>\s*$', raw, re.IGNORECASE):
            return []
        parsed = FeishuFSBase._parse_details_summary(raw)
        if parsed:
            summary_text, body_content = parsed
            out: List[Dict[str, Any]] = []
            title_blk = FeishuFSBase._docx_block_with_content(2, summary_text)
            if title_blk:
                out.append(title_blk)
            if body_content:
                out.extend(FeishuFSBase._markdown_to_docx_blocks(body_content))
            return out
        return [{'block_type': 14, 'code': {'elements': [{'text_run': {'content': raw}}]}}]

    @staticmethod
    def _ast_list_to_blocks(node: Dict[str, Any]) -> List[Dict[str, Any]]:
        ordered = (node.get('attrs') or {}).get('ordered', False)
        btype = 13 if ordered else 12
        out: List[Dict[str, Any]] = []
        for item in (node.get('children') or []):
            if isinstance(item, dict) and item.get('type') == 'list_item':
                blk = FeishuFSBase._docx_block_with_elements(
                    btype, FeishuFSBase._elements_from_inline_children(item.get('children') or []))
                if blk:
                    out.append(blk)
        return out

    @staticmethod
    def _ast_node_to_docx_block(node: Dict[str, Any]) -> List[Dict[str, Any]]:
        t = node.get('type')
        if t == 'blank_line':
            return []
        if t == 'heading':
            level = max(1, min(9, int((node.get('attrs') or {}).get('level', 1))))
            return FeishuFSBase._inline_children_to_block(2 + level, node)
        if t == 'paragraph':
            return FeishuFSBase._inline_children_to_block(2, node)
        if t == 'block_code':
            content = (node.get('raw') or '').rstrip()
            return [{'block_type': 14, 'code': {'elements': [{'text_run': {'content': content}}]}}]
        if t == 'block_quote':
            return FeishuFSBase._inline_children_to_block(15, node)
        if t == 'list':
            return FeishuFSBase._ast_list_to_blocks(node)
        if t == 'table':
            return FeishuFSBase._ast_table_to_block(node)
        if t == 'block_html':
            return FeishuFSBase._ast_block_html_to_blocks(node)
        blk = FeishuFSBase._docx_block_with_elements(
            2, FeishuFSBase._elements_from_inline_children(node.get('children') or []))
        if blk:
            return [blk]
        content = FeishuFSBase._text_from_ast_node(node).strip()
        return [FeishuFSBase._docx_block_with_content(2, content)] if content else []

    @staticmethod
    def _normalize_md_tables(md_text: str) -> str:
        lines = md_text.splitlines(keepends=True)
        result = []
        for line in lines:
            stripped = line.rstrip('\n\r')
            if stripped.lstrip().startswith('|') and not stripped.rstrip().endswith('|'):
                stripped = stripped.rstrip() + ' |'
                line = stripped + ('\n' if line.endswith('\n') else '')
            result.append(line)
        return ''.join(result)

    @staticmethod
    def _markdown_to_docx_blocks(md_text: str) -> List[Dict[str, Any]]:
        md = mistune.create_markdown(renderer='ast', plugins=['table'])
        ast = md(FeishuFSBase._normalize_md_tables(md_text))
        if not ast:
            return []
        blocks: List[Dict[str, Any]] = []
        for node in ast:
            if not isinstance(node, dict):
                continue
            blocks.extend(FeishuFSBase._ast_node_to_docx_block(node))
        return blocks

    def _append_docx_blocks(self, document_id: str, blocks: List[Dict[str, Any]]) -> None:
        if not blocks:
            return
        url = f'{self._base_url}/docx/v1/documents/{document_id}/blocks/{document_id}/children'
        batch_size = 50
        index = 0
        chunk: List[Dict[str, Any]] = []
        for blk in blocks:
            if blk.get('_table_cells') is not None:
                if chunk:
                    self._post(url, json={'index': index, 'children': chunk})
                    index += len(chunk)
                    chunk = []
                payload = {k: v for k, v in blk.items() if k != '_table_cells'}
                resp = self._request('POST', url, json={'index': index, 'children': [payload]})
                created_list = (resp.json().get('data') or {}).get('children') or []
                table_info = created_list[0] if created_list and isinstance(created_list[0], dict) else {}
                table_block_id = table_info.get('block_id')
                cell_ids: List[str] = (table_info.get('table') or {}).get('cells') or []
                if table_block_id:
                    self._fill_table_cells(document_id, table_block_id, blk['_table_cells'], cell_ids)
                index += 1
            else:
                chunk.append(blk)
                if len(chunk) >= batch_size:
                    self._post(url, json={'index': index, 'children': chunk})
                    index += len(chunk)
                    chunk = []
        if chunk:
            self._post(url, json={'index': index, 'children': chunk})

    def _get_table_cells(self, document_id: str, table_block_id: str) -> List[Dict[str, Any]]:
        url = f'{self._base_url}/docx/v1/documents/{document_id}/blocks/{table_block_id}/children'
        items: List[Dict[str, Any]] = []
        page_token: Optional[str] = None
        while True:
            params: Dict[str, Any] = {'page_size': 500, 'with_descendants': 'true'}
            if page_token:
                params['page_token'] = page_token
            data = (self._get(url, params=params) or {}).get('data') or {}
            items.extend(data.get('items') or [])
            page_token = data.get('page_token')
            if not page_token:
                break
        cell_blocks = [x for x in items if x.get('block_type') == 32]
        if not cell_blocks:
            for row_block in (x for x in items if x.get('block_id')):
                rid = row_block['block_id']
                row_data = (self._get(
                    f'{self._base_url}/docx/v1/documents/{document_id}/blocks/{rid}/children',
                    params={'page_size': 100}) or {}).get('data') or {}
                cell_blocks.extend(x for x in (row_data.get('items') or []) if x.get('block_type') == 32)
        return cell_blocks

    def _write_cell_text(self, document_id: str, cell_block_id: str, text: str) -> None:
        cell_children_url = (
            f'{self._base_url}/docx/v1/documents/{document_id}/blocks/{cell_block_id}/children')
        cd = (self._get(cell_children_url, params={'page_size': 10}) or {}).get('data') or {}
        text_blocks = [x for x in (cd.get('items') or []) if x.get('block_type') == 2]
        if text_blocks:
            self._patch(
                f'{self._base_url}/docx/v1/documents/{document_id}/blocks/{text_blocks[0]["block_id"]}',
                json={'update_text_elements': {'elements': [{'text_run': {'content': text}}]}})
        else:
            self._post(cell_children_url, json={
                'index': 0,
                'children': [{'block_type': 2, 'text': {'elements': [{'text_run': {'content': text}}]}}],
            })

    def _fill_table_cells(self, document_id: str, table_block_id: str,
                          grid: List[List[str]], cell_ids: Optional[List[str]] = None) -> None:
        flat_cells = [c for row in grid for c in row]
        if not cell_ids:
            cell_blocks = self._get_table_cells(document_id, table_block_id)
            cell_ids = [b['block_id'] for b in cell_blocks if b.get('block_id')]
        for cell_id, text in zip(cell_ids[:len(flat_cells)], flat_cells):
            if cell_id and text:
                self._write_cell_text(document_id, cell_id, text)


class FeishuFS(FeishuFSBase):

    def __new__(cls, base_url: Optional[str] = None, app_id: Optional[str] = None, app_secret: Optional[str] = None,
                space_id: Optional[str] = None, user_refresh_token: Optional[str] = None,
                oauth_port: int = 9981, oauth_scope: str = _DEFAULT_OAUTH_SCOPE,
                asynchronous: bool = False, use_listings_cache: bool = False,
                skip_instance_cache: bool = False, loop: Optional[Any] = None,
                dynamic_auth: bool = False) -> LazyLLMFSBase:
        if space_id is not None and str(space_id).strip():
            return FeishuWikiFS(base_url=base_url, app_id=app_id, app_secret=app_secret, space_id=space_id,
                                user_refresh_token=user_refresh_token, oauth_port=oauth_port,
                                oauth_scope=oauth_scope, asynchronous=asynchronous,
                                use_listings_cache=use_listings_cache,
                                skip_instance_cache=skip_instance_cache, loop=loop, dynamic_auth=dynamic_auth)
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

    def _get_item_raw(self, path: str) -> Dict[str, Any]:
        parts = [p for p in path.strip('/').split('/') if p]
        if not parts:
            return {'type': 'folder', 'token': '', 'name': ''}
        name = parts[-1]
        parent_token = self._resolve_path_to_token('/' + '/'.join(parts[:-1])) if len(parts) > 1 else ''
        match = next((it for it in self._list_files_raw(parent_token) if it.get('name') == name), None)
        if match is None:
            raise FileNotFoundError(path)
        return match

    def copy(self, path1: str, path2: str, recursive: bool = False, **kwargs) -> None:
        src = self._get_item_raw(path1)
        if src.get('type') == 'folder':
            raise NotImplementedError('FeishuFS: the official Drive API does not support folder copy')
        src_token = src.get('token', '')
        parts2 = [p for p in path2.strip('/').split('/') if p]
        new_name = parts2[-1] if parts2 else src.get('name', 'copy')
        parent_path2 = '/' + '/'.join(parts2[:-1]) if len(parts2) > 1 else '/'
        folder_token = self._resolve_path_to_token(parent_path2)
        self._post(f'{self._base_url}/drive/v1/files/{src_token}/copy',
                   json={'name': new_name, 'type': src.get('type', 'file'), 'folder_token': folder_token})

    def move(self, path1: str, path2: str, recursive: bool = False, **kwargs) -> None:
        src = self._get_item_raw(path1)
        src_token = src.get('token', '')
        parts2 = [p for p in path2.strip('/').split('/') if p]
        parent_path2 = '/' + '/'.join(parts2[:-1]) if len(parts2) > 1 else '/'
        folder_token = self._resolve_path_to_token(parent_path2)
        self._post(f'{self._base_url}/drive/v1/files/{src_token}/move',
                   json={'type': src.get('type', 'file'), 'folder_token': folder_token})

    def _download_range(self, path: str, start: int, end: int) -> bytes:
        token = self._resolve_path_to_token(path)
        resp = self._request('GET', f'{self._base_url}/drive/v1/files/{token}/download',
                             headers={'Range': f'bytes={start}-{end - 1}'})
        return resp.content

    def _upload_data(self, path: str, data: bytes, **kwargs) -> None:
        parts = [p for p in path.strip('/').split('/') if p]
        name = parts[-1] if parts else 'untitled'
        parent_token = self._resolve_path_to_token('/' + '/'.join(parts[:-1])) if len(parts) > 1 else ''
        content_type = kwargs.get('content_type')
        if content_type is None:
            content_type = 'markdown' if (name.endswith('.md')) else 'text'
        if content_type in ('markdown', 'md'):
            try:
                text = data.decode('utf-8')
            except UnicodeDecodeError:
                self._upload_file_to_drive(name, data, folder_token=parent_token)
                return
            doc_title = name[:-3] if name.endswith('.md') else name
            folder_token = parent_token or self._drive_root_folder_token()
            create_resp = self._post(
                f'{self._base_url}/docx/v1/documents',
                params={'folder_token': folder_token},
                json={'title': doc_title},
            )
            doc_id = ((create_resp.get('data') or {}).get('document') or {}).get('document_id') or ''
            if not doc_id:
                LOG.warning('Feishu drive docx create returned no document_id, falling back to file upload')
                self._upload_file_to_drive(name, data, folder_token=parent_token)
                return
            blocks = self._markdown_to_docx_blocks(text)
            self._append_docx_blocks(doc_id, blocks)
        else:
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
    protocol = 'feishu'
    _fs_protocol_key = 'feishu'

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

    def copy(self, path1: str, path2: str, recursive: bool = False, **kwargs) -> None:
        src_token = self._resolve_path_to_token(path1)
        if not src_token:
            raise FileNotFoundError(path1)
        parts2 = [p for p in path2.strip('/').split('/') if p]
        title = parts2[-1] if parts2 else ''
        parent_path2 = '/' + '/'.join(parts2[:-1]) if len(parts2) > 1 else '/'
        parent_token = self._resolve_path_to_token(parent_path2)
        payload: Dict[str, Any] = {}
        if parent_token:
            payload['target_parent_token'] = parent_token
        if title:
            payload['title'] = title
        self._post(f'{self._base_url}/wiki/v2/spaces/{self._space_id}/nodes/{src_token}/copy', json=payload)

    def move(self, path1: str, path2: str, recursive: bool = False, **kwargs) -> None:
        src_token = self._resolve_path_to_token(path1)
        if not src_token:
            raise FileNotFoundError(path1)
        parts2 = [p for p in path2.strip('/').split('/') if p]
        parent_path2 = '/' + '/'.join(parts2[:-1]) if len(parts2) > 1 else '/'
        parent_token = self._resolve_path_to_token(parent_path2)
        payload: Dict[str, Any] = {}
        if parent_token:
            payload['target_parent_token'] = parent_token
        self._post(f'{self._base_url}/wiki/v2/spaces/{self._space_id}/nodes/{src_token}/move', json=payload)

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

    def _upload_data(self, path: str, data: bytes, **kwargs) -> None:
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
        content_type = kwargs.get('content_type')
        if content_type is None:
            content_type = 'markdown' if (path.rstrip('/').split('/')[-1].endswith('.md')) else 'text'
        if content_type in ('markdown', 'md'):
            blocks = self._markdown_to_docx_blocks(text)
            self._append_docx_blocks(doc_id, blocks)
        else:
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

    def get_document_id(self, path: str) -> str:
        self._require_space_id()
        node_token = self._resolve_path_to_token(path)
        if not node_token:
            raise FileNotFoundError(f'Path not found: {path}')
        node = self._get_node(node_token)
        obj_type = node.get('obj_type')
        obj_token = node.get('obj_token') or ''
        if obj_type not in ('doc', 'docx'):
            raise ValueError(
                f'Path is not a doc/docx node (obj_type={obj_type}), cannot get document_id'
            )
        return obj_token

    def _get_doc_blocks_raw(self, document_id: str, with_descendants: bool = True) -> List[Dict[str, Any]]:
        url = f'{self._base_url}/docx/v1/documents/{document_id}/blocks/{document_id}/children'
        params: Dict[str, Any] = {'page_size': 500}
        if with_descendants:
            params['with_descendants'] = 'true'
        results: List[Dict[str, Any]] = []
        page_token: Optional[str] = None
        while True:
            if page_token:
                params['page_token'] = page_token
            data = self._get(url, params=params)
            items = data.get('data', {}).get('items') or []
            results.extend(items)
            page_token = data.get('data', {}).get('page_token')
            if not page_token:
                break
        return results

    def get_doc_blocks(self, path: str, with_descendants: bool = True) -> List[Dict[str, Any]]:
        document_id = self.get_document_id(path)
        blocks = self._get_doc_blocks_raw(document_id, with_descendants=with_descendants)
        out: List[Dict[str, Any]] = []
        for b in blocks:
            entry: Dict[str, Any] = {
                'block_id': b.get('block_id', ''),
                'block_type': b.get('block_type'),
                'parent_id': b.get('parent_id', ''),
            }
            if 'text' in b and isinstance(b['text'], dict):
                plain = (b['text'].get('elements') or [])
                texts = [
                    e.get('text_run', {}).get('content', '')
                    for e in plain if isinstance(e, dict)
                ]
                entry['plain_text'] = ''.join(texts)
            out.append(entry)
        return out

    def update_doc_block_text(self, path: str, block_id: str, new_text: str) -> None:
        document_id = self.get_document_id(path)
        url = f'{self._base_url}/docx/v1/documents/{document_id}/blocks/{block_id}'
        payload = {
            'text': {
                'elements': [{'text_run': {'content': new_text}}],
                'style': {},
            },
        }
        self._patch(url, json=payload)

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
