# Copyright (c) 2026 LazyAGI. All rights reserved.
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlencode, urlparse

import requests


from lazyllm import LOG, config
from lazyllm import globals as lazyllm_globals
from lazyllm.common import Credential
from ..base import LazyLLMFSBase, LinkDocumentFSBase, CloudFSBufferedFile

config.add('feishu_app_id', str, None, 'FEISHU_APP_ID', description='Feishu App ID for tenant_access_token.')
config.add('feishu_app_secret', str, None, 'FEISHU_APP_SECRET',
           description='Feishu App Secret for tenant_access_token.')
lazyllm_globals.config.add(
    'feishu_wiki_space_id', str, None, 'FEISHU_WIKI_SPACE_ID',
    description='Default Feishu wiki space_id for tree ops (ls/mkdir) when not set in URI.',
)

_SPACE_ID_DYNAMIC = 'dynamic'
_FEISHU_BARE_HOST_RE = re.compile(
    r'^https?://(?:[^/@]+\.)*(?:feishu\.cn|larksuite\.com)(?::\d+)?/', re.IGNORECASE,
)
_FEISHU_WIKI_TILDE_PREFIXES = ('~link/', '~node/', '~docx/', '~doc/')


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


def _parse_feishu_browser_url(url: str) -> Optional[Dict[str, str]]:
    parsed = urlparse(url)
    hostname = (parsed.hostname or '').lower()
    if parsed.scheme not in ('http', 'https') or not (
        hostname in ('feishu.cn', 'larksuite.com')
        or hostname.endswith(('.feishu.cn', '.larksuite.com'))
    ):
        return None
    parts = parsed.path.strip('/').split('/')
    if len(parts) != 2 or not parts[1]:
        return None
    kind = {'wiki': 'wiki_node', 'docx': 'docx', 'docs': 'doc'}.get(parts[0].lower())
    return {'kind': kind, 'token': parts[1]} if kind else None


def _is_wiki_locator_path(path: str) -> bool:
    norm = path.lstrip('/')
    if any(norm.startswith(p) for p in _FEISHU_WIKI_TILDE_PREFIXES):
        return True
    return bool(_FEISHU_BARE_HOST_RE.match(path))


def _iter_docx_elements(blocks: List[Dict[str, Any]]):
    for block in blocks:
        bt = block.get('block_type')
        if bt in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
            key = {2: 'text', 3: 'heading1', 4: 'heading2', 5: 'heading3', 6: 'heading4',
                   7: 'heading5', 8: 'heading6', 9: 'heading7', 10: 'heading8', 11: 'heading9',
                   12: 'bullet', 13: 'ordered', 14: 'code', 15: 'quote'}.get(bt, 'text')
            for el in (block.get(key) or {}).get('elements') or []:
                yield el
        elif bt == 48:
            yield {'link_preview': block.get('link_preview') or {}}


def _ref_from_element(el: Dict[str, Any]) -> Optional[str]:
    if 'mention_doc' in el:
        md = el['mention_doc']
        url = md.get('url') or ''
        if url:
            return url
        token = md.get('token') or ''
        obj_type = md.get('obj_type') or ''
        if token and obj_type:
            path_seg = 'wiki' if obj_type == 'wiki_node' else ('docx' if obj_type == 'docx' else 'docs')
            return f'https://open.feishu.cn/{path_seg}/{token}'
    if 'text_run' in el:
        link = (el['text_run'].get('text_element_style') or {}).get('link') or {}
        url = link.get('url') or ''
        if url and _FEISHU_BARE_HOST_RE.match(url):
            return url
    if 'link_preview' in el:
        url = el['link_preview'].get('url') or ''
        if url and _FEISHU_BARE_HOST_RE.match(url):
            return url
    return None


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


class FeishuFSBase(LinkDocumentFSBase):
    __lazyllm_registry_disable__ = True

    def __init__(self, base_url: Optional[str] = None, app_id: Optional[str] = None, app_secret: Optional[str] = None,
                 space_id: Optional[str] = None, user_refresh_token: Optional[str] = None,
                 oauth_port: int = 9981, oauth_scope: str = _DEFAULT_OAUTH_SCOPE,
                 asynchronous: bool = False, use_listings_cache: bool = False,
                 skip_instance_cache: bool = False, loop: Optional[Any] = None, dynamic_auth: bool = False):
        if dynamic_auth and (app_id or app_secret or user_refresh_token):
            raise ValueError('app_id/app_secret/user_refresh_token must be None when dynamic_auth=True')
        if not dynamic_auth:
            app_id = app_id or config['feishu_app_id']
            app_secret = app_secret or config['feishu_app_secret']
            assert app_id and app_secret, 'feishu_app_id and feishu_app_secret are required'
        self._oauth_port: int = oauth_port
        self._oauth_scope: str = oauth_scope
        self._init_user_refresh_token: str = '' if dynamic_auth else (user_refresh_token or '')
        super().__init__(
            token={} if dynamic_auth else {'app_id': app_id, 'app_secret': app_secret},
            base_url=base_url or _API_BASE,
            asynchronous=asynchronous, use_listings_cache=use_listings_cache,
            skip_instance_cache=skip_instance_cache, loop=loop,
            dynamic_auth=dynamic_auth,
        )
        self._space_id = (space_id or '').strip() if space_id is not None else ''

    def _make_credential(self, token: Any, dynamic_auth: bool):
        if not dynamic_auth and self._init_user_refresh_token:
            rt = self._init_user_refresh_token
            return Credential(kind='oauth2', secret_key=token, refresh_token=rt,
                              oauth_auto=(rt == 'auto'))
        return super()._make_credential(token, dynamic_auth)

    @property
    def _app_id(self) -> str:
        return self._secret_key.get('app_id', '')

    @property
    def _app_secret(self) -> str:
        return self._secret_key.get('app_secret', '')

    @property
    def _user_refresh_token(self) -> str:
        return self._credential.refresh_token

    def get_user_refresh_token(self) -> str:
        return self._user_refresh_token

    def _setup_auth(self) -> None:
        self._session.headers.update({'Content-Type': 'application/json; charset=utf-8'})

    def _get_persist_key(self) -> str:
        return f'feishu:{self._app_id}'

    def _do_refresh_token(self, refresh_token: str) -> Tuple[str, Optional[float], str]:
        return _feishu_refresh_user_token(self._session, self._app_id, self._app_secret, refresh_token)

    def _do_acquire_without_refresh(self) -> Tuple[str, Optional[float], str]:
        token, expires_at = _feishu_acquire_access_token(self._session, self._app_id, self._app_secret)
        return token, expires_at, ''

    def _do_oauth_flow(self) -> Tuple[str, Optional[float], str]:
        redirect_uri = f'http://localhost:{self._oauth_port}/callback'
        authorize_url = f'{_OAUTH_AUTHORIZE_URL}?' + urlencode({
            'client_id': self._app_id, 'redirect_uri': redirect_uri,
            'scope': self._oauth_scope, 'response_type': 'code',
        })
        LOG.success(
            f'Feishu OAuth: open the link below in a browser to authorize access.\n'
            f'  Prerequisites:\n'
            f'    1. Register {redirect_uri} in your Feishu app Security Settings -> Redirect URL.\n'
            f'    2. Enable user-identity drive permissions (e.g. drive:drive) in Permission Management\n'
            f'       and enable offline_access, then publish the app.\n\n'
            f'  {authorize_url}\n'
        )
        code = self._run_local_oauth_server(self._oauth_port, _OAUTH_TIMEOUT)
        return _feishu_exchange_code(self._session, self._app_id, self._app_secret, code, redirect_uri)

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
            self._request(
                'POST', f'{self._base_url}/drive/v1/files/upload_part',
                files={
                    'upload_id': (None, upload_id),
                    'seq': (None, str(i)),
                    'size': (None, str(len(chunk))),
                    'file': (name, chunk, 'application/octet-stream'),
                },
                headers={'Content-Type': None},
            )
        result = self._post(f'{self._base_url}/drive/v1/files/upload_finish',
                            json={'upload_id': upload_id, 'block_num': num_blocks})
        return result.get('data', {}).get('file_token', '')

    _DOCX_BLOCK_TYPE_KEY: Dict[int, str] = {
        1: 'page', 2: 'text', 3: 'heading1', 4: 'heading2', 5: 'heading3', 6: 'heading4',
        7: 'heading5', 8: 'heading6', 9: 'heading7', 10: 'heading8', 11: 'heading9',
        12: 'bullet', 13: 'ordered', 14: 'code', 15: 'quote', 17: 'todo',
        19: 'callout', 22: 'divider', 24: 'grid', 25: 'grid_column',
        31: 'table', 32: 'table_cell', 34: 'quote_container',
    }

    @staticmethod
    def _docx_element_plain_text(element: Dict[str, Any]) -> str:
        for element_type in (
            'text_run', 'mention_user', 'mention_doc', 'reminder',
            'file', 'inline_block', 'equation', 'undefined_element',
        ):
            value = element.get(element_type)
            if not isinstance(value, dict):
                continue
            for field in ('content', 'title', 'name', 'text'):
                text = value.get(field)
                if isinstance(text, str):
                    return text
        return ''

    @staticmethod
    def _docx_block_plain_text(block: Dict[str, Any]) -> str:
        key = FeishuFSBase._DOCX_BLOCK_TYPE_KEY.get(block.get('block_type'))
        if not key:
            return ''
        elements = (block.get(key) or {}).get('elements') or []
        return ''.join(
            FeishuFSBase._docx_element_plain_text(element)
            for element in elements
            if isinstance(element, dict)
        )

    @staticmethod
    def _extract_table_grid(
        cell_ids: List[str], block_map: Dict[str, Dict[str, Any]],
        rows: int, cols: int,
    ) -> List[List[Optional[List[Dict[str, Any]]]]]:
        grid: List[Optional[List[Dict[str, Any]]]] = []
        for cid in cell_ids:
            cell = block_map.get(cid)
            if cell is None:
                grid.append(None)
                continue
            elements: List[Dict[str, Any]] = []
            for tcid in (cell.get('children') or []):
                txt_blk = block_map.get(tcid)
                if txt_blk:
                    elements.extend(txt_blk.get('text', {}).get('elements', []))
            grid.append(elements if elements else None)
        return [
            [grid[r * cols + c] if r * cols + c < len(grid) else None
             for c in range(cols)]
            for r in range(rows)
        ]

    def _convert_markdown_via_api(self, text: str) -> Tuple[
            List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        resp = self._post(
            f'{self._base_url}/docx/v1/documents/blocks/convert',
            json={'content_type': 'markdown', 'content': text})
        if resp.get('code', 0) != 0:
            raise RuntimeError(
                'Feishu Convert API failed: %s' % resp.get('msg', resp))

        blocks_raw = (resp.get('data') or {}).get('blocks', [])
        first_level = (resp.get('data') or {}).get('first_level_block_ids', [])
        if not blocks_raw:
            return [], {}

        block_map = {b['block_id']: b for b in blocks_raw}

        block_children: Dict[str, List[Dict[str, Any]]] = {}

        def _transform(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[List[str]]]:
            bt = raw.get('block_type', 0)
            if bt == 22:  # divider
                return ({'block_type': 22, 'divider': {}}, None)
            if bt == 27:  # image
                return ({'block_type': 2, 'text': {
                    'elements': [{'text_run': {'content': '[Image]'}}]}}, None)

            b = dict(raw)
            children = list(b.pop('children', [])) or None
            b['_temp_id'] = b.pop('block_id')
            b.pop('parent_id', None)

            if bt == 31:  # table
                b['table'] = dict(b.get('table') or {})
                b['table'].pop('merge_info', None)
                prop = b['table'].get('property', {})
                rows, cols = prop.get('row_size', 0), prop.get('column_size', 0)
                b['table'] = {'property': {'row_size': rows, 'column_size': cols}}
                b['_table_cells'] = FeishuFSBase._extract_table_grid(
                    children or [], block_map, rows, cols)
                children = None

            return (b, children)

        def _walk(block_ids: List[str]) -> List[Dict[str, Any]]:
            block_result: List[Dict[str, Any]] = []
            for bid in block_ids:
                if bid not in block_map:
                    continue
                block, children = _transform(block_map[bid])
                block_result.append(block)
                if children:
                    nested = _walk(children)
                    if nested:
                        block_children[bid] = nested
            return block_result

        top_blocks = _walk(first_level)
        return top_blocks, block_children

    @classmethod
    def _prepare_docx_descendants(
        cls, blocks: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        content_blocks = [block for block in blocks if block.get('block_type') != 1]
        raw_by_id = {block['block_id']: block for block in content_blocks}
        children_by_parent: Dict[str, List[str]] = {}
        for block in content_blocks:
            parent_id = block.get('parent_id')
            if parent_id in raw_by_id:
                children_by_parent.setdefault(parent_id, []).append(block['block_id'])

        root_block_ids: List[str] = []
        descendants: List[Dict[str, Any]] = []
        for block in content_blocks:
            block_id = block['block_id']
            block_type = block.get('block_type')
            content_key = cls._DOCX_BLOCK_TYPE_KEY.get(block_type)
            if content_key is None:
                raise ValueError(f'Feishu block type {block_type!r} is not supported for structured writing.')
            content = block.get(content_key)
            if block_type == 22 and content is None:
                content = {}
            content = dict(content)
            if block_type == 31:
                content.pop('cells', None)
                content.pop('merge_info', None)
                prop = dict(content.get('property') or {})
                prop.pop('merge_info', None)
                content['property'] = prop

            descendant = {
                'block_id': block_id,
                'block_type': block_type,
                content_key: content,
            }
            children = children_by_parent.get(block_id)
            if children:
                descendant['children'] = children
            descendants.append(descendant)
            if block.get('parent_id') not in raw_by_id:
                root_block_ids.append(block_id)
        return root_block_ids, descendants

    def _resolve_and_insert_children(
        self,
        document_id: str,
        block_children: Dict[str, List[Dict[str, Any]]],
        block_id_map: Dict[str, str],
    ) -> Dict[str, str]:
        temp_to_real = dict(block_id_map)
        pending = dict(block_children)
        while pending:
            resolved = [parent_id for parent_id in pending if parent_id in temp_to_real]
            if not resolved:
                raise RuntimeError(
                    'Feishu create descendant blocks has unresolved parent block ids: '
                    f'{sorted(pending)}.'
                )
            for temp_parent_id in resolved:
                _, child_map = self._append_docx_blocks(
                    document_id,
                    pending.pop(temp_parent_id),
                    parent_block_id=temp_to_real[temp_parent_id],
                )
                temp_to_real.update(child_map)
        return temp_to_real

    def _get_docx_children(self, document_id: str, parent_block_id: str) -> List[Dict[str, Any]]:
        url = f'{self._base_url}/docx/v1/documents/{document_id}/blocks/{parent_block_id}/children'
        items: List[Dict[str, Any]] = []
        page_token: Optional[str] = None
        while True:
            params: Dict[str, Any] = {'page_size': 500}
            if page_token:
                params['page_token'] = page_token
            data = (self._get(url, params=params) or {}).get('data') or {}
            items.extend(data.get('items') or [])
            page_token = data.get('page_token')
            if not page_token:
                break
        return items

    def _reuse_docx_default_text(
        self,
        document_id: str,
        blocks: List[Dict[str, Any]],
        existing_blocks: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        if not blocks or not existing_blocks:
            return blocks, {}
        source, existing = blocks[0], existing_blocks[0]
        source_id, existing_id = source.get('_temp_id'), existing.get('block_id')
        if (
            source.get('block_type') != 2
            or existing.get('block_type') != 2
            or self._docx_block_plain_text(existing)
            or not source_id
            or not existing_id
        ):
            return blocks, {}
        elements = (source.get('text') or {}).get('elements') or []
        if elements:
            self._patch(
                f'{self._base_url}/docx/v1/documents/{document_id}/blocks/{existing_id}',
                json={'update_text_elements': {'elements': elements}},
            )
        return blocks[1:], {source_id: existing_id}

    def _append_docx_blocks(
        self,
        document_id: str,
        blocks: List[Dict[str, Any]],
        *,
        parent_block_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        if not blocks:
            return [], {}
        inserted_blocks: List[Dict[str, Any]] = []
        block_id_map: Dict[str, str] = {}
        parent_id = parent_block_id or document_id
        url = f'{self._base_url}/docx/v1/documents/{document_id}/blocks/{parent_id}/children'
        if parent_id != document_id and blocks[0].get('block_type') == 2:
            existing_blocks = self._get_docx_children(document_id, parent_id)
            insert_index = len(existing_blocks)
            blocks, reused_map = self._reuse_docx_default_text(document_id, blocks, existing_blocks)
            block_id_map.update(reused_map)
        else:
            insert_index = -1
        chunk: List[Dict[str, Any]] = []

        def insert_blocks(source_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            nonlocal insert_index
            payload = [{
                key: value for key, value in block.items()
                if key not in ('_temp_id', '_table_cells')
            } for block in source_blocks]
            response = self._post(url, json={'index': insert_index, 'children': payload})
            created = (response.get('data') or {}).get('children') or []
            inserted_blocks.extend(created)
            if insert_index != -1:
                insert_index += len(source_blocks)
            for source, real in zip(source_blocks, created):
                if source.get('_temp_id') and real.get('block_id'):
                    block_id_map[source['_temp_id']] = real['block_id']
            return created

        for block in blocks:
            if block.get('_table_cells') is not None:
                if chunk:
                    insert_blocks(chunk)
                    chunk = []
                created_list = insert_blocks([block])
                table_info = created_list[0] if created_list and isinstance(created_list[0], dict) else {}
                table_block_id = table_info.get('block_id')
                cell_ids = (table_info.get('table') or {}).get('cells') or []
                if table_block_id:
                    self._fill_table_cells(
                        document_id, table_block_id, block['_table_cells'], cell_ids)
            else:
                chunk.append(block)
                if len(chunk) == 50:
                    insert_blocks(chunk)
                    chunk = []
        if chunk:
            insert_blocks(chunk)
        return inserted_blocks, block_id_map

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

    def _create_descendant_blocks(
        self,
        document_id: str,
        parent_block_id: str,
        index: int,
        children_id: List[str],
        descendants: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        url = f'{self._base_url}/docx/v1/documents/{document_id}/blocks/{parent_block_id}/descendant'
        response = self._post(
            url,
            json={
                'index': index,
                'children_id': children_id,
                'descendants': descendants,
            },
        )
        return (response or {}).get('data') or {}

    def _batch_update_blocks(
        self,
        document_id: str,
        requests: List[Dict[str, Any]],
        *,
        document_revision_id: int = -1,
    ) -> Dict[str, Any]:
        url = f'{self._base_url}/docx/v1/documents/{document_id}/blocks/batch_update'
        response = self._patch(
            url,
            params={'document_revision_id': document_revision_id},
            json={'requests': requests},
        )
        return (response or {}).get('data') or {}

    def _batch_delete_child_blocks(
        self,
        document_id: str,
        parent_block_id: str,
        start_index: int,
        end_index: int,
        *,
        document_revision_id: int = -1,
    ) -> Dict[str, Any]:
        url = (
            f'{self._base_url}/docx/v1/documents/{document_id}/blocks/'
            f'{parent_block_id}/children/batch_delete'
        )
        response = self._delete(
            url,
            params={'document_revision_id': document_revision_id},
            json={'start_index': start_index, 'end_index': end_index},
        )
        return (response or {}).get('data') or {}

    def create_block(
        self,
        document_id: str,
        parent_block_id: str,
        index: int,
        children_id: List[str],
        descendants: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return self._create_descendant_blocks(
            document_id, parent_block_id, index, children_id, descendants)

    def update_block(
        self,
        document_id: str,
        requests: List[Dict[str, Any]],
        *,
        document_revision_id: int = -1,
    ) -> Dict[str, Any]:
        return self._batch_update_blocks(
            document_id, requests, document_revision_id=document_revision_id)

    def delete_block(
        self,
        document_id: str,
        parent_block_id: str,
        start_index: int,
        end_index: int,
        *,
        document_revision_id: int = -1,
    ) -> Dict[str, Any]:
        return self._batch_delete_child_blocks(
            document_id,
            parent_block_id,
            start_index,
            end_index,
            document_revision_id=document_revision_id,
        )

    def move_block(
        self,
        document_id: str,
        source_parent_block_id: str,
        source_index: int,
        target_parent_block_id: str,
        target_index: int,
        children_id: List[str],
        descendants: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if len(children_id) != 1:
            raise ValueError('move_block requires exactly one root block.')
        create_index = target_index
        if source_parent_block_id == target_parent_block_id and source_index < target_index:
            create_index += 1
        create_data = self._create_descendant_blocks(
            document_id,
            target_parent_block_id,
            create_index,
            children_id,
            descendants,
        )
        delete_index = source_index
        if source_parent_block_id == target_parent_block_id and create_index <= source_index:
            delete_index += len(children_id)
        delete_data = self._batch_delete_child_blocks(
            document_id,
            source_parent_block_id,
            delete_index,
            delete_index + 1,
            document_revision_id=create_data.get('document_revision_id', -1),
        )
        return {'create': create_data, 'delete': delete_data}

    def write_doc_blocks(
        self,
        document_id: str,
        blocks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        '''Append native blocks to an existing Feishu document.'''
        if not isinstance(blocks, list):
            raise TypeError(f'blocks must be a list, got {type(blocks).__name__}.')
        if not blocks:
            raise ValueError('blocks must not be empty.')
        children_id, descendants = self._prepare_docx_descendants(blocks)
        if descendants:
            url = f'{self._base_url}/docx/v1/documents/{document_id}/blocks/{document_id}/descendant'
            self._post(url, json={
                'index': -1,
                'children_id': children_id,
                'descendants': descendants,
            })
        return self._get_doc_blocks_raw(document_id, with_descendants=True)

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

    def _write_cell_text(self, document_id: str, cell_block_id: str,
                         elements: List[Dict[str, Any]]) -> None:
        cell_children_url = (
            f'{self._base_url}/docx/v1/documents/{document_id}/blocks/{cell_block_id}/children')
        cd = (self._get(cell_children_url, params={'page_size': 10}) or {}).get('data') or {}
        text_blocks = [x for x in (cd.get('items') or []) if x.get('block_type') == 2]
        if text_blocks:
            self._patch(
                f'{self._base_url}/docx/v1/documents/{document_id}/blocks/{text_blocks[0]["block_id"]}',
                json={'update_text_elements': {'elements': elements}})
        else:
            self._post(cell_children_url, json={
                'index': 0,
                'children': [{'block_type': 2, 'text': {'elements': elements}}],
            })

    def _fill_table_cells(
        self,
        document_id: str,
        table_block_id: str,
        grid: List[List[Optional[List[Dict[str, Any]]]]],
        cell_ids: Optional[List[str]] = None,
    ) -> None:
        flat_cells = [c for row in grid for c in row]
        if not cell_ids:
            cell_blocks = self._get_table_cells(document_id, table_block_id)
            cell_ids = [b['block_id'] for b in cell_blocks if b.get('block_id')]
        for cell_id, elements in zip(cell_ids[:len(flat_cells)], flat_cells):
            if cell_id and elements:
                self._write_cell_text(document_id, cell_id, elements)

class FeishuFS(FeishuFSBase):
    __tool_auto_activate__ = [
        r'https?://[^\s/]+\.(?:feishu\.(?:cn|com)|larksuite\.com)(?:[/:?#]|$)',
        r'飞书|(?<!\w)feishu(?!\w)',
    ]
    __public_apis__ = LazyLLMFSBase.__public_apis__

    def __new__(cls, base_url: Optional[str] = None, app_id: Optional[str] = None, app_secret: Optional[str] = None,
                space_id: Optional[str] = None, user_refresh_token: Optional[str] = None,
                oauth_port: int = 9981, oauth_scope: str = _DEFAULT_OAUTH_SCOPE,
                asynchronous: bool = False, use_listings_cache: bool = False,
                skip_instance_cache: bool = False, loop: Optional[Any] = None,
                dynamic_auth: bool = False) -> LazyLLMFSBase:
        sid = (space_id or '').strip() if space_id is not None else ''
        if sid:
            wiki_sid = '' if sid == _SPACE_ID_DYNAMIC else sid
            return FeishuWikiFS(base_url=base_url, app_id=app_id, app_secret=app_secret, space_id=wiki_sid,
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
            try:
                top_blocks, block_children = self._convert_markdown_via_api(text)
                _, block_id_map = self._append_docx_blocks(doc_id, top_blocks)
                if block_children:
                    self._resolve_and_insert_children(doc_id, block_children, block_id_map)
            except (RuntimeError, requests.HTTPError):
                LOG.warning('Convert API failed, falling back to file upload')
                self._upload_file_to_drive(name, data, folder_token=parent_token)
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

    def __init__(self, fs: 'FeishuWikiFS', path: str, include_references: bool = False, **kwargs) -> None:
        content = fs._fetch_wiki_content(path, include_references=include_references)
        self._wiki_content: bytes = content
        super().__init__(fs, path, size=len(content), **kwargs)

    def _fetch_range(self, start: int, end: int) -> bytes:
        return self._wiki_content[start:end]


class FeishuWikiFS(FeishuFSBase):
    __tool_auto_activate__ = [
        r'https?://[^\s/]+\.(?:feishu\.(?:cn|com)|larksuite\.com)(?:[/:?#]|$)',
        r'飞书|(?<!\w)feishu(?!\w)',
    ]
    '''Read and manage authenticated Feishu/Lark Wiki spaces and documents.

    Select this Toolkit for Feishu or Lark browser URLs, especially `/wiki/`
    links. Use resolve_link to inspect a URL, read or read_with_references to
    load its content, search/find to locate Wiki nodes, and editing methods only
    when the user requests a change.
    '''

    __lazyllm_registry_disable__ = True
    protocol = 'feishu'
    _fs_protocol_key = 'feishu'
    document_provider = 'feishu'
    __public_apis__ = LinkDocumentFSBase.build_public_apis(extra=['search', 'find'])

    def _create_docx_node(self, title: str, parent_token: str = '') -> str:
        url = f'{self._base_url}/wiki/v2/spaces/{self._effective_space_id()}/nodes'
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

    def _effective_space_id(self) -> str:
        if self._space_id and self._space_id != _SPACE_ID_DYNAMIC:
            return self._space_id
        return (lazyllm_globals.config['feishu_wiki_space_id'] or '').strip()

    def _resolve_space_id(self, space_id: str = '') -> str:
        sid = (space_id or '').strip()
        if sid and sid != _SPACE_ID_DYNAMIC:
            return sid
        return self._effective_space_id()

    def _require_space_id(self) -> None:
        if not self._effective_space_id():
            raise ValueError(
                "space_id is required for FeishuWikiFS: pass space_id='wikcn...' to the constructor, "
                "set globals.config['feishu_wiki_space_id'], or use feishu@<space_id>:/ URI"
            )

    def _list_nodes_raw(self, parent_token: str = '', space_id: str = '') -> List[Dict[str, Any]]:
        sid = self._resolve_space_id(space_id)
        if not sid:
            raise ValueError(
                'space_id is required for Feishu wiki node listing: pass space_id or '
                "set globals.config['feishu_wiki_space_id']"
            )
        url = f'{self._base_url}/wiki/v2/spaces/{sid}/nodes'
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
            page_token = data.get('data', {}).get('page_token') or data.get('data', {}).get('next_page_token')
            if not page_token:
                break
        return results

    def _list_spaces_raw(self) -> List[Dict[str, Any]]:
        url = f'{self._base_url}/wiki/v2/spaces'
        params: Dict[str, Any] = {'page_size': 50}
        results: List[Dict[str, Any]] = []
        page_token: Optional[str] = None
        while True:
            if page_token:
                params['page_token'] = page_token
            data = self._get(url, params=params)
            payload = data.get('data') or {}
            results.extend(payload.get('items') or [])
            page_token = payload.get('page_token') or payload.get('next_page_token')
            if not payload.get('has_more') or not page_token:
                break
        return results

    def _resolve_path_to_token(self, path: str) -> str:
        self._require_space_id()
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

    def resolve_wiki_ref(self, url_or_path: str) -> Dict[str, Any]:
        norm = url_or_path.lstrip('/')
        if self.is_link_path(norm):
            url = self.decode_link_path(norm)
            parsed = _parse_feishu_browser_url(url)
            if not parsed:
                raise ValueError(f'Cannot parse Feishu browser URL: {url!r}')
            if parsed['kind'] == 'wiki_node':
                node = self._get_node(parsed['token'])
                return self._node_to_ref_dict(node, parsed['token'])
            return {'node_token': '', 'space_id': '', 'title': '', 'obj_type': parsed['kind'],
                    'obj_token': parsed['token'], 'has_child': False}
        if norm.startswith('~node/'):
            token = norm[len('~node/'):].rstrip('/').split('/')[0]
            node = self._get_node(token)
            return self._node_to_ref_dict(node, token)
        if _FEISHU_BARE_HOST_RE.match(url_or_path):
            parsed = _parse_feishu_browser_url(url_or_path)
            if not parsed:
                raise ValueError(f'Cannot parse Feishu browser URL: {url_or_path!r}')
            if parsed['kind'] == 'wiki_node':
                node = self._get_node(parsed['token'])
                return self._node_to_ref_dict(node, parsed['token'])
            return {'node_token': '', 'space_id': '', 'title': '', 'obj_type': parsed['kind'],
                    'obj_token': parsed['token'], 'has_child': False}
        node_token = self._resolve_path_to_token(url_or_path)
        node = self._get_node(node_token)
        return self._node_to_ref_dict(node, node_token)

    def _resolve_document_ref(self, url_or_path: str) -> Dict[str, Any]:
        return self.resolve_wiki_ref(url_or_path)

    @staticmethod
    def _node_to_ref_dict(node: Dict[str, Any], fallback_token: str = '') -> Dict[str, Any]:
        node_token = node.get('node_token') or fallback_token
        return {
            'node_token': node_token,
            'space_id': node.get('space_id') or '',
            'title': node.get('title') or '',
            'obj_type': node.get('obj_type') or '',
            'obj_token': node.get('obj_token') or '',
            'has_child': bool(node.get('has_child')),
            'creator': node.get('creator') or '',
            'owner': node.get('owner') or '',
            'node_creator': node.get('node_creator') or '',
        }

    def _list_child_nodes(self, node_token: str) -> List[Dict[str, Any]]:
        return self._list_nodes_raw(node_token)

    def ls(self, path: str = '/', detail: bool = True, **kwargs) -> List:
        if _is_wiki_locator_path(path):
            ref = self.resolve_wiki_ref(path)
            node_token = ref.get('node_token') or ''
            obj_type = ref.get('obj_type') or ''
            if not node_token:
                if obj_type in ('docx', 'doc'):
                    raise ValueError(f'ls: only wiki nodes can be listed; got obj_type={obj_type!r} for {path!r}')
                return []
            items = self._list_child_nodes(node_token)
            entries = [self._node_to_entry(item) for item in items]
            return entries if detail else [e['name'] for e in entries]
        self._require_space_id()
        node_token = self._resolve_path_to_token(path)
        items = self._list_nodes_raw(node_token)
        entries = [self._node_to_entry(item) for item in items]
        return entries if detail else [e['name'] for e in entries]

    def info(self, path: str, **kwargs) -> Dict[str, Any]:
        if _is_wiki_locator_path(path):
            ref = self.resolve_wiki_ref(path)
            node_token = ref.get('node_token') or ''
            if not node_token:
                return self._entry(path, ftype='directory')
            node = self._get_node(node_token)
            return self._node_to_entry(node, default_name=ref.get('title') or path)
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
        self._post(f'{self._base_url}/wiki/v2/spaces/{self._effective_space_id()}/nodes/{src_token}/copy', json=payload)

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
        self._post(f'{self._base_url}/wiki/v2/spaces/{self._effective_space_id()}/nodes/{src_token}/move', json=payload)

    def _resolve_link_content(self, parsed: Dict[str, str]) -> bytes:
        kind = parsed['kind']
        token = parsed['token']
        if kind == 'wiki_node':
            node = self._get_node(token)
            obj_type = node.get('obj_type')
            obj_token = node.get('obj_token') or ''
            if not obj_type or not obj_token:
                return b''
            if obj_type in {'doc', 'docx'}:
                return self._download_doc_raw(obj_token, obj_type=obj_type)
            if obj_type == 'file':
                return self._download_file_raw(obj_token)
            raise NotImplementedError(
                f'Feishu wiki node obj_type {obj_type!r} is not yet supported for link-based fetch'
            )
        if kind == 'docx':
            return self._download_doc_raw(token, obj_type='docx')
        if kind == 'doc':
            return self._download_doc_raw(token, obj_type='doc')
        raise NotImplementedError(f'Unsupported link kind: {kind!r}')

    def fetch_url(self, url: str) -> bytes:
        parsed = _parse_feishu_browser_url(url)
        if not parsed:
            raise ValueError(f'Cannot parse Feishu browser URL: {url!r}')
        return self._resolve_link_content(parsed)

    def _resolve_doc_token_from_path(self, path: str) -> Tuple[str, str]:
        norm = path.lstrip('/')
        if self.is_link_path(norm):
            url = self.decode_link_path(norm)
            if not _FEISHU_BARE_HOST_RE.match(url):
                raise ValueError(f'Expected a Feishu document URL: {url!r}')
            parsed = _parse_feishu_browser_url(url)
            if not parsed:
                return '', ''
            if parsed['kind'] == 'wiki_node':
                node = self._get_node(parsed['token'])
                return node.get('obj_token') or '', node.get('obj_type') or ''
            return parsed['token'], parsed['kind']
        if norm.startswith('~node/'):
            token = norm[len('~node/'):].rstrip('/').split('/')[0]
            node = self._get_node(token)
            return node.get('obj_token') or '', node.get('obj_type') or ''
        if norm.startswith('~docx/'):
            return norm[len('~docx/'):].split('/')[0], 'docx'
        if norm.startswith('~doc/'):
            return norm[len('~doc/'):].split('/')[0], 'doc'
        return '', ''

    def _list_document_references(self, path: str) -> List[Dict[str, Any]]:
        doc_token, obj_type = self._resolve_doc_token_from_path(path)
        if not doc_token or obj_type not in ('doc', 'docx'):
            return []
        try:
            blocks = self._get_doc_blocks_raw(doc_token, with_descendants=True)
        except Exception as exc:
            LOG.warning(f'_list_document_references: failed to get blocks for {path!r}: {exc}')
            return []
        refs: List[Dict[str, Any]] = []
        for el in _iter_docx_elements(blocks):
            url = _ref_from_element(el)
            if url:
                parsed = _parse_feishu_browser_url(url)
                ref_type = 'mention_doc' if 'mention_doc' in el else (
                    'link_preview' if 'link_preview' in el else 'hyperlink')
                refs.append({'url': url, 'ref_type': ref_type,
                             'kind': parsed['kind'] if parsed else 'external'})
        return LinkDocumentFSBase.dedupe_document_references(refs)

    def _fetch_wiki_content(self, path: str, include_references: bool = False) -> bytes:  # noqa C901
        norm = path.lstrip('/')
        # Bare feishu URL (https://xxx.feishu.cn/...) — convert to ~link/ path
        if _FEISHU_BARE_HOST_RE.match(path):
            norm = self.to_link_path(path).lstrip('/')
        if self.is_link_path(norm):
            url = self.decode_link_path(norm)
            parsed = _parse_feishu_browser_url(url)
            if not parsed:
                raise ValueError(f'Cannot parse Feishu browser URL from path: {path!r}')
            body = self._resolve_link_content(parsed)
        elif norm.startswith('~node/'):
            token = norm[len('~node/'):].rstrip('/').split('/')[0]
            body = self._resolve_link_content({'kind': 'wiki_node', 'token': token})
        elif norm.startswith('~docx/'):
            token = norm[len('~docx/'):].split('/')[0]
            body = self._download_doc_raw(token, obj_type='docx')
        elif norm.startswith('~doc/'):
            token = norm[len('~doc/'):].split('/')[0]
            body = self._download_doc_raw(token, obj_type='doc')
        else:
            node_token = self._resolve_path_to_token(path)
            if not node_token:
                return b''
            node = self._get_node(node_token)
            obj_type = node.get('obj_type')
            obj_token = node.get('obj_token') or ''
            if not obj_type or not obj_token:
                return b''
            if obj_type in {'doc', 'docx'}:
                body = self._download_doc_raw(obj_token, obj_type=obj_type)
            elif obj_type == 'file':
                body = self._download_file_raw(obj_token)
            else:
                return b''
        if include_references:
            body = self._append_document_references_footer_bytes(body, path)
        return body

    def cat_file(self, path: str, start: Optional[int] = None, end: Optional[int] = None,
                 **kwargs) -> bytes:
        data = self._fetch_wiki_content(path)
        return data[start:end] if (start is not None or end is not None) else data

    def read_bytes(self, path: str, include_references: bool = False) -> bytes:
        return self._fetch_wiki_content(path, include_references=include_references)

    def _open(self, path: str, mode: str = 'rb', block_size: Optional[int] = None,
              autocommit: bool = True, cache_options: Optional[Dict] = None,
              include_references: bool = False, **kwargs) -> CloudFSBufferedFile:
        if 'b' not in mode:
            raise ValueError('FeishuWikiFS only supports binary mode')
        if 'r' in mode:
            return FeishuWikiFile(self, path, include_references=include_references,
                                  mode=mode, block_size=block_size or self.blocksize,
                                  autocommit=autocommit, cache_options=cache_options)
        norm = path.lstrip('/')
        if norm.startswith(('~link/', '~node/', '~docx/', '~doc/')):
            raise NotImplementedError('FeishuWikiFS: write mode is not supported for link/node paths')
        return CloudFSBufferedFile(self, path, mode=mode, block_size=block_size or self.blocksize,
                                   autocommit=autocommit, cache_options=cache_options)

    def mkdir(self, path: str, create_parents: bool = True, **kwargs) -> None:
        parts = [p for p in path.strip('/').split('/') if p]
        if not parts:
            return
        title = parts[-1]
        parent_token = self._resolve_path_to_token('/' + '/'.join(parts[:-1])) if len(parts) > 1 else ''
        url = f'{self._base_url}/wiki/v2/spaces/{self._effective_space_id()}/nodes'
        payload: Dict[str, Any] = {'title': title, 'obj_type': 'docx', 'node_type': 'origin'}
        if parent_token:
            payload['parent_node_token'] = parent_token
        self._post(url, json=payload)

    def rm_file(self, path: str) -> None:
        token = self._resolve_path_to_token(path)
        if not token:
            raise FileNotFoundError(path)
        self._delete(f'{self._base_url}/wiki/v2/spaces/{self._effective_space_id()}/nodes/{token}')

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
            try:
                top_blocks, block_children = self._convert_markdown_via_api(text)
                _, block_id_map = self._append_docx_blocks(doc_id, top_blocks)
                if block_children:
                    self._resolve_and_insert_children(doc_id, block_children, block_id_map)
            except (RuntimeError, requests.HTTPError):
                LOG.warning('Convert API failed, falling back to plain text append')
                self._append_docx_text(doc_id, text)
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
        if node and not self._space_id:
            sid = (node.get('space_id') or '').strip()
            if sid:
                self._space_id = sid
        return node or {}

    def search(self, query: Union[str, List[str]], space_id: str = '', node_id: str = '',
               page_size: int = 20) -> List[Dict[str, Any]]:
        '''Search Feishu Wiki nodes by text.

        Args:
            query: Text to search for.
            space_id: Optional Wiki space id.
            node_id: Optional parent node scope.
            page_size: Maximum result count.
        '''
        if isinstance(query, list):
            query = ' '.join(str(item).strip() for item in query if str(item).strip())
        query = (query or '').strip()
        if not query:
            raise ValueError('query is required')
        page_size = max(1, min(int(page_size), 50))
        sid = self._resolve_space_id(space_id)
        node_id = (node_id or '').strip()
        if node_id and not sid:
            raise ValueError('space_id is required when node_id is specified')
        url = f'{self._base_url}/wiki/v2/nodes/search'
        payload: Dict[str, Any] = {'query': query}
        if sid:
            payload['space_id'] = sid
        if node_id:
            payload['node_id'] = node_id
        params: Dict[str, Any] = {'page_size': page_size}
        results: List[Dict[str, Any]] = []
        page_token: Optional[str] = None
        while True:
            if page_token:
                params['page_token'] = page_token
            data = self._post(url, params=params, json=payload)
            response_data = data.get('data') or {}
            items = response_data.get('items') or []
            for item in items:
                results.append({
                    'title': item.get('title') or '',
                    'node_token': item.get('node_id') or item.get('node_token') or '',
                    'obj_type': item.get('obj_type') or '',
                    'url': item.get('url') or '',
                    'snippet': item.get('snippet') or '',
                    'space_id': item.get('space_id') or sid,
                })
            page_token = response_data.get('page_token') or response_data.get('next_page_token')
            if not response_data.get('has_more') or not page_token or len(results) >= page_size:
                break
        return results[:page_size]

    @staticmethod
    def _compile_find_pattern(pattern: str) -> 're.Pattern[str]':
        pattern = (pattern or '').strip()
        if not pattern:
            raise ValueError('pattern is required')
        try:
            return re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            raise ValueError(f'Invalid regex pattern: {e}') from e

    def _enumerate_find_spaces(self, sid: str) -> List[str]:
        if sid:
            return [sid]
        return [
            str(item.get('space_id') or '').strip()
            for item in self._list_spaces_raw()
            if str(item.get('space_id') or '').strip()
        ]

    @staticmethod
    def _feishu_node_to_find_result(node: Dict[str, Any], current_space_id: str) -> Dict[str, Any]:
        return {
            'title': node.get('title') or '',
            'node_token': node.get('node_token') or '',
            'obj_type': node.get('obj_type') or '',
            'url': node.get('url') or '',
            'space_id': node.get('space_id') or current_space_id,
            'has_child': bool(node.get('has_child')),
        }

    def _collect_find_in_space(
        self,
        regex: 're.Pattern[str]',
        current_space_id: str,
        max_results: int,
        results: List[Dict[str, Any]],
        visited: set,
    ) -> None:
        self._walk_feishu_space(
            regex, '', 0, current_space_id, max_results, results, visited,
        )

    def _walk_feishu_space(
        self,
        regex: 're.Pattern[str]',
        parent_token: str,
        depth: int,
        current_space_id: str,
        max_results: int,
        results: List[Dict[str, Any]],
        visited: set,
    ) -> None:
        if len(results) >= max_results or depth > 8:
            return
        try:
            nodes = self._list_nodes_raw(parent_token, space_id=current_space_id)
        except Exception as exc:
            LOG.warning(
                f'Feishu wiki traversal failed for space_id={current_space_id!r} '
                f'parent_token={parent_token!r}: {exc}'
            )
            return
        for node in nodes:
            if self._process_feishu_find_node(
                node, regex, current_space_id, depth, max_results, results, visited,
            ):
                return

    def _process_feishu_find_node(
        self,
        node: Dict[str, Any],
        regex: 're.Pattern[str]',
        current_space_id: str,
        depth: int,
        max_results: int,
        results: List[Dict[str, Any]],
        visited: set,
    ) -> bool:
        if len(results) >= max_results:
            return True
        title = node.get('title') or ''
        if not title:
            return False
        if regex.search(title):
            results.append(self._feishu_node_to_find_result(node, current_space_id))
            if len(results) >= max_results:
                return True
        nt = node.get('node_token') or ''
        if nt and nt not in visited and bool(node.get('has_child')):
            visited.add(nt)
            self._walk_feishu_space(
                regex, nt, depth + 1, current_space_id, max_results, results, visited,
            )
        return len(results) >= max_results

    def find(self, pattern: str, space_id: str = '', max_results: int = 50) -> List[Dict[str, Any]]:
        '''Find Feishu Wiki nodes whose paths or titles match a pattern.

        Args:
            pattern: Glob or regular-expression pattern.
            space_id: Optional Wiki space id.
            max_results: Maximum result count.
        '''
        regex = self._compile_find_pattern(pattern)
        max_results = max(1, min(int(max_results), 200))
        sid = self._resolve_space_id(space_id)
        space_ids = self._enumerate_find_spaces(sid)

        results: List[Dict[str, Any]] = []
        visited: set = set()
        for current_space_id in space_ids:
            self._collect_find_in_space(regex, current_space_id, max_results, results, visited)
            if len(results) >= max_results:
                break
        return results[:max_results]

    def get_document_id(self, path: str) -> str:
        if self.is_link_path(path):
            path = self.decode_link_path(path)

        norm = path.lstrip('/')
        if norm.startswith('~docx/'):
            return norm[len('~docx/'):].rstrip('/').split('/')[0]
        if norm.startswith('~doc/'):
            return norm[len('~doc/'):].rstrip('/').split('/')[0]
        if norm.startswith('~node/'):
            node_token = norm[len('~node/'):].rstrip('/').split('/')[0]
        else:
            parsed = _parse_feishu_browser_url(path)
            if parsed:
                if parsed['kind'] in ('doc', 'docx'):
                    return parsed['token']
                node_token = parsed['token']
            elif urlparse(path).scheme:
                raise ValueError(f'Expected a Feishu document URL: {path!r}')
            else:
                node_token = self._resolve_path_to_token(path)
                if not node_token:
                    raise FileNotFoundError(f'Path not found: {path}')

        node = self._get_node(node_token)
        obj_type = node.get('obj_type')
        document_id = node.get('obj_token') or ''
        if obj_type not in ('doc', 'docx') or not document_id:
            raise ValueError(
                f'Feishu target does not point to a document (obj_type={obj_type!r})'
            )
        return document_id

    def get_doc_blocks(self, path: str, with_descendants: bool = True) -> List[Dict[str, Any]]:
        '''Return native Feishu blocks with an added derived ``plain_text`` field.'''
        document_id = self.get_document_id(path)
        blocks = self._get_doc_blocks_raw(document_id, with_descendants=with_descendants)
        for block in blocks:
            block.setdefault('plain_text', self._docx_block_plain_text(block))
        return blocks

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
            creator=node.get('creator') or '', owner=node.get('owner') or '',
            node_creator=node.get('node_creator') or '',
        )
