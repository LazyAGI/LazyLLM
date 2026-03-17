# Copyright (c) 2026 LazyAGI. All rights reserved.
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from lazyllm import config

from ..base import LazyLLMFSBase, CloudFSBufferedFile

config.add('feishu_app_id', str, None, 'FEISHU_APP_ID', description='Feishu App ID for tenant_access_token.')
config.add('feishu_app_secret', str, None, 'FEISHU_APP_SECRET', description='Feishu App Secret for tenant_access_token.')


_API_BASE = 'https://open.feishu.cn/open-apis'
_TOKEN_URL = 'https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal'
_TOKEN_REFRESH_BUFFER = 300  # refresh if remaining life <= 5 min


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


class FeishuFS(LazyLLMFSBase):

    def __new__(cls, base_url: Optional[str] = None, app_id: Optional[str] = None, app_secret: Optional[str] = None,
                space_id: Optional[str] = None, asynchronous: bool = False, use_listings_cache: bool = False,
                skip_instance_cache: bool = False, loop: Optional[Any] = None) -> LazyLLMFSBase:
        if not app_id or not app_secret:
            app_id = app_id or config['feishu_app_id']
            app_secret = app_secret or config['feishu_app_secret']
            assert app_id and app_secret, 'feishu_app_id and feishu_app_secret are required'
        if space_id is not None and str(space_id).strip():
            return FeishuWikiFS(base_url=base_url, app_id=app_id, app_secret=app_secret, space_id=space_id,
                                asynchronous=asynchronous, use_listings_cache=use_listings_cache,
                                skip_instance_cache=skip_instance_cache, loop=loop)
        return super().__new__(cls)

    def __init__(self, base_url: Optional[str] = None, app_id: Optional[str] = None, app_secret: Optional[str] = None,
                 asynchronous: bool = False, use_listings_cache: bool = False, skip_instance_cache: bool = False,
                 loop: Optional[Any] = None):
        super().__init__(
            token={'app_id': app_id, 'app_secret': app_secret},
            base_url=base_url or _API_BASE,
            asynchronous=asynchronous,
            use_listings_cache=use_listings_cache,
            skip_instance_cache=skip_instance_cache,
            loop=loop,
        )
        if self._app_id and self._app_secret:
            self._ensure_token()

    @property
    def _app_id(self) -> str:
        return str(getattr(self, '_token', {}).get('app_id', ''))  # type: ignore[union-attr]

    @property
    def _app_secret(self) -> str:
        return str(getattr(self, '_token', {}).get('app_secret', ''))  # type: ignore[union-attr]

    def _setup_auth(self) -> None:
        self._session.headers.update({
            'Content-Type': 'application/json; charset=utf-8',
        })

    def _acquire_access_token(self) -> Tuple[str, Optional[float]]:
        return _feishu_acquire_access_token(self._session, self._app_id, self._app_secret)

    def _apply_access_token(self, token: str) -> None:
        self._session.headers.update({'Authorization': f'Bearer {token}'})

    def ls(self, path: str, detail: bool = True, **kwargs) -> List:
        folder_token = self._token_from_path(path)
        url = f'{self._base_url}/drive/v1/files'
        params: Dict[str, Any] = {'page_size': 200}
        if folder_token:
            params['folder_token'] = folder_token

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
            next_token = data.get('data', {}).get('next_page_token')
            if not next_token:
                break
            page_token = next_token
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
    def _token_from_path(path: str) -> str:
        stripped = path.lstrip('/')
        if not stripped:
            return ''
        return stripped.split('/')[-1]

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


class FeishuWikiFS(LazyLLMFSBase):
    __lazyllm_registry_disable__ = True

    def __init__(self, base_url: Optional[str] = None, app_id: Optional[str] = None, app_secret: Optional[str] = None,
                 space_id: Optional[str] = None, asynchronous: bool = False, use_listings_cache: bool = False,
                 skip_instance_cache: bool = False, loop: Optional[Any] = None):
        self._space_id = (space_id or '').strip()
        if not self._space_id:
            raise ValueError('space_id is required for FeishuWikiFS')
        super().__init__(
            token={'app_id': app_id, 'app_secret': app_secret},
            base_url=base_url or _API_BASE,
            asynchronous=asynchronous,
            use_listings_cache=use_listings_cache,
            skip_instance_cache=skip_instance_cache,
            loop=loop,
        )
        if self._app_id and self._app_secret:
            self._ensure_token()

    @property
    def _app_id(self) -> str:
        return str(getattr(self, '_token', {}).get('app_id', ''))  # type: ignore[union-attr]

    @property
    def _app_secret(self) -> str:
        return str(getattr(self, '_token', {}).get('app_secret', ''))  # type: ignore[union-attr]

    def _setup_auth(self) -> None:
        self._session.headers.update({
            'Content-Type': 'application/json; charset=utf-8',
        })

    def _acquire_access_token(self) -> Tuple[str, Optional[float]]:
        return _feishu_acquire_access_token(self._session, self._app_id, self._app_secret)

    def _apply_access_token(self, token: str) -> None:
        self._session.headers.update({'Authorization': f'Bearer {token}'})

    def ls(self, path: str, detail: bool = True, **kwargs) -> List:
        node_token = self._token_from_path(path)
        url = f'{self._base_url}/wiki/v2/spaces/{self._space_id}/nodes'
        params: Dict[str, Any] = {'page_size': 200}
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
            page_token = data.get('data', {}).get('page_token') or data.get('data', {}).get('next_page_token')
            if not page_token:
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
        obj_token = node.get('obj_token') or node.get('obj_token', '')
        if not obj_type or not obj_token:
            return b''
        if obj_type in {'doc', 'docx'}:
            data = self._download_doc_raw(obj_token)
        elif obj_type == 'file':
            data = self._download_file_raw(obj_token)
        else:
            data = b''
        return data[start:end]

    def _upload_data(self, path: str, data: bytes) -> None:
        raise NotImplementedError('FeishuWikiFS does not support write yet')

    def _download_doc_raw(self, doc_token: str) -> bytes:
        try:
            url = f'{self._base_url}/docx/v1/documents/{doc_token}/raw_content'
            resp = self._request('GET', url)
            return resp.content
        except Exception:
            url = f'{self._base_url}/doc/v2/{doc_token}/raw_content'
            resp = self._request('GET', url)
            return resp.content

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
    def _token_from_path(path: str) -> str:
        stripped = path.lstrip('/')
        if not stripped:
            return ''
        return stripped.split('/')[-1]

    @staticmethod
    def _node_to_entry(node: Dict[str, Any], default_name: Optional[str] = None) -> Dict[str, Any]:
        obj_type = node.get('obj_type')
        is_dir = obj_type in {'folder', 'wiki', 'space'}
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
