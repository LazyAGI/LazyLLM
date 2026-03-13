# Copyright (c) 2026 LazyAGI. All rights reserved.
from typing import Any, Dict, List, Optional

from ..base import LazyLLMFSBase, CloudFSBufferedFile


_API_BASE = 'https://open.feishu.cn/open-apis'


class FeishuFS(LazyLLMFSBase):

    protocol = 'feishu'

    def __init__(self, token: str, base_url: Optional[str] = None,
                 app_id: Optional[str] = None, app_secret: Optional[str] = None,
                 **storage_options):
        self._app_id = app_id
        self._app_secret = app_secret
        super().__init__(token=token, base_url=base_url or _API_BASE, **storage_options)

    def _setup_auth(self) -> None:
        self._session.headers.update({
            'Authorization': f'Bearer {self._token}',
            'Content-Type': 'application/json; charset=utf-8',
        })

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
