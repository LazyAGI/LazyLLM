# Copyright (c) 2026 LazyAGI. All rights reserved.
from datetime import datetime
from typing import Any, Dict, List, Optional

import lazyllm

from ..base import LazyLLMFSBase, CloudFSBufferedFile


_GRAPH_BASE = 'https://graph.microsoft.com/v1.0'


class OneDriveFS(LazyLLMFSBase):

    def __init__(self, token: Optional[str] = None,
                 client_id: Optional[str] = None,
                 client_secret: Optional[str] = None,
                 tenant_id: str = 'common',
                 base_url: Optional[str] = None,
                 **storage_options):
        self._client_id = client_id
        self._client_secret = client_secret
        self._tenant_id = tenant_id
        if not token and client_id and client_secret:
            token = self._acquire_app_token(client_id, client_secret, tenant_id)
            if not token or not str(token).strip():
                raise ValueError('Failed to acquire OneDrive token')
        super().__init__(token=token or '', base_url=base_url or _GRAPH_BASE, **storage_options)

    def _setup_auth(self) -> None:
        if self._token:
            self._session.headers.update({
                'Authorization': f'Bearer {self._token}',
                'Accept': 'application/json',
                'Content-Type': 'application/json',
            })

    def ls(self, path: str, detail: bool = True, **kwargs) -> List:
        url = self._make_children_url(path)
        results = []
        next_url: Optional[str] = url
        params: Dict[str, Any] = {'$top': 200}
        while next_url:
            if next_url == url:
                data = self._get(next_url, params=params)
            else:
                data = self._get(next_url)
            for item in data.get('value', []):
                entry = self._item_to_entry(item)
                results.append(entry if detail else entry['name'])
            next_url = data.get('@odata.nextLink')
        return results

    def info(self, path: str, **kwargs) -> Dict[str, Any]:
        url = self._make_item_url(path)
        if not url:
            return self._entry('/', ftype='directory')
        data = self._get(url)
        return self._item_to_entry(data)

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
        name = parts[-1] if parts else 'New Folder'
        parent_path = '/'.join(parts[:-1]) if len(parts) > 1 else 'me'
        parent_url = self._make_children_url(parent_path)
        payload = {
            'name': name,
            'folder': {},
            '@microsoft.graph.conflictBehavior': 'rename',
        }
        self._post(parent_url, json=payload)

    def rm_file(self, path: str) -> None:
        url = self._make_item_url(path)
        if not url:
            raise FileNotFoundError(path)
        self._delete(url)

    def rmdir(self, path: str) -> None:
        self.rm_file(path)

    def _download_range(self, path: str, start: int, end: int) -> bytes:
        url = self._make_item_url(path)
        if not url:
            raise FileNotFoundError(path)
        download_url = self._get(url).get('@microsoft.graph.downloadUrl', '')
        if not download_url:
            download_url = url + '/content'
        headers = {'Range': f'bytes={start}-{end - 1}'}
        resp = self._request('GET', download_url, headers=headers)
        return resp.content

    def _upload_data(self, path: str, data: bytes) -> None:
        url = self._make_upload_url(path)
        self._request('PUT', url, data=data,
                      headers={'Content-Type': 'application/octet-stream'})

    def _platform_supports_webhook(self) -> bool:
        return True

    def _register_webhook(self, webhook_url: str, events: List[str], path: str) -> Dict[str, Any]:
        import datetime
        expiry = (datetime.datetime.utcnow() + datetime.timedelta(hours=1)).isoformat() + 'Z'
        resource = self._graph_resource(path)
        payload = {
            'changeType': ','.join(events) if events and events != ['*'] else 'updated,deleted,created',
            'notificationUrl': webhook_url,
            'resource': resource,
            'expirationDateTime': expiry,
            'clientState': 'lazyllm-fs',
        }
        return self._post(f'{self._base_url}/subscriptions', json=payload)

    def _make_item_url(self, path: str) -> str:
        parts = self._parse_path(path)
        if not parts:
            return f'{self._base_url}/me/drive/root'
        if parts[0] == 'me':
            item_id = parts[-1] if len(parts) > 1 else 'root'
            if item_id == 'me':
                return f'{self._base_url}/me/drive/root'
            return f'{self._base_url}/me/drive/items/{item_id}'
        if parts[0] == 'sites' and len(parts) >= 4:
            site_id, drive_id = parts[1], parts[3]
            item_id = parts[4] if len(parts) > 4 else 'root'
            return f'{self._base_url}/sites/{site_id}/drives/{drive_id}/items/{item_id}'
        item_id = parts[-1]
        return f'{self._base_url}/me/drive/items/{item_id}'

    def _make_children_url(self, path: str) -> str:
        return self._make_item_url(path) + '/children'

    def _make_upload_url(self, path: str) -> str:
        return self._make_item_url(path) + '/content'

    def _graph_resource(self, path: str) -> str:
        parts = self._parse_path(path)
        if not parts or parts[0] == 'me':
            return 'me/drive/root'
        return f'me/drive/items/{parts[-1]}'

    @staticmethod
    def _acquire_app_token(client_id: str, client_secret: str, tenant_id: str) -> str:
        try:
            import msal
            app = msal.ConfidentialClientApplication(
                client_id,
                authority=f'https://login.microsoftonline.com/{tenant_id}',
                client_credential=client_secret,
            )
            result = app.acquire_token_for_client(
                scopes=['https://graph.microsoft.com/.default'])
            return result.get('access_token', '')
        except ImportError:
            import requests as req
            url = f'https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token'
            data = {
                'grant_type': 'client_credentials',
                'client_id': client_id,
                'client_secret': client_secret,
                'scope': 'https://graph.microsoft.com/.default',
            }
            resp = req.post(url, data=data)
            resp.raise_for_status()
            return resp.json().get('access_token', '')

    @staticmethod
    def _item_to_entry(item: Dict[str, Any]) -> Dict[str, Any]:
        is_folder = 'folder' in item
        mtime = None
        ts = item.get('lastModifiedDateTime')
        if ts:
            try:
                mtime = datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
            except (ValueError, TypeError) as e:
                lazyllm.LOG.debug(f"Failed to parse timestamp '{ts}': {e}")
        return LazyLLMFSBase._entry(
            name=item.get('id', ''),
            size=int(item.get('size', 0) or 0),
            ftype='directory' if is_folder else 'file',
            mtime=mtime, title=item.get('name', ''),
        )
