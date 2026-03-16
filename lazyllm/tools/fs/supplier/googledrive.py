# Copyright (c) 2026 LazyAGI. All rights reserved.
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple

import lazyllm

from ..base import LazyLLMFSBase, CloudFSBufferedFile


_API_BASE = 'https://www.googleapis.com/drive/v3'
_UPLOAD_BASE = 'https://www.googleapis.com/upload/drive/v3'
_SCOPES = ['https://www.googleapis.com/auth/drive']
_SA_TOKEN_BUFFER = 300  # refresh 5 min before expiry


class GoogleDriveFS(LazyLLMFSBase):

    def __init__(
        self,
        credentials: Optional[Union[str, dict]] = None,
        base_url: Optional[str] = None,
        asynchronous: bool = False,
        use_listings_cache: bool = False,
        skip_instance_cache: bool = False,
        loop: Optional[Any] = None,
    ):
        secret_payload: Optional[Dict] = None
        if credentials:
            if isinstance(credentials, str):
                with open(credentials) as fh:
                    secret_payload = json.load(fh)
            else:
                secret_payload = credentials
        super().__init__(
            token=secret_payload or {},
            base_url=base_url or _API_BASE,
            asynchronous=asynchronous,
            use_listings_cache=use_listings_cache,
            skip_instance_cache=skip_instance_cache,
            loop=loop,
        )

    @property
    def _service_account_info(self) -> Optional[Dict]:
        if isinstance(self._secret_key, dict):
            return self._secret_key
        return None

    def _setup_auth(self) -> None:
        # Short-lived access tokens are injected into headers by _ensure_token.
        return None

    def _acquire_access_token(self) -> Tuple[str, Optional[float]]:
        if not self._service_account_info:
            return '', None
        token = self._fetch_sa_token()
        if not token:
            return '', None
        expires_at = time.time() + 3600 - _SA_TOKEN_BUFFER
        return token, expires_at

    def ls(self, path: str, detail: bool = True, **kwargs) -> List:
        parts = self._parse_path(path)
        if parts and parts[0] == 'drive' and len(parts) >= 2:
            drive_id = parts[1]
            parent_id = parts[2] if len(parts) > 2 else drive_id
        elif parts:
            drive_id = None
            parent_id = parts[-1]
        else:
            drive_id = None
            parent_id = 'root'

        params: Dict[str, Any] = {
            'q': f"'{parent_id}' in parents and trashed = false",
            'fields': 'nextPageToken,files(id,name,mimeType,size,modifiedTime,createdTime)',
            'pageSize': 200,
        }
        if drive_id:
            params.update({
                'driveId': drive_id,
                'includeItemsFromAllDrives': 'true',
                'supportsAllDrives': 'true',
                'corpora': 'drive',
            })

        results = []
        page_token = None
        while True:
            if page_token:
                params['pageToken'] = page_token
            data = self._get(f'{self._base_url}/files', params=params)
            for item in data.get('files', []):
                entry = self._item_to_entry(item)
                results.append(entry if detail else entry['name'])
            page_token = data.get('nextPageToken')
            if not page_token:
                break
        return results

    def info(self, path: str, **kwargs) -> Dict[str, Any]:
        parts = self._parse_path(path)
        if not parts:
            return self._entry('/', ftype='directory')
        file_id = parts[-1]
        url = f'{self._base_url}/files/{file_id}'
        params = {
            'fields': 'id,name,mimeType,size,modifiedTime,createdTime',
            'supportsAllDrives': 'true',
        }
        data = self._get(url, params=params)
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
        parent_id = parts[-2] if len(parts) >= 2 else 'root'
        metadata: Dict[str, Any] = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id],
        }
        self._post(f'{self._base_url}/files', json=metadata,
                   params={'supportsAllDrives': 'true'})

    def rm_file(self, path: str) -> None:
        parts = self._parse_path(path)
        if not parts:
            raise FileNotFoundError(path)
        file_id = parts[-1]
        url = f'{self._base_url}/files/{file_id}'
        self._delete(url, params={'supportsAllDrives': 'true'})

    def rmdir(self, path: str) -> None:
        self.rm_file(path)

    def _download_range(self, path: str, start: int, end: int) -> bytes:
        parts = self._parse_path(path)
        file_id = parts[-1] if parts else path
        url = f'{self._base_url}/files/{file_id}'
        headers = {'Range': f'bytes={start}-{end - 1}'}
        resp = self._request('GET', url,
                             params={'alt': 'media', 'supportsAllDrives': 'true'},
                             headers=headers)
        return resp.content

    def _upload_data(self, path: str, data: bytes) -> None:
        parts = self._parse_path(path)
        name = parts[-1] if parts else 'untitled'
        parent_id = parts[-2] if len(parts) >= 2 else 'root'

        metadata = json.dumps({'name': name, 'parents': [parent_id]})
        boundary = 'lazyllm_fs_boundary'
        body = (
            f'--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n'
            f'{metadata}\r\n'
            f'--{boundary}\r\nContent-Type: application/octet-stream\r\n\r\n'
        ).encode() + data + f'\r\n--{boundary}--'.encode()

        self._request(
            'POST', f'{_UPLOAD_BASE}/files',
            params={'uploadType': 'multipart', 'supportsAllDrives': 'true'},
            headers={'Content-Type': f'multipart/related; boundary={boundary}'},
            data=body,
        )

    def _platform_supports_webhook(self) -> bool:
        return True

    def _register_webhook(self, webhook_url: str, events: List[str], path: str) -> Dict[str, Any]:
        import uuid
        parts = self._parse_path(path)
        file_id = parts[-1] if parts else 'root'
        channel_id = str(uuid.uuid4())
        payload = {
            'id': channel_id,
            'type': 'web_hook',
            'address': webhook_url,
            'payload': True,
        }
        url = f'{self._base_url}/files/{file_id}/watch'
        result = self._post(url, json=payload, params={'supportsAllDrives': 'true'})
        result['channel_id'] = channel_id
        return result

    def _fetch_sa_token(self) -> str:
        if not self._service_account_info:
            return ''
        try:
            import jwt as pyjwt
            import time as _time
            info = self._service_account_info
            now = int(_time.time())
            claim = {
                'iss': info['client_email'],
                'scope': ' '.join(_SCOPES),
                'aud': 'https://oauth2.googleapis.com/token',
                'iat': now,
                'exp': now + 3600,
            }
            signed = pyjwt.encode(claim, info['private_key'], algorithm='RS256')
            resp = self._session.post(
                'https://oauth2.googleapis.com/token',
                data={
                    'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer',
                    'assertion': signed,
                },
            )
            resp.raise_for_status()
            return resp.json().get('access_token', '')
        except Exception as e:
            lazyllm.LOG.debug(f'Failed to fetch service account token: {e}')
            return ''

    @staticmethod
    def _item_to_entry(item: Dict[str, Any]) -> Dict[str, Any]:
        mime = item.get('mimeType', '')
        ftype = 'directory' if mime == 'application/vnd.google-apps.folder' else 'file'
        mtime = None
        ts = item.get('modifiedTime')
        if ts:
            try:
                mtime = datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
            except (ValueError, TypeError) as e:
                lazyllm.LOG.debug(f"Failed to parse timestamp '{ts}': {e}")
        return LazyLLMFSBase._entry(
            name=item.get('id', ''),
            size=int(item.get('size', 0) or 0),
            ftype=ftype, mtime=mtime,
            title=item.get('name', ''), mime_type=mime,
        )
