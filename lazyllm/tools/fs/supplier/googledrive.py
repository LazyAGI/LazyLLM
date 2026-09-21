# Copyright (c) 2026 LazyAGI. All rights reserved.
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Union, Tuple

import lazyllm
from lazyllm.tools.agent.toolsManager import fc_register
from lazyllm import config

from ..base import LazyLLMFSBase, CloudFSBufferedFile

config.add('googledrive_credentials', str, None, 'GOOGLE_APPLICATION_CREDENTIALS',
           description='Path to Google service account JSON (ADC).')

_API_BASE = 'https://www.googleapis.com/drive/v3'
_UPLOAD_BASE = 'https://www.googleapis.com/upload/drive/v3'
_SCOPES = ['https://www.googleapis.com/auth/drive']
_SA_TOKEN_BUFFER = 300  # refresh 5 min before expiry
_LIST_FIELDS = (
    'nextPageToken,incompleteSearch,'
    'files(id,name,mimeType,size,modifiedTime,createdTime,parents,driveId,webViewLink,description)'
)
_GOOGLE_WORKSPACE_EXPORT_TYPES = {
    'application/vnd.google-apps.document': 'text/plain',
    'application/vnd.google-apps.spreadsheet': 'text/csv',
}


def _search_tool_schema(
    keywords: List[str],
    file_name: str = '',
    drive_id: str = '',
    folder_id: str = '',
    limit: int = 20,
) -> List[Dict[str, Any]]:
    return []


def _adapt_search_tool_input(tool_input: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    if isinstance(tool_input, str):
        return {'keywords': [tool_input]}
    adapted = dict(tool_input)
    if 'keywords' not in adapted and 'query' in adapted:
        adapted['keywords'] = adapted.pop('query')
    if isinstance(adapted.get('keywords'), str):
        adapted['keywords'] = [adapted['keywords']]
    return adapted


class GoogleDriveFS(LazyLLMFSBase):
    __tool_auto_activate__ = [
        r'https?://(?:drive|docs)\.google\.com(?:[/:?#]|$)',
        r'谷歌云端硬盘|谷歌(?:文档|表格|幻灯片)|Google\s*云端硬盘',
        r'(?<!\w)google\s+(?:drive|docs|sheets|slides)(?!\w)',
    ]
    __public_apis__ = ['search', 'find', 'read', 'read_file']
    __tool_schema_overrides__ = {'search': _search_tool_schema}
    __tool_input_adapters__ = {'search': _adapt_search_tool_input}

    def __init__(
        self,
        credentials: Optional[Union[str, dict]] = None,
        base_url: Optional[str] = None,
        asynchronous: bool = False,
        use_listings_cache: bool = False,
        skip_instance_cache: bool = False,
        loop: Optional[Any] = None,
        dynamic_auth: bool = False,
    ):
        self._mime_type_cache: Dict[str, str] = {}
        if dynamic_auth:
            super().__init__(
                token={},
                base_url=base_url or _API_BASE,
                asynchronous=asynchronous,
                use_listings_cache=use_listings_cache,
                skip_instance_cache=skip_instance_cache,
                loop=loop,
                dynamic_auth=True,
            )
            return
        credentials = (credentials or config['googledrive_credentials']
                       or os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
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
        # Short-lived access tokens are injected into headers by inject_auth_header().
        return None

    def _do_acquire_without_refresh(self) -> Tuple[str, Optional[float], str]:
        if not self._service_account_info:
            raise ValueError(f'{type(self).__name__} failed to acquire access token: '
                             'service_account_info is not configured.')
        token = self._fetch_sa_token()
        if not token:
            raise ValueError(f'{type(self).__name__} failed to acquire access token from service account.')
        expires_at = time.time() + 3600 - _SA_TOKEN_BUFFER
        return token, expires_at, ''

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
            'fields': 'id,name,mimeType,size,modifiedTime,createdTime,webViewLink',
            'supportsAllDrives': 'true',
        }
        data = self._get(url, params=params)
        return self._item_to_entry(data)

    @fc_register(host_file='NONE')
    def search(
        self,
        keywords: Union[str, List[str]],
        file_name: str = '',
        drive_id: str = '',
        folder_id: str = '',
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        normalized = self._normalize_keywords(keywords)
        limit = self._normalize_limit(limit, default=20, maximum=1000)
        terms = ['trashed = false']
        terms.extend(f"fullText contains '{self._escape_query_literal(item)}'" for item in normalized)
        if file_name := (file_name or '').strip():
            terms.append(f"name = '{self._escape_query_literal(file_name)}'")
        if folder_id := (folder_id or '').strip():
            terms.append(f"'{self._escape_query_literal(folder_id)}' in parents")
        return [
            self._item_to_entry(item)
            for item in self._iter_files(
                ' and '.join(terms),
                drive_id=(drive_id or '').strip(),
                max_items=limit,
            )
        ]

    @fc_register(host_file='NONE')
    def find(
        self,
        pattern: str,
        drive_id: str = '',
        folder_id: str = '',
        limit: int = 50,
        max_scan: int = 1000,
    ) -> List[Dict[str, Any]]:
        pattern = (pattern or '').strip()
        if not pattern:
            raise ValueError('pattern is required')
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f'invalid regular expression: {exc}') from exc
        limit = self._normalize_limit(limit, default=50, maximum=1000)
        max_scan = self._normalize_limit(max_scan, default=1000, maximum=10000)
        terms = ['trashed = false']
        if folder_id := (folder_id or '').strip():
            terms.append(f"'{self._escape_query_literal(folder_id)}' in parents")

        matches = []
        for item in self._iter_files(
            ' and '.join(terms),
            drive_id=(drive_id or '').strip(),
            max_items=max_scan,
        ):
            if regex.search(item.get('name', '')):
                matches.append(self._item_to_entry(item))
                if len(matches) >= limit:
                    break
        return matches

    @fc_register(host_file='NONE')
    def read(self, path: str) -> str:
        return super().read(path)

    def read_bytes(self, path: str) -> bytes:
        metadata = self.info(path)
        mime_type = metadata.get('mime_type') or ''
        if mime_type.startswith('application/vnd.google-apps.'):
            # Workspace metadata has no exported byte size; buffered reads would return empty content.
            return self._export_document(metadata['name'], mime_type)
        return super().read_bytes(path)

    @fc_register(host_file='NONE')
    def read_file(self, path: str) -> str:
        return super().read_file(path)

    def write(self, path: str, content: str) -> None:
        return super().write(path, content)

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

    def rm(self, path: str, recursive: bool = False) -> None:
        return super().rm(path, recursive=recursive)

    def copy(self, path1: str, path2: str, recursive: bool = False, **kwargs) -> None:
        if self.isdir(path1):  # type: ignore[attr-defined]
            raise NotImplementedError('GoogleDriveFS does not support folder copy via the official API')
        parts1, parts2 = self._parse_path(path1), self._parse_path(path2)
        file_id = parts1[-1] if parts1 else ''
        if not file_id:
            raise FileNotFoundError(path1)
        parent_id = parts2[-2] if len(parts2) >= 2 else 'root'
        metadata: Dict[str, Any] = {'parents': [parent_id], 'name': parts2[-1]} if parts2 else {'parents': ['root']}
        self._post(f'{self._base_url}/files/{file_id}/copy',
                   json=metadata, params={'supportsAllDrives': 'true'})

    def move(self, path1: str, path2: str, recursive: bool = False, **kwargs) -> None:
        parts1, parts2 = self._parse_path(path1), self._parse_path(path2)
        file_id = parts1[-1] if parts1 else ''
        if not file_id:
            raise FileNotFoundError(path1)
        new_parent = parts2[-2] if len(parts2) >= 2 else 'root'
        info = self._get(f'{self._base_url}/files/{file_id}',
                         params={'fields': 'parents', 'supportsAllDrives': 'true'})
        old_parents = ','.join(info.get('parents', []))
        body: Dict[str, Any] = {'name': parts2[-1]} if parts2 else {}
        self._patch(f'{self._base_url}/files/{file_id}',
                    params={'addParents': new_parent, 'removeParents': old_parents,
                            'supportsAllDrives': 'true', 'fields': 'id'},
                    json=body or None)

    def _download_range(self, path: str, start: int, end: int) -> bytes:
        parts = self._parse_path(path)
        file_id = parts[-1] if parts else path
        mime_type = self._mime_type_cache.get(file_id)
        if mime_type is None:
            metadata = self._get(
                f'{self._base_url}/files/{file_id}',
                params={'fields': 'mimeType', 'supportsAllDrives': 'true'},
            )
            mime_type = metadata.get('mimeType') or ''
            self._mime_type_cache[file_id] = mime_type
        if mime_type.startswith('application/vnd.google-apps.'):
            return self._export_document(file_id, mime_type)[start:end]
        url = f'{self._base_url}/files/{file_id}'
        headers = {'Range': f'bytes={start}-{end - 1}'}
        resp = self._request('GET', url,
                             params={'alt': 'media', 'supportsAllDrives': 'true'},
                             headers=headers)
        return resp.content

    def _export_document(self, file_id: str, mime_type: str) -> bytes:
        if mime_type not in _GOOGLE_WORKSPACE_EXPORT_TYPES:
            raise NotImplementedError(f'GoogleDriveFS cannot export {mime_type} as text')
        return self._request(
            'GET', f'{self._base_url}/files/{file_id}/export',
            params={'mimeType': _GOOGLE_WORKSPACE_EXPORT_TYPES[mime_type]},
        ).content

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
    def _escape_query_literal(value: str) -> str:
        return value.replace('\\', '\\\\').replace("'", "\\'")

    @staticmethod
    def _normalize_keywords(keywords: Union[str, List[str]]) -> List[str]:
        values = [keywords] if isinstance(keywords, str) else keywords
        if not isinstance(values, (list, tuple)):
            raise ValueError('keywords must be a string or a list of strings')
        normalized = []
        for item in values:
            if not isinstance(item, str):
                raise ValueError('keywords must contain only strings')
            if value := item.strip():
                normalized.append(value)
        if not normalized:
            raise ValueError('at least one keyword is required')
        return normalized

    @staticmethod
    def _normalize_limit(value: int, default: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(1, min(parsed, maximum))

    def list_page(self, folder_id: str = '', query: str = '', query_mode: str = 'name',
                  drive_id: str = '', page_size: int = 100, page_token: str = '') -> Dict[str, Any]:
        if query_mode not in {'name', 'full_text'}:
            raise ValueError('query_mode must be name or full_text')
        if isinstance(page_size, bool) or not isinstance(page_size, int) or not 1 <= page_size <= 1000:
            raise ValueError('page_size must be between 1 and 1000')
        for value in (folder_id, drive_id):
            if value and not re.fullmatch(r'[A-Za-z0-9_-]+', value):
                raise ValueError('Invalid Google Drive folder or drive ID')
        terms = ['trashed = false']
        if folder_id:
            terms.append(f"'{self._escape_query_literal(folder_id)}' in parents")
        if query := query.strip():
            field = 'name' if query_mode == 'name' else 'fullText'
            terms.append(f"{field} contains '{self._escape_query_literal(query)}'")
        data = self._files_page(' and '.join(terms), drive_id, page_size, page_token)
        return {
            'items': [self._item_to_entry(item) for item in data.get('files', [])],
            'next_page_token': data.get('nextPageToken') or '',
            'incomplete_search': bool(data.get('incompleteSearch')),
        }

    def _files_page(self, query: str, drive_id: str, page_size: int, page_token: str) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            'q': query,
            'spaces': 'drive',
            'fields': _LIST_FIELDS,
            'pageSize': page_size,
            'orderBy': 'modifiedTime desc',
            'includeItemsFromAllDrives': 'true',
            'supportsAllDrives': 'true',
        }
        if drive_id:
            params.update({'corpora': 'drive', 'driveId': drive_id})
        if page_token:
            params['pageToken'] = page_token
        return self._get(f'{self._base_url}/files', params=params)

    def _iter_files(self, query: str, drive_id: str = '', max_items: int = 1000) -> Iterator[Dict[str, Any]]:
        seen = 0
        page_token = ''
        while seen < max_items:
            data = self._files_page(query, drive_id, min(max_items - seen, 1000), page_token)
            if data.get('incompleteSearch'):
                lazyllm.LOG.warning(
                    'Google Drive files.list returned incompleteSearch=true; '
                    'search results may be incomplete'
                )
            for item in data.get('files', []):
                yield item
                seen += 1
                if seen >= max_items:
                    return
            page_token = data.get('nextPageToken') or ''
            if not page_token:
                return

    @staticmethod
    def _item_to_entry(item: Dict[str, Any]) -> Dict[str, Any]:
        mime = item.get('mimeType') or ''
        file_id = item.get('id') or ''
        ftype = 'directory' if mime == 'application/vnd.google-apps.folder' else 'file'
        mtime = None
        ts = item.get('modifiedTime')
        if ts:
            try:
                mtime = datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
            except (ValueError, TypeError) as e:
                lazyllm.LOG.debug(f"Failed to parse timestamp '{ts}': {e}")
        return LazyLLMFSBase._entry(
            name=file_id,
            size=int(item.get('size', 0) or 0),
            ftype=ftype, mtime=mtime,
            title=item.get('name') or '', mime_type=mime,
            google_drive_path=f'googledrive:/{file_id}',
            web_url=item.get('webViewLink') or '',
            parents=item.get('parents') or [],
            drive_id=item.get('driveId') or '',
            description=item.get('description') or '',
        )
