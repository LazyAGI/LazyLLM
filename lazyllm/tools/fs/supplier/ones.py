# Copyright (c) 2026 LazyAGI. All rights reserved.
from typing import Any, Dict, List, Optional

from ..base import LazyLLMFSBase, CloudFSBufferedFile


_CLOUD_BASE = 'https://ones.ai/project/api/project'


class OnesFS(LazyLLMFSBase):

    protocol = 'ones'

    def __init__(self, token: str, user_id: Optional[str] = None,
                 base_url: Optional[str] = None, **storage_options):
        if ':' in token and not user_id:
            uid, tok = token.split(':', 1)
            self._user_id = uid
            token = tok
        else:
            self._user_id = user_id or ''
        super().__init__(token=token, base_url=base_url or _CLOUD_BASE, **storage_options)

    def _setup_auth(self) -> None:
        headers: Dict[str, str] = {
            'Ones-Auth-Token': self._token,
            'Content-Type': 'application/json',
        }
        if self._user_id:
            headers['Ones-User-Id'] = self._user_id
        self._session.headers.update(headers)

    def ls(self, path: str, detail: bool = True, **kwargs) -> List:
        parts = self._parse_path(path)
        if not parts:
            raise ValueError('path must start with /<team_uuid>')
        team_uuid = parts[0]
        if len(parts) == 1:
            return self._list_spaces(team_uuid, detail)
        space_uuid = parts[1]
        parent_uuid = parts[2] if len(parts) > 2 else None
        return self._list_pages(team_uuid, space_uuid, parent_uuid, detail)

    def info(self, path: str, **kwargs) -> Dict[str, Any]:
        parts = self._parse_path(path)
        if not parts:
            return self._entry('/', ftype='directory')
        team_uuid = parts[0]
        if len(parts) == 1:
            return self._entry(team_uuid, ftype='directory')
        if len(parts) == 2:
            space_uuid = parts[1]
            url = f'{self._base_url}/team/{team_uuid}/wiki/spaces/{space_uuid}'
            data = self._get(url)
            return self._space_to_entry(data.get('space', data))
        page_uuid = parts[2]
        url = f'{self._base_url}/team/{team_uuid}/wiki/spaces/{parts[1]}/pages/{page_uuid}'
        data = self._get(url)
        return self._page_to_entry(data.get('page', data))

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
        if len(parts) < 3:
            raise ValueError('path must be /<team_uuid>/<space_uuid>/<title>')
        team_uuid, space_uuid = parts[0], parts[1]
        title = parts[-1]
        parent_uuid = parts[2] if len(parts) >= 4 else None
        payload: Dict[str, Any] = {'title': title, 'content': ''}
        if parent_uuid:
            payload['parent_uuid'] = parent_uuid
        url = f'{self._base_url}/team/{team_uuid}/wiki/spaces/{space_uuid}/pages'
        self._post(url, json=payload)

    def rm_file(self, path: str) -> None:
        parts = self._parse_path(path)
        if len(parts) < 3:
            raise ValueError('path must be /<team_uuid>/<space_uuid>/<page_uuid>')
        team_uuid, space_uuid, page_uuid = parts[0], parts[1], parts[2]
        url = f'{self._base_url}/team/{team_uuid}/wiki/spaces/{space_uuid}/pages/{page_uuid}'
        self._delete(url)

    def _download_range(self, path: str, start: int, end: int) -> bytes:
        parts = self._parse_path(path)
        if len(parts) < 3:
            raise FileNotFoundError(path)
        team_uuid, space_uuid, page_uuid = parts[0], parts[1], parts[2]
        url = f'{self._base_url}/team/{team_uuid}/wiki/spaces/{space_uuid}/pages/{page_uuid}'
        data = self._get(url)
        page = data.get('page', data)
        content = page.get('content', '')
        encoded = content.encode('utf-8')
        return encoded[start:end]

    def _upload_data(self, path: str, data: bytes) -> None:
        parts = self._parse_path(path)
        if len(parts) < 3:
            raise ValueError('path must be /<team_uuid>/<space_uuid>/<page_uuid_or_title>')
        team_uuid, space_uuid, page_uuid = parts[0], parts[1], parts[2]
        content = data.decode('utf-8', errors='replace')
        url = f'{self._base_url}/team/{team_uuid}/wiki/spaces/{space_uuid}/pages/{page_uuid}'
        try:
            self._get(url)
            self._put(url, json={'content': content})
        except Exception:
            self._post(
                f'{self._base_url}/team/{team_uuid}/wiki/spaces/{space_uuid}/pages',
                json={'title': page_uuid, 'content': content},
            )

    def _platform_supports_webhook(self) -> bool:
        return False

    def _list_spaces(self, team_uuid: str, detail: bool) -> List:
        url = f'{self._base_url}/team/{team_uuid}/wiki/spaces'
        data = self._get(url)
        spaces = data.get('spaces', data.get('data', []))
        if detail:
            return [self._space_to_entry(s) for s in spaces]
        return [s.get('uuid', '') for s in spaces]

    def _list_pages(self, team_uuid: str, space_uuid: str,
                    parent_uuid: Optional[str], detail: bool) -> List:
        url = f'{self._base_url}/team/{team_uuid}/wiki/spaces/{space_uuid}/pages'
        params: Dict[str, Any] = {}
        if parent_uuid:
            params['parent_uuid'] = parent_uuid
        data = self._get(url, params=params)
        pages = data.get('pages', data.get('data', []))
        if detail:
            return [self._page_to_entry(p) for p in pages]
        return [p.get('uuid', '') for p in pages]

    @staticmethod
    def _space_to_entry(space: Dict[str, Any]) -> Dict[str, Any]:
        return LazyLLMFSBase._entry(
            name=space.get('uuid', ''), ftype='directory',
            title=space.get('title', space.get('name', '')),
        )

    @staticmethod
    def _page_to_entry(page: Dict[str, Any]) -> Dict[str, Any]:
        mtime = None
        ts = page.get('updated_time') or page.get('update_time')
        if ts:
            try:
                mtime = float(ts)
            except (TypeError, ValueError):
                pass
        return LazyLLMFSBase._entry(
            name=page.get('uuid', ''), ftype='file',
            mtime=mtime, title=page.get('title', ''),
        )
