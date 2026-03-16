# Copyright (c) 2026 LazyAGI. All rights reserved.
import base64
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import lazyllm

from ..base import LazyLLMFSBase, CloudFSBufferedFile


_CLOUD_BASE = 'https://api.atlassian.com/ex/confluence'


class ConfluenceFS(LazyLLMFSBase):
    def __init__(self, token: str, base_url: Optional[str] = None,
                 email: Optional[str] = None, cloud: bool = True,
                 cloud_id: Optional[str] = None, **storage_options):
        self._email = email
        self._cloud = cloud
        self._cloud_id = cloud_id
        if base_url:
            resolved_base = base_url.rstrip('/')
        elif cloud and cloud_id:
            resolved_base = f'{_CLOUD_BASE}/{cloud_id}'
        elif not cloud:
            resolved_base = ''
        else:
            resolved_base = _CLOUD_BASE
        super().__init__(token=token, base_url=resolved_base, **storage_options)

    def _setup_auth(self) -> None:
        if self._cloud and self._email:
            cred = base64.b64encode(
                f'{self._email}:{self._secret_key}'.encode()
            ).decode()
            self._session.headers.update({
                'Authorization': f'Basic {cred}',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            })
        else:
            self._session.headers.update({
                'Authorization': f'Bearer {self._secret_key}',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            })

    @property
    def _rest(self) -> str:
        return f'{self._base_url}/rest/api'

    def ls(self, path: str, detail: bool = True, **kwargs) -> List:
        parts = self._parse_path(path)
        if not parts:
            return self._list_spaces(detail)
        space_key = parts[0]
        if len(parts) == 1:
            return self._list_space_pages(space_key, detail)
        page_id = parts[-1]
        return self._list_child_pages(page_id, detail, space_key=space_key)

    def info(self, path: str, **kwargs) -> Dict[str, Any]:
        parts = self._parse_path(path)
        if not parts:
            return self._entry('/', ftype='directory')
        if len(parts) == 1:
            url = f'{self._rest}/space/{parts[0]}'
            data = self._get(url)
            return self._entry(
                name=path, ftype='directory',
                title=data.get('name', ''), key=data.get('key', ''),
            )
        page_id = parts[-1]
        url = f'{self._rest}/content/{page_id}'
        data = self._get(url, params={'expand': 'version,space,body.storage'})
        return self._page_to_entry(data)

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
        if len(parts) < 2:
            raise ValueError('path must be /<space_key>/<title> or /<space_key>/<parent_id>/<title>')
        space_key = parts[0]
        title = parts[-1]
        ancestor_id = parts[-2] if len(parts) >= 3 else None
        payload: Dict[str, Any] = {
            'type': 'page',
            'title': title,
            'space': {'key': space_key},
            'body': {'storage': {'value': '', 'representation': 'storage'}},
        }
        if ancestor_id:
            payload['ancestors'] = [{'id': ancestor_id}]
        self._post(f'{self._rest}/content', json=payload)

    def rm_file(self, path: str) -> None:
        parts = self._parse_path(path)
        if not parts:
            raise FileNotFoundError(path)
        page_id = parts[-1]
        self._delete(f'{self._rest}/content/{page_id}')

    def rmdir(self, path: str) -> None:
        parts = self._parse_path(path)
        if not parts:
            return
        if len(parts) == 1:
            self._delete(f'{self._rest}/space/{parts[0]}')
        else:
            self._delete(f'{self._rest}/content/{parts[-1]}')

    def _download_range(self, path: str, start: int, end: int) -> bytes:
        parts = self._parse_path(path)
        page_id = parts[-1] if parts else path
        url = f'{self._rest}/content/{page_id}'
        data = self._get(url, params={'expand': 'body.storage'})
        body = data.get('body', {}).get('storage', {}).get('value', '')
        encoded = body.encode('utf-8')
        return encoded[start:end]

    def _upload_data(self, path: str, data: bytes) -> None:
        parts = self._parse_path(path)
        if len(parts) < 2:
            raise ValueError('path must include space_key and page title/id')
        space_key = parts[0]
        page_id_or_title = parts[-1]
        title = page_id_or_title
        try:
            url = f'{self._rest}/content/{page_id_or_title}'
            existing = self._get(url, params={'expand': 'version'})
            version = existing.get('version', {}).get('number', 0) + 1
            self._put(url, json={
                'version': {'number': version},
                'title': existing.get('title', title),
                'type': 'page',
                'body': {'storage': {
                    'value': data.decode('utf-8', errors='replace'),
                    'representation': 'storage',
                }},
            })
        except Exception:
            payload: Dict[str, Any] = {
                'type': 'page',
                'title': title,
                'space': {'key': space_key},
                'body': {'storage': {
                    'value': data.decode('utf-8', errors='replace'),
                    'representation': 'storage',
                }},
            }
            if len(parts) >= 3:
                payload['ancestors'] = [{'id': parts[-2]}]
            self._post(f'{self._rest}/content', json=payload)

    def _platform_supports_webhook(self) -> bool:
        return True

    def _register_webhook(self, webhook_url: str, events: List[str], path: str) -> Dict[str, Any]:
        url = f'{self._rest}/webhooks'
        payload = {
            'name': f'lazyllm-fs-{int(time.time())}',
            'url': webhook_url,
            'events': events or ['page_created', 'page_updated', 'page_deleted'],
            'active': True,
        }
        return self._post(url, json=payload)

    def _list_spaces(self, detail: bool) -> List:
        url = f'{self._rest}/space'
        data = self._get(url, params={'limit': 200})
        results = data.get('results', [])
        if detail:
            return [
                self._entry(r.get('key', ''), ftype='directory', title=r.get('name', ''))
                for r in results
            ]
        return [r.get('key', '') for r in results]

    def _list_space_pages(self, space_key: str, detail: bool) -> List:
        url = f'{self._rest}/space/{space_key}/content'
        data = self._get(url, params={'limit': 200, 'expand': 'version'})
        pages = data.get('page', {}).get('results', []) or []
        if detail:
            entries = [self._page_to_entry(p) for p in pages]
            for e in entries:
                e['name'] = f'{space_key}/{e["name"]}'
            return entries
        return [f'{space_key}/{p.get("id", "")}' for p in pages]

    def _list_child_pages(self, page_id: str, detail: bool, space_key: str = '') -> List:
        url = f'{self._rest}/content/{page_id}/child/page'
        data = self._get(url, params={'limit': 200, 'expand': 'version'})
        pages = data.get('results', [])
        if detail:
            entries = [self._page_to_entry(p) for p in pages]
            for e in entries:
                e['name'] = f'{space_key}/{e["name"]}' if space_key else e['name']
            return entries
        return [
            f'{space_key}/{p.get("id", "")}' if space_key else p.get('id', '')
            for p in pages
        ]

    @staticmethod
    def _page_to_entry(page: Dict[str, Any]) -> Dict[str, Any]:
        ver = page.get('version', {})
        mtime_str = ver.get('when')
        mtime = None
        if mtime_str:
            try:
                mtime = datetime.fromisoformat(
                    mtime_str.replace('Z', '+00:00')).timestamp()
            except (ValueError, TypeError) as e:
                lazyllm.LOG.debug(f"Failed to parse timestamp '{mtime_str}': {e}")
        return LazyLLMFSBase._entry(
            name=page.get('id', ''), ftype='file', mtime=mtime,
            title=page.get('title', ''),
            space=page.get('space', {}).get('key', ''),
            version=ver.get('number'),
        )
