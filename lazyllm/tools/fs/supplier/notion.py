# Copyright (c) 2026 LazyAGI. All rights reserved.
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import lazyllm
from lazyllm import config

from ..base import LazyLLMFSBase, CloudFSBufferedFile

config.add('notion_token', str, None, 'NOTION_TOKEN', description='Notion API token (notion-client official env).')

_API_BASE = 'https://api.notion.com/v1'
_NOTION_VERSION = '2022-06-28'


class NotionFS(LazyLLMFSBase):

    def __init__(self, token: Optional[str] = None, base_url: Optional[str] = None,
                 dynamic_auth: bool = False, **storage_options):
        if dynamic_auth:
            token = ''
        else:
            token = (token or config['notion_token'] or os.environ.get('NOTION_TOKEN')
                     or os.environ.get('NOTION_API_KEY') or '')
        super().__init__(token=token, base_url=base_url or _API_BASE, dynamic_auth=dynamic_auth, **storage_options)

    def _setup_auth(self) -> None:
        self._session.headers.update({
            'Notion-Version': _NOTION_VERSION,
            'Content-Type': 'application/json',
        })
        self._access_token = self._secret_key

    def ls(self, path: str, detail: bool = True, **kwargs) -> List:
        parts = self._parse_path(path)
        if not parts:
            return self._search_all(detail)
        block_id = parts[-1]
        return self._list_children(block_id, detail)

    def info(self, path: str, **kwargs) -> Dict[str, Any]:
        parts = self._parse_path(path)
        if not parts:
            return self._entry('/', ftype='directory')
        block_id = parts[-1]
        try:
            url = f'{self._base_url}/pages/{block_id}'
            data = self._get(url)
            return self._page_to_entry(data)
        except Exception:
            pass
        try:
            url = f'{self._base_url}/databases/{block_id}'
            data = self._get(url)
            return self._db_to_entry(data)
        except Exception:
            return self._entry(name=path, ftype='file')

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
            raise ValueError('path must be /<parent_page_id>/<title>')
        parent_id = parts[0]
        title = parts[-1]
        payload: Dict[str, Any] = {
            'parent': {'page_id': parent_id},
            'properties': {
                'title': {'title': [{'text': {'content': title}}]}
            },
        }
        self._post(f'{self._base_url}/pages', json=payload)

    def rm_file(self, path: str) -> None:
        parts = self._parse_path(path)
        if not parts:
            raise FileNotFoundError(path)
        page_id = parts[-1]
        self._patch(f'{self._base_url}/pages/{page_id}', json={'archived': True})

    def rmdir(self, path: str) -> None:
        parts = self._parse_path(path)
        if not parts:
            return
        block_id = parts[-1]
        try:
            self._patch(f'{self._base_url}/databases/{block_id}', json={'archived': True})
        except Exception:
            self._patch(f'{self._base_url}/pages/{block_id}', json={'archived': True})

    def copy(self, path1: str, path2: str, recursive: bool = False, **kwargs) -> None:
        raise NotImplementedError('NotionFS: Notion official API does not support copy')

    def move(self, path1: str, path2: str, recursive: bool = False, **kwargs) -> None:
        raise NotImplementedError('NotionFS: Notion official API does not support move')

    def _download_range(self, path: str, start: int, end: int) -> bytes:
        parts = self._parse_path(path)
        block_id = parts[-1] if parts else path
        text = self._extract_page_text(block_id)
        encoded = text.encode('utf-8')
        return encoded[start:end]

    def _upload_data(self, path: str, data: bytes) -> None:
        parts = self._parse_path(path)
        if not parts:
            raise ValueError('path must include a page_id')
        page_id = parts[-1]
        text = data.decode('utf-8', errors='replace')
        payload = {
            'children': [{
                'object': 'block',
                'type': 'paragraph',
                'paragraph': {
                    'rich_text': [{'type': 'text', 'text': {'content': text[:2000]}}]
                }
            }]
        }
        self._patch(f'{self._base_url}/blocks/{page_id}/children', json=payload)

    def _platform_supports_webhook(self) -> bool:
        return False

    def _search_all(self, detail: bool) -> List:
        url = f'{self._base_url}/search'
        data = self._post(url, json={'page_size': 100})
        results = data.get('results', [])
        if detail:
            return [self._object_to_entry(r) for r in results]
        return [r.get('id', '') for r in results]

    def _list_children(self, block_id: str, detail: bool) -> List:
        url = f'{self._base_url}/blocks/{block_id}/children'
        data = self._get(url, params={'page_size': 100})
        results = data.get('results', [])
        if detail:
            return [self._block_to_entry(r) for r in results]
        return [r.get('id', '') for r in results]

    def _extract_page_text(self, block_id: str) -> str:
        url = f'{self._base_url}/blocks/{block_id}/children'
        data = self._get(url, params={'page_size': 100})
        lines = []
        for block in data.get('results', []):
            btype = block.get('type', '')
            content = block.get(btype, {})
            rich = content.get('rich_text', [])
            text = ''.join(t.get('plain_text', '') for t in rich)
            if text:
                lines.append(text)
        return '\n'.join(lines)

    @staticmethod
    def _page_to_entry(page: Dict[str, Any]) -> Dict[str, Any]:
        pid = page.get('id', '')
        title = ''
        props = page.get('properties', {})
        for key in ('title', 'Title', 'Name'):
            if key in props:
                rich = props[key].get('title', [])
                title = ''.join(t.get('plain_text', '') for t in rich)
                break
        mtime = None
        ts = page.get('last_edited_time')
        if ts:
            try:
                mtime = datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
            except (ValueError, TypeError) as e:
                lazyllm.LOG.debug(f"Failed to parse timestamp '{ts}': {e}")
        return LazyLLMFSBase._entry(name=pid, ftype='file', mtime=mtime, title=title)

    @staticmethod
    def _db_to_entry(db: Dict[str, Any]) -> Dict[str, Any]:
        did = db.get('id', '')
        titles = db.get('title', [])
        title = ''.join(t.get('plain_text', '') for t in titles)
        return LazyLLMFSBase._entry(name=did, ftype='directory', title=title)

    @staticmethod
    def _object_to_entry(obj: Dict[str, Any]) -> Dict[str, Any]:
        if obj.get('object') == 'database':
            return NotionFS._db_to_entry(obj)
        return NotionFS._page_to_entry(obj)

    @staticmethod
    def _block_to_entry(block: Dict[str, Any]) -> Dict[str, Any]:
        bid = block.get('id', '')
        btype = block.get('type', 'paragraph')
        has_children = block.get('has_children', False)
        return LazyLLMFSBase._entry(
            name=bid,
            ftype='directory' if has_children else 'file',
            block_type=btype,
        )
