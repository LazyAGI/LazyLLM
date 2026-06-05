# Copyright (c) 2026 LazyAGI. All rights reserved.
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import lazyllm
from lazyllm import config

from ..base import LazyLLMFSBase, LinkDocumentFSBase, CloudFSBufferedFile

config.add('notion_token', str, None, 'NOTION_TOKEN', description='Notion API token (notion-client official env).')

_API_BASE = 'https://api.notion.com/v1'
_NOTION_VERSION = '2022-06-28'
_NOTION_MARKDOWN_VERSION = '2026-03-11'
_PAGE_SIZE = 100
_MAX_RECURSION_DEPTH = 3

_NOTION_HOST_RE = re.compile(r'(^|\.)notion\.(so|site)$', re.IGNORECASE)
_UUID_RE = re.compile(
    r'(?<![0-9a-fA-F])('
    r'[0-9a-fA-F]{32}|'
    r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-'
    r'[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
    r')(?![0-9a-fA-F])'
)


def _normalize_notion_id(value: str) -> str:
    raw = (value or '').strip().strip('/').replace('-', '')
    if not re.fullmatch(r'[0-9a-fA-F]{32}', raw):
        raise ValueError(f'Invalid Notion object id: {value!r}')
    raw = raw.lower()
    return f'{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}'


def _find_notion_ids(text: str) -> List[str]:
    ids: List[str] = []
    for match in _UUID_RE.finditer(text or ''):
        ids.append(_normalize_notion_id(match.group(1)))
    return ids


def _parse_notion_browser_url(url: str) -> Optional[Dict[str, str]]:
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        return None
    host = (parsed.hostname or '').lower()
    if not _NOTION_HOST_RE.search(host):
        return None

    path_ids = _find_notion_ids(unquote(parsed.path or ''))
    query = parse_qs(parsed.query or '')
    query_ids: List[str] = []
    for key in ('p', 'page', 'page_id', 'pageId', 'database_id', 'databaseId', 'id'):
        for value in query.get(key, []):
            query_ids.extend(_find_notion_ids(value))
    fragment_ids = _find_notion_ids(unquote(parsed.fragment or ''))

    object_id = (path_ids or query_ids or fragment_ids or [''])[-1]
    if not object_id:
        return None
    result = {'kind': 'object', 'id': object_id}
    if fragment_ids and fragment_ids[-1] != object_id:
        result['block_id'] = fragment_ids[-1]
    return result


def _is_notion_browser_url(path: str) -> bool:
    return _parse_notion_browser_url(path) is not None


def _strip_notion_protocol(path: str) -> str:
    if path.startswith('notion:/'):
        return path[len('notion:'):]
    return path


def _dedupe_refs(refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return LinkDocumentFSBase.dedupe_document_references(refs)


def _format_references_footer(refs: List[Dict[str, Any]]) -> str:
    return LinkDocumentFSBase.format_document_references_footer(refs, 'notion')


class NotionFile(CloudFSBufferedFile):

    def __init__(self, fs: 'NotionFS', path: str, include_references: bool = False, **kwargs) -> None:
        content = fs._fetch_content(path, include_references=include_references)
        self._notion_content: bytes = content
        super().__init__(fs, path, size=len(content), **kwargs)

    def _fetch_range(self, start: int, end: int) -> bytes:
        return self._notion_content[start:end]


class NotionFS(LinkDocumentFSBase):
    document_provider = 'notion'
    __public_apis__ = ['ls', 'info', 'mkdir', 'rm', 'exists',
                       'read', 'read_file', 'search', 'write', 'move',
                       *LinkDocumentFSBase.__document_public_apis__]

    def __init__(self, token: Optional[str] = None, base_url: Optional[str] = None,
                 dynamic_auth: bool = False, **storage_options):
        if dynamic_auth:
            if token:
                raise ValueError('token must be None when dynamic_auth=True')
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

    def ls(self, path: str, detail: bool = True, **kwargs) -> List:
        kind, object_id = self._resolve_ref(path)
        if kind == 'root':
            return self._search_all(detail)
        if kind == 'database':
            entries = [self._page_to_entry(p) for p in self._query_database(object_id)]
            return entries if detail else [e['name'] for e in entries]
        if kind == 'block':
            entries = [self._block_to_entry(b) for b in self._list_children_raw(object_id)]
            return entries if detail else [e['name'] for e in entries]

        try:
            self._retrieve_page(object_id)
            entries = [self._block_to_entry(b) for b in self._list_children_raw(object_id)]
        except Exception:
            entries = [self._page_to_entry(p) for p in self._query_database(object_id)]
        return entries if detail else [e['name'] for e in entries]

    def info(self, path: str, **kwargs) -> Dict[str, Any]:
        kind, object_id = self._resolve_ref(path)
        if kind == 'root':
            return self._entry('/', ftype='directory')
        if kind == 'database':
            return self._db_to_entry(self._retrieve_database(object_id))
        if kind == 'block':
            return self._block_to_entry(self._retrieve_block(object_id))
        if kind == 'page':
            return self._page_to_entry(self._retrieve_page(object_id))

        try:
            return self._page_to_entry(self._retrieve_page(object_id))
        except Exception as page_exc:
            try:
                return self._db_to_entry(self._retrieve_database(object_id))
            except Exception:
                raise page_exc

    def _open(self, path: str, mode: str = 'rb',
              block_size: Optional[int] = None,
              autocommit: bool = True,
              cache_options: Optional[Dict] = None,
              include_references: bool = False,
              **kwargs) -> CloudFSBufferedFile:
        if 'b' not in mode:
            raise ValueError('NotionFS only supports binary mode')
        if 'r' in mode:
            return NotionFile(
                self, path, include_references=include_references,
                mode=mode, block_size=block_size or self.blocksize,
                autocommit=autocommit, cache_options=cache_options,
            )
        return CloudFSBufferedFile(
            self, path, mode=mode,
            block_size=block_size or self.blocksize,
            autocommit=autocommit, cache_options=cache_options,
        )

    def read_bytes(self, path: str, include_references: bool = False) -> bytes:
        return self._fetch_content(path, include_references=include_references)

    def cat_file(self, path: str, start: Optional[int] = None, end: Optional[int] = None,
                 include_references: bool = False, **kwargs) -> bytes:
        data = self._fetch_content(path, include_references=include_references)
        return data[start:end] if (start is not None or end is not None) else data

    def fetch_url(self, url: str) -> bytes:
        parsed = _parse_notion_browser_url(url)
        if not parsed:
            raise ValueError(f'Cannot parse Notion browser URL: {url!r}')
        return self._fetch_content(f'/{parsed["id"]}')

    def search(self, query: str, object_type: str = '', limit: int = 20,
               sort_direction: str = 'descending') -> List[Dict[str, Any]]:
        """Search Notion pages/databases by title.

        Args:
            query (str): Title keyword to search in Notion.
            object_type (str): Optional object filter: ``page`` or ``database``.
            limit (int): Maximum number of results to return.
            sort_direction (str): Sort by last edited time, ``ascending`` or ``descending``.

        Returns:
            List[Dict[str, Any]]: Matching Notion objects with ``title``, ``id``, and ``notion_path``.
        """
        query = (query or '').strip()
        if not query:
            raise ValueError('query is required')
        object_type = (object_type or '').strip().lower()
        if object_type and object_type not in {'page', 'database'}:
            raise ValueError('object_type must be page or database')
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 20
        limit = max(1, min(limit, _PAGE_SIZE))
        sort_direction = (sort_direction or 'descending').strip().lower()
        if sort_direction not in {'ascending', 'descending'}:
            sort_direction = 'descending'

        payload: Dict[str, Any] = {
            'query': query,
            'page_size': limit,
            'sort': {'direction': sort_direction, 'timestamp': 'last_edited_time'},
        }
        if object_type:
            payload['filter'] = {'property': 'object', 'value': object_type}
        results = self._paginate_post(f'{self._base_url}/search', payload)
        return [self._object_to_entry(item) for item in results[:limit]]

    def mkdir(self, path: str, create_parents: bool = True, **kwargs) -> None:
        parent_id, title = self._parse_parent_title_path(path)
        if not parent_id or not title:
            raise ValueError('path must be /<parent_page_id>/<title>')
        payload: Dict[str, Any] = {
            'parent': {'page_id': parent_id},
            'properties': {
                'title': {'title': [{'text': {'content': title}}]}
            },
        }
        self._post(f'{self._base_url}/pages', json=payload)

    def rm_file(self, path: str) -> None:
        kind, object_id = self._resolve_ref(path)
        if kind in ('root', 'database'):
            raise FileNotFoundError(path)
        self._patch(f'{self._base_url}/pages/{object_id}', json={'archived': True})

    def rmdir(self, path: str) -> None:
        kind, object_id = self._resolve_ref(path)
        if kind == 'root':
            return
        try:
            self._patch(f'{self._base_url}/databases/{object_id}', json={'archived': True})
        except Exception:
            self._patch(f'{self._base_url}/pages/{object_id}', json={'archived': True})

    def copy(self, path1: str, path2: str, recursive: bool = False, **kwargs) -> None:
        raise NotImplementedError('NotionFS: Notion official API does not support copy')

    def move(self, path1: str, path2: str, recursive: bool = False, **kwargs) -> None:
        src_kind, page_id = self._resolve_ref(path1)
        if src_kind not in ('page', 'object'):
            raise NotImplementedError('NotionFS.move only supports moving pages')
        parent_id, new_title = self._parse_move_destination(path2)
        self._post(
            f'{self._base_url}/pages/{page_id}/move',
            json={'parent': {'type': 'page_id', 'page_id': parent_id}},
            headers={'Notion-Version': _NOTION_MARKDOWN_VERSION},
        )
        if new_title:
            self.update_page_title(page_id, new_title)

    def _download_range(self, path: str, start: int, end: int) -> bytes:
        return self._fetch_content(path)[start:end]

    def _upload_data(self, path: str, data: bytes, **kwargs) -> None:
        kind, object_id = self._resolve_ref(path)
        if kind in ('root', 'database'):
            raise ValueError('path must include a page_id or block_id')
        text = data.decode('utf-8', errors='replace')
        content_type = kwargs.get('content_type')
        if kind in ('page', 'object') and content_type in ('markdown', 'md'):
            self.replace_page_markdown(object_id, text)
            return
        blocks = self._text_to_paragraph_blocks(text)
        for i in range(0, len(blocks), _PAGE_SIZE):
            self._patch(f'{self._base_url}/blocks/{object_id}/children',
                        json={'children': blocks[i:i + _PAGE_SIZE]})

    def _platform_supports_webhook(self) -> bool:
        return False

    def _resolve_ref(self, path: str) -> Tuple[str, str]:
        path = _strip_notion_protocol(path)
        if not path or path == '/':
            return 'root', ''
        if _is_notion_browser_url(path):
            parsed = _parse_notion_browser_url(path)
            assert parsed is not None
            return 'object', parsed['id']

        norm = path.lstrip('/')
        if self.is_link_path(norm):
            url = self.decode_link_path(norm)
            parsed = _parse_notion_browser_url(url)
            if not parsed:
                raise ValueError(f'Cannot parse Notion browser URL: {url!r}')
            return 'object', parsed['id']
        for prefix, kind in (
            ('~page/', 'page'),
            ('~database/', 'database'),
            ('~block/', 'block'),
        ):
            if norm.startswith(prefix):
                token = norm[len(prefix):].rstrip('/').split('/')[0]
                return kind, _normalize_notion_id(token)

        parts = self._parse_path(path)
        if not parts:
            return 'root', ''
        return 'object', _normalize_notion_id(parts[-1])

    def _parse_parent_title_path(self, path: str) -> Tuple[str, str]:
        path = _strip_notion_protocol(path)
        if _is_notion_browser_url(path):
            parsed = _parse_notion_browser_url(path)
            return (parsed['id'], '') if parsed else ('', '')

        norm = path.strip('/')
        if norm.startswith(('~page/', '~database/', '~block/')):
            _, rest = norm.split('/', 1)
            parts = [p for p in rest.split('/') if p]
        else:
            parts = [p for p in norm.split('/') if p]
        if not parts:
            return '', ''
        parent_id = _normalize_notion_id(unquote(parts[0]))
        title = unquote('/'.join(parts[1:])) if len(parts) > 1 else ''
        return parent_id, title

    def _parse_move_destination(self, path: str) -> Tuple[str, str]:
        parent_id, title = self._parse_parent_title_path(path)
        if not parent_id:
            raise ValueError('move destination must include a parent page id')
        return parent_id, title

    def _search_all(self, detail: bool) -> List:
        results = self._paginate_post(f'{self._base_url}/search', {'page_size': _PAGE_SIZE})
        if detail:
            return [self._object_to_entry(r) for r in results]
        return [r.get('id', '') for r in results]

    def _list_children_raw(self, block_id: str) -> List[Dict[str, Any]]:
        return self._paginate_get(f'{self._base_url}/blocks/{block_id}/children', {'page_size': _PAGE_SIZE})

    def _query_database(self, database_id: str) -> List[Dict[str, Any]]:
        return self._paginate_post(f'{self._base_url}/databases/{database_id}/query', {'page_size': _PAGE_SIZE})

    def _retrieve_page(self, page_id: str) -> Dict[str, Any]:
        return self._get(f'{self._base_url}/pages/{page_id}')

    def _retrieve_database(self, database_id: str) -> Dict[str, Any]:
        return self._get(f'{self._base_url}/databases/{database_id}')

    def _retrieve_block(self, block_id: str) -> Dict[str, Any]:
        return self._get(f'{self._base_url}/blocks/{block_id}')

    def _retrieve_page_markdown(self, page_id: str) -> Optional[str]:
        try:
            data = self._get(
                f'{self._base_url}/pages/{page_id}/markdown',
                headers={'Notion-Version': _NOTION_MARKDOWN_VERSION},
            )
        except Exception as exc:
            lazyllm.LOG.debug(f'Notion markdown endpoint unavailable for {page_id}: {exc}')
            return None
        markdown = data.get('markdown')
        return markdown if isinstance(markdown, str) else None

    def replace_page_markdown(self, page_id: str, markdown: str,
                              allow_deleting_content: bool = False) -> Dict[str, Any]:
        return self._patch(
            f'{self._base_url}/pages/{_normalize_notion_id(page_id)}/markdown',
            json={
                'type': 'replace_content',
                'replace_content': {
                    'new_str': markdown,
                    'allow_deleting_content': allow_deleting_content,
                },
            },
            headers={'Notion-Version': _NOTION_MARKDOWN_VERSION},
        )

    def insert_page_markdown(self, page_id: str, markdown: str,
                             position: str = 'end') -> Dict[str, Any]:
        return self._patch(
            f'{self._base_url}/pages/{_normalize_notion_id(page_id)}/markdown',
            json={
                'type': 'insert_content',
                'insert_content': {
                    'new_str': markdown,
                    'position': {'type': position},
                },
            },
            headers={'Notion-Version': _NOTION_MARKDOWN_VERSION},
        )

    def update_page_title(self, page_id: str, title: str) -> None:
        page_id = _normalize_notion_id(page_id)
        page = self._retrieve_page(page_id)
        title_key = self._title_property_key(page)
        self._patch(
            f'{self._base_url}/pages/{page_id}',
            json={'properties': {title_key: {'title': self._text_to_rich_text(title)}}},
        )

    def resolve_notion_ref(self, url_or_path: str) -> Dict[str, Any]:
        kind, object_id = self._resolve_ref(url_or_path)
        if kind == 'root':
            return {'object_id': '', 'object_type': 'root', 'title': 'Notion'}
        if kind == 'database':
            entry = self._db_to_entry(self._retrieve_database(object_id))
        elif kind == 'block':
            entry = self._block_to_entry(self._retrieve_block(object_id))
        else:
            try:
                entry = self._page_to_entry(self._retrieve_page(object_id))
            except Exception:
                entry = self._db_to_entry(self._retrieve_database(object_id))
        return {
            'object_id': entry.get('id', object_id),
            'object_type': entry.get('object') or entry.get('block_type') or kind,
            'title': entry.get('title') or entry.get('name') or '',
            'notion_path': entry.get('notion_path') or f'notion:/~{kind}/{object_id}',
            'has_child': entry.get('type') == 'directory',
        }

    def _resolve_document_ref(self, url_or_path: str) -> Dict[str, Any]:
        return self.resolve_notion_ref(url_or_path)

    def get_document_id(self, path: str) -> str:
        kind, object_id = self._resolve_ref(path)
        if kind == 'root' or not object_id:
            raise FileNotFoundError(f'Path not found: {path}')
        return object_id

    def get_doc_blocks(self, path: str, with_descendants: bool = True) -> List[Dict[str, Any]]:
        kind, object_id = self._resolve_ref(path)
        if kind == 'root' or not object_id:
            return []
        blocks: List[Dict[str, Any]]
        if kind == 'database':
            blocks = []
            for page in self._query_database(object_id):
                page_entry = self._page_to_entry(page)
                blocks.append({
                    'block_id': page.get('id', ''),
                    'block_type': 'child_page',
                    'parent_id': object_id,
                    'plain_text': page_entry.get('title', ''),
                    'has_children': True,
                })
                if with_descendants and page.get('id'):
                    blocks.extend(self._get_doc_blocks_raw(page['id'], with_descendants=True))
        else:
            blocks = self._get_doc_blocks_raw(object_id, with_descendants=with_descendants)
        return [self._block_summary(block) for block in blocks]

    def update_doc_block_text(self, path: str, block_id: str, new_text: str) -> None:
        self.get_document_id(path)
        block_id = _normalize_notion_id(block_id)
        block = self._retrieve_block(block_id)
        btype = block.get('type', '')
        content = dict(block.get(btype) or {})
        if btype not in {
            'paragraph', 'heading_1', 'heading_2', 'heading_3',
            'bulleted_list_item', 'numbered_list_item', 'to_do',
            'toggle', 'quote', 'callout', 'code',
        }:
            raise NotImplementedError(f'NotionFS.update_doc_block_text does not support block type {btype!r}')
        content['rich_text'] = self._text_to_rich_text(new_text)
        if btype == 'to_do':
            content['checked'] = bool((block.get('to_do') or {}).get('checked'))
        if btype == 'code':
            content['language'] = (block.get('code') or {}).get('language') or 'plain text'
        self._patch(f'{self._base_url}/blocks/{block_id}', json={btype: content})

    def _get_doc_blocks_raw(self, block_id: str, with_descendants: bool = True,
                            depth: int = 0, visited: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        visited = visited or set()
        block_id = _normalize_notion_id(block_id)
        if block_id in visited or depth > _MAX_RECURSION_DEPTH:
            return []
        visited.add(block_id)
        blocks = self._list_children_raw(block_id)
        if not with_descendants:
            return blocks
        out: List[Dict[str, Any]] = []
        for block in blocks:
            out.append(block)
            child_id = block.get('id', '')
            if block.get('has_children') and child_id:
                out.extend(self._get_doc_blocks_raw(child_id, True, depth + 1, visited))
        return out

    def _list_document_references(self, path: str) -> List[Dict[str, Any]]:
        kind, object_id = self._resolve_ref(path)
        if kind == 'root' or not object_id:
            return []
        refs: List[Dict[str, Any]] = []
        try:
            if kind == 'database':
                for page in self._query_database(object_id):
                    refs.extend(self._refs_from_page_properties(page))
                    page_id = page.get('id')
                    if page_id:
                        refs.extend(self._refs_from_blocks(self._get_doc_blocks_raw(page_id, True)))
            else:
                if kind in ('page', 'object'):
                    try:
                        refs.extend(self._refs_from_page_properties(self._retrieve_page(object_id)))
                    except Exception as exc:
                        lazyllm.LOG.debug(f'Failed to get Notion page properties for references: {exc}')
                refs.extend(self._refs_from_blocks(self._get_doc_blocks_raw(object_id, True)))
        except Exception as exc:
            lazyllm.LOG.warning(f'_list_document_references: failed to get blocks for {path!r}: {exc}')
            return []
        return _dedupe_refs(refs)

    def _paginate_get(self, url: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        params = dict(params or {})
        results: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        while True:
            page_params = dict(params)
            if cursor:
                page_params['start_cursor'] = cursor
            data = self._get(url, params=page_params)
            results.extend(data.get('results') or [])
            cursor = data.get('next_cursor') if data.get('has_more') else None
            if not cursor:
                break
        return results

    def _paginate_post(self, url: str, payload: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        base_payload = dict(payload or {})
        results: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        while True:
            page_payload = dict(base_payload)
            if cursor:
                page_payload['start_cursor'] = cursor
            data = self._post(url, json=page_payload)
            results.extend(data.get('results') or [])
            cursor = data.get('next_cursor') if data.get('has_more') else None
            if not cursor:
                break
        return results

    def _fetch_content(self, path: str, include_references: bool = False) -> bytes:
        kind, object_id = self._resolve_ref(path)
        if kind == 'root':
            text = self._search_to_markdown()
        elif kind == 'database':
            text = self._database_to_markdown(object_id)
        elif kind == 'block':
            block = self._retrieve_block(object_id)
            text = '\n'.join(self._block_to_markdown(block, depth=0, visited=set()))
        elif kind == 'page':
            text = self._page_to_markdown(object_id)
        else:
            try:
                text = self._page_to_markdown(object_id)
            except Exception:
                text = self._database_to_markdown(object_id)
        if include_references:
            text = self._append_document_references_footer(text, path)
        return text.encode('utf-8')

    def _search_to_markdown(self) -> str:
        entries = self._search_all(detail=True)
        lines = ['# Notion']
        for entry in entries:
            title = entry.get('title') or entry.get('name') or entry.get('id') or ''
            object_id = entry.get('id') or entry.get('name') or ''
            object_type = entry.get('object') or entry.get('block_type') or entry.get('type') or ''
            if title or object_id:
                lines.append(f'- {title} ({object_type}: {object_id})')
        return '\n'.join(lines)

    def _page_to_markdown(self, page_id: str, heading_level: int = 1,
                          depth: int = 0, visited: Optional[Set[str]] = None,
                          include_title: bool = True) -> str:
        visited = visited or set()
        page_id = _normalize_notion_id(page_id)
        if page_id in visited:
            return ''
        visited.add(page_id)

        page = self._retrieve_page(page_id)
        title = self._page_title(page) or page_id
        heading = '#' * max(1, min(6, heading_level))
        lines = [f'{heading} {title}'] if include_title else []
        markdown = self._retrieve_page_markdown(page_id)
        if markdown:
            lines.append(markdown)
        else:
            children = self._list_children_raw(page_id)
            lines.extend(self._blocks_to_markdown(children, depth=depth, visited=visited))
        return self._join_markdown(lines)

    def _database_to_markdown(self, database_id: str, heading_level: int = 1,
                              depth: int = 0, visited: Optional[Set[str]] = None,
                              include_title: bool = True) -> str:
        visited = visited or set()
        database_id = _normalize_notion_id(database_id)
        if database_id in visited:
            return ''
        visited.add(database_id)

        db = self._retrieve_database(database_id)
        title = self._database_title(db) or database_id
        heading = '#' * max(1, min(6, heading_level))
        lines = [f'{heading} {title}'] if include_title else []
        pages = self._query_database(database_id)
        for page in pages:
            page_id = page.get('id', '')
            page_title = self._page_title(page) or page_id
            child_heading = '#' * max(1, min(6, heading_level + 1))
            lines.append(f'{child_heading} {page_title}')
            if page_id and depth < _MAX_RECURSION_DEPTH:
                try:
                    body = self._page_to_markdown(page_id, heading_level + 2, depth + 1, visited,
                                                  include_title=False)
                    if body:
                        lines.append(body)
                except Exception as exc:
                    lazyllm.LOG.debug(f'Failed to fetch Notion database child page {page_id}: {exc}')
        return self._join_markdown(lines)

    def _blocks_to_markdown(self, blocks: List[Dict[str, Any]],
                            depth: int, visited: Set[str]) -> List[str]:
        lines: List[str] = []
        table_rows: List[str] = []
        for block in blocks:
            if block.get('type') == 'table_row':
                table_rows.append(self._table_row_to_markdown(block))
                continue
            if table_rows:
                lines.extend(self._flush_table_rows(table_rows))
                table_rows = []
            lines.extend(self._block_to_markdown(block, depth=depth, visited=visited))
        if table_rows:
            lines.extend(self._flush_table_rows(table_rows))
        return lines

    def _block_to_markdown(self, block: Dict[str, Any], depth: int, visited: Set[str]) -> List[str]:  # noqa C901
        block_id = block.get('id', '')
        btype = block.get('type', '')
        content = block.get(btype) or {}
        lines: List[str] = []

        if btype == 'paragraph':
            text = self._rich_text_to_markdown(content.get('rich_text') or [])
            if text:
                lines.append(text)
        elif btype in ('heading_1', 'heading_2', 'heading_3'):
            level = {'heading_1': 1, 'heading_2': 2, 'heading_3': 3}[btype]
            text = self._rich_text_to_markdown(content.get('rich_text') or [])
            if text:
                lines.append(f'{"#" * level} {text}')
        elif btype == 'bulleted_list_item':
            text = self._rich_text_to_markdown(content.get('rich_text') or [])
            if text:
                lines.append(f'- {text}')
        elif btype == 'numbered_list_item':
            text = self._rich_text_to_markdown(content.get('rich_text') or [])
            if text:
                lines.append(f'1. {text}')
        elif btype == 'to_do':
            text = self._rich_text_to_markdown(content.get('rich_text') or [])
            checked = 'x' if content.get('checked') else ' '
            if text:
                lines.append(f'- [{checked}] {text}')
        elif btype == 'toggle':
            text = self._rich_text_to_markdown(content.get('rich_text') or [])
            if text:
                lines.append(f'- {text}')
        elif btype == 'quote':
            text = self._rich_text_to_markdown(content.get('rich_text') or [])
            if text:
                lines.append('\n'.join(f'> {line}' for line in text.splitlines()))
        elif btype == 'code':
            text = self._rich_text_to_markdown(content.get('rich_text') or [])
            language = content.get('language') or ''
            lines.append(f'```{language}\n{text}\n```')
        elif btype == 'callout':
            text = self._rich_text_to_markdown(content.get('rich_text') or [])
            if text:
                lines.append(f'> {text}')
        elif btype == 'child_page':
            title = content.get('title') or block_id
            lines.append(f'## {title}')
            if block_id and depth < _MAX_RECURSION_DEPTH:
                child = self._page_to_markdown(
                    block_id, heading_level=3, depth=depth + 1, visited=visited,
                    include_title=False,
                )
                if child:
                    lines.append(child)
        elif btype == 'child_database':
            title = content.get('title') or block_id
            lines.append(f'## {title}')
            if block_id and depth < _MAX_RECURSION_DEPTH:
                child = self._database_to_markdown(
                    block_id, heading_level=3, depth=depth + 1, visited=visited,
                    include_title=False,
                )
                if child:
                    lines.append(child)
        elif btype == 'divider':
            lines.append('---')
        elif btype == 'bookmark':
            url = content.get('url') or ''
            caption = self._rich_text_to_markdown(content.get('caption') or [])
            lines.append(f'[{caption or url}]({url})' if url else caption)
        elif btype in ('embed', 'image', 'video', 'file', 'pdf', 'audio'):
            text = self._file_block_to_markdown(content)
            if text:
                lines.append(text)
        elif btype == 'equation':
            expression = content.get('expression') or ''
            if expression:
                lines.append(f'$$\n{expression}\n$$')
        elif btype == 'table':
            pass
        elif btype == 'table_row':
            lines.append(self._table_row_to_markdown(block))
        else:
            text = self._rich_text_to_markdown(content.get('rich_text') or [])
            if text:
                lines.append(text)

        if block.get('has_children') and block_id and btype not in ('child_page', 'child_database'):
            child_key = _normalize_notion_id(block_id)
            if child_key not in visited and depth < _MAX_RECURSION_DEPTH:
                visited.add(child_key)
                children = self._list_children_raw(block_id)
                lines.extend(self._blocks_to_markdown(children, depth=depth + 1, visited=visited))
        return lines

    @staticmethod
    def _join_markdown(lines: List[str]) -> str:
        out: List[str] = []
        last_blank = False
        for line in lines:
            if line is None:
                continue
            text = str(line).strip('\n')
            if not text:
                if not last_blank:
                    out.append('')
                last_blank = True
                continue
            out.append(text)
            last_blank = False
        chunks: List[str] = []
        for text in out:
            if not chunks:
                chunks.append(text)
                continue
            previous = chunks[-1]
            if (
                (previous.startswith('|') and text.startswith('|'))
                or (previous.startswith('- ') and text.startswith('- '))
                or (previous.startswith('1. ') and text.startswith('1. '))
            ):
                chunks[-1] = previous + '\n' + text
            else:
                chunks.append(text)
        return '\n\n'.join(chunks).strip()

    @staticmethod
    def _flush_table_rows(rows: List[str]) -> List[str]:
        if not rows:
            return []
        first_cols = rows[0].count('|') - 1
        if first_cols <= 0:
            return rows
        separator = '|' + '|'.join(['---'] * first_cols) + '|'
        return [rows[0], separator, *rows[1:]]

    @staticmethod
    def _text_to_rich_text(text: str) -> List[Dict[str, Any]]:
        if text == '':
            return []
        return [
            {'type': 'text', 'text': {'content': text[i:i + 2000]}}
            for i in range(0, len(text), 2000)
        ]

    @classmethod
    def _text_to_paragraph_blocks(cls, text: str) -> List[Dict[str, Any]]:
        chunks = [text[i:i + 2000] for i in range(0, len(text), 2000)] or ['']
        return [{
            'object': 'block',
            'type': 'paragraph',
            'paragraph': {'rich_text': cls._text_to_rich_text(chunk)},
        } for chunk in chunks]

    @staticmethod
    def _title_property_key(page: Dict[str, Any]) -> str:
        props = page.get('properties', {}) or {}
        for key, prop in props.items():
            if isinstance(prop, dict) and prop.get('type') == 'title':
                return key
        for key in ('title', 'Title', 'Name'):
            if key in props:
                return key
        return 'title'

    @staticmethod
    def _block_plain_text(block: Dict[str, Any]) -> str:
        btype = block.get('type') or block.get('block_type') or ''
        content = block.get(btype) if btype else None
        if not isinstance(content, dict):
            return block.get('plain_text', '') or block.get('title', '')
        if btype in ('child_page', 'child_database'):
            return content.get('title') or ''
        if btype == 'table_row':
            cells = content.get('cells') or []
            return ' | '.join(NotionFS._rich_text_to_markdown(cell) for cell in cells)
        return NotionFS._rich_text_to_markdown(content.get('rich_text') or content.get('caption') or [])

    @classmethod
    def _block_summary(cls, block: Dict[str, Any]) -> Dict[str, Any]:
        btype = block.get('type') or block.get('block_type')
        return {
            'block_id': block.get('id') or block.get('block_id', ''),
            'block_type': btype,
            'parent_id': ((block.get('parent') or {}).get('page_id')
                          or (block.get('parent') or {}).get('block_id')
                          or block.get('parent_id', '')),
            'plain_text': cls._block_plain_text(block),
            'has_children': bool(block.get('has_children')),
        }

    @staticmethod
    def _ref_from_rich_text_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        text_obj = item.get('text') or {}
        href = item.get('href') or ((text_obj.get('link') or {}).get('url'))
        if href:
            parsed = _parse_notion_browser_url(href)
            return {
                'url': href,
                'ref_type': 'hyperlink',
                'kind': parsed['kind'] if parsed else 'external',
            }
        if item.get('type') != 'mention':
            return None
        mention = item.get('mention') or {}
        mtype = mention.get('type') or ''
        value = mention.get(mtype)
        if not isinstance(value, dict):
            return None
        object_id = value.get('id')
        if mtype in ('page', 'database') and object_id:
            return {
                'url': f'notion:/~{mtype}/{_normalize_notion_id(object_id)}',
                'ref_type': f'mention_{mtype}',
                'kind': mtype,
            }
        url = value.get('url')
        if url:
            parsed = _parse_notion_browser_url(url)
            return {
                'url': url,
                'ref_type': f'mention_{mtype}',
                'kind': parsed['kind'] if parsed else 'external',
            }
        return None

    @classmethod
    def _refs_from_rich_text(cls, rich: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        for item in rich or []:
            ref = cls._ref_from_rich_text_item(item)
            if ref:
                refs.append(ref)
        return refs

    @classmethod
    def _refs_from_page_properties(cls, page: Dict[str, Any]) -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        for prop in (page.get('properties') or {}).values():
            if not isinstance(prop, dict):
                continue
            ptype = prop.get('type')
            value = prop.get(ptype) if ptype else None
            if isinstance(value, list):
                refs.extend(cls._refs_from_rich_text(value))
            elif ptype == 'url' and isinstance(value, str) and value:
                parsed = _parse_notion_browser_url(value)
                refs.append({'url': value, 'ref_type': 'property_url',
                             'kind': parsed['kind'] if parsed else 'external'})
            elif ptype in ('relation',) and isinstance(value, list):
                for rel in value:
                    if isinstance(rel, dict) and rel.get('id'):
                        refs.append({'url': f'notion:/~page/{_normalize_notion_id(rel["id"])}',
                                     'ref_type': 'property_relation', 'kind': 'page'})
        return refs

    @classmethod
    def _refs_from_block_content(cls, btype: str, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        for key in ('rich_text', 'caption'):
            refs.extend(cls._refs_from_rich_text(content.get(key) or []))
        if btype == 'bookmark':
            url = content.get('url') or ''
            if url:
                parsed = _parse_notion_browser_url(url)
                refs.append({'url': url, 'ref_type': 'bookmark',
                             'kind': parsed['kind'] if parsed else 'external'})
        if btype in ('embed', 'link_preview'):
            url = content.get('url') or ''
            if url:
                parsed = _parse_notion_browser_url(url)
                refs.append({'url': url, 'ref_type': btype,
                             'kind': parsed['kind'] if parsed else 'external'})
        if btype in ('image', 'video', 'file', 'pdf', 'audio'):
            for file_type in ('external', 'file'):
                file_obj = content.get(file_type)
                if isinstance(file_obj, dict) and file_obj.get('url'):
                    refs.append({'url': file_obj['url'], 'ref_type': btype, 'kind': file_type})
        return refs

    @classmethod
    def _refs_from_blocks(cls, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        for block in blocks:
            btype = block.get('type') or ''
            content = block.get(btype) or {}
            if isinstance(content, dict):
                refs.extend(cls._refs_from_block_content(btype, content))
            if btype == 'child_page' and block.get('id'):
                refs.append({'url': f'notion:/~page/{_normalize_notion_id(block["id"])}',
                             'ref_type': 'child_page', 'kind': 'page'})
            elif btype == 'child_database' and block.get('id'):
                refs.append({'url': f'notion:/~database/{_normalize_notion_id(block["id"])}',
                             'ref_type': 'child_database', 'kind': 'database'})
        return refs

    def _table_row_to_markdown(self, block: Dict[str, Any]) -> str:
        cells = (block.get('table_row') or {}).get('cells') or []
        rendered = [self._rich_text_to_markdown(cell).replace('|', '\\|') for cell in cells]
        return '|' + '|'.join(rendered) + '|'

    def _file_block_to_markdown(self, content: Dict[str, Any]) -> str:
        caption = self._rich_text_to_markdown(content.get('caption') or [])
        url = ''
        if isinstance(content.get('external'), dict):
            url = content['external'].get('url') or ''
        if isinstance(content.get('file'), dict):
            url = content['file'].get('url') or url
        if isinstance(content.get('pdf'), dict):
            url = content['pdf'].get('url') or url
        if url:
            return f'[{caption or url}]({url})'
        return caption

    @staticmethod
    def _rich_text_to_markdown(rich: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for item in rich or []:
            text = item.get('plain_text') or ''
            if not text and item.get('type') == 'mention':
                text = NotionFS._mention_to_text(item.get('mention') or {})
            text_obj = item.get('text') or {}
            href = item.get('href') or ((text_obj.get('link') or {}).get('url'))
            annotations = item.get('annotations') or {}
            if annotations.get('code') and text:
                text = f'`{text}`'
            if annotations.get('bold') and text:
                text = f'**{text}**'
            if annotations.get('italic') and text:
                text = f'*{text}*'
            if href and text:
                text = f'[{text}]({href})'
            parts.append(text)
        return ''.join(parts)

    @staticmethod
    def _mention_to_text(mention: Dict[str, Any]) -> str:
        mtype = mention.get('type') or ''
        value = mention.get(mtype) if mtype else None
        if isinstance(value, dict):
            return value.get('id') or value.get('name') or value.get('url') or ''
        if isinstance(value, str):
            return value
        return ''

    @staticmethod
    def _page_title(page: Dict[str, Any]) -> str:
        props = page.get('properties', {})
        for prop in props.values():
            if prop.get('type') == 'title':
                return NotionFS._rich_text_to_markdown(prop.get('title') or [])
        for key in ('title', 'Title', 'Name'):
            prop = props.get(key)
            if isinstance(prop, dict):
                return NotionFS._rich_text_to_markdown(prop.get('title') or [])
        return ''

    @staticmethod
    def _database_title(db: Dict[str, Any]) -> str:
        return NotionFS._rich_text_to_markdown(db.get('title') or [])

    @staticmethod
    def _page_to_entry(page: Dict[str, Any]) -> Dict[str, Any]:
        pid = page.get('id', '')
        title = NotionFS._page_title(page)
        mtime = None
        ts = page.get('last_edited_time')
        if ts:
            try:
                mtime = datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
            except (ValueError, TypeError) as e:
                lazyllm.LOG.debug(f"Failed to parse timestamp '{ts}': {e}")
        return LazyLLMFSBase._entry(
            name=title or pid, ftype='directory', mtime=mtime, title=title,
            id=pid, object=page.get('object', 'page'), notion_path=f'notion:/~page/{pid}',
        )

    @staticmethod
    def _db_to_entry(db: Dict[str, Any]) -> Dict[str, Any]:
        did = db.get('id', '')
        title = NotionFS._database_title(db)
        return LazyLLMFSBase._entry(
            name=title or did, ftype='directory', title=title, id=did,
            object=db.get('object', 'database'), notion_path=f'notion:/~database/{did}',
        )

    @staticmethod
    def _object_to_entry(obj: Dict[str, Any]) -> Dict[str, Any]:
        if obj.get('object') == 'database':
            return NotionFS._db_to_entry(obj)
        return NotionFS._page_to_entry(obj)

    @staticmethod
    def _block_to_entry(block: Dict[str, Any]) -> Dict[str, Any]:
        bid = block.get('id', '')
        btype = block.get('type', 'paragraph')
        content = block.get(btype) or {}
        title = content.get('title') or NotionFS._rich_text_to_markdown(content.get('rich_text') or [])
        has_children = block.get('has_children', False) or btype in ('child_page', 'child_database')
        if btype == 'child_database':
            notion_path = f'notion:/~database/{bid}'
        elif btype == 'child_page':
            notion_path = f'notion:/~page/{bid}'
        else:
            notion_path = f'notion:/~block/{bid}'
        return LazyLLMFSBase._entry(
            name=title or bid,
            ftype='directory' if has_children else 'file',
            block_type=btype, title=title, id=bid, notion_path=notion_path,
        )
