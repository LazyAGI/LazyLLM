# Copyright (c) 2026 LazyAGI. All rights reserved.
import mimetypes
import os
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import parse_qs, unquote, urlparse

import lazyllm
import requests
from lazyllm import config

from ..base import LazyLLMFSBase, LinkDocumentFSBase, CloudFSBufferedFile, clean_document_ref

config.add('notion_token', str, None, 'NOTION_TOKEN', description='Notion API token (notion-client official env).')

_API_BASE = 'https://api.notion.com/v1'
_NOTION_VERSION = '2026-03-11'
_PAGE_SIZE = 100
_MAX_RECURSION_DEPTH = 3
_SINGLE_PART_UPLOAD_LIMIT = 20 * 1024 * 1024
_MULTI_PART_CHUNK_SIZE = 10 * 1024 * 1024

_BLOCK_CREATE_PAYLOAD_FIELDS = {
    'audio': {'caption', 'external', 'file', 'file_upload', 'type'},
    'bookmark': {'caption', 'url'},
    'breadcrumb': set(),
    'bulleted_list_item': {'color', 'rich_text'},
    'callout': {'color', 'icon', 'rich_text'},
    'code': {'caption', 'language', 'rich_text'},
    'column': {'width_ratio'},
    'column_list': set(),
    'divider': set(),
    'embed': {'caption', 'url'},
    'equation': {'expression'},
    'file': {'caption', 'external', 'file', 'file_upload', 'name', 'type'},
    'heading_1': {'color', 'is_toggleable', 'rich_text'},
    'heading_2': {'color', 'is_toggleable', 'rich_text'},
    'heading_3': {'color', 'is_toggleable', 'rich_text'},
    'heading_4': {'color', 'is_toggleable', 'rich_text'},
    'image': {'caption', 'external', 'file', 'file_upload', 'type'},
    'link_to_page': {'comment_id', 'database_id', 'page_id', 'type'},
    'numbered_list_item': {'color', 'rich_text'},
    'paragraph': {'color', 'icon', 'rich_text'},
    'pdf': {'caption', 'external', 'file', 'file_upload', 'type'},
    'quote': {'color', 'rich_text'},
    'synced_block': {'synced_from'},
    'tab': set(),
    'table': {'has_column_header', 'has_row_header', 'table_width'},
    'table_of_contents': {'color'},
    'table_row': {'cells'},
    'template': {'rich_text'},
    'to_do': {'checked', 'color', 'rich_text'},
    'toggle': {'color', 'rich_text'},
    'video': {'caption', 'external', 'file', 'file_upload', 'type'},
}
_MEDIA_BLOCK_TYPES = {'audio', 'file', 'image', 'pdf', 'video'}
_BLOCK_UPDATE_TYPES = {
    'paragraph', 'heading_1', 'heading_2', 'heading_3', 'heading_4',
    'bulleted_list_item', 'numbered_list_item', 'to_do', 'quote',
    'callout', 'code', 'table_row', 'image',
}
_RICH_TEXT_PAYLOAD_FIELDS = {'caption', 'rich_text', 'title'}
_ANNOTATION_FIELDS = {'bold', 'code', 'color', 'italic', 'strikethrough', 'underline'}

_NOTION_HOST_RE = re.compile(r'(^|\.)notion\.(so|site|com)$', re.IGNORECASE)
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
    url = clean_document_ref(url)
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
    path_parts = [p for p in (parsed.path or '').strip('/').split('/') if p]
    if path_parts[:1] == ['p']:
        result['mode_hint'] = 'page'
    elif any(query.get(key) for key in ('p', 'page', 'page_id', 'pageId')):
        result['mode_hint'] = 'page'
    elif any(query.get(key) for key in ('database_id', 'databaseId')):
        result['mode_hint'] = 'database'
    return result


def _is_notion_browser_url(path: str) -> bool:
    return _parse_notion_browser_url(path) is not None


def _strip_notion_protocol(path: str) -> str:
    path = clean_document_ref(path)
    if path.startswith('notion:/'):
        return path[len('notion:'):]
    return path


def _parsed_notion_ref_to_path(parsed: Dict[str, str]) -> str:
    mode_hint = parsed.get('mode_hint')
    if mode_hint in ('page', 'database'):
        return f'/~{mode_hint}/{parsed["id"]}'
    return f'/{parsed["id"]}'


def _parsed_notion_ref_to_kind(parsed: Dict[str, str]) -> Tuple[str, str]:
    mode_hint = parsed.get('mode_hint')
    return (mode_hint, parsed['id']) if mode_hint in ('page', 'database') else ('object', parsed['id'])


def _is_notion_object_not_found(exc: Exception) -> bool:
    if not isinstance(exc, requests.HTTPError):
        return False
    response = getattr(exc, 'response', None)
    if getattr(response, 'status_code', None) != 404:
        return False
    try:
        body = response.json()
    except Exception:
        return False
    return isinstance(body, dict) and body.get('code') == 'object_not_found'


def _ls_tool_schema(path: str = '/', detail: bool = True) -> List:
    return []


def _adapt_ls_tool_input(tool_input: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    if isinstance(tool_input, str):
        return {'path': tool_input.strip() or '/'}
    adapted = dict(tool_input)
    adapted['path'] = str(adapted.get('path') or '/').strip() or '/'
    return adapted


class NotionFile(CloudFSBufferedFile):
    def __init__(self, fs: 'NotionFS', path: str, include_references: bool = False, **kwargs) -> None:
        content = fs._fetch_content(path, include_references=include_references)
        self._notion_content: bytes = content
        super().__init__(fs, path, size=len(content), **kwargs)

    def _fetch_range(self, start: int, end: int) -> bytes:
        return self._notion_content[start:end]


class NotionFS(LinkDocumentFSBase):
    __tool_auto_activate__ = [r'https?://(?:[^\s/]+\.)?(?:notion\.(?:so|site|com))(?:[/:?#]|$)']
    '''Read and manage authenticated Notion pages, databases, and documents.

    Select this Toolkit for notion.so, notion.site, or notion.com browser URLs.
    Resolve or read a supplied URL directly; use search/find when the exact page
    is not known.
    '''

    document_provider = 'notion'
    __public_apis__ = LinkDocumentFSBase.build_public_apis(extra=['search', 'find'], exclude=['copy'])
    __tool_schema_overrides__ = {'ls': _ls_tool_schema}
    __tool_input_adapters__ = {'ls': _adapt_ls_tool_input}

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
        self._kind_cache: Dict[str, str] = {}

    def _setup_auth(self) -> None:
        self._session.headers.update({
            'Notion-Version': _NOTION_VERSION,
            'Content-Type': 'application/json',
        })

    def ls(self, path: str, detail: bool = True, **kwargs) -> List:
        kind, object_id = self._resolve_access_ref(path)
        if kind == 'root':
            return self._search_all(detail)
        if kind in ('database', 'data_source'):
            entries = [self._object_to_entry(p) for p in self._query_collection(kind, object_id)]
            return entries if detail else [e['name'] for e in entries]
        if kind == 'block':
            entries = [self._block_to_entry(b) for b in self._list_children_raw(object_id)]
            return entries if detail else [e['name'] for e in entries]

        entries = [self._block_to_entry(b) for b in self._list_children_raw(object_id)]
        return entries if detail else [e['name'] for e in entries]

    def info(self, path: str, **kwargs) -> Dict[str, Any]:
        kind, object_id = self._resolve_access_ref(path)
        if kind == 'root':
            return self._entry('/', ftype='directory')
        if kind == 'database':
            return self._db_to_entry(self._retrieve_database(object_id))
        if kind == 'data_source':
            return self._data_source_to_entry(self._retrieve_data_source(object_id))
        if kind == 'block':
            return self._block_to_entry(self._retrieve_block(object_id))
        return self._page_to_entry(self._retrieve_page(object_id))

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
        return self._fetch_content(_parsed_notion_ref_to_path(parsed))

    def search(self, query: str, object_type: str = '', limit: int = 20,
               sort_direction: str = 'descending', scope: str = '',
               title_pattern: str = '') -> List[Dict[str, Any]]:
        '''Search connected Notion pages and databases.

        Args:
            query: Text to search for.
            object_type: Optional object type filter: page, data_source, or the database compatibility alias.
            limit: Maximum result count.
            sort_direction: Sort by last-edited time in ascending or descending order.
            scope: Optional page, database, or data-source scope.
            title_pattern: Optional title pattern filter.
        '''
        query = (query or '').strip()
        if not query:
            raise ValueError('query is required')
        object_type = self._normalize_search_object_type(object_type)
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 20
        limit = max(1, min(limit, _PAGE_SIZE))
        sort_direction = (sort_direction or 'descending').strip().lower()
        if sort_direction not in {'ascending', 'descending'}:
            sort_direction = 'descending'
        title_regex = self._compile_title_regex(title_pattern)

        scope_kind, scope_id = self._resolve_search_scope(scope)
        if scope_kind in ('database', 'data_source'):
            entries = [
                self._object_to_entry(item)
                for item in self._query_collection(scope_kind, scope_id)
            ]
            return [
                entry for entry in entries
                if self._entry_matches_query(entry, query)
                and self._entry_matches_title_regex(entry, title_regex)
            ][:limit]

        payload: Dict[str, Any] = {
            'query': query,
            'page_size': limit,
            'sort': {'direction': sort_direction, 'timestamp': 'last_edited_time'},
        }
        if object_type:
            payload['filter'] = {'property': 'object', 'value': object_type}
        results = self._paginate_post(f'{self._base_url}/search', payload)
        entries = [self._object_to_entry(item) for item in results]
        return [
            entry for entry in entries
            if self._entry_matches_title_regex(entry, title_regex)
        ][:limit]

    def find(self, pattern: str, object_type: str = '', limit: int = 50,
             scope: str = '') -> List[Dict[str, Any]]:
        '''Find connected Notion objects matching a pattern.

        Args:
            pattern: Pattern to match.
            object_type: Optional object type filter: page, data_source, or the database compatibility alias.
            limit: Maximum result count.
            scope: Optional page, database, or data-source scope.
        '''
        pattern = (pattern or '').strip()
        if not pattern:
            raise ValueError('pattern is required')
        limit = self._clamp_find_limit(limit)
        object_type = self._validate_find_object_type(object_type)
        regex = self._compile_title_regex(pattern)

        scope_kind, scope_id = self._resolve_search_scope(scope)
        if scope_kind in ('database', 'data_source'):
            return self._find_in_collection(scope_kind, scope_id, regex, limit)

        return self._find_via_search_api(object_type, regex, limit)

    @staticmethod
    def _clamp_find_limit(limit: int) -> int:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 50
        return max(1, min(limit, _PAGE_SIZE))

    @staticmethod
    def _validate_find_object_type(object_type: str) -> str:
        return NotionFS._normalize_search_object_type(object_type)

    @staticmethod
    def _normalize_search_object_type(object_type: str) -> str:
        object_type = (object_type or '').strip().lower()
        if object_type == 'database':
            return 'data_source'
        if object_type and object_type not in {'page', 'data_source'}:
            raise ValueError('object_type must be page, data_source, or database')
        return object_type

    def _find_in_collection(
        self, kind: str, object_id: str, regex, limit: int,
    ) -> List[Dict[str, Any]]:
        entries = [
            self._object_to_entry(item)
            for item in self._query_collection(kind, object_id)
        ]
        return [
            entry for entry in entries
            if self._entry_matches_title_regex(entry, regex)
        ][:limit]

    def _find_via_search_api(
        self, object_type: str, regex, limit: int,
    ) -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {
            'page_size': _PAGE_SIZE,
            'sort': {'direction': 'descending', 'timestamp': 'last_edited_time'},
        }
        if object_type:
            payload['filter'] = {'property': 'object', 'value': object_type}
        results: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        while len(results) < limit:
            page = self._fetch_find_search_page(payload, cursor)
            results.extend(self._collect_find_matches(page.get('results') or [], regex, limit, results))
            cursor = self._next_find_cursor(page, results, limit)
            if self._find_page_done(cursor, results, limit):
                break
        return results[:limit]

    @staticmethod
    def _next_find_cursor(
        page: Dict[str, Any], results: List[Dict[str, Any]], limit: int,
    ) -> Optional[str]:
        if len(results) >= limit:
            return None
        return page.get('next_cursor') if page.get('has_more') else None

    @staticmethod
    def _find_page_done(
        cursor: Optional[str], results: List[Dict[str, Any]], limit: int,
    ) -> bool:
        return not cursor or len(results) >= limit

    def _fetch_find_search_page(
        self, payload: Dict[str, Any], cursor: Optional[str],
    ) -> Dict[str, Any]:
        page_payload = dict(payload)
        if cursor:
            page_payload['start_cursor'] = cursor
        return self._post(f'{self._base_url}/search', json=page_payload)

    def _collect_find_matches(
        self,
        items: List[Dict[str, Any]], regex,
        limit: int, results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if len(results) >= limit:
            return []
        collected: List[Dict[str, Any]] = []
        for item in items:
            if len(results) + len(collected) >= limit:
                break
            entry = self._object_to_entry(item)
            title = entry.get('title') or entry.get('name') or ''
            if title and regex.search(title):
                collected.append(entry)
        return collected

    def _resolve_search_scope(self, scope: str = '') -> Tuple[str, str]:
        scope = (scope or '').strip()
        if not scope:
            return '', ''
        for prefix, kind in (('database:', 'database'), ('data_source:', 'data_source')):
            if scope.startswith(prefix):
                return kind, _normalize_notion_id(scope[len(prefix):])
        kind, object_id = self._resolve_access_ref(scope)
        if kind not in ('database', 'data_source'):
            raise ValueError('scope must be a Notion database or data_source id/path')
        return kind, object_id

    @staticmethod
    def _compile_title_regex(pattern: str):
        pattern = (pattern or '').strip()
        if not pattern:
            return None
        try:
            return re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            raise ValueError(f'Invalid regex pattern: {e}') from e

    @staticmethod
    def _entry_title(entry: Dict[str, Any]) -> str:
        return entry.get('title') or entry.get('name') or ''

    @classmethod
    def _entry_matches_title_regex(cls, entry: Dict[str, Any], regex) -> bool:
        return regex is None or bool(regex.search(cls._entry_title(entry)))

    @classmethod
    def _entry_matches_query(cls, entry: Dict[str, Any], query: str) -> bool:
        title = cls._entry_title(entry).lower()
        return all(part.lower() in title for part in query.split())

    def mkdir(self, path: str, create_parents: bool = True, **kwargs) -> None:
        parent_kind, parent_id, title = self._resolve_parent_ref(path)
        if parent_kind not in ('page', 'database', 'data_source') or not parent_id or not title:
            raise ValueError('path must be /<parent_page_database_or_data_source_id>/<title>')
        parent, title_key = self._build_page_parent_and_title_key(parent_kind, parent_id)
        payload: Dict[str, Any] = {
            'parent': parent,
            'properties': {
                title_key: {'title': [{'text': {'content': title}}]}
            },
        }
        self._post(f'{self._base_url}/pages', json=payload)

    def rm_file(self, path: str) -> None:
        kind, object_id = self._resolve_access_ref(path)
        if kind in ('root', 'database', 'data_source'):
            raise FileNotFoundError(path)
        if kind == 'block':
            self._delete(f'{self._base_url}/blocks/{object_id}')
            return
        self._patch(f'{self._base_url}/pages/{object_id}', json={'in_trash': True})

    def rmdir(self, path: str) -> None:
        kind, object_id = self._resolve_access_ref(path)
        if kind == 'root':
            return
        if kind == 'block':
            self._delete(f'{self._base_url}/blocks/{object_id}')
            return
        if kind == 'data_source':
            self._patch(f'{self._base_url}/data_sources/{object_id}', json={'in_trash': True})
            return
        if kind == 'database':
            self._patch(f'{self._base_url}/databases/{object_id}', json={'in_trash': True})
            return
        self._patch(f'{self._base_url}/pages/{object_id}', json={'in_trash': True})

    def copy(self, path1: str, path2: str, recursive: bool = False, **kwargs) -> None:
        raise NotImplementedError('NotionFS: Notion official API does not support copy')

    def move(self, path1: str, path2: str, recursive: bool = False, **kwargs) -> None:
        src_kind, page_id = self._resolve_access_ref(path1)
        if src_kind != 'page':
            raise NotImplementedError('NotionFS.move only supports moving pages')
        parent_kind, parent_id, new_title = self._parse_move_destination(path2)
        self._post(
            f'{self._base_url}/pages/{page_id}/move',
            json={'parent': self._build_move_parent(parent_kind, parent_id)},
        )
        if new_title:
            self.update_page_title(page_id, new_title)

    def _download_range(self, path: str, start: int, end: int) -> bytes:
        return self._fetch_content(path)[start:end]

    def _upload_data(self, path: str, data: bytes, **kwargs) -> None:
        kind, object_id = self._resolve_access_ref(path)
        if kind in ('root', 'database', 'data_source'):
            raise ValueError('path must include a page_id or block_id')
        text = data.decode('utf-8', errors='replace')
        content_type = kwargs.get('content_type')
        if kind == 'page' and content_type in ('markdown', 'md'):
            self.replace_page_markdown(object_id, text, allow_deleting_content=True)
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
        norm = path.lstrip('/')
        if self.is_link_path(norm):
            path = self.decode_link_path(norm)
            norm = path.lstrip('/')

        parsed = _parse_notion_browser_url(path)
        if parsed:
            return _parsed_notion_ref_to_kind(parsed)

        for prefix, kind in (
            ('~page/', 'page'),
            ('~database/', 'database'),
            ('~data_source/', 'data_source'),
            ('~block/', 'block'),
        ):
            if norm.startswith(prefix):
                token = norm[len(prefix):].rstrip('/').split('/')[0]
                return kind, _normalize_notion_id(token)

        parts = self._parse_path(path)
        if not parts:
            return 'root', ''
        return 'object', _normalize_notion_id(parts[-1])

    def _resolve_access_ref(self, path: str) -> Tuple[str, str]:
        kind, object_id = self._resolve_ref(path)
        if kind != 'object':
            return kind, object_id
        cached = self._kind_cache.get(object_id)
        if cached is not None:
            return cached, object_id
        resolved = self._resolve_object_kind(object_id)
        self._kind_cache[object_id] = resolved
        return resolved, object_id

    def _resolve_object_kind(self, object_id: str) -> str:
        try:
            self._retrieve_page(object_id)
            return 'page'
        except requests.HTTPError as exc:
            if not _is_notion_object_not_found(exc):
                raise
        try:
            self._retrieve_database(object_id)
            return 'database'
        except requests.HTTPError as exc:
            if not _is_notion_object_not_found(exc):
                raise
        self._retrieve_data_source(object_id)
        return 'data_source'

    def _resolve_parent_ref(self, path: str) -> Tuple[str, str, str]:
        path = _strip_notion_protocol(path)
        if _is_notion_browser_url(path):
            parsed = _parse_notion_browser_url(path)
            if not parsed:
                return '', '', ''
            kind, object_id = self._resolve_access_ref(_parsed_notion_ref_to_path(parsed))
            return kind, object_id, ''

        norm = path.strip('/')
        if not norm:
            return 'root', '', ''
        explicit_kind = ''
        if norm.startswith(('~page/', '~database/', '~data_source/', '~block/')):
            prefix, rest = norm.split('/', 1)
            explicit_kind = {
                '~page': 'page',
                '~database': 'database',
                '~data_source': 'data_source',
                '~block': 'block',
            }[prefix]
            parts = [p for p in rest.split('/') if p]
        else:
            parts = [p for p in norm.split('/') if p]
        if not parts:
            return explicit_kind or 'root', '', ''
        parent_id = _normalize_notion_id(unquote(parts[0]))
        title = unquote('/'.join(parts[1:])) if len(parts) > 1 else ''
        if explicit_kind:
            return explicit_kind, parent_id, title
        kind, object_id = self._resolve_access_ref(f'/{parent_id}')
        return kind, object_id, title

    def _parse_move_destination(self, path: str) -> Tuple[str, str, str]:
        parent_kind, parent_id, title = self._resolve_parent_ref(path)
        if parent_kind not in ('page', 'database', 'data_source') or not parent_id:
            raise ValueError('move destination must include a parent page, database, or data source id')
        return parent_kind, parent_id, title

    def _resolve_data_source_id(self, database_id: str) -> str:
        database = self._retrieve_database(database_id)
        data_sources = database.get('data_sources') or []
        data_source_ids = [
            _normalize_notion_id(item.get('id', ''))
            for item in data_sources
            if isinstance(item, dict) and item.get('id')
        ]
        if len(data_source_ids) == 1:
            return data_source_ids[0]
        if not data_source_ids:
            raise ValueError(
                'database parent does not expose a child data source; '
                'use a page parent or a database with one data source'
            )
        raise ValueError('database parent has multiple data sources; specify the intended data source explicitly')

    def _build_page_parent(self, parent_kind: str, parent_id: str) -> Dict[str, str]:
        if parent_kind == 'page':
            return {'type': 'page_id', 'page_id': parent_id}
        if parent_kind == 'database':
            data_source_id = self._resolve_data_source_id(parent_id)
            return {'type': 'data_source_id', 'data_source_id': data_source_id}
        if parent_kind == 'data_source':
            return {'type': 'data_source_id', 'data_source_id': parent_id}
        raise ValueError('parent must be a page, database, or data source')

    def _build_page_parent_and_title_key(self, parent_kind: str, parent_id: str) -> Tuple[Dict[str, str], str]:
        parent = self._build_page_parent(parent_kind, parent_id)
        title_key = 'title'
        data_source_id = parent.get('data_source_id')
        if data_source_id:
            title_key = self._data_source_title_property_key(data_source_id)
        return parent, title_key

    def _build_move_parent(self, parent_kind: str, parent_id: str) -> Dict[str, str]:
        if parent_kind == 'page':
            return {'type': 'page_id', 'page_id': parent_id}
        if parent_kind == 'database':
            return {'type': 'data_source_id', 'data_source_id': self._resolve_data_source_id(parent_id)}
        if parent_kind == 'data_source':
            return {'type': 'data_source_id', 'data_source_id': parent_id}
        raise ValueError('parent must be a page, database, or data source')

    def _search_all(self, detail: bool) -> List:
        results = self._paginate_post(f'{self._base_url}/search', {'page_size': _PAGE_SIZE})
        if detail:
            return [self._object_to_entry(r) for r in results]
        return [r.get('id', '') for r in results]

    def _list_children_raw(self, block_id: str) -> List[Dict[str, Any]]:
        return self._paginate_get(f'{self._base_url}/blocks/{block_id}/children', {'page_size': _PAGE_SIZE})

    def _query_collection(self, kind: str, object_id: str) -> List[Dict[str, Any]]:
        if kind == 'data_source':
            return self._query_data_source(object_id)
        return self._query_database(object_id)

    def _query_database(self, database_id: str) -> List[Dict[str, Any]]:
        return self._query_data_source(self._resolve_data_source_id(database_id))

    def _query_data_source(self, data_source_id: str) -> List[Dict[str, Any]]:
        return self._paginate_post(
            f'{self._base_url}/data_sources/{_normalize_notion_id(data_source_id)}/query',
            {'page_size': _PAGE_SIZE},
        )

    def _retrieve_page(self, page_id: str) -> Dict[str, Any]:
        return self._get(f'{self._base_url}/pages/{page_id}')

    def _retrieve_database(self, database_id: str) -> Dict[str, Any]:
        return self._get(f'{self._base_url}/databases/{database_id}')

    def _retrieve_data_source(self, data_source_id: str) -> Dict[str, Any]:
        return self._get(f'{self._base_url}/data_sources/{_normalize_notion_id(data_source_id)}')

    def _data_source_title_property_key(self, data_source_id: str) -> str:
        data_source = self._retrieve_data_source(data_source_id)
        props = data_source.get('properties') or {}
        for key, prop in props.items():
            if isinstance(prop, dict) and prop.get('type') == 'title':
                return key
        for key in ('title', 'Title', 'Name'):
            if key in props:
                return key
        return 'title'

    def _retrieve_block(self, block_id: str) -> Dict[str, Any]:
        return self._get(f'{self._base_url}/blocks/{block_id}')

    def _retrieve_page_markdown(self, page_id: str) -> Optional[str]:
        try:
            data = self._get(
                f'{self._base_url}/pages/{page_id}/markdown',
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
        )

    def insert_page_markdown(self, page_id: str, markdown: str,
                             position: str = 'end') -> Dict[str, Any]:
        return self._patch(
            f'{self._base_url}/pages/{_normalize_notion_id(page_id)}/markdown',
            json={
                'type': 'insert_content',
                'insert_content': {
                    'content': markdown,
                    'position': {'type': position},
                },
            },
        )

    def update_page_title(self, page_id: str, title: str) -> None:
        page_id = _normalize_notion_id(page_id)
        page = self._retrieve_page(page_id)
        title_key = self._title_property_key(page)
        self._patch(
            f'{self._base_url}/pages/{page_id}',
            json={'properties': {title_key: {'title': self._text_to_rich_text(title)}}},
        )

    def get_document_metadata(self, path: str) -> Dict[str, Any]:
        '''Return the page identity and conflict baseline required by Writer providers.'''
        kind, object_id = self._resolve_access_ref(path)
        if kind != 'page' or not object_id:
            return {
                'document_id': object_id,
                'object_type': kind,
                'title': '',
                'browser_url': '',
                'internal_uri': f'notion:/~{kind}/{object_id}' if object_id else '',
                'last_edited_time': '',
            }
        page = self._retrieve_page(object_id)
        page_id = _normalize_notion_id(str(page.get('id') or object_id))
        return {
            'document_id': page_id,
            'object_type': 'page',
            'title': self._page_title(page),
            'browser_url': str(page.get('url') or ''),
            'internal_uri': f'notion:/~page/{page_id}',
            'last_edited_time': str(page.get('last_edited_time') or ''),
        }

    def create_document(self, title: str, parent_path: str = '') -> Dict[str, Any]:
        '''Create a Notion page and return its normalized Writer target metadata.'''
        title = str(title or '').strip()
        if not title:
            raise ValueError('title is required')
        parent_kind, parent_id = self._resolve_access_ref(parent_path)
        if parent_kind == 'root':
            # Public OAuth connections and personal access tokens may create a
            # private page owned by the authorizing user at workspace level.
            parent = {'type': 'workspace', 'workspace': True}
            title_key = 'title'
        elif parent_kind in ('page', 'database', 'data_source') and parent_id:
            parent, title_key = self._build_page_parent_and_title_key(parent_kind, parent_id)
        else:
            raise ValueError('Notion document parent must be a page, database, or data source.')
        page = self._post(
            f'{self._base_url}/pages',
            json={
                'parent': parent,
                'properties': {
                    title_key: {'title': [{'text': {'content': title}}]},
                },
            },
        )
        if not isinstance(page, dict) or not page.get('id'):
            raise RuntimeError('Notion did not return the created page.')
        page_id = _normalize_notion_id(str(page['id']))
        return {
            'document_id': page_id,
            'title': self._page_title(page) or title,
            'browser_url': str(page.get('url') or ''),
            'internal_uri': f'notion:/~page/{page_id}',
            'last_edited_time': str(page.get('last_edited_time') or ''),
        }

    def resolve_notion_ref(self, url_or_path: str) -> Dict[str, Any]:
        kind, object_id = self._resolve_access_ref(url_or_path)
        if kind == 'root':
            return {'object_id': '', 'object_type': 'root', 'title': 'Notion'}
        if kind == 'database':
            entry = self._db_to_entry(self._retrieve_database(object_id))
        elif kind == 'data_source':
            entry = self._data_source_to_entry(self._retrieve_data_source(object_id))
        elif kind == 'block':
            entry = self._block_to_entry(self._retrieve_block(object_id))
        else:
            entry = self._page_to_entry(self._retrieve_page(object_id))
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
        kind, object_id = self._resolve_access_ref(path)
        if kind == 'root' or not object_id:
            return []
        blocks: List[Dict[str, Any]]
        if kind in ('database', 'data_source'):
            blocks = []
            for page in self._query_collection(kind, object_id):
                page_entry = self._object_to_entry(page)
                child_type = 'child_database' if page.get('object') == 'data_source' else 'child_page'
                blocks.append({
                    'block_id': page.get('id', ''),
                    'block_type': child_type,
                    'parent_id': object_id,
                    'plain_text': page_entry.get('title', ''),
                    'has_children': True,
                })
                if with_descendants and page.get('id'):
                    if page.get('object') == 'page':
                        blocks.extend(self._get_doc_blocks_raw(page['id'], with_descendants=True))
        else:
            blocks = self._get_doc_blocks_raw(object_id, with_descendants=with_descendants)
        return [{**block, **self._block_summary(block)} for block in blocks]

    def create_block(self, document_id: str, parent_block_id: str,
                     block: Optional[Dict[str, Any]] = None,
                     index: Optional[int] = None,
                     blocks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        '''Create one semantic Notion subtree as one or more physical sibling blocks.'''
        document_id = _normalize_notion_id(document_id)
        parent_block_id = _normalize_notion_id(parent_block_id)
        if blocks is not None and block is not None:
            raise ValueError('provide either block or blocks, not both.')
        source_blocks = blocks if blocks is not None else [block]
        if not isinstance(source_blocks, list) or not source_blocks:
            raise ValueError('blocks must be a non-empty list.')
        if any(not isinstance(item, dict) for item in source_blocks):
            raise TypeError('every block must be a dict.')
        if parent_block_id != document_id:
            self._ensure_block_belongs_to_document(f'/~page/{document_id}', parent_block_id)
        position = self._position_for_index(parent_block_id, index)
        relations: List[Dict[str, str]] = []
        created: List[Dict[str, Any]] = []
        try:
            created = self._append_block_trees(
                parent_block_id, source_blocks, position=position, relations=relations)
            self._rebind_internal_links(source_blocks, relations, require_all=True)
        except Exception:
            for created_block in reversed(created):
                created_id = created_block.get('id') if isinstance(created_block, dict) else None
                if not isinstance(created_id, str) or not created_id:
                    continue
                try:
                    self._delete(f'{self._base_url}/blocks/{_normalize_notion_id(created_id)}')
                except Exception as rollback_exc:
                    lazyllm.LOG.warning(
                        f'Failed to roll back created Notion block {created_id!r}: '
                        f'{rollback_exc}')
            raise
        return {
            'block': created[0],
            'block_id': created[0]['id'],
            'blocks': created,
            'block_ids': [item['id'] for item in created],
            'block_id_relations': relations,
        }

    def update_block(self, document_id: str, block_id: str,
                     block: Dict[str, Any]) -> Dict[str, Any]:
        '''Update one editable native Notion block after validating page ownership.'''
        document_id = _normalize_notion_id(document_id)
        block_id = _normalize_notion_id(block_id)
        self._ensure_block_belongs_to_document(f'/~page/{document_id}', block_id)
        if not isinstance(block, dict):
            raise TypeError(f'block must be a dict, got {type(block).__name__}.')
        block_type = block.get('type') or block.get('block_type')
        if block_type not in _BLOCK_UPDATE_TYPES:
            raise ValueError(f'Notion block type {block_type!r} does not support updates.')
        payload = block.get(block_type)
        if not isinstance(payload, dict):
            raise ValueError(f'native Notion block {block_type!r} must have an object payload.')
        if 'children' in payload:
            raise ValueError('Notion block updates cannot include children.')
        if block_type in _MEDIA_BLOCK_TYPES \
                and set(payload).issubset({'caption'}):
            cleaned = {}
            if 'caption' in payload:
                cleaned['caption'] = self._sanitize_rich_text_array(
                    payload['caption'], f'{block_type}.caption')
        else:
            cleaned = self._sanitize_create_payload(block_type, payload)
        updated = self._patch(
            f'{self._base_url}/blocks/{block_id}',
            json={block_type: cleaned},
        )
        if not isinstance(updated, dict):
            updated = {}
        returned_id = updated.get('id')
        if returned_id and _normalize_notion_id(returned_id) != block_id:
            raise RuntimeError(f'Notion updated unexpected block {returned_id!r}; expected {block_id!r}.')
        returned_type = updated.get('type')
        if returned_type and returned_type != block_type:
            raise RuntimeError(
                f'Notion returned block type {returned_type!r}; expected {block_type!r}.')
        return {**updated, 'block_id': block_id, 'block_type': block_type}

    def delete_block(self, document_id: str, block_id: str) -> Dict[str, Any]:
        '''Delete a block after proving that it belongs to the target Notion page.'''
        document_id = _normalize_notion_id(document_id)
        block_id = _normalize_notion_id(block_id)
        self._ensure_block_belongs_to_document(f'/~page/{document_id}', block_id)
        deleted = self._delete(f'{self._base_url}/blocks/{block_id}')
        if not isinstance(deleted, dict):
            deleted = {}
        returned_id = deleted.get('id')
        if returned_id and _normalize_notion_id(returned_id) != block_id:
            raise RuntimeError(f'Notion deleted unexpected block {returned_id!r}; expected {block_id!r}.')
        return {**deleted, 'block_id': block_id}

    def move_block(
        self,
        document_id: str,
        source_block_id: str,
        source_parent_block_id: str,
        source_index: int,
        target_parent_block_id: str,
        target_index: int,
        block: Dict[str, Any],
    ) -> Dict[str, Any]:
        '''Move a Notion block subtree by cloning, verifying, then deleting the source.'''
        document_id = _normalize_notion_id(document_id)
        source_block_id = _normalize_notion_id(source_block_id)
        source_parent_block_id = _normalize_notion_id(source_parent_block_id)
        target_parent_block_id = _normalize_notion_id(target_parent_block_id)
        self._ensure_block_belongs_to_document(f'/~page/{document_id}', source_block_id)
        if source_parent_block_id != document_id:
            self._ensure_block_belongs_to_document(
                f'/~page/{document_id}', source_parent_block_id)
        if target_parent_block_id != document_id:
            self._ensure_block_belongs_to_document(
                f'/~page/{document_id}', target_parent_block_id)
        if not isinstance(source_index, int) or isinstance(source_index, bool) or source_index < 0:
            raise ValueError('source_index must be a non-negative integer.')
        if not isinstance(target_index, int) or isinstance(target_index, bool) or target_index < 0:
            raise ValueError('target_index must be a non-negative integer.')
        source_children = self._list_children_raw(source_parent_block_id)
        if source_index >= len(source_children) \
                or _normalize_notion_id(source_children[source_index].get('id', '')) != source_block_id:
            raise RuntimeError('Notion move source index no longer matches the persisted document.')

        create_index = target_index
        if source_parent_block_id == target_parent_block_id and target_index > source_index:
            create_index += 1
        position = self._position_for_index(target_parent_block_id, create_index)
        relations: List[Dict[str, str]] = []
        created: Dict[str, Any] = {}
        source_delete_requested = False
        try:
            created = self._append_block_tree(
                target_parent_block_id, block, position=position, relations=relations)
            self._rebind_internal_links([block], relations, require_all=False)
            created_id = _normalize_notion_id(str(created.get('id') or ''))
            visible_ids = {
                _normalize_notion_id(str(child.get('id') or ''))
                for child in self._list_children_raw(target_parent_block_id)
                if isinstance(child, dict) and child.get('id')
            }
            if created_id not in visible_ids:
                raise RuntimeError(f'Notion did not persist moved block clone {created_id!r}.')
            source_delete_requested = True
            deleted = self._delete(f'{self._base_url}/blocks/{source_block_id}')
            returned_id = deleted.get('id') if isinstance(deleted, dict) else None
            if returned_id and _normalize_notion_id(returned_id) != source_block_id:
                raise RuntimeError(
                    f'Notion deleted unexpected block {returned_id!r}; '
                    f'expected {source_block_id!r}.')
        except Exception:
            created_id = created.get('id') if isinstance(created, dict) else None
            if not source_delete_requested and isinstance(created_id, str) and created_id:
                try:
                    self._delete(f'{self._base_url}/blocks/{_normalize_notion_id(created_id)}')
                except Exception as rollback_exc:
                    lazyllm.LOG.warning(
                        f'Failed to roll back moved Notion block {created_id!r}: '
                        f'{rollback_exc}')
            raise
        return {
            'block': created,
            'block_id': created['id'],
            'source_block_id': source_block_id,
            'block_id_relations': relations,
        }

    def write_doc_blocks(self, document_id: str, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        '''Append native Writer blocks to an existing Notion page.'''
        document_id = _normalize_notion_id(document_id)
        self._validate_write_blocks(blocks)
        relations: List[Dict[str, str]] = []
        for block in blocks:
            self._append_block_tree(document_id, block, relations=relations)
        self._rebind_internal_links(blocks, relations, require_all=False)
        return self._get_doc_blocks_raw(document_id, with_descendants=True)

    def replace_doc_blocks(self, document_id: str, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        '''Replace all root blocks, creating and verifying new content before deletion.'''
        document_id = _normalize_notion_id(document_id)
        self._validate_write_blocks(blocks)
        existing = self._list_children_raw(document_id)
        existing_ids = [_normalize_notion_id(block['id']) for block in existing
                        if isinstance(block, dict) and block.get('id')]
        created_root_ids: List[str] = []
        relations: List[Dict[str, str]] = []
        try:
            for block in blocks:
                created = self._append_block_tree(document_id, block, relations=relations)
                created_root_ids.append(_normalize_notion_id(created['id']))
            visible_ids = {
                _normalize_notion_id(block['id'])
                for block in self._list_children_raw(document_id)
                if isinstance(block, dict) and block.get('id')
            }
            missing = [block_id for block_id in created_root_ids if block_id not in visible_ids]
            if missing:
                raise RuntimeError(f'Notion did not persist created root blocks: {missing!r}.')
            self._rebind_internal_links(blocks, relations, require_all=True)
        except Exception:
            # The original page remains intact. Remove only roots created by this attempt.
            for block_id in reversed(created_root_ids):
                try:
                    self._delete(f'{self._base_url}/blocks/{block_id}')
                except Exception as rollback_exc:
                    lazyllm.LOG.warning(f'Failed to roll back created Notion block {block_id!r}: {rollback_exc}')
            raise

        for block_id in existing_ids:
            deleted = self._delete(f'{self._base_url}/blocks/{block_id}')
            returned_id = deleted.get('id') if isinstance(deleted, dict) else None
            if returned_id and _normalize_notion_id(returned_id) != block_id:
                raise RuntimeError(f'Notion deleted unexpected block {returned_id!r}; expected {block_id!r}.')
        return self._get_doc_blocks_raw(document_id, with_descendants=True)

    def _rebind_internal_links(self, blocks: List[Dict[str, Any]], relations: List[Dict[str, str]], *,
                               require_all: bool) -> None:
        created_by_temporary_id = {
            relation['temporary_block_id']: relation['block_id']
            for relation in relations
            if relation.get('temporary_block_id') and relation.get('block_id')
        }

        def rewrite_rich_text(value: Any, path: str) -> Tuple[List[Dict[str, Any]], bool]:
            if not isinstance(value, list):
                raise TypeError(f'Notion {path} must be a list.')
            rewritten = deepcopy(value)
            changed = False
            for index, item in enumerate(rewritten):
                if not isinstance(item, dict):
                    continue
                text = item.get('text')
                link = text.get('link') if isinstance(text, dict) else None
                target_node_id = link.get('_target_node_id') if isinstance(link, dict) else None
                if not isinstance(target_node_id, str) or not target_node_id:
                    continue
                target_block_id = created_by_temporary_id.get(target_node_id)
                if not target_block_id:
                    if require_all:
                        raise RuntimeError(f'Notion internal reference target {target_node_id!r} '
                                           'was not created during replacement.')
                    continue
                url = link.get('url')
                if not isinstance(url, str) or not url:
                    raise ValueError(f'Notion {path}[{index}] internal link has no URL.')
                link['url'] = urlparse(url)._replace(fragment=target_block_id.replace('-', '')).geturl()
                changed = True
            return rewritten, changed

        def visit(block: Dict[str, Any]) -> None:
            block_type = block.get('type')
            payload = block.get(block_type) if isinstance(block_type, str) else None
            if not isinstance(payload, dict):
                return
            temporary_id = block.get('_temporary_node_id') or block.get('temporary_node_id')
            created_id = created_by_temporary_id.get(temporary_id)
            update: Dict[str, Any] = {}
            for field in ('caption', 'rich_text'):
                if field not in payload:
                    continue
                rewritten, changed = rewrite_rich_text(payload[field], f'{block_type}.{field}')
                if changed:
                    update[field] = self._sanitize_rich_text_array(rewritten, f'{block_type}.{field}')
            if block_type == 'table_row' and isinstance(payload.get('cells'), list):
                cells: List[List[Dict[str, Any]]] = []
                cells_changed = False
                for index, cell in enumerate(payload['cells']):
                    rewritten, changed = rewrite_rich_text(cell, f'table_row.cells[{index}]')
                    cells.append(self._sanitize_rich_text_array(rewritten, f'table_row.cells[{index}]'))
                    cells_changed = cells_changed or changed
                if cells_changed:
                    update['cells'] = cells
            if update:
                if not created_id:
                    raise RuntimeError(f'Notion linked source block {temporary_id!r} was not created.')
                self._patch(f'{self._base_url}/blocks/{created_id}', json={block_type: update})
            children = payload.get('children') or []
            if isinstance(children, list):
                for child in children:
                    if isinstance(child, dict):
                        visit(child)

        for block in blocks:
            visit(block)

    @staticmethod
    def _validate_write_blocks(blocks: List[Dict[str, Any]]) -> None:
        if not isinstance(blocks, list):
            raise TypeError(f'blocks must be a list, got {type(blocks).__name__}.')
        if not blocks:
            raise ValueError('blocks must not be empty.')
        if any(not isinstance(block, dict) for block in blocks):
            raise TypeError('every block must be a dict.')

    def _position_for_index(self, parent_block_id: str, index: Optional[int]) -> Dict[str, Any]:
        if index is None:
            return {'type': 'end'}
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError('index must be a non-negative integer or None.')
        children = self._list_children_raw(parent_block_id)
        if index == 0:
            return {'type': 'start'}
        if index > len(children):
            raise ValueError(f'index {index} exceeds child count {len(children)} for {parent_block_id!r}.')
        if index == len(children):
            return {'type': 'end'}
        previous_id = children[index - 1].get('id')
        if not isinstance(previous_id, str) or not previous_id:
            raise RuntimeError('Notion child response is missing an ID required for positioning.')
        return {'type': 'after_block', 'after_block': {'id': _normalize_notion_id(previous_id)}}

    def _append_block_tree(self, parent_block_id: str, block: Dict[str, Any], *,
                           position: Optional[Dict[str, Any]] = None,
                           relations: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        return self._append_block_trees(
            parent_block_id, [block], position=position, relations=relations)[0]

    def _append_block_trees(self, parent_block_id: str, blocks: List[Dict[str, Any]], *,
                            position: Optional[Dict[str, Any]] = None,
                            relations: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, Any]]:
        parent_block_id = _normalize_notion_id(parent_block_id)
        if not isinstance(blocks, list) or not blocks:
            raise ValueError('blocks must be a non-empty list.')
        if len(blocks) > _PAGE_SIZE:
            raise ValueError(f'Notion can append at most {_PAGE_SIZE} sibling blocks per request.')
        request_blocks: List[Dict[str, Any]] = []
        nested_children: List[List[Dict[str, Any]]] = []
        inline_children_by_root: List[List[Dict[str, Any]]] = []
        for block in blocks:
            prepared_block = self._prepare_block_media(block)
            request_block, children = self._split_native_children(prepared_block)
            block_type = request_block['type']
            inline_children: List[Dict[str, Any]] = []
            if block_type == 'table':
                if not children or children[0].get('type') != 'table_row':
                    raise ValueError('Notion table creation requires at least one table_row child.')
                inline_children = [children.pop(0)]
                inline_request, inline_descendants = self._split_native_children(inline_children[0])
                if inline_descendants:
                    raise ValueError('Notion table_row cannot contain nested block children.')
                request_block[block_type]['children'] = [inline_request]
            request_blocks.append(request_block)
            nested_children.append(children)
            inline_children_by_root.append(inline_children)

        body: Dict[str, Any] = {'children': request_blocks}
        if position is not None:
            body['position'] = deepcopy(position)
        response = self._patch(f'{self._base_url}/blocks/{parent_block_id}/children', json=body)
        results = response.get('results') if isinstance(response, dict) else None
        valid_envelope = isinstance(response, dict) \
            and response.get('object') == 'list' \
            and response.get('type') == 'block'
        created_results = results[:len(request_blocks)] if isinstance(results, list) else []
        valid_results = len(created_results) == len(request_blocks) and all(
            isinstance(item, dict)
            and item.get('object') == 'block'
            and bool(item.get('id'))
            and item.get('type') == request_block.get('type')
            for request_block, item in zip(request_blocks, created_results)
        )
        if not valid_envelope or not valid_results:
            response_summary = {
                'type': response.get('type'),
                'object': response.get('object'),
                'keys': sorted(response),
                'result_count': len(results) if isinstance(results, list) else None,
                'results': [
                    {
                        'id': item.get('id'),
                        'type': item.get('type'),
                        'parent': item.get('parent'),
                        'plain_text': ''.join(
                            str(fragment.get('plain_text') or '')
                            for fragment in (
                                item.get(item.get('type'), {}).get('rich_text', [])
                                if isinstance(item.get(item.get('type')), dict)
                                else []
                            )
                            if isinstance(fragment, dict)
                        ),
                    } if isinstance(item, dict) else {
                        'result_type': type(item).__name__,
                    }
                    for item in results
                ] if isinstance(results, list) else None,
            } if isinstance(response, dict) else {'response_type': type(response).__name__}
            raise RuntimeError(
                'Notion append block children returned an invalid response; '
                f'response={response_summary!r}.')
        created_blocks: List[Dict[str, Any]] = []
        for block, created, children, inline_children in zip(
                blocks, created_results, nested_children, inline_children_by_root):
            created_id = _normalize_notion_id(created['id'])
            normalized_created = {**created, 'id': created_id}
            created_blocks.append(normalized_created)
            self._record_block_relation(block, created_id, relations)
            if inline_children:
                created_inline = self._list_children_raw(created_id)
                if len(created_inline) < len(inline_children):
                    raise RuntimeError('Notion did not return the required inline table row.')
                for source_child, created_child in zip(inline_children, created_inline):
                    child_id = created_child.get('id') if isinstance(created_child, dict) else None
                    if not isinstance(child_id, str) or not child_id:
                        raise RuntimeError('Created Notion table row is missing its block ID.')
                    self._record_block_relation(
                        source_child, _normalize_notion_id(child_id), relations)
            for child in children:
                self._append_block_tree(created_id, child, relations=relations)
        return created_blocks

    def _prepare_block_media(self, block: Dict[str, Any]) -> Dict[str, Any]:
        prepared = deepcopy(block)
        block_type = prepared.get('type')
        if block_type != 'image':
            return prepared
        payload = prepared.get('image')
        if not isinstance(payload, dict):
            return prepared
        private_media = prepared.pop('_media', None)
        data: Optional[bytes] = None
        filename = ''
        content_type = ''
        if isinstance(private_media, dict) and private_media.get('local_path'):
            local_path = Path(str(private_media['local_path'])).expanduser()
            if not local_path.is_file():
                raise ValueError(f'Notion image file does not exist: {local_path}.')
            data = local_path.read_bytes()
            filename = str(private_media.get('file_name') or local_path.name)
            content_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        elif payload.get('type') == 'file':
            file_object = payload.get('file')
            url = file_object.get('url') if isinstance(file_object, dict) else None
            if not isinstance(url, str) or not url:
                raise ValueError('Notion image file payload has no downloadable URL.')
            data, filename, content_type = self._download_notion_file(url)
        if data is None:
            return prepared
        upload_id = self._upload_notion_file(data, filename, content_type)
        for field in ('external', 'file', 'file_upload'):
            payload.pop(field, None)
        payload['type'] = 'file_upload'
        payload['file_upload'] = {'id': upload_id}
        return prepared

    @staticmethod
    def _download_notion_file(url: str) -> Tuple[bytes, str, str]:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '').split(';', 1)[0].strip()
        path_name = Path(unquote(urlparse(url).path)).name
        filename = path_name or 'notion-image'
        if not Path(filename).suffix:
            extension = mimetypes.guess_extension(content_type) or ''
            filename += extension
        return response.content, filename, content_type or 'application/octet-stream'

    def _upload_notion_file(self, data: bytes, filename: str, content_type: str) -> str:
        if not data:
            raise ValueError('Cannot upload an empty Notion image.')
        filename = (filename or 'notion-image').strip() or 'notion-image'
        content_type = content_type or mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        if len(data) <= _SINGLE_PART_UPLOAD_LIMIT:
            created = self._post(
                f'{self._base_url}/file_uploads',
                json={
                    'mode': 'single_part',
                    'filename': filename,
                    'content_type': content_type,
                },
            )
            upload_id = created.get('id') if isinstance(created, dict) else None
            if not isinstance(upload_id, str) or not upload_id:
                raise RuntimeError('Notion create file upload returned no upload ID.')
            uploaded = self._post(
                f'{self._base_url}/file_uploads/{upload_id}/send',
                files={'file': (filename, data, content_type)},
                headers={'Content-Type': None},
            )
        else:
            parts = [
                data[offset:offset + _MULTI_PART_CHUNK_SIZE]
                for offset in range(0, len(data), _MULTI_PART_CHUNK_SIZE)
            ]
            created = self._post(
                f'{self._base_url}/file_uploads',
                json={
                    'mode': 'multi_part',
                    'number_of_parts': len(parts),
                    'filename': filename,
                    'content_type': content_type,
                },
            )
            upload_id = created.get('id') if isinstance(created, dict) else None
            if not isinstance(upload_id, str) or not upload_id:
                raise RuntimeError('Notion create file upload returned no upload ID.')
            uploaded = {}
            for index, part in enumerate(parts, start=1):
                uploaded = self._post(
                    f'{self._base_url}/file_uploads/{upload_id}/send',
                    files={'file': (filename, part, content_type)},
                    data={'part_number': str(index)},
                    headers={'Content-Type': None},
                )
            uploaded = self._post(
                f'{self._base_url}/file_uploads/{upload_id}/complete', json={})
        status = uploaded.get('status') if isinstance(uploaded, dict) else None
        if status not in (None, 'uploaded'):
            raise RuntimeError(f'Notion file upload {upload_id!r} ended with status {status!r}.')
        return upload_id

    @classmethod
    def _split_native_children(
        cls,
        block: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        raw = deepcopy(block)
        block_type = raw.get('type')
        if not isinstance(block_type, str) or not block_type:
            raise ValueError('native Notion block must have a non-empty type.')
        payload = raw.get(block_type)
        if not isinstance(payload, dict):
            raise ValueError(f'native Notion block {block_type!r} must have an object payload.')
        children = payload.pop('children', [])
        if not isinstance(children, list) or any(not isinstance(child, dict) for child in children):
            raise TypeError(f'native Notion block {block_type!r} children must be a list of dicts.')
        raw[block_type] = cls._sanitize_create_payload(block_type, payload)
        for field in (
            'id', 'block_id', 'parent', 'parent_id', 'created_time', 'last_edited_time',
            'created_by', 'last_edited_by', 'archived', 'in_trash', 'has_children',
            'block_type', 'plain_text', '_temporary_node_id', 'temporary_node_id',
        ):
            raw.pop(field, None)
        raw['object'] = 'block'
        return raw, children

    @classmethod
    def _sanitize_create_payload(cls, block_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        allowed = _BLOCK_CREATE_PAYLOAD_FIELDS.get(block_type)
        if allowed is None:
            raise ValueError(f'Notion block type {block_type!r} cannot be created through the API.')
        cleaned = {
            key: deepcopy(value)
            for key, value in payload.items()
            if key in allowed and value is not None
        }
        if block_type == 'synced_block' and 'synced_from' in payload:
            synced_from = payload['synced_from']
            if synced_from is None:
                cleaned['synced_from'] = None
            elif isinstance(synced_from, dict) \
                    and isinstance(synced_from.get('block_id'), str) \
                    and synced_from['block_id']:
                cleaned['synced_from'] = {'block_id': synced_from['block_id']}
            else:
                raise ValueError('Notion synced_block.synced_from requires a block ID or null.')
        if block_type in _MEDIA_BLOCK_TYPES:
            cleaned = cls._sanitize_media_payload(block_type, cleaned)
        if block_type == 'link_to_page':
            cleaned = cls._sanitize_link_to_page(cleaned)
        if block_type in {'paragraph', 'callout'} and 'icon' in cleaned:
            cleaned['icon'] = cls._sanitize_icon(cleaned['icon'], block_type)
        for field in _RICH_TEXT_PAYLOAD_FIELDS & cleaned.keys():
            cleaned[field] = cls._sanitize_rich_text_array(cleaned[field], f'{block_type}.{field}')
        if block_type == 'table_row' and 'cells' in cleaned:
            cells = cleaned['cells']
            if not isinstance(cells, list):
                raise TypeError('Notion table_row.cells must be a list.')
            cleaned['cells'] = [
                cls._sanitize_rich_text_array(cell, f'table_row.cells[{index}]')
                for index, cell in enumerate(cells)
            ]
        return cleaned

    @staticmethod
    def _sanitize_link_to_page(payload: Dict[str, Any]) -> Dict[str, Any]:
        link_type = payload.get('type')
        if link_type is None:
            present_types = [field for field in ('comment_id', 'database_id', 'page_id') if field in payload]
            if len(present_types) == 1:
                link_type = present_types[0]
        if link_type not in {'comment_id', 'database_id', 'page_id'}:
            raise ValueError(f'Notion link_to_page has unsupported type {link_type!r}.')
        object_id = payload.get(link_type)
        if not isinstance(object_id, str) or not object_id:
            raise ValueError(f'Notion link_to_page.{link_type} requires an ID.')
        return {'type': link_type, link_type: object_id}

    @staticmethod
    def _sanitize_media_payload(block_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        media_type = payload.get('type')
        if media_type is None:
            present_types = [field for field in ('external', 'file_upload', 'file') if field in payload]
            if len(present_types) == 1:
                media_type = present_types[0]
                payload['type'] = media_type
        if media_type == 'file':
            internal = payload.pop('file', None)
            url = internal.get('url') if isinstance(internal, dict) else None
            if not isinstance(url, str) or not url:
                raise ValueError(f'Notion {block_type} file payload has no URL to reuse as an external file.')
            payload['type'] = 'external'
            payload['external'] = {'url': url}
        if payload.get('type') == 'external':
            external = payload.get('external')
            if not isinstance(external, dict) or not isinstance(external.get('url'), str) \
                    or not external['url']:
                raise ValueError(f'Notion {block_type} external payload requires a URL.')
            payload['external'] = {'url': external['url']}
            payload.pop('file_upload', None)
        elif payload.get('type') == 'file_upload':
            upload = payload.get('file_upload')
            if not isinstance(upload, dict) or not isinstance(upload.get('id'), str) or not upload['id']:
                raise ValueError(f'Notion {block_type} file_upload payload requires an ID.')
            payload['file_upload'] = {'id': upload['id']}
            payload.pop('external', None)
        else:
            raise ValueError(f'Notion {block_type} payload must use external or file_upload media.')
        return payload

    @staticmethod
    def _sanitize_icon(icon: Any, block_type: str) -> Dict[str, Any]:
        if not isinstance(icon, dict):
            raise TypeError(f'Notion {block_type}.icon must be an object when provided.')
        icon_type = icon.get('type')
        if icon_type == 'file':
            internal = icon.get('file')
            url = internal.get('url') if isinstance(internal, dict) else None
            if not isinstance(url, str) or not url:
                raise ValueError(f'Notion {block_type}.icon file has no reusable URL.')
            return {'type': 'external', 'external': {'url': url}}
        field = icon_type if icon_type in {
            'custom_emoji', 'emoji', 'external', 'file_upload', 'icon',
        } else None
        value = icon.get(field) if field else None
        if field is None or value is None:
            raise ValueError(f'Notion {block_type}.icon has an unsupported type {icon_type!r}.')
        if field in {'custom_emoji', 'file_upload'}:
            if not isinstance(value, dict) or not isinstance(value.get('id'), str) or not value['id']:
                raise ValueError(f'Notion {block_type}.icon {field} requires an ID.')
            value = {'id': value['id']}
        elif field == 'external':
            if not isinstance(value, dict) or not isinstance(value.get('url'), str) or not value['url']:
                raise ValueError(f'Notion {block_type}.icon external requires a URL.')
            value = {'url': value['url']}
        return {'type': icon_type, field: deepcopy(value)}

    @classmethod
    def _sanitize_rich_text_array(cls, value: Any, path: str) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            raise TypeError(f'Notion {path} must be a list.')
        return [cls._sanitize_rich_text_item(item, f'{path}[{index}]') for index, item in enumerate(value)]

    @classmethod
    def _sanitize_rich_text_item(cls, item: Any, path: str) -> Dict[str, Any]:
        if not isinstance(item, dict):
            raise TypeError(f'Notion {path} must be an object.')
        rich_type = item.get('type')
        if rich_type not in {'equation', 'mention', 'text'}:
            raise ValueError(f'Notion {path} has unsupported rich text type {rich_type!r}.')
        typed_value = item.get(rich_type)
        if not isinstance(typed_value, dict):
            raise TypeError(f'Notion {path}.{rich_type} must be an object.')
        cleaned: Dict[str, Any] = {
            'type': rich_type,
        }
        annotations = item.get('annotations')
        if isinstance(annotations, dict):
            cleaned['annotations'] = {
                key: deepcopy(value)
                for key, value in annotations.items()
                if key in _ANNOTATION_FIELDS and value is not None
            }
        if rich_type == 'text':
            content = typed_value.get('content')
            if not isinstance(content, str):
                raise TypeError(f'Notion {path}.text.content must be a string.')
            text = {'content': content}
            if 'link' in typed_value:
                link = typed_value['link']
                if link is not None and (
                    not isinstance(link, dict)
                    or not isinstance(link.get('url'), str)
                    or not link['url']
                ):
                    raise ValueError(f'Notion {path}.text.link requires a URL or null.')
                text['link'] = None if link is None else {'url': link['url']}
            cleaned['text'] = text
        elif rich_type == 'equation':
            expression = typed_value.get('expression')
            if not isinstance(expression, str):
                raise TypeError(f'Notion {path}.equation.expression must be a string.')
            cleaned['equation'] = {'expression': expression}
        else:
            cleaned['mention'] = cls._sanitize_mention(typed_value, path)
        return cleaned

    @staticmethod
    def _sanitize_mention(mention: Dict[str, Any], path: str) -> Dict[str, Any]:
        mention_type = mention.get('type')
        value = mention.get(mention_type) if isinstance(mention_type, str) else None
        if not isinstance(value, dict):
            raise TypeError(f'Notion {path}.mention must contain its typed object.')
        if mention_type in {'database', 'page', 'user'}:
            object_id = value.get('id')
            if not isinstance(object_id, str) or not object_id:
                raise ValueError(f'Notion {path}.mention.{mention_type} requires an ID.')
            return {'type': mention_type, mention_type: {'id': object_id}}
        if mention_type == 'date':
            start = value.get('start')
            if not isinstance(start, str) or not start:
                raise ValueError(f'Notion {path}.mention.date requires a start value.')
            date = {'start': start}
            for field in ('end', 'time_zone'):
                if field in value:
                    field_value = value[field]
                    if field_value is not None and not isinstance(field_value, str):
                        raise TypeError(
                            f'Notion {path}.mention.date.{field} must be a string or null.')
                    date[field] = field_value
            return {'type': 'date', 'date': date}
        if mention_type == 'custom_emoji':
            emoji_id = value.get('id')
            if not isinstance(emoji_id, str) or not emoji_id:
                raise ValueError(f'Notion {path}.mention.custom_emoji requires an ID.')
            custom_emoji = {'id': emoji_id}
            for field in ('name', 'url'):
                if isinstance(value.get(field), str) and value[field]:
                    custom_emoji[field] = value[field]
            return {'type': 'custom_emoji', 'custom_emoji': custom_emoji}
        if mention_type == 'template_mention':
            template_type = value.get('type')
            template_value = value.get(template_type) if isinstance(template_type, str) else None
            if template_type not in {'template_mention_date', 'template_mention_user'} \
                    or not isinstance(template_value, str) or not template_value:
                raise ValueError(f'Notion {path}.mention.template_mention is invalid.')
            return {
                'type': 'template_mention',
                'template_mention': {
                    'type': template_type,
                    template_type: template_value,
                },
            }
        raise ValueError(
            f'Notion {path}.mention type {mention_type!r} cannot be used in a create request.')

    @staticmethod
    def _record_block_relation(
        source: Dict[str, Any],
        created_id: str,
        relations: Optional[List[Dict[str, str]]],
    ) -> None:
        if relations is None:
            return
        temporary_id = source.get('_temporary_node_id') or source.get('temporary_node_id')
        if isinstance(temporary_id, str) and temporary_id:
            relations.append({
                'temporary_block_id': temporary_id,
                'block_id': created_id,
            })

    def update_doc_block_text(self, path: str, block_id: str, new_text: str) -> None:
        block_id = _normalize_notion_id(block_id)
        self._ensure_block_belongs_to_document(path, block_id)
        block = self._retrieve_block(block_id)
        btype = block.get('type', '')
        content = dict(block.get(btype) or {})
        if btype not in {
            'paragraph', 'heading_1', 'heading_2', 'heading_3', 'heading_4',
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

    def _ensure_block_belongs_to_document(self, path: str, block_id: str) -> None:
        kind, document_id = self._resolve_access_ref(path)
        if kind == 'root' or not document_id:
            raise FileNotFoundError(f'Path not found: {path}')
        if kind == 'block' and _normalize_notion_id(document_id) == block_id:
            return
        visible_ids = {
            _normalize_notion_id(block.get('block_id') or block.get('id') or '')
            for block in self.get_doc_blocks(path, with_descendants=True)
            if block.get('block_id') or block.get('id')
        }
        if block_id not in visible_ids:
            raise ValueError(f'block_id {block_id!r} is not under document {path!r}')

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
            if block.get('type') == 'meeting_notes':
                for section_id in self._meeting_notes_child_ids(block):
                    if section_id in visited:
                        continue
                    section = self._retrieve_block(section_id)
                    out.append(section)
                    if section.get('has_children'):
                        out.extend(self._get_doc_blocks_raw(section_id, True, depth + 1, visited))
            elif block.get('has_children') and child_id:
                out.extend(self._get_doc_blocks_raw(child_id, True, depth + 1, visited))
        return out

    @staticmethod
    def _meeting_notes_child_ids(block: Dict[str, Any]) -> List[str]:
        children = (block.get('meeting_notes') or {}).get('children') or {}
        if not isinstance(children, dict):
            return []
        ids: List[str] = []
        for value in children.values():
            if not isinstance(value, str) or not value:
                continue
            child_id = _normalize_notion_id(value)
            if child_id not in ids:
                ids.append(child_id)
        return ids

    def _list_document_references(self, path: str) -> List[Dict[str, Any]]:
        kind, object_id = self._resolve_access_ref(path)
        if kind == 'root' or not object_id:
            return []
        try:
            if kind in ('database', 'data_source'):
                refs = self._list_collection_references(kind, object_id)
            else:
                refs = self._list_page_or_block_references(kind, object_id)
        except Exception as exc:
            lazyllm.LOG.warning(f'_list_document_references: failed to get blocks for {path!r}: {exc}')
            return []
        return self.dedupe_document_references(refs)

    def _list_collection_references(self, kind: str, object_id: str) -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        for item in self._query_collection(kind, object_id):
            refs.extend(self._refs_from_page_properties(item))
            item_id = item.get('id')
            if item.get('object') == 'page' and item_id:
                refs.extend(self._refs_from_blocks(self._get_doc_blocks_raw(item_id, True)))
            elif item.get('object') == 'data_source' and item_id:
                refs.extend(self._safe_property_refs(item_id, self._retrieve_data_source, 'data source'))
        return refs

    def _list_page_or_block_references(self, kind: str, object_id: str) -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        if kind == 'page':
            refs.extend(self._safe_property_refs(object_id, self._retrieve_page, 'page'))
        refs.extend(self._refs_from_blocks(self._get_doc_blocks_raw(object_id, True)))
        return refs

    def _safe_property_refs(self, object_id: str, retrieve: Callable[[str], Dict[str, Any]],
                            label: str) -> List[Dict[str, Any]]:
        try:
            return self._refs_from_page_properties(retrieve(object_id))
        except Exception as exc:
            lazyllm.LOG.debug(f'Failed to get Notion {label} properties for references: {exc}')
            return []

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

    def _paginate_post(self, url: str, payload: Optional[Dict[str, Any]] = None, **kwargs) -> List[Dict[str, Any]]:
        base_payload = dict(payload or {})
        results: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        while True:
            page_payload = dict(base_payload)
            if cursor:
                page_payload['start_cursor'] = cursor
            data = self._post(url, json=page_payload, **kwargs)
            results.extend(data.get('results') or [])
            cursor = data.get('next_cursor') if data.get('has_more') else None
            if not cursor:
                break
        return results

    def _fetch_content(self, path: str, include_references: bool = False) -> bytes:
        kind, object_id = self._resolve_access_ref(path)
        if kind == 'root':
            text = self._search_to_markdown()
        elif kind == 'database':
            text = self._database_to_markdown(object_id)
        elif kind == 'data_source':
            text = self._data_source_to_markdown(object_id)
        elif kind == 'block':
            block = self._retrieve_block(object_id)
            text = '\n'.join(self._block_to_markdown(block, depth=0, visited=set()))
        else:
            text = self._page_to_markdown(object_id)
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
        return self._collection_to_markdown(
            database_id, self._retrieve_database, self._query_database, self._database_title,
            heading_level, depth, visited, include_title, failure_label='Notion database child page',
        )

    def _data_source_to_markdown(self, data_source_id: str, heading_level: int = 1,
                                 depth: int = 0, visited: Optional[Set[str]] = None,
                                 include_title: bool = True) -> str:
        return self._collection_to_markdown(
            data_source_id, self._retrieve_data_source, self._query_data_source, self._data_source_title,
            heading_level, depth, visited, include_title, failure_label='Notion data source child page',
        )

    def _collection_to_markdown(self, object_id: str, retrieve: Callable[[str], Dict[str, Any]],
                                query: Callable[[str], List[Dict[str, Any]]],
                                title_getter: Callable[[Dict[str, Any]], str],
                                heading_level: int, depth: int, visited: Optional[Set[str]],
                                include_title: bool, failure_label: str) -> str:
        visited = visited or set()
        object_id = _normalize_notion_id(object_id)
        if object_id in visited:
            return ''
        visited.add(object_id)

        obj = retrieve(object_id)
        title = title_getter(obj) or object_id
        pages = query(object_id)
        return self._collection_pages_to_markdown(
            title, pages, heading_level, depth, visited, include_title,
            failure_label=failure_label,
        )

    def _collection_pages_to_markdown(self, title: str, pages: List[Dict[str, Any]],
                                      heading_level: int, depth: int, visited: Set[str],
                                      include_title: bool, failure_label: str) -> str:
        heading = '#' * max(1, min(6, heading_level))
        lines = [f'{heading} {title}'] if include_title else []
        for page in pages:
            entry = self._object_to_entry(page)
            page_id = page.get('id', '')
            page_title = entry.get('title') or entry.get('name') or page_id
            child_heading = '#' * max(1, min(6, heading_level + 1))
            lines.append(f'{child_heading} {page_title}')
            if page_id and depth < _MAX_RECURSION_DEPTH:
                try:
                    if page.get('object') == 'data_source':
                        body = self._data_source_to_markdown(
                            page_id, heading_level + 2, depth + 1, visited,
                            include_title=False,
                        )
                    else:
                        body = self._page_to_markdown(
                            page_id, heading_level + 2, depth + 1, visited,
                            include_title=False,
                        )
                    if body:
                        lines.append(body)
                except Exception as exc:
                    lazyllm.LOG.debug(f'Failed to fetch {failure_label} {page_id}: {exc}')
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
        elif btype == 'meeting_notes':
            title = self._rich_text_to_markdown(content.get('title') or [])
            if title:
                lines.append(f'## {title}')
            if depth < _MAX_RECURSION_DEPTH:
                for section_id in self._meeting_notes_child_ids(block):
                    if section_id in visited:
                        continue
                    section = self._retrieve_block(section_id)
                    lines.extend(self._block_to_markdown(section, depth + 1, visited))
                    visited.add(section_id)
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

        if block.get('has_children') and block_id and btype not in (
            'child_page', 'child_database', 'meeting_notes',
        ):
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
        props = page.get('properties') or {}
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
        if btype == 'meeting_notes':
            return NotionFS._rich_text_to_markdown(content.get('title') or [])
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
                          or (block.get('parent') or {}).get('data_source_id')
                          or (block.get('parent') or {}).get('database_id')
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
            if ptype in ('relation',) and isinstance(value, list):
                for rel in value:
                    if isinstance(rel, dict) and rel.get('id'):
                        refs.append({'url': f'notion:/~page/{_normalize_notion_id(rel["id"])}',
                                     'ref_type': 'property_relation', 'kind': 'page'})
            elif isinstance(value, list):
                refs.extend(cls._refs_from_rich_text(value))
            elif ptype == 'url' and isinstance(value, str) and value:
                parsed = _parse_notion_browser_url(value)
                refs.append({'url': value, 'ref_type': 'property_url',
                             'kind': parsed['kind'] if parsed else 'external'})
        return refs

    @classmethod
    def _refs_from_block_content(cls, btype: str, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        for key in ('rich_text', 'caption', 'title'):
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
        props = page.get('properties') or {}
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
    def _data_source_title(data_source: Dict[str, Any]) -> str:
        title = data_source.get('title')
        if isinstance(title, list):
            return NotionFS._rich_text_to_markdown(title)
        return data_source.get('name') or ''

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
            url=page.get('url') or '',
        )

    @staticmethod
    def _db_to_entry(db: Dict[str, Any]) -> Dict[str, Any]:
        did = db.get('id', '')
        title = NotionFS._database_title(db)
        return LazyLLMFSBase._entry(
            name=title or did, ftype='directory', title=title, id=did,
            object=db.get('object', 'database'), notion_path=f'notion:/~database/{did}',
            url=db.get('url') or '',
        )

    @staticmethod
    def _data_source_to_entry(data_source: Dict[str, Any]) -> Dict[str, Any]:
        did = data_source.get('id', '')
        title = NotionFS._data_source_title(data_source)
        return LazyLLMFSBase._entry(
            name=title or did, ftype='directory', title=title, id=did,
            object=data_source.get('object', 'data_source'), notion_path=f'notion:/~data_source/{did}',
            url=data_source.get('url') or '',
        )

    @staticmethod
    def _object_to_entry(obj: Dict[str, Any]) -> Dict[str, Any]:
        if obj.get('object') == 'database':
            return NotionFS._db_to_entry(obj)
        if obj.get('object') == 'data_source':
            return NotionFS._data_source_to_entry(obj)
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
