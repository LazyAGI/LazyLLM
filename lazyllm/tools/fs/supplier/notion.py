# Copyright (c) 2026 LazyAGI. All rights reserved.
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import lazyllm
from lazyllm import config

from ..base import LazyLLMFSBase, CloudFSBufferedFile

config.add('notion_token', str, None, 'NOTION_TOKEN', description='Notion API token (notion-client official env).')

_API_BASE = 'https://api.notion.com/v1'
_NOTION_VERSION = '2022-06-28'
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


class NotionFile(CloudFSBufferedFile):

    def __init__(self, fs: 'NotionFS', path: str, **kwargs) -> None:
        content = fs._fetch_content(path)
        self._notion_content: bytes = content
        super().__init__(fs, path, size=len(content), **kwargs)

    def _fetch_range(self, start: int, end: int) -> bytes:
        return self._notion_content[start:end]


class NotionFS(LazyLLMFSBase):

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
              **kwargs) -> CloudFSBufferedFile:
        if 'b' not in mode:
            raise ValueError('NotionFS only supports binary mode')
        if 'r' in mode:
            return NotionFile(
                self, path, mode=mode, block_size=block_size or self.blocksize,
                autocommit=autocommit, cache_options=cache_options,
            )
        return CloudFSBufferedFile(
            self, path, mode=mode,
            block_size=block_size or self.blocksize,
            autocommit=autocommit, cache_options=cache_options,
        )

    def read_bytes(self, path: str) -> bytes:
        return self._fetch_content(path)

    def cat_file(self, path: str, start: Optional[int] = None, end: Optional[int] = None,
                 **kwargs) -> bytes:
        data = self._fetch_content(path)
        return data[start:end] if (start is not None or end is not None) else data

    def fetch_url(self, url: str) -> bytes:
        parsed = _parse_notion_browser_url(url)
        if not parsed:
            raise ValueError(f'Cannot parse Notion browser URL: {url!r}')
        return self._fetch_content(f'/{parsed["id"]}')

    def mkdir(self, path: str, create_parents: bool = True, **kwargs) -> None:
        parts = self._parse_path(path)
        if len(parts) < 2:
            raise ValueError('path must be /<parent_page_id>/<title>')
        parent_id = _normalize_notion_id(parts[0])
        title = parts[-1]
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
        raise NotImplementedError('NotionFS: Notion official API does not support move')

    def _download_range(self, path: str, start: int, end: int) -> bytes:
        return self._fetch_content(path)[start:end]

    def _upload_data(self, path: str, data: bytes) -> None:
        kind, object_id = self._resolve_ref(path)
        if kind in ('root', 'database'):
            raise ValueError('path must include a page_id or block_id')
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
        self._patch(f'{self._base_url}/blocks/{object_id}/children', json=payload)

    def _platform_supports_webhook(self) -> bool:
        return False

    def _resolve_ref(self, path: str) -> Tuple[str, str]:
        if not path or path == '/':
            return 'root', ''
        if _is_notion_browser_url(path):
            parsed = _parse_notion_browser_url(path)
            assert parsed is not None
            return 'object', parsed['id']

        norm = path.lstrip('/')
        if norm.startswith('~link/'):
            url = unquote(norm[len('~link/'):])
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

    def _fetch_content(self, path: str) -> bytes:
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
