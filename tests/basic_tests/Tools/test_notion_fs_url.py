# Copyright (c) 2026 LazyAGI. All rights reserved.
import unittest
from urllib.parse import quote

from lazyllm.tools.fs.client import _FSRouter, dynamic_fs_config
from lazyllm.tools.fs.supplier.notion import (
    NotionFS,
    _normalize_notion_id,
    _parse_notion_browser_url,
)


PAGE_RAW = '0123456789abcdef0123456789abcdef'
PAGE_ID = '01234567-89ab-cdef-0123-456789abcdef'
CHILD_RAW = 'fedcba9876543210fedcba9876543210'
CHILD_ID = 'fedcba98-7654-3210-fedc-ba9876543210'
DB_RAW = '11111111222233334444555555555555'
DB_ID = '11111111-2222-3333-4444-555555555555'


class TestParseNotionBrowserUrl(unittest.TestCase):

    def test_notion_so_page_url(self):
        result = _parse_notion_browser_url(f'https://www.notion.so/Project-Plan-{PAGE_RAW}?pvs=4')
        self.assertEqual(result, {'kind': 'object', 'id': PAGE_ID})

    def test_notion_site_page_url(self):
        result = _parse_notion_browser_url(f'https://team.notion.site/Project-Plan-{PAGE_RAW}')
        self.assertEqual(result, {'kind': 'object', 'id': PAGE_ID})

    def test_hyphenated_id(self):
        result = _parse_notion_browser_url(f'https://www.notion.so/{PAGE_ID}')
        self.assertEqual(result, {'kind': 'object', 'id': PAGE_ID})

    def test_query_page_id_fallback(self):
        result = _parse_notion_browser_url(f'https://www.notion.so/page?p={PAGE_RAW}')
        self.assertEqual(result, {'kind': 'object', 'id': PAGE_ID})

    def test_invalid_url(self):
        self.assertIsNone(_parse_notion_browser_url('https://example.com/Project-Plan-' + PAGE_RAW))
        self.assertIsNone(_parse_notion_browser_url('not-a-url'))
        self.assertIsNone(_parse_notion_browser_url(''))

    def test_normalize_invalid_id_raises(self):
        with self.assertRaises(ValueError):
            _normalize_notion_id('not-a-notion-id')


class TestFSRouterParseNotion(unittest.TestCase):

    def setUp(self):
        self.router = _FSRouter()

    def test_bare_notion_url_normalized(self):
        protocol, space_id, real_path = self.router._parse(f'https://www.notion.so/Project-Plan-{PAGE_RAW}')
        self.assertEqual(protocol, 'notion')
        self.assertEqual(space_id, 'dynamic')
        self.assertTrue(real_path.startswith('/~link/'))
        self.assertIn('notion.so', real_path)

    def test_notion_tilde_link_gets_dynamic_space(self):
        protocol, space_id, real_path = self.router._parse(f'notion:/~link/{quote(f"https://www.notion.so/{PAGE_RAW}", safe="")}')
        self.assertEqual(protocol, 'notion')
        self.assertEqual(space_id, 'dynamic')
        self.assertTrue(real_path.startswith('/~link/'))

    def test_notion_page_path_gets_dynamic_space(self):
        protocol, space_id, real_path = self.router._parse(f'notion:/~page/{PAGE_RAW}')
        self.assertEqual(protocol, 'notion')
        self.assertEqual(space_id, 'dynamic')
        self.assertEqual(real_path, f'/~page/{PAGE_RAW}')


class TestNotionDynamicAuth(unittest.TestCase):

    def test_dynamic_token_is_required_and_injected(self):
        fs = NotionFS(dynamic_auth=True)
        with self.assertRaises(ValueError):
            fs.inject_auth_header()

        with dynamic_fs_config({'notion': 'secret-token'}):
            headers = fs.inject_auth_header()
        self.assertEqual(headers['Authorization'], 'Bearer secret-token')


class TestNotionMarkdownFetch(unittest.TestCase):

    def _make_fs(self) -> NotionFS:
        fs = NotionFS(token='secret-token')

        def retrieve_page(page_id):
            if page_id == PAGE_ID:
                return {
                    'id': PAGE_ID,
                    'object': 'page',
                    'properties': {
                        'Name': {'type': 'title', 'title': [{'plain_text': 'Project Plan'}]},
                    },
                }
            if page_id == CHILD_ID:
                return {
                    'id': CHILD_ID,
                    'object': 'page',
                    'properties': {
                        'Name': {'type': 'title', 'title': [{'plain_text': 'Child Notes'}]},
                    },
                }
            raise AssertionError(page_id)

        def list_children(block_id):
            if block_id == PAGE_ID:
                return [
                    {
                        'id': 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
                        'type': 'heading_1',
                        'heading_1': {'rich_text': [{'plain_text': 'Overview'}]},
                    },
                    {
                        'id': 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb',
                        'type': 'paragraph',
                        'paragraph': {'rich_text': [{'plain_text': 'Build Notion integration.'}]},
                    },
                    {
                        'id': CHILD_ID,
                        'type': 'child_page',
                        'child_page': {'title': 'Child Notes'},
                        'has_children': True,
                    },
                ]
            if block_id == CHILD_ID:
                return [
                    {
                        'id': 'cccccccc-cccc-cccc-cccc-cccccccccccc',
                        'type': 'bulleted_list_item',
                        'bulleted_list_item': {'rich_text': [{'plain_text': 'Nested item'}]},
                    },
                ]
            return []

        fs._retrieve_page = retrieve_page
        fs._list_children_raw = list_children
        return fs

    def test_fetch_bare_link_as_markdown(self):
        fs = self._make_fs()
        path = '/~link/' + quote(f'https://www.notion.so/Project-Plan-{PAGE_RAW}?pvs=4', safe='')
        text = fs.read_bytes(path).decode('utf-8')
        self.assertIn('# Project Plan', text)
        self.assertIn('# Overview', text)
        self.assertIn('Build Notion integration.', text)
        self.assertIn('## Child Notes', text)
        self.assertIn('- Nested item', text)

    def test_child_page_entry_contains_fetchable_path(self):
        fs = self._make_fs()
        entries = fs.ls(f'/~page/{PAGE_RAW}')
        child = [e for e in entries if e.get('block_type') == 'child_page'][0]
        self.assertEqual(child['notion_path'], f'notion:/~page/{CHILD_ID}')
        self.assertEqual(child['title'], 'Child Notes')


class TestNotionDatabaseMarkdown(unittest.TestCase):

    def test_database_fallback_when_object_is_not_page(self):
        fs = NotionFS(token='secret-token')
        fs._retrieve_page = lambda _page_id: (_ for _ in ()).throw(RuntimeError('not a page'))
        fs._retrieve_database = lambda _database_id: {
            'id': DB_ID,
            'object': 'database',
            'title': [{'plain_text': 'Roadmap DB'}],
        }
        fs._query_database = lambda _database_id: [{
            'id': PAGE_ID,
            'object': 'page',
            'properties': {
                'Name': {'type': 'title', 'title': [{'plain_text': 'Q2 Plan'}]},
            },
        }]
        fs._list_children_raw = lambda _block_id: []
        text = fs.read_bytes(f'/~database/{DB_RAW}').decode('utf-8')
        self.assertIn('# Roadmap DB', text)
        self.assertIn('## Q2 Plan', text)
        linked = fs.fetch_url(f'https://www.notion.so/Roadmap-{DB_RAW}').decode('utf-8')
        self.assertIn('# Roadmap DB', linked)


if __name__ == '__main__':
    unittest.main()
