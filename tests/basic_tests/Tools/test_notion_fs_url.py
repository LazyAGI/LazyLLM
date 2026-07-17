# Copyright (c) 2026 LazyAGI. All rights reserved.
import unittest
from urllib.parse import quote

import requests

from lazyllm import init_session, locals as lazyllm_locals
from lazyllm.tools.fs.client import _FSRouter, dynamic_fs_config
from lazyllm.tools.fs.supplier.notion import (
    NotionFS,
    _adapt_ls_tool_input,
    _normalize_notion_id,
    _parse_notion_browser_url,
)
from lazyllm.tools.fs.base import LinkDocumentFSBase
from lazyllm.tools.agent.toolsManager import ToolManager
from tests.basic_tests.Tools.fs_test_utils import load_fs_docs_only


PAGE_RAW = '0123456789abcdef0123456789abcdef'
PAGE_ID = '01234567-89ab-cdef-0123-456789abcdef'
CHILD_RAW = 'fedcba9876543210fedcba9876543210'
CHILD_ID = 'fedcba98-7654-3210-fedc-ba9876543210'
DB_RAW = '11111111222233334444555555555555'
DB_ID = '11111111-2222-3333-4444-555555555555'
BLOCK_RAW = 'aaaaaaaaaaaabbbbccccdddddddddddd'
BLOCK_ID = 'aaaaaaaa-aaaa-bbbb-cccc-dddddddddddd'


def _notion_object_not_found() -> requests.HTTPError:
    response = requests.Response()
    response.status_code = 404
    response.reason = 'Not Found'
    response._content = b'{"code":"object_not_found"}'
    response.headers['Content-Type'] = 'application/json'
    return requests.HTTPError('404 Not Found', response=response)


class TestParseNotionBrowserUrl(unittest.TestCase):

    def test_notion_so_page_url(self):
        result = _parse_notion_browser_url(f'https://www.notion.so/Project-Plan-{PAGE_RAW}?pvs=4')
        self.assertEqual(result, {'kind': 'object', 'id': PAGE_ID})

    def test_notion_site_page_url(self):
        result = _parse_notion_browser_url(f'https://team.notion.site/Project-Plan-{PAGE_RAW}')
        self.assertEqual(result, {'kind': 'object', 'id': PAGE_ID})

    def test_app_notion_com_page_url(self):
        result = _parse_notion_browser_url(f'https://app.notion.com/p/{PAGE_RAW}?source=copy_link')
        self.assertEqual(result, {'kind': 'object', 'id': PAGE_ID, 'mode_hint': 'page'})

    def test_hyphenated_id(self):
        result = _parse_notion_browser_url(f'https://www.notion.so/{PAGE_ID}')
        self.assertEqual(result, {'kind': 'object', 'id': PAGE_ID})

    def test_query_page_id_fallback(self):
        result = _parse_notion_browser_url(f'https://www.notion.so/page?p={PAGE_RAW}')
        self.assertEqual(result, {'kind': 'object', 'id': PAGE_ID, 'mode_hint': 'page'})

    def test_query_database_id_hint(self):
        result = _parse_notion_browser_url(f'https://www.notion.so/db?database_id={DB_RAW}')
        self.assertEqual(result, {'kind': 'object', 'id': DB_ID, 'mode_hint': 'database'})

    def test_block_fragment_is_ignored_for_document_mode(self):
        result = _parse_notion_browser_url(
            f'https://www.notion.so/Project-Plan-{PAGE_RAW}#{BLOCK_RAW}'
        )
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

    def test_app_notion_com_bare_url_normalized(self):
        protocol, space_id, real_path = self.router._parse(f'https://app.notion.com/p/{PAGE_RAW}?source=copy_link')
        self.assertEqual(protocol, 'notion')
        self.assertEqual(space_id, 'dynamic')
        self.assertTrue(real_path.startswith('/~link/'))
        self.assertIn('notion.com', real_path)

    def test_notion_tilde_link_gets_dynamic_space(self):
        encoded_url = quote(f'https://www.notion.so/{PAGE_RAW}', safe='')
        protocol, space_id, real_path = self.router._parse(f'notion:/~link/{encoded_url}')
        self.assertEqual(protocol, 'notion')
        self.assertEqual(space_id, 'dynamic')
        self.assertTrue(real_path.startswith('/~link/'))

    def test_notion_page_path_gets_dynamic_space(self):
        protocol, space_id, real_path = self.router._parse(f'notion:/~page/{PAGE_RAW}')
        self.assertEqual(protocol, 'notion')
        self.assertEqual(space_id, 'dynamic')
        self.assertEqual(real_path, f'/~page/{PAGE_RAW}')

    def test_notion_data_source_path_gets_dynamic_space(self):
        protocol, space_id, real_path = self.router._parse(f'notion:/~data_source/{DB_RAW}')
        self.assertEqual(protocol, 'notion')
        self.assertEqual(space_id, 'dynamic')
        self.assertEqual(real_path, f'/~data_source/{DB_RAW}')

    def test_link_path_helper_round_trip(self):
        url = f'https://www.notion.so/Project-Plan-{PAGE_RAW}?pvs=4'
        path = LinkDocumentFSBase.to_link_path(url)
        self.assertTrue(LinkDocumentFSBase.is_link_path(path))
        self.assertEqual(LinkDocumentFSBase.decode_link_path(path), url)

    def test_document_public_apis_are_composed_by_base(self):
        apis = LinkDocumentFSBase.build_public_apis(extra=['search'], exclude=['copy'])
        self.assertIn('resolve_link', apis)
        self.assertIn('read_with_references', apis)
        self.assertIn('get_doc_blocks', apis)
        self.assertIn('search', apis)
        self.assertNotIn('copy', apis)


class TestNotionDynamicAuth(unittest.TestCase):

    def test_dynamic_token_is_required_and_injected(self):
        fs = NotionFS(dynamic_auth=True)
        with self.assertRaises(ValueError):
            fs.inject_auth_header()

        with dynamic_fs_config({'notion': 'secret-token'}):
            headers = fs.inject_auth_header()
        self.assertEqual(headers['Authorization'], 'Bearer secret-token')


class TestNotionToolRegistration(unittest.TestCase):

    def setUp(self):
        init_session()
        lazyllm_locals['_lazyllm_agent'] = {'workspace': {}}
        load_fs_docs_only(NotionFS.search)

    def test_document_flow_tools_are_registered(self):
        fs = NotionFS(dynamic_auth=True)
        manager = ToolManager([(fs, lambda _instance: 'secret-token')])
        names = {item['function']['name'] for item in manager.tools_description}

        self.assertEqual(names, {'get_NotionFS_methods'})
        manager._tool_call['get_NotionFS_methods']({})
        names = {item['function']['name'] for item in manager.tools_description}

        self.assertIn('NotionFS_ls', names)
        self.assertIn('NotionFS_search', names)
        self.assertIn('NotionFS_find', names)
        self.assertIn('NotionFS_resolve_link', names)
        self.assertIn('NotionFS_read_with_references', names)
        self.assertIn('NotionFS_get_doc_blocks', names)
        self.assertNotIn('NotionFS_copy', names)

    def test_ls_tool_defaults_empty_inputs_to_root(self):
        self.assertEqual(_adapt_ls_tool_input({}), {'path': '/'})
        self.assertEqual(_adapt_ls_tool_input({'path': ''}), {'path': '/'})
        self.assertEqual(_adapt_ls_tool_input(''), {'path': '/'})

        fs = NotionFS(dynamic_auth=True)
        manager = ToolManager([(fs, lambda _instance: 'secret-token')])
        manager._tool_call['get_NotionFS_methods']({})
        tool = manager._tool_call['NotionFS_ls']
        self.assertEqual(tool._validate_input({}), {'path': '/'})


class TestNotionSearch(unittest.TestCase):

    def test_search_builds_official_title_query_payload(self):
        fs = NotionFS(token='secret-token', skip_instance_cache=True)
        calls = []

        def paginate_post(url, payload):
            calls.append((url, payload))
            return [{
                'id': PAGE_ID,
                'object': 'page',
                'properties': {
                    'Name': {'type': 'title', 'title': [{'plain_text': 'Project Plan'}]},
                },
                'last_edited_time': '2026-06-01T00:00:00.000Z',
            }]

        fs._paginate_post = paginate_post
        results = fs.search('Project', object_type='page', limit=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['title'], 'Project Plan')
        self.assertEqual(results[0]['notion_path'], f'notion:/~page/{PAGE_ID}')
        url, payload = calls[0]
        self.assertTrue(url.endswith('/search'))
        self.assertEqual(payload['query'], 'Project')
        self.assertEqual(payload['page_size'], 5)
        self.assertEqual(payload['filter'], {'property': 'object', 'value': 'page'})
        self.assertEqual(payload['sort'], {'direction': 'descending', 'timestamp': 'last_edited_time'})

    def test_search_rejects_empty_query(self):
        fs = NotionFS(token='secret-token', skip_instance_cache=True)
        with self.assertRaises(ValueError):
            fs.search('')

    def test_search_can_be_scoped_to_data_source(self):
        fs = NotionFS(token='secret-token', skip_instance_cache=True)
        fs._query_data_source = lambda data_source_id: [
            {
                'id': PAGE_ID,
                'object': 'page',
                'properties': {
                    'Name': {'type': 'title', 'title': [{'plain_text': 'Project Plan'}]},
                },
            },
            {
                'id': CHILD_ID,
                'object': 'page',
                'properties': {
                    'Name': {'type': 'title', 'title': [{'plain_text': 'Meeting Notes'}]},
                },
            },
        ]

        results = fs.search(
            'Project',
            scope=f'notion:/~data_source/{DB_RAW}',
            title_pattern='Plan$',
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['title'], 'Project Plan')
        self.assertEqual(results[0]['notion_path'], f'notion:/~page/{PAGE_ID}')

    def test_find_can_be_scoped_to_data_source(self):
        fs = NotionFS(token='secret-token', skip_instance_cache=True)
        fs._query_data_source = lambda data_source_id: [
            {
                'id': PAGE_ID,
                'object': 'page',
                'properties': {
                    'Name': {'type': 'title', 'title': [{'plain_text': 'Project Plan'}]},
                },
            },
            {
                'id': CHILD_ID,
                'object': 'page',
                'properties': {
                    'Name': {'type': 'title', 'title': [{'plain_text': 'Meeting Notes'}]},
                },
            },
        ]

        results = fs.find('^Project', scope=f'notion:/~data_source/{DB_RAW}')

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['title'], 'Project Plan')


class TestNotionMarkdownFetch(unittest.TestCase):

    def _make_fs(self) -> NotionFS:
        fs = NotionFS(token='markdown-fetch-token', skip_instance_cache=True)

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
        fs._retrieve_page_markdown = lambda _page_id: None
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

    def test_include_references_adds_footer(self):
        fs = self._make_fs()

        def list_children(block_id):
            if block_id == PAGE_ID:
                return [{
                    'id': BLOCK_ID,
                    'type': 'paragraph',
                    'paragraph': {
                        'rich_text': [
                            {
                                'plain_text': 'linked',
                                'href': f'https://www.notion.so/Child-{CHILD_RAW}',
                            },
                            {
                                'type': 'mention',
                                'plain_text': 'Child Notes',
                                'mention': {'type': 'page', 'page': {'id': CHILD_ID}},
                            },
                        ],
                    },
                }]
            return []

        fs._list_children_raw = list_children
        text = fs.read_bytes(f'/~page/{PAGE_RAW}', include_references=True).decode('utf-8')
        self.assertIn('--- lazyllm-notion-references ---', text)
        self.assertIn(f'https://www.notion.so/Child-{CHILD_RAW}', text)
        self.assertIn(f'notion:/~page/{CHILD_ID}', text)


class TestNotionDatabaseMarkdown(unittest.TestCase):

    def test_database_fallback_when_object_is_not_page(self):
        fs = NotionFS(token='database-markdown-token', skip_instance_cache=True)
        fs._retrieve_page = lambda _page_id: (_ for _ in ()).throw(_notion_object_not_found())
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
        fs._retrieve_page_markdown = lambda _page_id: None
        text = fs.read_bytes(f'/~database/{DB_RAW}').decode('utf-8')
        self.assertIn('# Roadmap DB', text)
        self.assertIn('## Q2 Plan', text)
        linked = fs.fetch_url(f'https://www.notion.so/Roadmap-{DB_RAW}').decode('utf-8')
        self.assertIn('# Roadmap DB', linked)

    def test_database_url_doc_blocks_resolves_to_database(self):
        fs = NotionFS(token='database-doc-blocks-token', skip_instance_cache=True)
        fs._retrieve_page = lambda _page_id: (_ for _ in ()).throw(_notion_object_not_found())
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

        blocks = fs.get_doc_blocks(f'https://www.notion.so/Roadmap-{DB_RAW}')

        self.assertEqual(blocks[0]['block_id'], PAGE_ID)
        self.assertEqual(blocks[0]['block_type'], 'child_page')
        self.assertEqual(blocks[0]['parent_id'], DB_ID)
        self.assertEqual(blocks[0]['plain_text'], 'Q2 Plan')


class TestNotionWriteAndBlocks(unittest.TestCase):

    def test_replace_page_markdown_uses_markdown_endpoint(self):
        fs = NotionFS(token='mkdir-db-token', skip_instance_cache=True)
        calls = []
        fs._patch = lambda url, **kwargs: calls.append((url, kwargs)) or {'ok': True}
        fs.replace_page_markdown(PAGE_RAW, '# Hello')

        self.assertEqual(calls[0][0], f'https://api.notion.com/v1/pages/{PAGE_ID}/markdown')
        self.assertEqual(calls[0][1]['headers']['Notion-Version'], '2026-03-11')
        self.assertEqual(calls[0][1]['json']['type'], 'replace_content')
        self.assertEqual(calls[0][1]['json']['replace_content']['new_str'], '# Hello')

    def test_upload_markdown_replaces_page_markdown(self):
        fs = NotionFS(token='query-db-token', skip_instance_cache=True)
        calls = []
        fs.replace_page_markdown = lambda page_id, markdown, **kwargs: calls.append((page_id, markdown, kwargs))
        fs._upload_data(f'/~page/{PAGE_RAW}', b'# New body', content_type='markdown')
        self.assertEqual(calls, [(PAGE_ID, '# New body', {'allow_deleting_content': True})])

    def test_upload_text_appends_all_chunks(self):
        fs = NotionFS(token='mkdir-ds-token', skip_instance_cache=True)
        calls = []
        fs._patch = lambda url, **kwargs: calls.append((url, kwargs)) or {}
        fs._upload_data(f'/~block/{BLOCK_RAW}', ('a' * 2500).encode('utf-8'))

        self.assertEqual(calls[0][0], f'https://api.notion.com/v1/blocks/{BLOCK_ID}/children')
        children = calls[0][1]['json']['children']
        self.assertEqual(len(children), 2)
        self.assertEqual(len(children[0]['paragraph']['rich_text'][0]['text']['content']), 2000)
        self.assertEqual(len(children[1]['paragraph']['rich_text'][0]['text']['content']), 500)

    def test_move_posts_move_and_renames(self):
        fs = NotionFS(token='secret-token')
        calls = []
        fs._post = lambda url, **kwargs: calls.append(('post', url, kwargs)) or {}
        fs.update_page_title = lambda page_id, title: calls.append(('rename', page_id, title))

        fs.move(f'/~page/{PAGE_RAW}', f'/~page/{CHILD_RAW}/New Title')

        self.assertEqual(calls[0][1], f'https://api.notion.com/v1/pages/{PAGE_ID}/move')
        self.assertEqual(calls[0][2]['json']['parent']['page_id'], CHILD_ID)
        self.assertEqual(calls[0][2]['headers']['Notion-Version'], '2026-03-11')
        self.assertEqual(calls[1], ('rename', PAGE_ID, 'New Title'))

    def test_mkdir_under_database_uses_data_source_parent(self):
        fs = NotionFS(token='secret-token')
        calls = []
        data_source_id = '22222222-3333-4444-5555-666666666666'
        fs._get = lambda url, **kwargs: (
            {
                'id': DB_ID,
                'object': 'database',
                'data_sources': [{'id': data_source_id}],
            } if url.endswith(f'/databases/{DB_ID}') else {
                'id': data_source_id,
                'object': 'data_source',
                'properties': {'Name': {'type': 'title'}},
            } if url.endswith(f'/data_sources/{data_source_id}') else {}
        )
        fs._post = lambda url, **kwargs: calls.append((url, kwargs)) or {}

        fs.mkdir(f'/~database/{DB_RAW}/New Page')

        self.assertEqual(calls[0][0], 'https://api.notion.com/v1/pages')
        self.assertEqual(calls[0][1]['json']['parent'], {'data_source_id': data_source_id})
        self.assertEqual(calls[0][1]['headers']['Notion-Version'], '2026-03-11')
        self.assertEqual(
            calls[0][1]['json']['properties']['Name']['title'][0]['text']['content'],
            'New Page',
        )

    def test_query_database_uses_data_source_endpoint(self):
        fs = NotionFS(token='secret-token')
        calls = []
        data_source_id = '22222222-3333-4444-5555-666666666666'
        fs._get = lambda url, **kwargs: {
            'id': DB_ID,
            'object': 'database',
            'data_sources': [{'id': data_source_id}],
        } if url.endswith(f'/databases/{DB_ID}') else {}
        fs._post = lambda url, **kwargs: calls.append((url, kwargs)) or {
            'results': [],
            'has_more': False,
        }

        self.assertEqual(fs._query_database(DB_ID), [])

        self.assertEqual(calls[0][0], f'https://api.notion.com/v1/data_sources/{data_source_id}/query')
        self.assertEqual(calls[0][1]['headers']['Notion-Version'], '2026-03-11')

    def test_mkdir_under_explicit_data_source(self):
        fs = NotionFS(token='secret-token')
        calls = []
        data_source_id = '22222222-3333-4444-5555-666666666666'
        fs._get = lambda url, **kwargs: {
            'id': data_source_id,
            'object': 'data_source',
            'properties': {'Task': {'type': 'title'}},
        } if url.endswith(f'/data_sources/{data_source_id}') else {}
        fs._post = lambda url, **kwargs: calls.append((url, kwargs)) or {}

        fs.mkdir(f'/~data_source/{data_source_id}/New Task')

        self.assertEqual(calls[0][1]['json']['parent'], {'data_source_id': data_source_id})
        self.assertEqual(
            calls[0][1]['json']['properties']['Task']['title'][0]['text']['content'],
            'New Task',
        )

    def test_upload_data_rejects_data_source_path(self):
        fs = NotionFS(token='secret-token')

        with self.assertRaises(ValueError):
            fs._upload_data(f'/~data_source/{DB_RAW}', b'body')

    def test_move_to_database_uses_data_source_parent(self):
        fs = NotionFS(token='secret-token')
        calls = []
        data_source_id = '22222222-3333-4444-5555-666666666666'
        fs._post = lambda url, **kwargs: calls.append(('post', url, kwargs)) or {}
        fs.update_page_title = lambda page_id, title: calls.append(('rename', page_id, title))
        fs._get = lambda url, **kwargs: {
            'id': DB_ID,
            'object': 'database',
            'data_sources': [{'id': data_source_id}],
        } if url.endswith(f'/databases/{DB_ID}') else {}

        fs.move(f'/~page/{PAGE_RAW}', f'/~database/{DB_RAW}/New Title')

        self.assertEqual(calls[0][1], f'https://api.notion.com/v1/pages/{PAGE_ID}/move')
        self.assertEqual(
            calls[0][2]['json']['parent'],
            {'type': 'data_source_id', 'data_source_id': data_source_id},
        )
        self.assertEqual(calls[0][2]['headers']['Notion-Version'], '2026-03-11')
        self.assertEqual(calls[1], ('rename', PAGE_ID, 'New Title'))

    def test_rmdir_rejects_data_source_paths(self):
        fs = NotionFS(token='secret-token')
        fs._retrieve_page = lambda _page_id: (_ for _ in ()).throw(_notion_object_not_found())
        fs._retrieve_database = lambda _database_id: (_ for _ in ()).throw(_notion_object_not_found())
        fs._retrieve_data_source = lambda _data_source_id: {
            'id': DB_ID,
            'object': 'data_source',
            'properties': {'Task': {'type': 'title'}},
        }

        with self.assertRaises(NotImplementedError):
            fs.rmdir(f'/{DB_RAW}')

    def test_get_document_id_and_doc_blocks(self):
        fs = NotionFS(token='secret-token')
        fs._list_children_raw = lambda block_id: [{
            'id': BLOCK_ID,
            'type': 'paragraph',
            'parent': {'type': 'page_id', 'page_id': PAGE_ID},
            'paragraph': {'rich_text': [{'plain_text': 'Hello'}]},
        }] if block_id == PAGE_ID else []

        self.assertEqual(fs.get_document_id(f'/~page/{PAGE_RAW}'), PAGE_ID)
        blocks = fs.get_doc_blocks(f'/~page/{PAGE_RAW}')
        self.assertEqual(blocks[0]['block_id'], BLOCK_ID)
        self.assertEqual(blocks[0]['block_type'], 'paragraph')
        self.assertEqual(blocks[0]['plain_text'], 'Hello')
        self.assertEqual(blocks[0]['parent_id'], PAGE_ID)

    def test_update_doc_block_text_patches_rich_text(self):
        fs = NotionFS(token='secret-token')
        calls = []
        fs._list_children_raw = lambda block_id: [{
            'id': BLOCK_ID,
            'type': 'paragraph',
            'parent': {'type': 'page_id', 'page_id': PAGE_ID},
            'paragraph': {'rich_text': [{'plain_text': 'Old'}]},
        }] if block_id == PAGE_ID else []
        fs._retrieve_block = lambda block_id: {
            'id': block_id,
            'type': 'paragraph',
            'paragraph': {'rich_text': [{'plain_text': 'Old'}]},
        }
        fs._patch = lambda url, **kwargs: calls.append((url, kwargs)) or {}
        fs.update_doc_block_text(f'/~page/{PAGE_RAW}', BLOCK_RAW, 'New text')

        self.assertEqual(calls[0][0], f'https://api.notion.com/v1/blocks/{BLOCK_ID}')
        rich = calls[0][1]['json']['paragraph']['rich_text']
        self.assertEqual(rich[0]['text']['content'], 'New text')

    def test_update_doc_block_text_rejects_block_outside_document(self):
        fs = NotionFS(token='secret-token')
        fs._list_children_raw = lambda _block_id: []

        with self.assertRaises(ValueError):
            fs.update_doc_block_text(f'/~page/{PAGE_RAW}', BLOCK_RAW, 'New text')

    def test_rm_block_uses_block_delete_endpoint(self):
        fs = NotionFS(token='secret-token')
        calls = []
        fs._delete = lambda url, **kwargs: calls.append((url, kwargs)) or {}

        fs.rm_file(f'/~block/{BLOCK_RAW}')

        self.assertEqual(calls, [(f'https://api.notion.com/v1/blocks/{BLOCK_ID}', {})])

    def test_insert_page_markdown_uses_content_field(self):
        fs = NotionFS(token='secret-token')
        calls = []
        fs._patch = lambda url, **kwargs: calls.append((url, kwargs)) or {}

        fs.insert_page_markdown(PAGE_RAW, '## Added')

        self.assertEqual(calls[0][0], f'https://api.notion.com/v1/pages/{PAGE_ID}/markdown')
        self.assertEqual(calls[0][1]['json']['insert_content']['content'], '## Added')

    def test_resolve_notion_ref(self):
        fs = NotionFS(token='secret-token')
        fs._retrieve_page = lambda _page_id: {
            'id': PAGE_ID,
            'object': 'page',
            'properties': {'Name': {'type': 'title', 'title': [{'plain_text': 'Project Plan'}]}},
        }
        result = fs.resolve_notion_ref(f'https://www.notion.so/Project-{PAGE_RAW}')
        self.assertEqual(result['object_id'], PAGE_ID)
        self.assertEqual(result['object_type'], 'page')
        self.assertEqual(result['title'], 'Project Plan')

    def test_resolve_notion_ref_supports_data_source(self):
        fs = NotionFS(token='secret-token')
        fs._retrieve_page = lambda _page_id: (_ for _ in ()).throw(_notion_object_not_found())
        fs._retrieve_database = lambda _database_id: (_ for _ in ()).throw(_notion_object_not_found())
        fs._retrieve_data_source = lambda _data_source_id: {
            'id': DB_ID,
            'object': 'data_source',
            'title': [{'plain_text': 'Tasks'}],
        }

        result = fs.resolve_notion_ref(f'notion:/~data_source/{DB_RAW}')

        self.assertEqual(result['object_id'], DB_ID)
        self.assertEqual(result['object_type'], 'data_source')
        self.assertEqual(result['title'], 'Tasks')
        self.assertEqual(result['notion_path'], f'notion:/~data_source/{DB_ID}')

    def test_resolve_link_returns_standard_fields(self):
        fs = NotionFS(token='secret-token')
        fs._retrieve_page = lambda _page_id: {
            'id': PAGE_ID,
            'object': 'page',
            'properties': {'Name': {'type': 'title', 'title': [{'plain_text': 'Project Plan'}]}},
        }
        result = fs.resolve_link(f'https://www.notion.so/Project-{PAGE_RAW}')
        self.assertEqual(result['provider'], 'notion')
        self.assertEqual(result['object_id'], PAGE_ID)
        self.assertEqual(result['object_type'], 'page')
        self.assertEqual(result['title'], 'Project Plan')

    def test_read_with_references_uses_standard_document_flow(self):
        fs = NotionFS(token='secret-token')
        fs.read_bytes = lambda path, include_references=False: (
            f'{path}|refs={include_references}'.encode('utf-8')
        )

        self.assertEqual(fs.read_with_references('/~page/' + PAGE_RAW),
                         '/~page/' + PAGE_RAW + '|refs=True')


if __name__ == '__main__':
    unittest.main()
