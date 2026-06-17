# Copyright (c) 2026 LazyAGI. All rights reserved.
import unittest
from unittest.mock import MagicMock, patch

from lazyllm import init_session, locals as lazyllm_locals
from lazyllm.tools.fs.supplier.feishu import (
    _parse_feishu_browser_url,
    _SPACE_ID_DYNAMIC,
    FeishuFS,
    FeishuWikiFS,
)
from lazyllm.tools.fs.client import _FSRouter, _feishu_needs_wiki, _FEISHU_WIKI_PATH_PREFIXES
from lazyllm.tools.agent.toolsManager import ToolManager


class TestParseFeishuBrowserUrl(unittest.TestCase):

    def test_wiki_url(self):
        result = _parse_feishu_browser_url('https://sensetime.feishu.cn/wiki/MCjOwGxwSimztPkO5X6cv8uxnwb')
        self.assertEqual(result, {'kind': 'wiki_node', 'token': 'MCjOwGxwSimztPkO5X6cv8uxnwb'})

    def test_wiki_url_with_query(self):
        result = _parse_feishu_browser_url(
            'https://sensetime.feishu.cn/wiki/MCjOwGxwSimztPkO5X6cv8uxnwb?renamingWikiNode=true'
        )
        self.assertEqual(result, {'kind': 'wiki_node', 'token': 'MCjOwGxwSimztPkO5X6cv8uxnwb'})

    def test_docx_url(self):
        result = _parse_feishu_browser_url('https://company.feishu.cn/docx/AbCdEfGh12345678')
        self.assertEqual(result, {'kind': 'docx', 'token': 'AbCdEfGh12345678'})

    def test_docs_url(self):
        result = _parse_feishu_browser_url('https://company.larksuite.com/docs/doccnOldStyleToken')
        self.assertEqual(result, {'kind': 'doc', 'token': 'doccnOldStyleToken'})

    def test_wiki_with_fragment(self):
        result = _parse_feishu_browser_url(
            'https://company.feishu.cn/wiki/SomeToken123#anchor'
        )
        self.assertEqual(result, {'kind': 'wiki_node', 'token': 'SomeToken123'})

    def test_invalid_url(self):
        self.assertIsNone(_parse_feishu_browser_url('https://feishu.cn/'))
        self.assertIsNone(_parse_feishu_browser_url('not-a-url'))
        self.assertIsNone(_parse_feishu_browser_url(''))

    def test_uppercase_path(self):
        result = _parse_feishu_browser_url('https://xxx.feishu.cn/WIKI/Token999')
        self.assertEqual(result, {'kind': 'wiki_node', 'token': 'Token999'})


class TestSpaceIdDynamic(unittest.TestCase):

    def test_dynamic_constant(self):
        self.assertEqual(_SPACE_ID_DYNAMIC, 'dynamic')

    @patch('lazyllm.tools.fs.supplier.feishu.FeishuWikiFS.__init__', return_value=None)
    def test_feishu_fs_new_dynamic_routes_to_wiki(self, mock_init):
        # FeishuFS.__new__ with space_id='dynamic' should instantiate FeishuWikiFS with space_id=''
        with patch.object(FeishuWikiFS, '__init__', return_value=None) as wiki_init:
            instance = FeishuFS.__new__(
                FeishuFS,
                app_id='test_id', app_secret='test_secret',
                space_id='dynamic', dynamic_auth=True,
            )
            self.assertIsInstance(instance, FeishuWikiFS)
            call_kwargs = wiki_init.call_args[1]
            self.assertEqual(call_kwargs.get('space_id'), '')

    @patch('lazyllm.tools.fs.supplier.feishu.FeishuWikiFS.__init__', return_value=None)
    def test_feishu_fs_new_real_space_id_unchanged(self, mock_init):
        with patch.object(FeishuWikiFS, '__init__', return_value=None) as wiki_init:
            instance = FeishuFS.__new__(
                FeishuFS,
                app_id='test_id', app_secret='test_secret',
                space_id='wikcnRealSpaceId', dynamic_auth=True,
            )
            self.assertIsInstance(instance, FeishuWikiFS)
            call_kwargs = wiki_init.call_args[1]
            self.assertEqual(call_kwargs.get('space_id'), 'wikcnRealSpaceId')

    def test_feishu_fs_new_no_space_id_returns_drive(self):
        with patch('lazyllm.tools.fs.supplier.feishu.FeishuFSBase.__init__', return_value=None):
            instance = FeishuFS.__new__(FeishuFS, dynamic_auth=True)
            self.assertIs(type(instance), FeishuFS)


class TestEffectiveSpaceId(unittest.TestCase):

    def _make_wiki_fs(self, space_id: str = '') -> FeishuWikiFS:
        fs = object.__new__(FeishuWikiFS)
        fs._space_id = space_id
        return fs

    def test_returns_constructor_space_id_first(self):
        fs = self._make_wiki_fs('wikcnABC')
        with patch('lazyllm.tools.fs.supplier.feishu.lazyllm_globals') as mock_globals:
            mock_globals.config = {'feishu_wiki_space_id': 'wikcnOther'}
            self.assertEqual(fs._effective_space_id(), 'wikcnABC')

    def test_falls_back_to_globals_config(self):
        fs = self._make_wiki_fs('')
        with patch('lazyllm.tools.fs.supplier.feishu.lazyllm_globals') as mock_globals:
            mock_globals.config = {'feishu_wiki_space_id': 'wikcnFromGlobals'}
            self.assertEqual(fs._effective_space_id(), 'wikcnFromGlobals')

    def test_returns_empty_when_both_absent(self):
        fs = self._make_wiki_fs('')
        with patch('lazyllm.tools.fs.supplier.feishu.lazyllm_globals') as mock_globals:
            mock_globals.config = {'feishu_wiki_space_id': None}
            self.assertEqual(fs._effective_space_id(), '')

    def test_require_space_id_raises_when_empty(self):
        fs = self._make_wiki_fs('')
        with patch('lazyllm.tools.fs.supplier.feishu.lazyllm_globals') as mock_globals:
            mock_globals.config = {'feishu_wiki_space_id': None}
            with self.assertRaises(ValueError):
                fs._require_space_id()

    def test_require_space_id_ok_when_globals_set(self):
        fs = self._make_wiki_fs('')
        with patch('lazyllm.tools.fs.supplier.feishu.lazyllm_globals') as mock_globals:
            mock_globals.config = {'feishu_wiki_space_id': 'wikcnXYZ'}
            fs._require_space_id()  # should not raise


class TestFetchWikiContentPaths(unittest.TestCase):

    def _make_wiki_fs(self) -> FeishuWikiFS:
        fs = object.__new__(FeishuWikiFS)
        fs._space_id = ''
        return fs

    def test_tilde_node_path(self):
        fs = self._make_wiki_fs()
        node_token = 'MCjOwGxwSimztPkO5X6cv8uxnwb'
        fs._resolve_link_content = MagicMock(return_value=b'node content')
        result = fs._fetch_wiki_content(f'/~node/{node_token}')
        fs._resolve_link_content.assert_called_once_with({'kind': 'wiki_node', 'token': node_token})
        self.assertEqual(result, b'node content')

    def test_tilde_docx_path(self):
        fs = self._make_wiki_fs()
        fs._download_doc_raw = MagicMock(return_value=b'docx content')
        result = fs._fetch_wiki_content('/~docx/DocId123')
        fs._download_doc_raw.assert_called_once_with('DocId123', obj_type='docx')
        self.assertEqual(result, b'docx content')

    def test_tilde_doc_path(self):
        fs = self._make_wiki_fs()
        fs._download_doc_raw = MagicMock(return_value=b'doc content')
        result = fs._fetch_wiki_content('/~doc/OldDocToken')
        fs._download_doc_raw.assert_called_once_with('OldDocToken', obj_type='doc')
        self.assertEqual(result, b'doc content')

    def test_tilde_link_path(self):
        from urllib.parse import quote
        fs = self._make_wiki_fs()
        url = 'https://sensetime.feishu.cn/wiki/MCjOwGxwSimztPkO5X6cv8uxnwb'
        fs._resolve_link_content = MagicMock(return_value=b'linked content')
        result = fs._fetch_wiki_content('/~link/' + quote(url, safe=''))
        fs._resolve_link_content.assert_called_once_with({'kind': 'wiki_node', 'token': 'MCjOwGxwSimztPkO5X6cv8uxnwb'})
        self.assertEqual(result, b'linked content')

    def test_tilde_link_invalid_url_raises(self):
        fs = self._make_wiki_fs()
        with self.assertRaises(ValueError):
            fs._fetch_wiki_content('/~link/not-a-valid-feishu-url')

    def test_title_path_requires_space_id(self):
        fs = self._make_wiki_fs()
        with patch('lazyllm.tools.fs.supplier.feishu.lazyllm_globals') as mock_globals:
            mock_globals.config = {'feishu_wiki_space_id': None}
            with self.assertRaises(ValueError):
                fs._fetch_wiki_content('/some/path')


class TestGetNodeSpaceIdBackfill(unittest.TestCase):

    def _make_wiki_fs(self, space_id='') -> FeishuWikiFS:
        fs = object.__new__(FeishuWikiFS)
        fs._space_id = space_id
        fs._base_url = 'https://open.feishu.cn/open-apis'
        return fs

    def test_backfills_space_id_from_node_response(self):
        fs = self._make_wiki_fs('')
        node_data = {'data': {'node': {'space_id': 'wikcnBackfilled', 'obj_type': 'docx', 'obj_token': 'tok'}}}
        fs._get = MagicMock(return_value=node_data)
        node = fs._get_node('someToken')
        self.assertEqual(fs._space_id, 'wikcnBackfilled')
        self.assertEqual(node['space_id'], 'wikcnBackfilled')

    def test_does_not_overwrite_existing_space_id(self):
        fs = self._make_wiki_fs('wikcnOriginal')
        node_data = {'data': {'node': {'space_id': 'wikcnOther', 'obj_type': 'docx', 'obj_token': 'tok'}}}
        fs._get = MagicMock(return_value=node_data)
        fs._get_node('someToken')
        self.assertEqual(fs._space_id, 'wikcnOriginal')


class TestFSRouterParse(unittest.TestCase):

    def setUp(self):
        self.router = _FSRouter()

    def test_bare_feishu_url_normalized(self):
        protocol, space_id, real_path = self.router._parse(
            'https://sensetime.feishu.cn/wiki/MCjOwGxwSimztPkO5X6cv8uxnwb'
        )
        self.assertEqual(protocol, 'feishu')
        self.assertEqual(space_id, 'dynamic')
        self.assertTrue(real_path.startswith('/~link/'))
        self.assertIn('feishu.cn', real_path)

    def test_feishu_tilde_node_path_gets_dynamic_space(self):
        protocol, space_id, real_path = self.router._parse('feishu:/~node/MCjOwG')
        self.assertEqual(protocol, 'feishu')
        self.assertEqual(space_id, 'dynamic')
        self.assertEqual(real_path, '/~node/MCjOwG')

    def test_feishu_at_space_preserved(self):
        protocol, space_id, real_path = self.router._parse('feishu@wikcnXXX:/some/path')
        self.assertEqual(protocol, 'feishu')
        self.assertEqual(space_id, 'wikcnXXX')
        self.assertEqual(real_path, '/some/path')

    def test_feishu_at_dynamic_preserved(self):
        protocol, space_id, real_path = self.router._parse('feishu@dynamic:/~node/Tok')
        self.assertEqual(protocol, 'feishu')
        self.assertEqual(space_id, 'dynamic')
        self.assertEqual(real_path, '/~node/Tok')

    def test_larksuite_bare_url(self):
        protocol, space_id, real_path = self.router._parse(
            'https://company.larksuite.com/wiki/SomeToken'
        )
        self.assertEqual(protocol, 'feishu')
        self.assertEqual(space_id, 'dynamic')

    def test_local_path_unchanged(self):
        protocol, space_id, real_path = self.router._parse('/tmp/file.txt')
        self.assertEqual(protocol, 'file')
        self.assertIsNone(space_id)
        self.assertEqual(real_path, '/tmp/file.txt')

    def test_regular_feishu_cloud_drive_path(self):
        protocol, space_id, real_path = self.router._parse('feishu:/folder/file.txt')
        self.assertEqual(protocol, 'feishu')
        self.assertIsNone(space_id)
        self.assertEqual(real_path, '/folder/file.txt')


class TestFeishuToolRegistration(unittest.TestCase):

    def setUp(self):
        init_session()
        lazyllm_locals['_lazyllm_agent'] = {'workspace': {}}

    def test_document_flow_tools_are_registered(self):
        fs = FeishuFS(space_id='dynamic', dynamic_auth=True)
        manager = ToolManager([(fs, lambda _instance: 'secret-token')])
        names = {item['function']['name'] for item in manager.tools_description}

        self.assertEqual(names, {'get_FeishuWikiFS_methods'})
        manager._tool_call['get_FeishuWikiFS_methods']({})
        names = {item['function']['name'] for item in manager.tools_description}

        self.assertIn('FeishuWikiFS_resolve_link', names)
        self.assertIn('FeishuWikiFS_read_with_references', names)
        self.assertIn('FeishuWikiFS_get_doc_blocks', names)
        self.assertIn('FeishuWikiFS_copy', names)

    def test_resolve_link_returns_standard_fields(self):
        fs = FeishuFS(space_id='dynamic', dynamic_auth=True)
        fs._get_node = MagicMock(return_value={
            'node_token': 'node-1',
            'space_id': 'space-1',
            'title': 'Project Plan',
            'obj_type': 'docx',
            'obj_token': 'doc-1',
            'has_child': True,
        })

        result = fs.resolve_link('/~node/node-1')

        self.assertEqual(result['provider'], 'feishu')
        self.assertEqual(result['object_id'], 'node-1')
        self.assertEqual(result['object_type'], 'docx')
        self.assertEqual(result['title'], 'Project Plan')
        self.assertTrue(result['has_child'])


class TestFeishuNeedsWiki(unittest.TestCase):

    def test_with_space_id(self):
        self.assertTrue(_feishu_needs_wiki('wikcnXXX', '/'))

    def test_with_dynamic(self):
        self.assertTrue(_feishu_needs_wiki('dynamic', '/'))

    def test_tilde_prefix(self):
        for prefix in _FEISHU_WIKI_PATH_PREFIXES:
            self.assertTrue(_feishu_needs_wiki(None, f'/{prefix}token'))

    def test_plain_path_no_space_no_config(self):
        mock_config = MagicMock()
        mock_config.get = MagicMock(return_value=None)
        with patch('lazyllm.tools.fs.client.globals') as mock_globals:
            mock_globals.config = mock_config
            self.assertFalse(_feishu_needs_wiki(None, '/folder/file'))

    def test_plain_path_with_globals_config(self):
        with patch('lazyllm.tools.fs.client.globals') as mock_globals:
            mock_globals.config.get = MagicMock(return_value='wikcnFromGlobals')
            self.assertTrue(_feishu_needs_wiki(None, '/folder/file'))


if __name__ == '__main__':
    unittest.main()


class TestIsWikiLocatorPath(unittest.TestCase):

    def test_bare_feishu_url(self):
        from lazyllm.tools.fs.supplier.feishu import _is_wiki_locator_path
        self.assertTrue(_is_wiki_locator_path('https://company.feishu.cn/wiki/SomeToken'))

    def test_tilde_node(self):
        from lazyllm.tools.fs.supplier.feishu import _is_wiki_locator_path
        self.assertTrue(_is_wiki_locator_path('/~node/SomeToken'))
        self.assertTrue(_is_wiki_locator_path('~node/SomeToken'))

    def test_tilde_link(self):
        from lazyllm.tools.fs.supplier.feishu import _is_wiki_locator_path
        self.assertTrue(_is_wiki_locator_path('/~link/https%3A%2F%2Fxxx.feishu.cn%2Fwiki%2FTok'))

    def test_plain_path_not_locator(self):
        from lazyllm.tools.fs.supplier.feishu import _is_wiki_locator_path
        self.assertFalse(_is_wiki_locator_path('/some/title/path'))
        self.assertFalse(_is_wiki_locator_path('https://example.com/wiki/Tok'))


class TestRefFromElement(unittest.TestCase):

    def test_mention_doc_with_url(self):
        from lazyllm.tools.fs.supplier.feishu import _ref_from_element
        el = {'mention_doc': {'url': 'https://company.feishu.cn/wiki/TokA', 'token': 'TokA', 'obj_type': 'wiki_node'}}
        self.assertEqual(_ref_from_element(el), 'https://company.feishu.cn/wiki/TokA')

    def test_mention_doc_without_url_builds_path(self):
        from lazyllm.tools.fs.supplier.feishu import _ref_from_element
        el = {'mention_doc': {'token': 'TokB', 'obj_type': 'docx'}}
        result = _ref_from_element(el)
        self.assertIn('TokB', result)

    def test_text_run_with_feishu_link(self):
        from lazyllm.tools.fs.supplier.feishu import _ref_from_element
        el = {'text_run': {'content': 'click', 'text_element_style': {
            'link': {'url': 'https://company.feishu.cn/wiki/TokC'}}}}
        self.assertEqual(_ref_from_element(el), 'https://company.feishu.cn/wiki/TokC')

    def test_text_run_with_external_link_ignored(self):
        from lazyllm.tools.fs.supplier.feishu import _ref_from_element
        el = {'text_run': {'content': 'click', 'text_element_style': {
            'link': {'url': 'https://example.com/page'}}}}
        self.assertIsNone(_ref_from_element(el))

    def test_link_preview_feishu(self):
        from lazyllm.tools.fs.supplier.feishu import _ref_from_element
        el = {'link_preview': {'url': 'https://company.feishu.cn/docx/DocId'}}
        self.assertEqual(_ref_from_element(el), 'https://company.feishu.cn/docx/DocId')

    def test_no_ref_returns_none(self):
        from lazyllm.tools.fs.supplier.feishu import _ref_from_element
        self.assertIsNone(_ref_from_element({'text_run': {'content': 'plain text'}}))


class TestDedupeRefs(unittest.TestCase):

    def test_deduplication(self):
        from lazyllm.tools.fs.base import LinkDocumentFSBase
        refs = [
            {'url': 'https://a.feishu.cn/wiki/T1', 'ref_type': 'mention_doc'},
            {'url': 'https://a.feishu.cn/wiki/T1', 'ref_type': 'hyperlink'},
            {'url': 'https://a.feishu.cn/wiki/T2', 'ref_type': 'hyperlink'},
        ]
        result = LinkDocumentFSBase.dedupe_document_references(refs)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['url'], 'https://a.feishu.cn/wiki/T1')
        self.assertEqual(result[1]['url'], 'https://a.feishu.cn/wiki/T2')


class TestFormatReferencesFooter(unittest.TestCase):

    def test_empty_refs(self):
        from lazyllm.tools.fs.base import LinkDocumentFSBase
        self.assertEqual(LinkDocumentFSBase.format_document_references_footer([], 'feishu'), '')

    def test_footer_format(self):
        from lazyllm.tools.fs.base import LinkDocumentFSBase
        refs = [
            {'url': 'https://a.feishu.cn/wiki/T1', 'ref_type': 'mention_doc'},
            {'url': 'https://a.feishu.cn/docx/D2', 'ref_type': 'hyperlink'},
        ]
        footer = LinkDocumentFSBase.format_document_references_footer(refs, 'feishu')
        self.assertIn('lazyllm-feishu-references', footer)
        self.assertIn('[1] mention_doc | https://a.feishu.cn/wiki/T1', footer)
        self.assertIn('[2] hyperlink | https://a.feishu.cn/docx/D2', footer)
        self.assertIn('end lazyllm-feishu-references', footer)


class TestWikiLsWithLocatorPath(unittest.TestCase):

    def _make_wiki_fs(self) -> FeishuWikiFS:
        fs = object.__new__(FeishuWikiFS)
        fs._space_id = ''
        fs._base_url = 'https://open.feishu.cn/open-apis'
        return fs

    def test_ls_bare_wiki_url_returns_children(self):
        fs = self._make_wiki_fs()
        node_token = 'MCjOwGxwSimztPkO5X6cv8uxnwb'
        node_data = {'node_token': node_token, 'space_id': 'wikcnABC', 'title': 'Root',
                     'obj_type': 'docx', 'obj_token': 'ObjTok', 'has_child': True}
        child_nodes = [
            {'node_token': 'child1', 'title': 'Child1', 'obj_type': 'docx', 'has_child': False},
            {'node_token': 'child2', 'title': 'Child2', 'obj_type': 'docx', 'has_child': False},
        ]
        fs._get_node = MagicMock(return_value=node_data)
        fs._list_nodes_raw = MagicMock(return_value=child_nodes)
        url = f'https://company.feishu.cn/wiki/{node_token}'
        entries = fs.ls(url, detail=True)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]['name'], 'Child1')
        self.assertEqual(entries[1]['name'], 'Child2')
        fs._list_nodes_raw.assert_called_once_with(node_token)

    def test_ls_tilde_node_path(self):
        fs = self._make_wiki_fs()
        node_token = 'NodeTok123'
        node_data = {'node_token': node_token, 'space_id': 'wikcnABC', 'title': 'Doc',
                     'obj_type': 'docx', 'obj_token': 'ObjTok', 'has_child': True}
        child_nodes = [{'node_token': 'c1', 'title': 'Sub', 'obj_type': 'docx', 'has_child': False}]
        fs._get_node = MagicMock(return_value=node_data)
        fs._list_nodes_raw = MagicMock(return_value=child_nodes)
        entries = fs.ls(f'/~node/{node_token}', detail=False)
        self.assertEqual(entries, ['Sub'])

    def test_ls_node_no_children_returns_empty(self):
        fs = self._make_wiki_fs()
        node_token = 'LeafTok'
        node_data = {'node_token': node_token, 'space_id': 'wikcnABC', 'title': 'Leaf',
                     'obj_type': 'docx', 'obj_token': 'ObjTok', 'has_child': False}
        fs._get_node = MagicMock(return_value=node_data)
        fs._list_nodes_raw = MagicMock(return_value=[])
        entries = fs.ls(f'/~node/{node_token}')
        self.assertEqual(entries, [])

    def test_ls_docx_direct_link_raises(self):
        fs = self._make_wiki_fs()
        parsed_docx = {'kind': 'docx', 'token': 'DocxTok'}
        with patch('lazyllm.tools.fs.supplier.feishu._parse_feishu_browser_url', return_value=parsed_docx):
            with self.assertRaises(ValueError):
                fs.ls('https://company.feishu.cn/docx/DocxTok')


class TestWikiInfoWithLocatorPath(unittest.TestCase):

    def _make_wiki_fs(self) -> FeishuWikiFS:
        fs = object.__new__(FeishuWikiFS)
        fs._space_id = ''
        fs._base_url = 'https://open.feishu.cn/open-apis'
        return fs

    def test_info_bare_wiki_url(self):
        fs = self._make_wiki_fs()
        node_token = 'InfoTok'
        node_data = {'node_token': node_token, 'space_id': 'wikcnABC', 'title': 'MyDoc',
                     'obj_type': 'docx', 'obj_token': 'ObjTok', 'has_child': False,
                     'creator': 'user1', 'owner': 'user2'}
        fs._get_node = MagicMock(return_value=node_data)
        entry = fs.info(f'https://company.feishu.cn/wiki/{node_token}')
        self.assertEqual(entry['name'], 'MyDoc')
        self.assertEqual(entry['creator'], 'user1')
        self.assertEqual(entry['owner'], 'user2')
        self.assertEqual(entry['node_token'], node_token)


class TestFetchWikiContentWithReferences(unittest.TestCase):

    def _make_wiki_fs(self) -> FeishuWikiFS:
        fs = object.__new__(FeishuWikiFS)
        fs._space_id = ''
        fs._base_url = 'https://open.feishu.cn/open-apis'
        return fs

    def test_include_references_false_no_footer(self):
        fs = self._make_wiki_fs()
        fs._resolve_link_content = MagicMock(return_value=b'body text')
        result = fs._fetch_wiki_content('/~node/SomeTok', include_references=False)
        self.assertEqual(result, b'body text')

    def test_include_references_true_appends_footer(self):
        fs = self._make_wiki_fs()
        node_data = {'node_token': 'SomeTok', 'obj_type': 'docx', 'obj_token': 'DocId'}
        fs._get_node = MagicMock(return_value=node_data)
        fs._download_doc_raw = MagicMock(return_value=b'body text')
        refs = [{'url': 'https://a.feishu.cn/wiki/T1', 'ref_type': 'mention_doc', 'kind': 'wiki_node'}]
        fs._list_document_references = MagicMock(return_value=refs)
        result = fs._fetch_wiki_content('/~node/SomeTok', include_references=True)
        text = result.decode('utf-8')
        self.assertIn('body text', text)
        self.assertIn('lazyllm-feishu-references', text)
        self.assertIn('https://a.feishu.cn/wiki/T1', text)

    def test_include_references_true_no_refs_no_footer(self):
        fs = self._make_wiki_fs()
        node_data = {'node_token': 'SomeTok', 'obj_type': 'docx', 'obj_token': 'DocId'}
        fs._get_node = MagicMock(return_value=node_data)
        fs._download_doc_raw = MagicMock(return_value=b'body text')
        fs._list_document_references = MagicMock(return_value=[])
        result = fs._fetch_wiki_content('/~node/SomeTok', include_references=True)
        self.assertEqual(result, b'body text')


class TestReadBytesWithKwargs(unittest.TestCase):

    def _make_wiki_fs(self) -> FeishuWikiFS:
        fs = object.__new__(FeishuWikiFS)
        fs._space_id = ''
        fs._base_url = 'https://open.feishu.cn/open-apis'
        return fs

    def test_read_bytes_default_no_references(self):
        fs = self._make_wiki_fs()
        fs._fetch_wiki_content = MagicMock(return_value=b'content')
        result = fs.read_bytes('/~node/Tok')
        fs._fetch_wiki_content.assert_called_once_with('/~node/Tok', include_references=False)
        self.assertEqual(result, b'content')

    def test_read_bytes_with_include_references(self):
        fs = self._make_wiki_fs()
        fs._fetch_wiki_content = MagicMock(return_value=b'content+refs')
        result = fs.read_bytes('/~node/Tok', include_references=True)
        fs._fetch_wiki_content.assert_called_once_with('/~node/Tok', include_references=True)
        self.assertEqual(result, b'content+refs')


class TestNodeToEntryCreatorOwner(unittest.TestCase):

    def test_creator_owner_in_entry(self):
        node = {
            'node_token': 'Tok1', 'title': 'Doc', 'obj_type': 'docx',
            'obj_token': 'ObjTok', 'has_child': False,
            'creator': 'user_a', 'owner': 'user_b', 'node_creator': 'user_c',
        }
        entry = FeishuWikiFS._node_to_entry(node)
        self.assertEqual(entry['creator'], 'user_a')
        self.assertEqual(entry['owner'], 'user_b')
        self.assertEqual(entry['node_creator'], 'user_c')

    def test_missing_creator_defaults_to_empty(self):
        node = {'node_token': 'Tok2', 'title': 'Doc2', 'obj_type': 'docx', 'has_child': False}
        entry = FeishuWikiFS._node_to_entry(node)
        self.assertEqual(entry['creator'], '')
        self.assertEqual(entry['owner'], '')
        self.assertEqual(entry['node_creator'], '')
