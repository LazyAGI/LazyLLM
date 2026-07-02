# Copyright (c) 2026 LazyAGI. All rights reserved.
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from lazyllm import init_session, locals as lazyllm_locals
from lazyllm.tools.agent.toolsManager import ToolManager
from lazyllm.tools.fs.supplier.googledrive import GoogleDriveFS
from tests.basic_tests.Tools.fs_test_utils import load_fs_docs_only


class TestGoogleDriveSearch(unittest.TestCase):

    def _make_fs(self):
        return GoogleDriveFS(dynamic_auth=True, skip_instance_cache=True)

    def test_search_builds_keyword_and_scope_query(self):
        fs = self._make_fs()
        calls = []

        def get(url, params):
            calls.append((url, dict(params)))
            return {
                'files': [{
                    'id': 'file-1',
                    'name': 'Project Plan',
                    'mimeType': 'application/vnd.google-apps.document',
                    'modifiedTime': '2026-06-01T00:00:00Z',
                    'parents': ['folder-1'],
                    'driveId': 'drive-1',
                    'webViewLink': 'https://docs.google.com/document/d/file-1/edit',
                }]
            }

        fs._get = get
        results = fs.search(
            ['release', "owner's"],
            file_name='Project Plan',
            drive_id='drive-1',
            folder_id='folder-1',
            limit=5,
        )

        self.assertEqual(results[0]['title'], 'Project Plan')
        self.assertEqual(results[0]['google_drive_path'], 'googledrive:/file-1')
        self.assertEqual(results[0]['web_url'], 'https://docs.google.com/document/d/file-1/edit')
        _, params = calls[0]
        self.assertIn("fullText contains 'release'", params['q'])
        self.assertIn("fullText contains 'owner\\'s'", params['q'])
        self.assertIn("name = 'Project Plan'", params['q'])
        self.assertIn("'folder-1' in parents", params['q'])
        self.assertEqual(params['corpora'], 'drive')
        self.assertEqual(params['driveId'], 'drive-1')
        self.assertEqual(params['pageSize'], 5)

    def test_search_rejects_empty_keywords(self):
        fs = self._make_fs()
        with self.assertRaises(ValueError):
            fs.search([])

    def test_find_filters_file_names_with_regex(self):
        fs = self._make_fs()
        fs._get = lambda _url, params: {
            'files': [
                {'id': '1', 'name': 'Project Plan.md', 'mimeType': 'text/markdown'},
                {'id': '2', 'name': 'Meeting Notes.md', 'mimeType': 'text/markdown'},
                {'id': '3', 'name': 'Project Budget.xlsx', 'mimeType': 'application/vnd.ms-excel'},
            ]
        }

        results = fs.find(r'^Project.*\.(md|xlsx)$', limit=10)

        self.assertEqual([item['title'] for item in results], ['Project Plan.md', 'Project Budget.xlsx'])

    def test_find_rejects_invalid_regex(self):
        fs = self._make_fs()
        with self.assertRaises(ValueError):
            fs.find('[')

    def test_find_rejects_empty_regex_before_compile(self):
        fs = self._make_fs()
        with self.assertRaisesRegex(ValueError, 'pattern is required'):
            fs.find('   ')

    def test_google_workspace_document_uses_export(self):
        fs = self._make_fs()
        requests = []
        fs._get = lambda _url, params: {'mimeType': 'application/vnd.google-apps.document'}

        def request(method, url, **kwargs):
            requests.append((method, url, kwargs))
            return SimpleNamespace(content=b'hello Google Docs')

        fs._request = request
        self.assertEqual(fs._download_range('/file-1', 0, 5), b'hello')
        self.assertTrue(requests[0][1].endswith('/files/file-1/export'))
        self.assertEqual(requests[0][2]['params']['mimeType'], 'text/plain')

    def test_download_range_caches_mime_type(self):
        fs = self._make_fs()
        metadata_calls = []

        def get(url, params):
            metadata_calls.append((url, params))
            return {'mimeType': 'text/plain'}

        fs._get = get
        fs._request = lambda *_args, **_kwargs: SimpleNamespace(content=b'hello')

        fs._download_range('/file-1', 0, 2)
        fs._download_range('/file-1', 2, 5)

        self.assertEqual(len(metadata_calls), 1)

    def test_blob_file_uses_media_download(self):
        fs = self._make_fs()
        fs._get = lambda _url, params: {'mimeType': 'text/plain'}
        requests = []

        def request(method, url, **kwargs):
            requests.append((method, url, kwargs))
            return SimpleNamespace(content=b'hello')

        fs._request = request
        self.assertEqual(fs._download_range('/file-1', 0, 5), b'hello')
        self.assertEqual(requests[0][2]['params']['alt'], 'media')
        self.assertEqual(requests[0][2]['headers']['Range'], 'bytes=0-4')

    def test_iter_files_uses_remaining_count_for_next_page(self):
        fs = self._make_fs()
        page_sizes = []

        def get(_url, params):
            page_sizes.append(params['pageSize'])
            if len(page_sizes) == 1:
                return {'files': [{'id': '1'}, {'id': '2'}], 'nextPageToken': 'next'}
            return {'files': [{'id': '3'}]}

        fs._get = get
        self.assertEqual(len(list(fs._iter_files('trashed = false', max_items=3))), 3)
        self.assertEqual(page_sizes, [3, 1])

    def test_iter_files_warns_when_google_reports_incomplete_search(self):
        fs = self._make_fs()
        fs._get = lambda _url, params: {
            'files': [{'id': '1'}],
            'incompleteSearch': True,
        }

        with patch('lazyllm.tools.fs.supplier.googledrive.lazyllm.LOG.warning') as warning:
            self.assertEqual(len(list(fs._iter_files('trashed = false', max_items=1))), 1)

        warning.assert_called_once()
        self.assertIn('incompleteSearch=true', warning.call_args.args[0])

    def test_item_to_entry_normalizes_null_values(self):
        entry = GoogleDriveFS._item_to_entry({
            'id': None,
            'name': None,
            'mimeType': None,
            'webViewLink': None,
            'parents': None,
            'driveId': None,
            'description': None,
        })

        self.assertEqual(entry['name'], '')
        self.assertEqual(entry['title'], '')
        self.assertEqual(entry['web_url'], '')
        self.assertEqual(entry['parents'], [])

    def test_search_and_find_are_registered(self):
        init_session()
        lazyllm_locals['_lazyllm_agent'] = {'workspace': {}}
        load_fs_docs_only(GoogleDriveFS.search)
        fs = self._make_fs()
        manager = ToolManager([(fs, lambda _instance: 'secret-token')])
        manager._tool_call['get_GoogleDriveFS_methods']({})
        names = {item['function']['name'] for item in manager.tools_description}
        self.assertIn('GoogleDriveFS_search', names)
        self.assertIn('GoogleDriveFS_find', names)


if __name__ == '__main__':
    unittest.main()
