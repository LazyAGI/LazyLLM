from pathlib import Path
from unittest.mock import Mock

import pytest

from lazyllm.tools.fs.client import FS
from lazyllm.tools.fs.supplier.googledrive import GoogleDriveFS
from lazyllm.tools.writer.data_models import TargetDocument
from lazyllm.tools.writer.provider import (
    GoogleDriveWriterProvider, WriterProviderCapabilityError,
    get_writer_provider, list_writer_providers, match_writer_provider,
)
from lazyllm.tools.writer.tools.resource_tools import WriterResourceTools
from lazyllm.tools.writer.utils.export import export_writer_document


@pytest.mark.parametrize('locator', [
    'googledrive:/file-1', 'googledrive://file-1',
    'https://docs.google.com/document/d/file-1/edit?tab=t.0',
    'https://docs.google.com/spreadsheets/d/file-1/edit#gid=0',
    'https://drive.google.com/file/d/file-1/view',
    'https://drive.google.com/open?id=file-1',
])
def test_google_drive_registry_resolves_locators(locator):
    provider = match_writer_provider(locator)
    target = provider.resolve(locator)
    assert isinstance(provider, GoogleDriveWriterProvider)
    assert (target.doc_id, target.uri, target.adapter) == ('file-1', 'googledrive:/file-1', 'googledrive')
    assert next(p for p in list_writer_providers() if p['id'] == 'googledrive')['capabilities'] == ['load']
    with pytest.raises(WriterProviderCapabilityError):
        provider.require_capability('replace')
    with pytest.raises(NotImplementedError, match='write_document'):
        provider.write_document(None, target)


@pytest.mark.parametrize('locator', [
    '/tmp/file-1', 'file:///tmp/file-1', 'googledrive:/../file-1',
    'https://docs.google.com.attacker.example/document/d/file-1/edit',
    'https://user@docs.google.com/document/d/file-1/edit',
    'https://docs.google.com:444/document/d/file-1/edit',
    'https://drive.google.com/drive/folders/file-1',
    'https://docs.google.com/document/d/',
    'https://drive.google.com/open?id=file-1&id=file-2',
])
def test_google_drive_rejects_invalid_locators_before_io(locator, monkeypatch):
    lookup = Mock()
    monkeypatch.setattr(FS, '_get_or_create_fs', lookup)
    with pytest.raises(ValueError):
        GoogleDriveWriterProvider().resolve(locator)
    lookup.assert_not_called()


def google_fs(monkeypatch, mime='application/vnd.google-apps.document', content='正文\n```\n# literal'):
    fs = Mock()
    fs.info.return_value = {
        'name': 'file-1', 'title': '项目方案', 'mime_type': mime, 'type': 'file',
        'web_url': 'https://docs.google.com/document/d/file-1/edit', 'mtime': 123,
    }
    fs.read_file.return_value = content
    monkeypatch.setattr(FS, '_get_or_create_fs', Mock(return_value=fs))
    return fs


@pytest.mark.parametrize(('mime', 'fmt'), [
    ('application/vnd.google-apps.document', 'text'),
    ('application/vnd.google-apps.spreadsheet', 'csv'),
    ('text/plain', 'text'), ('text/markdown', 'markdown'),
])
def test_google_drive_load_preserves_text_and_works_through_resource_tool(monkeypatch, tmp_path, mime, fmt):
    original = '正文\n```\n# literal'
    fs = google_fs(monkeypatch, mime, original)
    provider = get_writer_provider('googledrive')
    target = provider.resolve('googledrive:/file-1')
    loaded = provider.load_document(target, stage='draft')
    resolved = loaded['target_document']
    assert resolved.title == '项目方案'
    assert resolved.meta['content_format'] == fmt
    assert resolved.meta['stage'] == 'draft'
    assert resolved.meta['browser_url'] == fs.info.return_value['web_url']
    assert original in loaded['source_document']
    if fmt == 'markdown':
        assert loaded['source_document'] == original
    else:
        assert loaded['source_document'].startswith(f'````{fmt}\n')
    if fmt == 'csv':
        assert any('one worksheet' in warning for warning in loaded['resource_warnings'])
    fs.read_file.assert_called_once_with('/file-1')
    assert target.title is None  # The caller's target must remain unchanged.
    result = WriterResourceTools(llm=None, artifact_store=str(tmp_path)).load_document(target)
    assert result['representation'] == 'markdown'
    paths = result['metadata']['artifact_paths']
    assert Path(paths['source_document']).read_text() == loaded['source_document']


@pytest.mark.parametrize('failure', ['folder', 'binary', 'slides', 'metadata', 'permission', 'id', 'adapter'])
def test_google_drive_read_failures_are_not_silenced(monkeypatch, failure):
    fs = google_fs(monkeypatch)
    target = TargetDocument(adapter='googledrive', uri='googledrive:/file-1')
    if failure == 'folder':
        fs.info.return_value['type'] = 'directory'
    elif failure in {'binary', 'slides'}:
        fs.info.return_value['mime_type'] = {
            'binary': 'application/pdf', 'slides': 'application/vnd.google-apps.presentation',
        }[failure]
    elif failure == 'metadata':
        fs.info.return_value['name'] = 'file-2'
    elif failure == 'permission':
        fs.info.side_effect = PermissionError('denied')
    elif failure == 'id':
        target.doc_id = 'file-2'
    else:
        target.adapter = 'notion'
    with pytest.raises((ValueError, NotImplementedError, PermissionError)):
        GoogleDriveWriterProvider().load_document(target)
    fs.read_file.assert_not_called()


@pytest.mark.parametrize('provider_name', ['feishu', 'notion'])
def test_structured_cloud_loads_return_title_body_and_target(monkeypatch, tmp_path, provider_name):
    fs = Mock()
    if provider_name == 'feishu':
        locator = 'https://example.feishu.cn/docx/doc-1'
        fs.get_document_id.return_value = 'doc-1'
        fs.get_doc_blocks.return_value = [
            {'block_id': 'doc-1', 'block_type': 1, 'children': ['p-1'],
             'page': {'elements': [{'text_run': {'content': '项目方案'}}]}},
            {'block_id': 'p-1', 'parent_id': 'doc-1', 'block_type': 2,
             'text': {'elements': [{'text_run': {'content': '正文内容'}}]}},
        ]
    else:
        locator = 'https://www.notion.so/0123456789abcdef0123456789abcdef'
        fs.get_document_metadata.return_value = {
            'object_type': 'page', 'document_id': 'doc-1', 'title': '项目方案',
            'browser_url': locator, 'last_edited_time': '2026-09-14T00:00:00Z',
        }
        fs.get_doc_blocks.return_value = [{
            'id': 'p-1', 'type': 'paragraph',
            'paragraph': {'rich_text': [{
                'type': 'text', 'text': {'content': '正文内容'}, 'plain_text': '正文内容',
            }]},
        }]
    monkeypatch.setattr(FS, '_get_or_create_fs', Mock(return_value=fs))
    provider = match_writer_provider(locator)
    target = provider.resolve(locator)
    loaded = provider.load_document(target)
    assert loaded['provider'] == provider_name
    assert loaded['representation'] == 'ir'
    assert loaded['target_document'].doc_id == 'doc-1'
    assert loaded['target_document'].title == loaded['source_document'].title == '项目方案'
    assert loaded['target_document'].uri == locator
    assert '正文内容' in export_writer_document(loaded['source_document'], 'markdown')
    assert target.title is None
    result = WriterResourceTools(llm=None, artifact_store=str(tmp_path)).load_document(target)
    assert result['representation'] == 'ir'


@pytest.mark.parametrize(('mime', 'export_type'), [
    ('application/vnd.google-apps.document', 'text/plain'),
    ('application/vnd.google-apps.spreadsheet', 'text/csv'),
])
def test_google_provider_reads_real_fs_export_without_size_metadata(monkeypatch, mime, export_type):
    fs = GoogleDriveFS(dynamic_auth=True, skip_instance_cache=True)
    fs._get = Mock(return_value={
        'id': 'file-1', 'name': '项目方案', 'mimeType': mime,
        'webViewLink': 'https://drive.google.com/file/d/file-1/view',
    })
    fs._request = Mock(return_value=Mock(content='非空正文'.encode()))
    monkeypatch.setattr(FS, '_get_or_create_fs', Mock(return_value=fs))
    provider = match_writer_provider('googledrive:/file-1')
    result = provider.load_document(provider.resolve('googledrive:/file-1'))
    assert '非空正文' in result['source_document']
    fs._request.assert_called_once_with(
        'GET', 'https://www.googleapis.com/drive/v3/files/file-1/export',
        params={'mimeType': export_type},
    )
    assert 'webViewLink' in fs._get.call_args.kwargs['params']['fields']


@pytest.mark.parametrize('content', [b'\x00\xffordinary file', b''])
def test_google_drive_ordinary_file_read_bytes_preserves_content(content):
    fs = GoogleDriveFS(dynamic_auth=True, skip_instance_cache=True)
    fs._get = Mock(return_value={
        'id': 'file-1', 'name': 'attachment.bin',
        'mimeType': 'application/octet-stream', 'size': str(len(content)),
    })
    fs._request = Mock(return_value=Mock(content=content))
    assert fs.read_bytes('/file-1') == content
    if content:
        assert fs._request.call_args.args == ('GET', 'https://www.googleapis.com/drive/v3/files/file-1')
        assert fs._request.call_args.kwargs['params']['alt'] == 'media'
    else:
        fs._request.assert_not_called()
