import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lazyllm.tools.writer.data_models import (
    MediaAsset,
    MediaAssetLibrary,
    TargetDocument,
    WriterBlock,
    WriterDocument,
    WriterSpan,
)
from lazyllm.tools.writer.provider import (
    FeishuWriterProvider,
    GitHubWriterProvider,
    NotionWriterProvider,
    ObsidianWriterProvider,
    WeChatWriterProvider,
    WriterProviderBase,
    WriterProviderCapabilities,
    WriterProviderCapabilityError,
    WriterProviderDocument,
    WriterProviderWriteOutcomeError,
    list_writer_providers,
    register_writer_provider,
)
from lazyllm.tools.writer.provider import registry as provider_registry
from lazyllm.tools.writer.tools.resource_tools import WriterResourceTools


def test_optional_provider_capabilities_default_to_unsupported():
    assert WriterProviderBase.capabilities.model_dump() == {
        'load': False,
        'create': False,
        'replace': False,
        'append': False,
        'patch': False,
        'revision_check': False,
        'media': False,
    }


@pytest.mark.parametrize(
    ('provider_class', 'expected'),
    [
        (FeishuWriterProvider, {**dict.fromkeys(
            ('load', 'create', 'replace', 'append', 'patch', 'revision_check', 'media'), True,
        )}),
        (NotionWriterProvider, {**dict.fromkeys(
            ('load', 'create', 'replace', 'append', 'patch', 'revision_check', 'media'), True,
        )}),
        (GitHubWriterProvider, {
            'load': True, 'create': True, 'replace': True, 'append': True,
            'patch': False, 'revision_check': True, 'media': True,
        }),
        (WeChatWriterProvider, {
            'load': True, 'create': True, 'replace': True, 'append': False,
            'patch': True, 'revision_check': True, 'media': True,
        }),
        (ObsidianWriterProvider, {
            'load': True, 'create': True, 'replace': True, 'append': False,
            'patch': False, 'revision_check': True, 'media': True,
        }),
    ],
)
def test_builtin_providers_declare_tested_capabilities(provider_class, expected):
    assert provider_class.capabilities.model_dump() == expected


def test_provider_registry_lists_new_provider_without_a_fixed_enum(monkeypatch):
    class TestWriterProvider(WriterProviderBase):
        provider = 'test'
        capabilities = WriterProviderCapabilities(load=True, append=True)

    monkeypatch.setattr(provider_registry, '_PROVIDERS', dict(provider_registry._PROVIDERS))
    register_writer_provider(TestWriterProvider)

    providers = {item['id']: item['capabilities'] for item in list_writer_providers()}
    assert providers['test'] == ['load', 'append']


def test_resource_create_requires_explicit_provider(tmp_path: Path):
    with pytest.raises(ValueError, match='adapter is required'):
        WriterResourceTools(artifact_store=str(tmp_path)).create_document('Document')


def test_provider_contract_separates_conversion_from_writing():
    markdown = '# Document\n\n```mermaid\nA --> B\n```\n'

    feishu_content = FeishuWriterProvider().convert_document(
        markdown,
        target=TargetDocument(adapter='feishu', doc_id='document-1'),
    )
    github = GitHubWriterProvider()
    github_content = github.convert_document(markdown)
    target = TargetDocument(adapter='github')
    editor_content = github.prepare_markdown_for_editor(markdown, target)

    assert feishu_content.provider == 'feishu'
    assert feishu_content.format == 'feishu_blocks'
    assert isinstance(feishu_content.content, list)
    assert isinstance(feishu_content.source_document, WriterDocument)
    assert feishu_content.source_document.document_id.startswith('writer-document-')
    assert feishu_content.source_document.provider_binding == {}
    assert github_content.provider == 'github'
    assert github_content.format == 'markdown'
    assert github_content.content == markdown
    assert editor_content == '# Document\n\n```text\nA --> B\n```\n'
    assert len(target.meta['github_writer_code_fences']) == 1
    assert target.meta['github_writer_code_fences'][0]['language'] == 'mermaid'


def test_wechat_conversion_is_pure_and_keeps_copyable_image_url(monkeypatch):
    provider = WeChatWriterProvider()
    monkeypatch.setattr(
        provider, '_access_token', lambda: pytest.fail('conversion must not authorize'),
    )
    document = WriterDocument(
        document_id='document-1',
        title='Document',
        blocks=[WriterBlock(
            node_id='image-1',
            type='image',
            references=[{'type': 'media_asset', 'id': 'asset-1'}],
        )],
    )
    media = MediaAssetLibrary(
        library_id='media-1',
        assets={'asset-1': MediaAsset(
            media_asset_id='asset-1',
            asset_type='image',
            source_type='input_resource',
            uri='https://example.test/image.png',
        )},
    )

    converted = provider.convert_document(document, media_assets=media)

    assert converted.format == 'html'
    assert 'src="https://example.test/image.png"' in converted.content
    assert converted.media_references == {
        'asset-1': 'https://example.test/image.png',
    }


@pytest.mark.parametrize(
    'provider', [FeishuWriterProvider(), NotionWriterProvider()],
)
def test_native_block_conversion_keeps_copyable_image_url(provider):
    document = WriterDocument(
        document_id='document-1',
        blocks=[WriterBlock(
            node_id='image-1',
            type='image',
            references=[{'type': 'media_asset', 'id': 'asset-1'}],
        )],
    )
    media = MediaAssetLibrary(
        library_id='media-1',
        assets={'asset-1': MediaAsset(
            media_asset_id='asset-1',
            asset_type='image',
            source_type='input_resource',
            uri='https://example.test/image.png',
        )},
    )

    converted = provider.convert_document(document, media_assets=media)

    assert 'https://example.test/image.png' in json.dumps(converted.content)
    assert converted.media_references == {
        'asset-1': 'https://example.test/image.png',
    }


@pytest.mark.parametrize(
    ('provider', 'target', 'document_id', 'expected_url'),
    [
        (
            FeishuWriterProvider(),
            TargetDocument(
                adapter='feishu',
                uri='https://example.feishu.cn/docx/target-document',
            ),
            'target-document',
            'https://example.feishu.cn/docx/target-document#target-heading',
        ),
        (
            NotionWriterProvider(),
            TargetDocument(
                adapter='notion',
                uri='https://www.notion.so/Target-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
            ),
            'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
            'https://www.notion.so/Target-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
            '#aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        ),
    ],
)
def test_native_write_materializes_internal_links_against_resolved_target(
        monkeypatch, provider, target, document_id, expected_url):
    document = WriterDocument(
        document_id='writer-document',
        stage='final',
        title='Document',
        blocks=[
            WriterBlock(
                node_id='target-heading', type='heading', numbering={'level': 1},
                content='Target', stage='final',
            ),
            WriterBlock(
                node_id='reference', type='paragraph', content='See target', stage='final',
                spans=[WriterSpan(text='See target', style={'link': {
                    'type': 'internal_ref',
                    'target_node_id': 'target-heading',
                }})],
            ),
        ],
    )
    converted = provider.convert_document(document)
    fs = MagicMock()
    monkeypatch.setattr(provider, '_resolve_document_target', lambda *_args, **_kwargs: (
        provider.provider,
        f'/~document/{document_id}',
        fs,
        provider._writer_adapter(),
        target.uri,
        document_id,
    ))

    provider.write_document(converted, target)

    written_blocks = fs.replace_doc_blocks.call_args.args[1]
    serialized = json.dumps(written_blocks)
    assert expected_url in serialized
    assert 'https://www.notion.so/#' not in serialized
    assert 'https://feishu.cn/docx/#' not in serialized


def test_legacy_replace_composes_the_two_provider_stages(monkeypatch):
    provider = GitHubWriterProvider()
    target = TargetDocument(adapter='github', doc_id='document-1')
    converted = WriterProviderDocument(
        provider='github',
        format='markdown',
        content='# Document',
        source_document=WriterDocument(document_id='document-1'),
    )
    calls = []

    def convert(content, *, target, media_assets):
        calls.append(('convert', content, target, media_assets))
        return converted

    def write(document, target, *, media_assets, mode):
        calls.append(('write', document, target, media_assets, mode))
        return {'success': True}

    monkeypatch.setattr(provider, 'convert_document', convert)
    monkeypatch.setattr(provider, 'write_document', write)

    assert provider.replace_document('# Document', target) == {'success': True}
    assert calls == [
        ('convert', '# Document', target, None),
        ('write', converted, target, None, 'replace'),
    ]


def test_resource_operation_rejects_unsupported_capability_before_provider_call(tmp_path: Path):
    with pytest.raises(WriterProviderCapabilityError) as captured:
        WriterResourceTools(artifact_store=str(tmp_path)).append_to_document(
            '# Document',
            {'adapter': 'wechat', 'doc_id': 'draft-1'},
        )

    assert captured.value.code == 'PROVIDER_CAPABILITY_UNSUPPORTED'
    assert captured.value.provider == 'wechat'
    assert captured.value.capability == 'append'
    assert captured.value.details == {
        'provider': 'wechat',
        'capability': 'append',
    }
    assert captured.value.retryable is False


def test_resource_write_marks_timeout_outcome_ambiguous(monkeypatch, tmp_path: Path):
    def timeout(*_args, **_kwargs):
        raise TimeoutError('provider response timed out')

    monkeypatch.setattr(WeChatWriterProvider, 'replace_document', timeout)

    with pytest.raises(WriterProviderWriteOutcomeError) as captured:
        WriterResourceTools(artifact_store=str(tmp_path)).replace_document(
            '# Document',
            {'adapter': 'wechat', 'doc_id': 'draft-1'},
        )

    assert captured.value.code == 'PROVIDER_WRITE_OUTCOME_AMBIGUOUS'
    assert captured.value.details == {
        'provider': 'wechat',
        'operation': 'replace',
    }
    assert captured.value.retryable is False


def test_resource_write_does_not_reclassify_deterministic_failure(monkeypatch, tmp_path: Path):
    def invalid_request(*_args, **_kwargs):
        raise ValueError('invalid document')

    monkeypatch.setattr(WeChatWriterProvider, 'replace_document', invalid_request)

    with pytest.raises(ValueError, match='invalid document'):
        WriterResourceTools(artifact_store=str(tmp_path)).replace_document(
            '# Document',
            {'adapter': 'wechat', 'doc_id': 'draft-1'},
        )
