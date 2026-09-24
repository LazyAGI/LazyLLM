import hashlib
from unittest.mock import MagicMock, patch

import pytest

from lazyllm.tools.fs.client import _FSRouter
from lazyllm.tools.fs.supplier.github import GitHubFSError, GitHubRepoFS, GitHubWikiFS
from lazyllm.tools.writer.data_models.multimodal import MediaAsset, MediaAssetLibrary
from lazyllm.tools.writer.data_models.task import TargetDocument
from lazyllm.tools.writer.provider import get_writer_provider, match_writer_provider
from lazyllm.tools.writer.provider.github import GitHubWriterProvider


def _resolved_target():
    return {
        'doc_id': 'acme/docs:main:guide.md',
        'uri': 'githubrepo:/acme/docs/guide.md?ref=main',
        'browser_url': 'https://github.com/acme/docs/blob/main/guide.md',
        'title': 'guide',
        'owner': 'acme',
        'repo': 'docs',
        'ref': 'main',
        'path': 'guide.md',
        'revision': 'commit-1',
        'blob_sha': 'blob-1',
        'target_type': 'repository',
        'fs_scheme': 'githubrepo',
    }


def _write_result(**updates):
    result = {
        'success': True,
        'provider': 'github',
        'target_type': 'repository',
        'doc_id': 'acme/docs:work:guide.md',
        'uri': 'githubrepo:/acme/docs/guide.md?ref=lazymind%2Fop',
        'browser_url': 'https://github.com/acme/docs/pull/9',
        'publish_mode': 'pull_request',
        'commit_sha': 'commit-2',
        'revision': 'commit-2',
        'work_branch': 'lazymind/op',
        'pull_request_url': 'https://github.com/acme/docs/pull/9',
        'warnings': [],
    }
    result.update(updates)
    return result


def test_github_repository_and_wiki_urls_are_routed():
    assert get_writer_provider('github').provider == 'github'
    assert match_writer_provider(
        'https://github.com/acme/docs/blob/main/guide.md'
    ).provider == 'github'
    assert match_writer_provider(
        'https://github.com/acme/docs/wiki/Guide'
    ).provider == 'github'
    assert not GitHubWriterProvider.matches(
        'https://github.com/acme/docs/blob/main/source.py'
    )

    router = _FSRouter()
    assert router._parse(
        'https://github.com/acme/docs/blob/main/guide.md'
    )[0] == 'githubrepo'
    assert router._parse(
        'https://github.com/acme/docs/wiki/Guide'
    )[0] == 'githubwiki'
    assert isinstance(router._get_or_create_fs('githubrepo', None), GitHubRepoFS)
    assert isinstance(router._get_or_create_fs('githubwiki', None), GitHubWikiFS)


def test_repository_document_loads_images_and_preserves_other_content():
    raw_url = 'https://raw.githubusercontent.com/acme/docs/main/assets/logo.png'
    markdown = (
        '# Guide\n\n'
        '[License](LICENSE)\n\n'
        '![diagram](assets/diagram.png)\n\n'
        f'![logo]({raw_url})\n\n'
        '![missing](assets/missing.png)\n\n'
        '<video src="assets/demo.mp4"></video>\n'
    )
    fs = MagicMock()
    fs.resolve_target.return_value = _resolved_target()
    fs.read_bytes.side_effect = [
        markdown.encode(), b'diagram', b'logo', FileNotFoundError('missing image'),
    ]
    provider = GitHubWriterProvider()

    with patch.object(provider, '_fs', return_value=fs):
        loaded = provider.load_document(TargetDocument(
            uri=_resolved_target()['uri'],
            adapter='github',
            meta={'target_type': 'repository'},
        ))

    assert loaded['source_document'] == markdown
    assert [item.meta['source_reference'] for item in loaded['input_resources']] == [
        'assets/diagram.png', raw_url,
    ]
    assert loaded['resource_warnings'] == [
        'assets/missing.png: FileNotFoundError',
    ]


def test_repository_content_round_trip_restores_layout_and_code_fences():
    layout = (
        '<table><tr><td><a href="docs/assets/artifact.jpg">'
        '<img src="docs/assets/artifact.jpg" alt="Artifact" width="100%" /></a>'
        '<br/><sub>Original caption</sub></td></tr></table>'
    )
    markdown = f'# Original\n\n{layout}\n\n```mermaid\nA --> B\n```\n'
    fs = MagicMock()
    fs.resolve_target.return_value = _resolved_target()
    fs.read_bytes.side_effect = [markdown.encode(), b'image']
    fs.apply_document_patch.return_value = _write_result()
    provider = GitHubWriterProvider()

    with patch.object(provider, '_fs', return_value=fs):
        loaded = provider.load_document(TargetDocument(
            uri=_resolved_target()['uri'],
            adapter='github',
            meta={'target_type': 'repository'},
        ))
        target = loaded['target_document']
        preview = '/static-files/writer-preview-assets/aa/artifact.jpg?sig=x'
        library = MediaAssetLibrary(
            library_id='library-1',
            assets={
                'asset-1': MediaAsset(
                    media_asset_id='asset-1',
                    asset_type='image',
                    source_type='input_resource',
                    meta={
                        'source_reference': 'docs/assets/artifact.jpg',
                        'preview_reference': preview,
                    },
                ),
            },
        )
        edited = loaded['source_document'].replace(
            '# Original', '# Changed',
        ).replace('docs/assets/artifact.jpg', preview)
        result = provider.replace_document(edited, target, media_assets=library)

    assert '```text\nA --> B' in loaded['source_document']
    assert target.meta['github_writer_code_fences'][0]['language'] == 'mermaid'
    assert fs.apply_document_patch.call_args.args[1] == markdown.replace(
        '# Original', '# Changed',
    )
    assert result['commit_sha'] == 'commit-2'
    assert target.meta['work_branch'] == 'lazymind/op'


def test_generated_image_is_uploaded_with_repository_patch(tmp_path):
    image = tmp_path / 'generated.png'
    image_data = b'\x89PNG\r\n\x1a\nwriter-generated-image'
    image.write_bytes(image_data)
    digest = hashlib.sha256(image_data).hexdigest()
    preview = f'/static-files/writer-preview-assets/{digest[:2]}/{digest}.png?sig=x'
    library = MediaAssetLibrary(
        library_id='library-1',
        assets={
            'generated-1': MediaAsset(
                media_asset_id='generated-1',
                asset_type='generated_image',
                source_type='image_generation',
                local_path=str(image),
            ),
        },
    )
    fs = MagicMock()
    fs.apply_document_patch.return_value = _write_result()
    provider = GitHubWriterProvider()
    target = GitHubWriterProvider._target_from_resolved(_resolved_target())

    with patch.object(provider, '_fs', return_value=fs):
        provider.replace_document(
            f'# Report\n\n![chart]({preview})\n', target, media_assets=library,
        )

    reference = f'assets/{digest[:2]}/{digest}.png'
    assert fs.apply_document_patch.call_args.args[1] == (
        f'# Report\n\n![chart]({reference})\n'
    )
    assert fs.apply_document_patch.call_args.kwargs['files'] == {
        reference: image_data,
    }


def test_repository_document_creation_writes_final_file_once():
    fs = MagicMock()
    fs.resolve_create_parent.return_value = {
        'uri': 'https://github.com/acme/docs/tree/main/articles',
        'browser_url': 'https://github.com/acme/docs/tree/main/articles',
        'owner': 'acme',
        'repo': 'docs',
        'ref': 'main',
        'base_ref': 'main',
        'directory': 'articles',
        'revision': 'commit-1',
        'target_type': 'repository',
        'create_pending': True,
    }
    fs.resolve_create_target.return_value = {
        **fs.resolve_create_parent.return_value,
        'doc_id': 'acme/docs:main:articles/new.md',
        'uri': 'githubrepo:/acme/docs/articles/new.md?ref=main',
        'title': 'new',
        'path': 'articles/new.md',
        'publish_mode': 'pull_request',
        'create_pending': False,
    }
    fs.apply_document_patch.return_value = _write_result()
    provider = GitHubWriterProvider()

    with patch('lazyllm.tools.writer.provider.github.GitHubRepoFS') as fs_type:
        fs_type.matches_create_parent.return_value = True
        fs_type.return_value = fs
        target = provider._resolve_create_target(
            'https://github.com/acme/docs/tree/main/articles',
        )
        provider.replace_document('# New\n', target)

    fs.resolve_create_parent.assert_called_once()
    fs.resolve_create_target.assert_called_once()
    assert fs.apply_document_patch.call_count == 1
    assert target.meta['create_pending'] is False


def test_repository_patch_rejects_stale_revision():
    fs = GitHubRepoFS(token='test-token', skip_instance_cache=True)
    fs._branch_head = MagicMock(return_value='new-head')
    fs._commit_has_operation = MagicMock(return_value=False)

    with pytest.raises(GitHubFSError) as captured:
        fs.apply_document_patch(
            'githubrepo:/acme/docs/guide.md?ref=main',
            '# Updated',
            expected_revision='old-head',
            publish_mode='direct',
        )

    assert captured.value.code == 'GITHUB_REVISION_CONFLICT'


def test_repository_patch_creates_and_reuses_pull_request_branch():
    fs = GitHubRepoFS(token='test-token', skip_instance_cache=True)
    fs._branch_head = MagicMock(side_effect=[None, 'base-head'])
    fs._create_ref = MagicMock()
    fs._commit_files = MagicMock(return_value='commit-2')
    fs._update_ref = MagicMock()
    fs._ensure_pr = MagicMock(return_value={
        'number': 7,
        'html_url': 'https://github.com/acme/docs/pull/7',
    })

    created = fs.apply_document_patch(
        'githubrepo:/acme/docs/guide.md?ref=main',
        '# Updated',
        operation_id='operation-1',
    )

    assert created['work_branch'] == 'lazymind/operation-1'
    assert created['pull_request_url'].endswith('/pull/7')

    fs._branch_head = MagicMock(return_value='commit-2')
    fs._commit_has_operation = MagicMock(return_value=False)
    fs._commit_files = MagicMock(return_value='commit-3')
    continued = fs.apply_document_patch(
        'githubrepo:/acme/docs/guide.md?ref=lazymind%2Foperation-1',
        '# Updated again',
        expected_revision='commit-2',
        operation_id='operation-1',
        work_branch='lazymind/operation-1',
        base_ref='main',
    )

    assert continued['work_branch'] == 'lazymind/operation-1'
    assert continued['commit_sha'] == 'commit-3'


def test_wiki_document_creation_uses_direct_commit():
    fs = MagicMock()
    fs.resolve_create_parent.return_value = {
        'uri': 'https://github.com/acme/docs/wiki',
        'browser_url': 'https://github.com/acme/docs/wiki',
        'owner': 'acme',
        'repo': 'docs',
        'ref': 'master',
        'base_ref': 'master',
        'revision': 'commit-1',
        'target_type': 'wiki',
        'publish_mode': 'direct',
        'create_pending': True,
    }
    fs.resolve_create_target.return_value = {
        **fs.resolve_create_parent.return_value,
        'doc_id': 'acme/docs.wiki:New-Page.md',
        'uri': 'githubwiki:/acme/docs/New-Page.md',
        'title': 'New-Page',
        'path': 'New-Page.md',
        'create_pending': False,
    }
    fs.apply_document_patch.return_value = {
        'success': True,
        'provider': 'github',
        'target_type': 'wiki',
        'uri': 'githubwiki:/acme/docs/New-Page.md',
        'revision': 'commit-2',
        'commit_sha': 'commit-2',
        'publish_mode': 'direct',
        'warnings': [],
    }
    provider = GitHubWriterProvider()

    with (
        patch('lazyllm.tools.writer.provider.github.GitHubRepoFS') as repo_type,
        patch('lazyllm.tools.writer.provider.github.GitHubWikiFS') as wiki_type,
    ):
        repo_type.matches_create_parent.return_value = False
        wiki_type.matches_create_parent.return_value = True
        wiki_type.return_value = fs
        target = provider._resolve_create_target('https://github.com/acme/docs/wiki')
        provider.replace_document('# New Page\n', target)

    assert target.meta['target_type'] == 'wiki'
    assert fs.apply_document_patch.call_args.kwargs['publish_mode'] == 'direct'
