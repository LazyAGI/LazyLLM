from unittest.mock import MagicMock

import pytest

from lazyllm.tools.fs.supplier import obsidian as obsidian_fs
from lazyllm.tools.fs.supplier.obsidian import ObsidianFS, ObsidianNote, ObsidianVault
from lazyllm.tools.writer.data_models.multimodal import MediaAsset, MediaAssetLibrary
from lazyllm.tools.writer.data_models.task import TargetDocument
from lazyllm.tools.writer.data_models.writer_ir import WriterDocument
from lazyllm.tools.writer.provider.base import WriterProviderDocument
from lazyllm.tools.writer.provider.obsidian import ObsidianWriterProvider


def _note(tmp_path) -> ObsidianNote:
    vault = ObsidianVault(
        vault_id='vlt_test',
        root=tmp_path,
        display_name='test',
    )
    path = tmp_path / 'note.md'
    path.write_text('', encoding='utf-8')
    return ObsidianNote(vault=vault, relative_path='note.md', path=path)


def _vault(path) -> None:
    (path / '.obsidian').mkdir(parents=True)


class TestObsidianVaultDiscovery:
    def test_discovers_nested_vaults(self, tmp_path):
        root = tmp_path / 'scan-root'
        first = root / 'Documents' / 'obs'
        second = root / 'Work' / 'knowledge'
        _vault(first)
        _vault(second)

        vaults = ObsidianFS.discover_vaults_for_root(str(root))

        assert {item.root for item in vaults} == {first, second}

    def test_discovers_a_scan_root_that_is_a_vault_without_descending_into_it(self, tmp_path):
        root = tmp_path / 'obs'
        nested = root / 'nested'
        _vault(root)
        _vault(nested)

        vaults = ObsidianFS.discover_vaults_for_root(str(root))

        assert [item.root for item in vaults] == [root]

    def test_vault_discovery_cache_expires(self, tmp_path, monkeypatch):
        root = tmp_path / 'scan-root'
        first = root / 'first'
        second = root / 'second'
        _vault(first)
        now = [100.0]
        monkeypatch.setattr(obsidian_fs.time, 'monotonic', lambda: now[0])

        first_scan = ObsidianFS.discover_vaults_for_root(str(root))
        _vault(second)
        cached_scan = ObsidianFS.discover_vaults_for_root(str(root))
        now[0] += 31.0
        refreshed_scan = ObsidianFS.discover_vaults_for_root(str(root))

        assert {item.root for item in first_scan} == {first}
        assert {item.root for item in cached_scan} == {first}
        assert {item.root for item in refreshed_scan} == {first, second}

    def test_scan_root_rejects_generic_file_operations(self, tmp_path):
        root = tmp_path / 'scan-root'
        _vault(root / 'obs')
        outside = root / 'outside.txt'
        outside.write_text('original', encoding='utf-8')
        fs = ObsidianFS(token=str(root))

        try:
            fs.ls('')
        except PermissionError as exc:
            assert 'scan-root mode' in str(exc)
        else:
            raise AssertionError('scan-root generic listing should be rejected')

        try:
            fs.write('outside.txt', 'changed')
        except PermissionError as exc:
            assert 'scan-root mode' in str(exc)
        else:
            raise AssertionError('scan-root generic writes should be rejected')
        assert outside.read_text(encoding='utf-8') == 'original'

    def test_resolve_image_reference_rejects_obsidian_internal_files(self, tmp_path):
        root = tmp_path / 'vault'
        _vault(root)
        note = _note(root)
        internal_image = root / '.obsidian' / 'plugins' / 'example' / 'icon.png'
        internal_image.parent.mkdir(parents=True)
        internal_image.write_bytes(b'internal')
        fs = ObsidianFS(token=str(root))

        with pytest.raises(FileNotFoundError):
            fs.resolve_image_reference(note, '.obsidian/plugins/example/icon.png')


class TestObsidianDisplayPath:
    def test_returns_the_real_path_without_a_host_mapping(self, tmp_path):
        root = tmp_path / 'scan-root'
        vault_root = root / 'obs'
        _vault(vault_root)
        note = _note(vault_root)
        fs = ObsidianFS(token=str(root))

        with obsidian_fs.config.temp('obsidian_host_root', None):
            assert fs.display_note_path(note) == str(note.path)

    def test_maps_a_container_note_path_to_the_host_scan_root(self, tmp_path):
        root = tmp_path / 'mounted-obsidian'
        vault_root = root / 'obs'
        _vault(vault_root)
        note = _note(vault_root)
        fs = ObsidianFS(token=str(root))

        with obsidian_fs.config.temp('obsidian_host_root', '/Users/test/Documents'):
            assert fs.display_note_path(note) == '/Users/test/Documents/obs/note.md'


class TestObsidianWriterProvider:
    def test_native_markdown_provider_contract(self):
        provider = ObsidianWriterProvider()
        converted = provider.convert_document('# Note\n')

        assert isinstance(converted, WriterProviderDocument)
        assert converted.provider == 'obsidian'
        assert converted.format == 'markdown'
        assert converted.content == '# Note\n'

        document = WriterDocument(
            document_id='writer-document',
            title='Note',
            blocks=[{'node_id': 'body', 'type': 'paragraph', 'content': 'Body'}],
        )
        converted_ir = provider.convert_document(document)
        assert converted_ir.format == 'markdown'
        assert converted_ir.content == '# Note\n\nBody\n'
        assert converted_ir.source_document.document_id == 'writer-document'

    def test_write_document_delegates_native_markdown_write(self, monkeypatch):
        provider = ObsidianWriterProvider()
        target = TargetDocument(adapter='obsidian', uri='obsidian://vlt_test/note.md')
        converted = provider.convert_document('# Note\n', target=target)
        captured = {}

        def replace_document(content, write_target, *, media_assets=None):
            captured.update({
                'content': content,
                'target': write_target,
                'media_assets': media_assets,
            })
            return {'doc_id': 'vlt_test:note.md', 'adapter': 'obsidian', 'locator': target.uri}

        monkeypatch.setattr(provider, 'replace_document', replace_document)

        result = provider.write_document(converted, target)

        assert captured['content'] == '# Note\n'
        assert captured['target'] is target
        assert result['persisted_document'] == '# Note\n'
        assert result['representation'] == 'markdown'
        assert result['published_link'] == ''

    def test_write_result_includes_the_host_local_path(self, tmp_path, monkeypatch):
        root = tmp_path / 'scan-root'
        vault_root = root / 'obs'
        _vault(vault_root)
        note = _note(vault_root)
        note.path.write_text('Before\n', encoding='utf-8')
        fs = ObsidianFS(token=str(root))
        provider = ObsidianWriterProvider()
        vault = fs.discover_vaults()[0]
        target = TargetDocument(
            uri=provider._canonical_uri(ObsidianNote(vault=vault, relative_path='note.md', path=note.path)),
            adapter='obsidian',
            meta={'obsidian_bridge': {'source_hash': provider._hash('Before\n')}},
        )
        monkeypatch.setattr(ObsidianWriterProvider, '_fs', staticmethod(lambda: fs))

        with obsidian_fs.config.temp('obsidian_host_root', '/Users/test/Documents'):
            result = provider.replace_document('After', target)

        assert result['local_path'] == '/Users/test/Documents/obs/note.md'
        assert target.meta['local_path'] == '/Users/test/Documents/obs/note.md'
        assert target.meta['obsidian_bridge']['source_hash'] == provider._hash('After\n')
        assert note.path.read_text(encoding='utf-8') == 'After\n'

    def test_create_target_keeps_the_host_local_path(self, tmp_path, monkeypatch):
        root = tmp_path / 'scan-root'
        vault_root = root / 'obs'
        _vault(vault_root)
        fs = ObsidianFS(token=str(root))
        provider = ObsidianWriterProvider()
        monkeypatch.setattr(ObsidianWriterProvider, '_fs', staticmethod(lambda: fs))

        with obsidian_fs.config.temp('obsidian_host_root', '/Users/test/Documents'):
            target = provider.create_document('New Note')

        assert target.meta['local_path'] == '/Users/test/Documents/obs/New Note.md'

    def test_canonical_uri_escapes_and_resolves_special_path(self, tmp_path, monkeypatch):
        provider = ObsidianWriterProvider()
        vault_root = tmp_path / 'vault'
        _vault(vault_root)
        note_path = vault_root / 'Folder' / 'Project Note #?.md'
        note_path.parent.mkdir()
        note_path.write_text('', encoding='utf-8')
        vault = ObsidianVault(vault_id='vlt_test', root=vault_root, display_name='vault')
        note = ObsidianNote(
            vault=vault,
            relative_path='Folder/Project Note #?.md',
            path=note_path,
        )

        uri = provider._canonical_uri(note)
        fs = ObsidianFS(token=str(vault_root))
        monkeypatch.setattr(fs, 'discover_vaults', lambda: [vault])

        assert uri == 'obsidian://vlt_test/Folder/Project%20Note%20%23%3F.md'
        assert fs.resolve_locator(uri) == note

    def test_native_obsidian_syntax_passes_through_without_bridge_markers(self, tmp_path):
        provider = ObsidianWriterProvider()
        note = _note(tmp_path)
        source = (
            '> [!note]+ Original title\n'
            '> Original body\n'
            '>\n'
            '> Second paragraph\n'
            '\n'
            'See [[Original target|Visible label]].\n'
            '%% comment %%\n'
            '^block-id\n'
            '```dataview\n'
            'LIST FROM #project\n'
            '```\n'
            '![[embedded-note]]\n'
        )

        markdown, bridge = provider._to_writer_markdown(source, note, MagicMock())

        assert markdown == source
        assert 'tokens' not in bridge
        assert 'block-obsidian-' not in markdown

        restored = provider._from_writer_markdown(
            markdown.replace('Original title', 'Edited title').replace('Original body', 'Edited body'),
            bridge,
            note,
            MagicMock(),
            None,
        )

        assert restored == (
            '> [!note]+ Edited title\n'
            '> Edited body\n'
            '>\n'
            '> Second paragraph\n'
            '\n'
            'See [[Original target|Visible label]].\n'
            '%% comment %%\n'
            '^block-id\n'
            '```dataview\n'
            'LIST FROM #project\n'
            '```\n'
            '![[embedded-note]]\n'
        )

    def test_writer_output_is_not_repaired_with_hidden_tokens(self, tmp_path):
        provider = ObsidianWriterProvider()
        note = _note(tmp_path)
        markdown, bridge = provider._to_writer_markdown('> [!warning]- Title\n> Body\n', note, MagicMock())
        output = markdown.replace('> [!warning]- ', '')

        restored = provider._from_writer_markdown(output, bridge, note, MagicMock(), None)

        assert restored == 'Title\n> Body\n'

    def test_load_returns_local_vault_images_as_file_resources(self, tmp_path, monkeypatch):
        provider = ObsidianWriterProvider()
        root = tmp_path / 'scan-root'
        vault_root = root / 'obs'
        _vault(vault_root)
        note_path = vault_root / 'note.md'
        image = vault_root / 'diagram.png'
        note_path.write_text('![[diagram.png]]\n', encoding='utf-8')
        image.write_bytes(b'not-inspected-by-this-bridge')
        fs = ObsidianFS(token=str(root))
        monkeypatch.setattr(ObsidianWriterProvider, '_fs', staticmethod(lambda: fs))
        target = TargetDocument(
            uri='obsidian://' + fs.discover_vaults()[0].vault_id + '/note.md',
            adapter='obsidian',
        )

        loaded = provider.load_document(target)

        assert loaded['source_document'] == '![diagram](diagram.png)\n'
        assert len(loaded['input_resources']) == 1
        assert loaded['input_resources'][0].uri == image.as_uri()
        assert loaded['input_resources'][0].meta['source_reference'] == 'diagram.png'
        assert loaded['resource_warnings'] == []
        assert loaded['target_document'].meta['local_path'] == str(note_path)

    def test_bridged_vault_image_restores_raw_embed_after_media_materialization(self, tmp_path):
        provider = ObsidianWriterProvider()
        note = _note(tmp_path)
        image = tmp_path / 'diagram.png'
        image.write_bytes(b'not-inspected-by-this-bridge')
        workspace_image = tmp_path / 'writer-media.png'
        workspace_image.write_bytes(b'writer-media')
        fs = MagicMock()
        fs.resolve_image_reference.return_value = image
        markdown, bridge = provider._to_writer_markdown('![[diagram.png]]\n', note, fs)
        media_assets = MediaAssetLibrary(
            library_id='media-library-test',
            assets={
                'asset-obsidian-test': MediaAsset(
                    media_asset_id='asset-obsidian-test',
                    asset_type='image',
                    source_type='input_resource',
                    uri=image.as_uri(),
                    local_path=str(workspace_image),
                    meta={'source_reference': 'diagram.png'},
                ),
            },
        )

        restored = provider._from_writer_markdown(
            f'![diagram]({workspace_image})\n', bridge, note, fs, media_assets,
        )

        assert restored == '![[diagram.png]]\n'

    def test_unmaterialized_vault_image_restores_raw_embed_before_presentation(self, tmp_path):
        provider = ObsidianWriterProvider()
        note = _note(tmp_path)
        image = tmp_path / 'diagram.png'
        image.write_bytes(b'not-inspected-by-this-bridge')
        fs = MagicMock()
        fs.resolve_image_reference.return_value = image
        markdown, bridge = provider._to_writer_markdown('![[diagram.png]]\n', note, fs)

        restored = provider._from_writer_markdown(
            markdown, bridge, note, fs, MediaAssetLibrary(library_id='media-library-test'),
        )

        assert restored == '![[diagram.png]]\n'

    def test_duplicate_image_aliases_round_trip_in_source_order(self, tmp_path):
        provider = ObsidianWriterProvider()
        note = _note(tmp_path)
        image = tmp_path / 'diagram.png'
        image.write_bytes(b'diagram')
        fs = MagicMock()
        fs.resolve_image_reference.return_value = image
        source = '![[diagram.png|原图]]\n![[diagram.png|放大图]]\n'

        markdown, bridge = provider._to_writer_markdown(source, note, fs)
        restored = provider._from_writer_markdown(markdown, bridge, note, fs, None)

        assert restored == source

    def test_restore_preserves_markdown_whitespace(self, tmp_path):
        provider = ObsidianWriterProvider()
        note = _note(tmp_path)
        content = '  first line\nsecond line  \n\n'

        restored = provider._from_writer_markdown(content, {}, note, MagicMock(), None)

        assert restored == content

    def test_existing_external_image_keeps_its_url_on_write_back(self, tmp_path):
        provider = ObsidianWriterProvider()
        note = _note(tmp_path)
        source_url = 'https://cdn.example.com/original.png'
        source = f'![Original]({source_url} "source title")\n'
        markdown, bridge = provider._to_writer_markdown(source, note, MagicMock())
        fs = MagicMock()

        restored = provider._from_writer_markdown(
            f'![Edited]({source_url} "edited title")\n',
            bridge,
            note,
            fs,
            None,
        )

        assert markdown == source
        assert bridge['external_images'] == {source_url: source_url}
        assert restored == f'![Edited]({source_url} "edited title")\n'
        fs.copy_attachment.assert_not_called()

    def test_existing_external_image_keeps_its_url_after_media_reuse(self, tmp_path):
        provider = ObsidianWriterProvider()
        note = _note(tmp_path)
        source_url = 'https://cdn.example.com/original.png'
        _, bridge = provider._to_writer_markdown(f'![Original]({source_url})\n', note, MagicMock())
        workspace_image = tmp_path / 'writer-media.png'
        workspace_image.write_bytes(b'writer-media')
        media_assets = MediaAssetLibrary(
            library_id='media-library-test',
            assets={
                'asset-external-test': MediaAsset(
                    media_asset_id='asset-external-test',
                    asset_type='image',
                    source_type='input_resource',
                    uri=source_url,
                    local_path=str(workspace_image),
                ),
            },
        )
        fs = MagicMock()

        restored = provider._from_writer_markdown(
            f'![Edited]({workspace_image})\n',
            bridge,
            note,
            fs,
            media_assets,
        )

        assert restored == f'![Edited]({source_url})\n'
        fs.copy_attachment.assert_not_called()

    def test_new_media_image_still_copies_into_the_vault(self, tmp_path):
        provider = ObsidianWriterProvider()
        note = _note(tmp_path)
        workspace_image = tmp_path / 'generated.png'
        workspace_image.write_bytes(b'generated')
        media_assets = MediaAssetLibrary(
            library_id='media-library-test',
            assets={
                'asset-generated-test': MediaAsset(
                    media_asset_id='asset-generated-test',
                    asset_type='image',
                    source_type='image_generation',
                    local_path=str(workspace_image),
                ),
            },
        )
        fs = MagicMock()
        fs.copy_attachment.return_value = 'assets/lazymind/generated.png'

        restored = provider._from_writer_markdown(
            f'![Generated]({workspace_image})\n',
            {},
            note,
            fs,
            media_assets,
        )

        assert restored == '![[assets/lazymind/generated.png]]\n'
        fs.copy_attachment.assert_called_once_with(note, workspace_image)

    def test_unregistered_local_image_is_not_copied_into_the_vault(self, tmp_path):
        provider = ObsidianWriterProvider()
        note = _note(tmp_path)
        local_image = tmp_path / 'unregistered.png'
        local_image.write_bytes(b'unregistered')
        fs = MagicMock()
        markdown = f'![Unregistered]({local_image.as_uri()})\n'

        restored = provider._from_writer_markdown(
            markdown,
            {},
            note,
            fs,
            MediaAssetLibrary(library_id='media-library-test'),
        )

        assert restored == markdown
        fs.copy_attachment.assert_not_called()
