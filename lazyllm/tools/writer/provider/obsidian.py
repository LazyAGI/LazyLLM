from __future__ import annotations

import hashlib
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote, unquote, urlparse

from lazyllm.tools.fs.supplier.obsidian import (
    OBSIDIAN_IMAGE_SUFFIXES,
    ObsidianFS,
    ObsidianNote,
)

from .base import (
    WriterProviderBase,
    WriterProviderCapabilities,
    WriterProviderDocument,
    WriterProviderWriteMode,
)
from ..data_models.multimodal import MediaAssetLibrary
from ..data_models.task import InputResource, TargetDocument
from ..data_models.writer_ir import WriterDocument, WriterStage
from ..utils import writer_document_to_markdown


_FRONTMATTER_RE = re.compile(r'\A---\r?\n.*?\r?\n---(?:\r?\n|\Z)', re.DOTALL)
_IMAGE_EMBED_RE = re.compile(r'!\[\[([^\]\n]+)\]\]')
_WRITER_SYSTEM_ANCHOR_LINE_RE = re.compile(
    r'^[ \t]*<a\s+id=(["\'])block-[^"\']+\1(?:[ \t]+[^>]*)?[ \t]*(?:/>|>[ \t]*</a>)[ \t]*(?:\r?\n|$)',
    re.MULTILINE,
)
_LOCAL_MARKDOWN_IMAGE_RE = re.compile(
    r'!\[(?P<alt>[^\]]*)\]\((?P<target><[^>\n]+>|[^)\s]+)(?:\s+["\'][^)]*["\'])?\)'
)


class ObsidianWriterProvider(WriterProviderBase):
    '''Bridge an Obsidian Markdown note through Writer's Markdown path.'''

    provider = 'obsidian'
    capabilities = WriterProviderCapabilities(
        load=True,
        create=True,
        replace=True,
        revision_check=True,
        media=True,
    )

    @classmethod
    def matches(cls, locator: str) -> bool:
        return str(locator or '').strip().lower().startswith('obsidian://')

    def resolve(self, locator: str) -> TargetDocument:
        value = str(locator or '').strip()
        if not self.matches(value):
            raise ValueError('Invalid Obsidian document locator.')
        return TargetDocument(uri=value, adapter=self.provider)

    def _convert_native_document(
        self,
        content: WriterDocument | str,
        *,
        target: TargetDocument | None = None,
        media_assets: MediaAssetLibrary | None = None,
    ) -> WriterProviderDocument:
        if isinstance(content, WriterDocument):
            markdown = self._serialize_writer_document(content, media_assets)
            source_document = content.model_copy(deep=True)
        elif isinstance(content, str):
            markdown = content
            source_document = self._writer_document(markdown, media_assets)
        else:
            raise TypeError('Obsidian Writer Provider accepts Markdown or WriterDocument content.')
        return WriterProviderDocument(
            provider=self.provider,
            format='markdown',
            content=markdown,
            source_document=source_document,
        )

    def write_document(
        self,
        converted: WriterProviderDocument,
        target: TargetDocument,
        *,
        media_assets: MediaAssetLibrary | None = None,
        mode: WriterProviderWriteMode = 'replace',
    ) -> dict:
        if converted.provider != self.provider or converted.format != 'markdown':
            raise ValueError('Obsidian write_document requires converted Obsidian Markdown.')
        if not isinstance(converted.content, str):
            raise TypeError('Converted Obsidian content must be a Markdown string.')
        if mode != 'replace':
            raise ValueError('Obsidian write_document only supports replace mode.')
        result = self.replace_document(
            converted.content,
            target,
            media_assets=media_assets,
        )
        return {
            **result,
            'persisted_document': converted.content,
            'representation': 'markdown',
            'published_link': '',
        }

    def load_document(
        self,
        target: TargetDocument,
        *,
        stage: WriterStage = 'final',
    ) -> dict:
        fs = self._fs()
        note, content = fs.read_note(str(target.uri or ''))
        markdown, bridge = self._to_writer_markdown(content, note, fs)
        resolved = target.model_copy(deep=True)
        resolved.doc_id = self._document_id(note)
        resolved.uri = self._canonical_uri(note)
        resolved.adapter = self.provider
        resolved.title = resolved.title or Path(note.relative_path).stem
        resolved.meta['obsidian_bridge'] = bridge
        resolved.meta['local_path'] = fs.display_note_path(note)
        resources = self._image_resources(bridge, resolved)
        return {
            'representation': 'markdown',
            'source_document': markdown,
            'target_document': resolved,
            'provider': self.provider,
            'block_count': len(markdown.splitlines()),
            'input_resources': resources,
            'resource_warnings': list(bridge.get('warnings') or []),
        }

    def create_document(self, title: str, parent_uri: str = '') -> TargetDocument:
        '''Create a note in the configured default Vault.

        Obsidian has no remote parent container to resolve here: the first
        discovered Vault is the explicit local default.
        '''
        fs = self._fs()
        note = fs.create_note(title)
        return TargetDocument(
            doc_id=self._document_id(note),
            uri=self._canonical_uri(note),
            adapter=self.provider,
            title=Path(note.relative_path).stem,
            meta={'local_path': fs.display_note_path(note)},
        )

    def replace_document(
        self,
        content: WriterDocument | str,
        target: TargetDocument,
        *,
        media_assets: MediaAssetLibrary | None = None,
    ) -> dict:
        if isinstance(content, WriterDocument):
            content = self._serialize_writer_document(content, media_assets)
        if not isinstance(content, str):
            raise TypeError('Obsidian only accepts Markdown content.')
        fs = self._fs()
        note = fs.resolve_locator(str(target.uri or ''))
        original = note.path.read_text(encoding='utf-8')
        bridge = dict(target.meta.get('obsidian_bridge') or {})
        restored = self._from_writer_markdown(content, bridge, note, fs, media_assets)
        warnings: List[str] = []
        if bridge.get('source_hash') and bridge['source_hash'] != self._hash(original):
            warnings.append('The Obsidian note changed after it was loaded; it was overwritten.')
        fs.write_note(note, restored)
        if bridge:
            bridge['source_hash'] = self._hash(restored)
            target.meta['obsidian_bridge'] = bridge
        local_path = fs.display_note_path(note)
        target.meta['local_path'] = local_path
        return {
            'doc_id': self._document_id(note),
            'adapter': self.provider,
            'locator': self._canonical_uri(note),
            'local_path': local_path,
            'block_count': len(restored.splitlines()),
            'warnings': warnings,
        }

    @staticmethod
    def _serialize_writer_document(
        document: WriterDocument,
        media_assets: MediaAssetLibrary | None,
    ) -> str:
        markdown_document = document.model_copy(deep=True)
        for block in markdown_document.iter_blocks():
            if block.type != 'image':
                continue
            reference = next(
                (
                    item for item in block.references
                    if item.get('type') == 'media_asset' and item.get('id')
                ),
                None,
            )
            asset = (
                media_assets.assets.get(str(reference['id']))
                if reference is not None and media_assets is not None
                else None
            )
            if asset is not None and asset.local_path:
                reference['path'] = asset.local_path
        return writer_document_to_markdown(markdown_document)

    @staticmethod
    def _fs() -> ObsidianFS:
        return ObsidianFS()

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    @staticmethod
    def _document_id(note: ObsidianNote) -> str:
        return f'{note.vault.vault_id}:{note.relative_path}'

    @staticmethod
    def _canonical_uri(note: ObsidianNote) -> str:
        return f'obsidian://{note.vault.vault_id}/{quote(note.relative_path, safe="/")}'

    @staticmethod
    def _writer_image_reference(note: ObsidianNote, source: Path) -> str:
        relative = Path(os.path.relpath(source, note.path.parent)).as_posix()
        return quote(relative, safe='/._-')

    @staticmethod
    def _image_resources(
        bridge: Dict[str, Any],
        target: TargetDocument,
    ) -> List[InputResource]:
        resources: List[InputResource] = []
        for index, (reference, item) in enumerate(
            dict(bridge.get('images') or {}).items(),
        ):
            if not isinstance(item, dict):
                continue
            resource_uri = str(item.get('resource_uri') or '').strip()
            if not resource_uri:
                continue
            path = Path(unquote(urlparse(resource_uri).path))
            resources.append(InputResource(
                resource_id=f'obsidian-image-{index:04d}',
                resource_type='image',
                uri=resource_uri,
                mime_type=mimetypes.guess_type(path.name)[0],
                title=path.name or None,
                summary=None,
                meta={
                    'provider': 'obsidian',
                    'origin': 'markdown',
                    'role': 'background',
                    'referenced_from': target.uri,
                    'source_reference': str(reference),
                },
            ))
        return resources

    def _to_writer_markdown(
        self,
        content: str,
        note: ObsidianNote,
        fs: ObsidianFS,
    ) -> tuple[str, Dict[str, Any]]:
        frontmatter = ''
        matched = _FRONTMATTER_RE.match(content)
        if matched:
            frontmatter = matched.group(0)
            content = content[matched.end():]
        bridge: Dict[str, Any] = {
            'source_hash': self._hash(frontmatter + content),
            'frontmatter': frontmatter,
            'images': {},
            'external_images': {},
            'warnings': [],
        }
        images: Dict[str, Dict[str, Any]] = bridge['images']
        external_images: Dict[str, str] = bridge['external_images']
        warnings: List[str] = bridge['warnings']

        def obsidian_image(match: re.Match[str]) -> str:
            raw = match.group(0)
            reference = match.group(1)
            target = reference.split('|', 1)[0].split('#', 1)[0].strip()
            suffix = Path(target).suffix.lower()
            if suffix in OBSIDIAN_IMAGE_SUFFIXES:
                alias = reference.partition('|')[2].strip()
                alt = alias if alias and not alias.isdigit() else ''
                return self._bridge_image(note, fs, images, warnings, raw, reference, alt, markdown_relative=False)
            return raw

        def local_markdown_image(match: re.Match[str]) -> str:
            raw = match.group(0)
            raw_target = match.group('target').strip()
            target = raw_target
            if target.startswith('<') and target.endswith('>'):
                target = target[1:-1].strip()
            if target in images:
                return raw
            parsed = urlparse(target)
            if parsed.scheme.lower() in {'http', 'https'} or target.startswith('//'):
                media_uri = f'https:{target}' if target.startswith('//') else target
                external_images[media_uri] = raw_target
                return raw
            if parsed.scheme:
                return raw
            suffix = Path(unquote(parsed.path)).suffix.lower()
            if suffix in OBSIDIAN_IMAGE_SUFFIXES:
                return self._bridge_image(
                    note, fs, images, warnings, raw,
                    target,
                    match.group('alt').strip(),
                    markdown_relative=True,
                )
            return raw

        content = _IMAGE_EMBED_RE.sub(obsidian_image, content)
        content = _LOCAL_MARKDOWN_IMAGE_RE.sub(local_markdown_image, content)
        return content, bridge

    def _from_writer_markdown(
        self,
        content: str,
        bridge: Dict[str, Any],
        note: ObsidianNote,
        fs: ObsidianFS,
        media_assets: MediaAssetLibrary | None,
    ) -> str:
        content = self._restore_images(content, bridge, note, fs, media_assets)
        content = _WRITER_SYSTEM_ANCHOR_LINE_RE.sub('', content)
        frontmatter = str(bridge.get('frontmatter') or '')
        if not content.endswith('\n'):
            content += '\n'
        return frontmatter + content

    def _restore_images(
        self,
        content: str,
        bridge: Dict[str, Any],
        note: ObsidianNote,
        fs: ObsidianFS,
        media_assets: MediaAssetLibrary | None,
    ) -> str:
        assets = list((media_assets.assets if media_assets else {}).values())
        images = {
            str(uri): {
                **dict(item),
                'raw_variants': list(item.get('raw_variants') or []),
            }
            for uri, item in dict(bridge.get('images') or {}).items()
            if isinstance(item, dict)
        }
        external_images = {
            str(uri): str(raw_uri)
            for uri, raw_uri in dict(bridge.get('external_images') or {}).items()
            if str(uri).strip() and str(raw_uri).strip()
        }

        def replacement(match: re.Match[str]) -> str:
            uri = match.group('target')
            reference = uri[1:-1].strip() if uri.startswith('<') and uri.endswith('>') else uri
            media_uri = f'https:{reference}' if reference.startswith('//') else reference
            original_external = external_images.get(media_uri)
            if original_external is None:
                for asset in assets:
                    if reference != str(asset.local_path or ''):
                        continue
                    original_external = external_images.get(str(asset.uri or ''))
                    if original_external is not None:
                        break
            if original_external is not None:
                return match.group(0).replace(uri, original_external, 1)
            original = self._bridged_image_raw(uri, images, assets)
            if original is not None:
                return original
            source = self._asset_path(reference, assets)
            if source is None:
                return match.group(0)
            return f'![[{fs.copy_attachment(note, source)}]]'

        return _LOCAL_MARKDOWN_IMAGE_RE.sub(replacement, content)

    @staticmethod
    def _bridged_image_raw(
        uri: str,
        images: Dict[str, Dict[str, Any]],
        assets: list[Any],
    ) -> str | None:
        def consume(item: Dict[str, Any]) -> str | None:
            variants = item.get('raw_variants') or []
            if variants:
                raw = str(variants.pop(0) or '')
            else:
                raw = str(item.get('raw') or '')
            return raw or None

        for candidate in (uri, unquote(uri)):
            item = images.get(candidate)
            raw = consume(item) if item else None
            if raw:
                return raw
        for asset in assets:
            local_path = str(asset.local_path or '')
            if uri != local_path and uri != str(asset.uri or ''):
                continue
            metadata = getattr(asset, 'meta', {}) or {}
            source_reference = str(metadata.get('source_reference') or '')
            item = images.get(source_reference)
            if not item:
                continue
            raw = consume(item)
            if raw:
                return raw
        return None

    @staticmethod
    def _asset_path(uri: str, assets: list[Any]) -> Path | None:
        raw = str(uri or '').strip()
        lowered = raw.lower()
        if lowered.startswith(('http://', 'https://')):
            candidates = {raw}
        elif lowered.startswith('file://'):
            candidates = {raw, unquote(urlparse(raw).path)}
        else:
            candidates = {raw, unquote(raw)}

        for asset in assets:
            values = {str(asset.uri or ''), str(asset.local_path or '')}
            if candidates.isdisjoint(values):
                continue
            local = Path(str(asset.local_path or ''))
            if local.is_file():
                return local
        return None

    def _bridge_image(
        self, note, fs, images, warnings,
        raw: str,
        reference: str,
        alt: str,
        *,
        markdown_relative: bool,
    ) -> str:
        try:
            source = fs.resolve_image_reference(
                note,
                reference,
                markdown_relative=markdown_relative,
            )
        except (FileNotFoundError, ValueError) as exc:
            warnings.append(f'Obsidian image was kept without import: {exc}')
            return raw
        writer_reference = self._writer_image_reference(note, source)
        image = images.setdefault(
            writer_reference,
            {
                'raw': raw,
                'resource_uri': source.as_uri(),
                'raw_variants': [],
            },
        )
        image.setdefault('raw_variants', []).append(raw)
        return f'![{alt or source.stem}]({writer_reference})'


__all__ = ['ObsidianWriterProvider']
