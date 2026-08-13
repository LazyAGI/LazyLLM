from __future__ import annotations

import hashlib
import ipaddress
import json
import socket
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urljoin, urlparse

import requests

from pydantic import BaseModel, Field

from lazyllm import config
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.thirdparty import PIL, mistune

from .base import WriterToolBase
from ..data_models.multimodal import (
    MediaAsset,
    MediaAssetLibrary,
    VisualPlan,
    _VISUAL_STRATEGY_ORDER,
)
from ..data_models.task import InputResource, WritingTask
from ..data_models.writer_ir import WriterDocument
from ..prompts import RESOLVE_VISUAL_NEEDS_PROMPT, VISION_SUMMARY_PROMPT


_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_IMAGE_SUFFIXES = {'.bmp', '.gif', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}
_MAX_REDIRECTS = 5


class _MediaSelections(BaseModel):
    selections: Dict[str, List[str]] = Field(default_factory=dict)


class WriterMultimodalTools(WriterToolBase):
    __public_apis__ = ['collect_available_media']

    def collect_available_media(  # noqa: C901
        self,
        task: Any,
        input_resources: Any = None,
        source_document: Any = None,
    ) -> dict:
        '''Collect user-provided and source-document images into the local media library.'''
        writing_task = self._unified_model(task, WritingTask)
        resources = [
            resource.model_copy(deep=True)
            for resource in [*writing_task.inputs, *self._unified_models(input_resources, InputResource)]
        ]
        library = MediaAssetLibrary(library_id=f'media-library-{writing_task.task_id or "task"}')
        warnings: List[str] = []
        pre_materialized_resource_ids: set[str] = set()

        if source_document is not None:
            try:
                document = self._unified_document(source_document)
                if isinstance(document, str):
                    embedded = self._markdown_image_resources(
                        document,
                        source_label=(
                            str(source_document) if isinstance(source_document, str) else 'source_document'
                        ),
                    )
                    self._extend_unique_resources(resources, embedded)
                else:
                    document_media, document_warnings = self._materialize_document_images(document)
                    warnings.extend(document_warnings)
                    for resource, asset in document_media:
                        self._register_asset(resource, asset, library, warnings)
                        if resource.resource_id:
                            pre_materialized_resource_ids.add(resource.resource_id)
                        self._extend_unique_resources(resources, [resource])
            except Exception as exc:
                warnings.append(f'Failed to collect source document images: {type(exc).__name__}: {exc}')

        for resource in resources:
            if resource.resource_id in pre_materialized_resource_ids or not self._is_image_resource(resource):
                continue
            label = resource.resource_id or resource.title or resource.uri or 'image resource'
            try:
                asset = self._materialize_input_resource(resource)
                self._register_asset(resource, asset, library, warnings)
            except Exception as exc:
                resource.resource_type = 'image'
                resource.meta['semantic_status'] = 'unknown'
                warnings.append(f'Failed to collect {label!r}: {type(exc).__name__}: {exc}')

        return self._save_artifacts(
            {'media_assets': library, 'profile_input_resources': resources},
            step_name='collect_available_media',
            primary_key='media_assets',
            context_key=None,
            summary='Collected available writing media.',
            warnings=warnings,
        ).model_dump()

    def resolve_visual_needs(
        self,
        visual_plan: Any,
        media_assets: Any,
        allowed_strategies: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        '''Reuse matching assets and request generated images for the remaining needs.'''
        plan = self._unified_model(visual_plan, VisualPlan)
        library = self._unified_model(media_assets, MediaAssetLibrary).model_copy(deep=True)
        needs = {need.need_id: need for need in plan.instructions}
        unresolved = []
        for need_id in needs:
            asset_ids = [
                asset_id for asset_id in library.visual_need_asset_ids.get(need_id, [])
                if asset_id in library.assets and self._asset_is_available(library.assets[asset_id])
            ]
            if asset_ids:
                library.visual_need_asset_ids[need_id] = asset_ids
            else:
                library.visual_need_asset_ids.pop(need_id, None)
                unresolved.append(need_id)

        available = [asset for asset in library.assets.values() if self._asset_is_available(asset)]
        warnings: List[str] = []
        if unresolved and available and self.llm is not None:
            try:
                selections = self._select_existing_assets(unresolved, needs, available)
                for need_id, asset_ids in selections.items():
                    selected = [
                        asset_id for asset_id in asset_ids
                        if asset_id in library.assets and self._asset_is_available(library.assets[asset_id])
                    ]
                    if need_id in unresolved and selected:
                        library.visual_need_asset_ids[need_id] = selected
            except Exception as exc:
                warnings.append(f'Existing media selection failed: {type(exc).__name__}: {exc}')

        acquisition_requests = []
        for need_id in unresolved:
            if library.visual_need_asset_ids.get(need_id):
                continue
            need = needs[need_id]
            strategies = self._visual_strategies(need, allowed_strategies)
            if not strategies:
                warnings.append(f'Visual need {need_id!r} has no MVP acquisition strategy.')
                continue
            acquisition_requests.append({
                'instruction_id': need_id,
                'visual_type': need.visual_type,
                'purpose': need.purpose,
                'strategies': strategies,
                'required': need.required,
            })

        return {
            'media_assets': library,
            'acquisition_requests': acquisition_requests,
            'warnings': warnings,
        }

    @staticmethod
    def _visual_strategies(need: Any, allowed_strategies: Optional[List[str]]) -> List[str]:
        strategies = list(_VISUAL_STRATEGY_ORDER.get(need.visual_type, []))
        if allowed_strategies is not None:
            strategies = [strategy for strategy in strategies if strategy in allowed_strategies]
        if need.preferred_strategy in strategies:
            strategies.remove(need.preferred_strategy)
            strategies.insert(0, need.preferred_strategy)
        return strategies

    def materialize_acquired_media(
        self,
        visual_plan: Any,
        media_assets: Any,
        acquired_resources: Any,
    ) -> Dict[str, Any]:
        '''Add acquired local images to the task library and bind them to visual needs.'''
        plan = self._unified_model(visual_plan, VisualPlan)
        library = self._unified_model(media_assets, MediaAssetLibrary).model_copy(deep=True)
        resources = self._unified_raw_data(acquired_resources) or {}
        if not isinstance(resources, dict):
            raise TypeError('acquired_resources must map visual need IDs to InputResource values.')

        warnings: List[str] = []
        for need in plan.instructions:
            if library.visual_need_asset_ids.get(need.need_id):
                continue
            resource = resources.get(need.need_id)
            if resource is None:
                if need.required:
                    warnings.append(f'Required visual need {need.need_id!r} remains unresolved.')
                continue
            try:
                asset = self._materialize_input_resource(self._unified_model(resource, InputResource))
                asset = library.assets.setdefault(asset.media_asset_id, asset)
                library.visual_need_asset_ids[need.need_id] = [asset.media_asset_id]
            except Exception as exc:
                warnings.append(f'Failed to materialize {need.need_id!r}: {type(exc).__name__}: {exc}')
        return {'media_assets': library, 'warnings': warnings}

    def _select_existing_assets(
        self,
        need_ids: List[str],
        needs: Dict[str, Any],
        assets: List[MediaAsset],
    ) -> Dict[str, List[str]]:
        prompt = RESOLVE_VISUAL_NEEDS_PROMPT.format(
            visual_needs_json=json.dumps([
                {'need_id': need_id, 'visual_type': needs[need_id].visual_type, 'purpose': needs[need_id].purpose}
                for need_id in need_ids
            ], ensure_ascii=False, indent=2),
            available_media_json=json.dumps([
                {
                    'media_asset_id': asset.media_asset_id,
                    'asset_type': asset.asset_type,
                    'caption': asset.caption,
                    'summary': asset.summary,
                    'semantic_status': asset.meta.get('semantic_status'),
                }
                for asset in assets
            ], ensure_ascii=False, indent=2),
        )
        return self._call_llm_structured(prompt, _MediaSelections).selections

    def _materialize_input_resource(self, resource: InputResource) -> MediaAsset:
        uri = str(resource.uri or '').strip()
        parsed = urlparse(uri)
        if not uri:
            raise ValueError('image resource URI is required.')
        if parsed.scheme in {'http', 'https'}:
            return self._materialize_image_bytes(
                self._download_external_image(uri),
                resource,
                suffix_hint=Path(parsed.path).suffix,
            )
        if parsed.scheme not in {'', 'file'}:
            raise ValueError('image inputs must use a local file path or an HTTP(S) URL.')
        source = Path(unquote(parsed.path) if parsed.scheme == 'file' else uri).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f'image file does not exist: {source}')
        if not 0 < source.stat().st_size <= _MAX_IMAGE_BYTES:
            raise ValueError('image file must be between 1 byte and 20 MB.')
        return self._materialize_image_bytes(source.read_bytes(), resource, suffix_hint=source.suffix)

    def _materialize_image_bytes(
        self,
        data: bytes,
        resource: InputResource,
        *,
        suffix_hint: str = '',
    ) -> MediaAsset:
        size = len(data)
        if not 0 < size <= _MAX_IMAGE_BYTES:
            raise ValueError('image file must be between 1 byte and 20 MB.')
        image_format, width, height = self._inspect_image_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        suffix = self._image_suffix(suffix_hint, image_format)
        destination = self._assets_dir() / f'{digest}{suffix}'
        if not destination.exists():
            destination.write_bytes(data)

        caption = str(resource.meta.get('caption') or '').strip() or None
        summary = str(resource.summary or '').strip()
        summary_source = 'resource_summary' if summary else 'filename'
        if not summary:
            summary = (
                caption or resource.title
                or f'Image {resource.uri!r}; image content has not been analyzed.'
            )
        source_type = str(resource.meta.get('source_type') or 'input_resource')
        source_meta = {
            key: resource.meta[key]
            for key in ('origin', 'provider', 'provider_block_id')
            if resource.meta.get(key) not in (None, '')
        }
        return MediaAsset(
            media_asset_id=f'asset-{digest}',
            asset_type='generated_image' if source_type == 'image_generation' else 'image',
            source_type=source_type,
            uri=resource.uri,
            local_path=str(destination),
            caption=caption,
            summary=summary,
            meta={
                **source_meta,
                'sha256': digest,
                'mime_type': PIL.Image.MIME.get(image_format),
                'byte_size': size,
                'width': width,
                'height': height,
                'summary_source': resource.meta.get('summary_source') or summary_source,
                'semantic_status': resource.meta.get('semantic_status') or (
                    'ready' if summary_source == 'resource_summary' else 'unknown'
                ),
            },
        )

    def _register_asset(
        self,
        resource: InputResource,
        asset: MediaAsset,
        library: MediaAssetLibrary,
        warnings: List[str],
    ) -> MediaAsset:
        label = resource.resource_id or resource.title or resource.uri or 'image resource'
        asset = library.assets.setdefault(asset.media_asset_id, asset)
        if asset.meta.get('semantic_status') != 'ready' and self.llm is not None:
            try:
                asset.summary = self._describe_image(asset.local_path or '')
                asset.meta.update(summary_source='vision_model', semantic_status='ready')
            except Exception as exc:
                warnings.append(f'Failed to understand {label!r}: {type(exc).__name__}: {exc}')
        resource.resource_type = 'image'
        resource.mime_type = asset.meta.get('mime_type') or resource.mime_type
        resource.summary = asset.summary
        resource.meta.update(
            summary_source=asset.meta.get('summary_source'),
            semantic_status=asset.meta.get('semantic_status'),
        )
        return asset

    def _materialize_document_images(
        self,
        document: WriterDocument,
    ) -> tuple[List[tuple[InputResource, MediaAsset]], List[str]]:
        provider = str(document.provider_binding.get('provider') or '').lower()
        image_blocks = [block for block in document.iter_blocks() if block.type == 'image']
        if not image_blocks or provider != 'feishu':
            return [], []

        locator = str(
            document.provider_binding.get('uri')
            or ((document.metadata.get('source') or {}).get('uri') if isinstance(
                document.metadata.get('source'), dict) else '')
            or f'feishu:/~docx/{document.provider_binding.get("document_id") or ""}'
        ).strip()
        import lazyllm.tools.fs.client as _fs_client
        protocol, space_id, real_path = _fs_client.FS._parse(locator)
        fs = _fs_client.FS._get_or_create_fs(protocol, space_id, real_path)
        download_media = getattr(fs, 'download_media', None)
        if not callable(download_media):
            return [], [f'{type(fs).__name__} does not support Feishu media downloads.']

        collected: List[tuple[InputResource, MediaAsset]] = []
        warnings: List[str] = []
        for block in image_blocks:
            raw = block.provider_payload.get('raw_block') or {}
            image = raw.get('image') if isinstance(raw, dict) else None
            token = str((image or {}).get('token') or '').strip()
            block_id = str(block.provider_binding.get('block_id') or block.node_id)
            if not token:
                warnings.append(f'Feishu image block {block_id!r} has no media token.')
                continue
            resource = InputResource(
                resource_id=f'feishu-image-{block_id}',
                resource_type='image',
                uri=f'{locator}#image={block_id}',
                title=block.content or f'Feishu image {block_id}',
                summary=block.content or None,
                meta={
                    'provider': 'feishu',
                    'provider_block_id': block_id,
                    'source_type': 'input_resource',
                    'origin': 'source_document',
                    'caption': block.content or None,
                },
            )
            try:
                asset = self._materialize_image_bytes(download_media(token), resource)
                collected.append((resource, asset))
            except Exception as exc:
                warnings.append(
                    f'Failed to download Feishu image {block_id!r}: '
                    f'{type(exc).__name__}: {exc}'
                )
        return collected, warnings

    @staticmethod
    def _markdown_image_resources(
        markdown: str,
        *,
        source_label: str,
    ) -> List[InputResource]:
        parser = mistune.create_markdown(renderer='ast')
        references: List[tuple[str, str, str]] = []

        def text_content(node: Dict[str, Any]) -> str:
            return str(node.get('raw') or '') + ''.join(
                text_content(child) for child in node.get('children') or []
                if isinstance(child, dict)
            )

        def walk(node: Dict[str, Any]) -> None:
            if node.get('type') == 'image':
                attrs = node.get('attrs') or {}
                url = str(attrs.get('url') or '').strip()
                if url:
                    references.append((url, text_content(node).strip(), str(attrs.get('title') or '').strip()))
            for child in node.get('children') or []:
                if isinstance(child, dict):
                    walk(child)

        for token in parser(markdown or ''):
            if isinstance(token, dict):
                walk(token)

        resources: List[InputResource] = []
        seen: set[str] = set()
        for raw_url, alt, title in references:
            uri = f'https:{raw_url}' if raw_url.startswith('//') else raw_url
            if urlparse(uri).scheme not in {'http', 'https'}:
                continue
            if uri in seen:
                continue
            seen.add(uri)
            digest = hashlib.sha256(f'{source_label}\0{uri}'.encode('utf-8')).hexdigest()[:16]
            resources.append(InputResource(
                resource_id=f'markdown-image-{digest}',
                resource_type='image',
                uri=uri,
                title=title or alt or Path(urlparse(uri).path).name or None,
                summary=alt or None,
                meta={
                    'caption': alt or None,
                    'source_type': 'input_resource',
                    'origin': 'markdown',
                },
            ))
        return resources

    @staticmethod
    def _extend_unique_resources(resources: List[InputResource], additions: List[InputResource]) -> None:
        known = {(resource.resource_id, resource.uri) for resource in resources}
        known_image_uris = {
            resource.uri for resource in resources
            if resource.resource_type == 'image' and resource.uri
        }
        for resource in additions:
            key = (resource.resource_id, resource.uri)
            if key not in known and not (
                resource.resource_type == 'image' and resource.uri in known_image_uris
            ):
                resources.append(resource)
                known.add(key)
                if resource.resource_type == 'image' and resource.uri:
                    known_image_uris.add(resource.uri)

    def _download_external_image(self, url: str) -> bytes:
        current = url
        for redirect_count in range(_MAX_REDIRECTS + 1):
            self._validate_remote_url(current)
            response = requests.get(
                current,
                timeout=(5, 30),
                stream=True,
                allow_redirects=False,
                headers={'Accept': 'image/*', 'User-Agent': 'LazyLLM-Writer/1.0'},
            )
            try:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get('Location')
                    if not location or redirect_count == _MAX_REDIRECTS:
                        raise ValueError(f'image URL exceeded {_MAX_REDIRECTS} redirects.')
                    current = urljoin(current, location)
                    continue
                response.raise_for_status()
                content_type = str(response.headers.get('Content-Type') or '').split(';', 1)[0].strip().lower()
                if content_type and not content_type.startswith('image/'):
                    raise ValueError(f'external image URL returned content type {content_type!r}.')
                content_length = response.headers.get('Content-Length')
                if content_length and int(content_length) > _MAX_IMAGE_BYTES:
                    raise ValueError('external image exceeds the 20 MB limit.')
                chunks: List[bytes] = []
                size = 0
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > _MAX_IMAGE_BYTES:
                        raise ValueError('external image exceeds the 20 MB limit.')
                    chunks.append(chunk)
                return b''.join(chunks)
            finally:
                response.close()
        raise ValueError(f'image URL exceeded {_MAX_REDIRECTS} redirects.')

    @staticmethod
    def _validate_remote_url(url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
            raise ValueError('external image URL must use HTTP or HTTPS.')
        if config['allow_internal_network']:
            return
        try:
            addresses = {
                item[4][0]
                for item in socket.getaddrinfo(
                    parsed.hostname,
                    parsed.port or (443 if parsed.scheme == 'https' else 80),
                    type=socket.SOCK_STREAM,
                )
            }
        except OSError as exc:
            raise ValueError(f'cannot resolve external image host {parsed.hostname!r}.') from exc
        if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
            raise ValueError(f'access to non-public image host {parsed.hostname!r} is not allowed.')

    def _describe_image(self, local_path: str) -> str:
        output = self.llm(
            encode_query_with_filepaths(VISION_SUMMARY_PROMPT, [local_path]),
            stream_output=False,
            llm_chat_history=[],
            lazyllm_files=None,
        )
        description = str(output).strip()
        if not description:
            raise ValueError('vision model returned an empty image description.')
        return description

    def _assets_dir(self) -> Path:
        if not self.artifact_store:
            raise ValueError('artifact_store is not set')
        path = Path(self.artifact_store).expanduser().resolve() / 'assets'
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _is_image_resource(resource: InputResource) -> bool:
        if resource.resource_type == 'image' or str(resource.mime_type or '').startswith('image/'):
            return True
        return (
            resource.resource_type == 'file'
            and Path(urlparse(resource.uri or '').path).suffix.lower() in _IMAGE_SUFFIXES
        )

    @staticmethod
    def _asset_is_available(asset: MediaAsset) -> bool:
        return bool(asset.local_path and Path(asset.local_path).is_file())

    @staticmethod
    def _inspect_image(path: Path) -> tuple[str, int, int]:
        return WriterMultimodalTools._inspect_image_bytes(path.read_bytes())

    @staticmethod
    def _inspect_image_bytes(data: bytes) -> tuple[str, int, int]:
        try:
            with PIL.Image.open(BytesIO(data)) as image:
                image_format = str(image.format or '').upper()
                width, height = image.size
                image.verify()
        except Exception as exc:
            raise ValueError('image data is not a valid image.') from exc
        if not image_format:
            raise ValueError('image format cannot be detected.')
        return image_format, width, height

    @staticmethod
    def _image_suffix(suffix_hint: str, image_format: str) -> str:
        suffix = str(suffix_hint or '').lower()
        if suffix in _IMAGE_SUFFIXES:
            return suffix
        return {
            'JPEG': '.jpg',
            'TIFF': '.tif',
        }.get(image_format, f'.{image_format.lower()}')
