from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.thirdparty import PIL

from .base import WriterToolBase
from ..data_models.multimodal import (
    MediaAsset,
    MediaAssetLibrary,
    VisualPlan,
    _VISUAL_STRATEGY_ORDER,
)
from ..data_models.task import InputResource, WritingTask
from ..prompts import RESOLVE_VISUAL_NEEDS_PROMPT, VISION_SUMMARY_PROMPT


_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_IMAGE_SUFFIXES = {'.bmp', '.gif', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}


class _MediaSelections(BaseModel):
    selections: Dict[str, List[str]] = Field(default_factory=dict)


class WriterMultimodalTools(WriterToolBase):
    __public_apis__ = ['collect_available_media']

    def collect_available_media(self, task: Any, input_resources: Any = None) -> dict:
        '''Collect local images already available to this writing task.'''
        writing_task = self._unified_model(task, WritingTask)
        resources = [
            resource.model_copy(deep=True)
            for resource in [*writing_task.inputs, *self._unified_models(input_resources, InputResource)]
        ]
        library = MediaAssetLibrary(library_id=f'media-library-{writing_task.task_id or "task"}')
        warnings: List[str] = []

        for resource in resources:
            if not self._is_image_resource(resource):
                continue
            label = resource.resource_id or resource.title or resource.uri or 'image resource'
            try:
                asset = self._materialize_input_resource(resource)
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
        if not uri or parsed.scheme not in {'', 'file'}:
            raise ValueError('MVP image inputs must use a local file path.')
        source = Path(parsed.path if parsed.scheme == 'file' else uri).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f'image file does not exist: {source}')
        size = source.stat().st_size
        if not 0 < size <= _MAX_IMAGE_BYTES:
            raise ValueError('image file must be between 1 byte and 20 MB.')
        image_format, width, height = self._inspect_image(source)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        suffix = source.suffix.lower() or f'.{image_format.lower()}'
        destination = self._assets_dir() / f'{digest}{suffix}'
        if not destination.exists():
            shutil.copyfile(source, destination)

        caption = str(resource.meta.get('caption') or '').strip() or None
        summary = str(resource.summary or '').strip()
        summary_source = 'resource_summary' if summary else 'filename'
        if not summary:
            summary = (
                caption or resource.title
                or f'Image file {source.name!r}; image content has not been analyzed.'
            )
        source_type = str(resource.meta.get('source_type') or 'input_resource')
        return MediaAsset(
            media_asset_id=f'asset-{digest}',
            asset_type='generated_image' if source_type == 'image_generation' else 'image',
            source_type=source_type,
            uri=uri,
            local_path=str(destination),
            caption=caption,
            summary=summary,
            meta={
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
        try:
            with PIL.Image.open(path) as image:
                image_format = str(image.format or '').upper()
                width, height = image.size
                image.verify()
        except Exception as exc:
            raise ValueError(f'file is not a valid image: {path}') from exc
        if not image_format:
            raise ValueError(f'image format cannot be detected: {path}')
        return image_format, width, height
