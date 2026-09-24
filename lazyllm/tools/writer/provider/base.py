from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field
import requests

from ..data_models.multimodal import MediaAssetLibrary
from ..data_models.revision import PatchSet
from ..data_models.task import InputResource, TargetDocument
from ..data_models.writer_ir import WriterDocument, WriterStage
from ..utils.conversion import writer_document_from_markdown
from ..utils.export import export_writer_document


WriterProviderCapability = Literal[
    'load', 'create', 'replace', 'append', 'patch', 'revision_check', 'media',
]
WriterProviderWriteMode = Literal['replace', 'append']


class WriterProviderCapabilities(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    load: bool = False
    create: bool = False
    replace: bool = False
    append: bool = False
    patch: bool = False
    revision_check: bool = False
    media: bool = False


class WriterProviderDocument(BaseModel):
    model_config = ConfigDict(extra='forbid')

    provider: str
    format: str
    content: Any
    source_document: WriterDocument
    media_references: dict[str, str] = Field(default_factory=dict)


class WriterProviderCapabilityError(RuntimeError):
    code = 'PROVIDER_CAPABILITY_UNSUPPORTED'
    retryable = False

    def __init__(self, provider: str, capability: WriterProviderCapability):
        self.provider = provider
        self.capability = capability
        self.details = {'provider': provider, 'capability': capability}
        super().__init__(provider, capability)

    def __str__(self) -> str:
        return f'Writer provider {self.provider!r} does not support {self.capability!r}.'


class WriterProviderRevisionError(RuntimeError):
    code = 'REVISION_CONFLICT'
    retryable = False

    def __init__(self, provider: str, expected: str | None, actual: str | None):
        self.provider = provider
        self.expected = expected
        self.actual = actual
        self.details = {
            'provider': provider,
            'expected_revision': expected,
            'actual_revision': actual,
        }
        super().__init__(provider, expected, actual)

    def __str__(self) -> str:
        return (
            f'Writer provider {self.provider!r} document changed since it was loaded: '
            f'expected {self.expected!r}, got {self.actual!r}.'
        )


class WriterProviderWriteOutcomeError(RuntimeError):
    code = 'PROVIDER_WRITE_OUTCOME_AMBIGUOUS'
    retryable = False

    def __init__(self, provider: str, operation: str):
        self.provider = provider
        self.operation = operation
        self.details = {'provider': provider, 'operation': operation}
        super().__init__(provider, operation)

    def __str__(self) -> str:
        return (
            f'Writer provider {self.provider!r} did not confirm the {self.operation!r} outcome; '
            'inspect the remote document before retrying.'
        )


def is_ambiguous_write_error(error: Exception) -> bool:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, WriterProviderWriteOutcomeError):
            return True
        if isinstance(current, (
            TimeoutError,
            ConnectionError,
            requests.Timeout,
            requests.ConnectionError,
            json.JSONDecodeError,
        )):
            return True
        status_code = getattr(current, 'status_code', None)
        response = getattr(current, 'response', None)
        if not isinstance(status_code, int):
            status_code = getattr(response, 'status_code', None)
        if isinstance(status_code, int) and status_code >= 500:
            return True
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    return False


class WriterProviderBase(ABC):
    '''Read and persist Writer content through one external document provider.'''

    provider: str = ''
    capabilities = WriterProviderCapabilities()

    def __init__(self, adapters=None):
        self.adapters = adapters or {}

    def require_capability(self, capability: WriterProviderCapability) -> None:
        if not getattr(self.capabilities, capability):
            raise WriterProviderCapabilityError(
                self.provider or type(self).__name__, capability,
            )

    @staticmethod
    def _writer_document(
        content: WriterDocument | str,
        media_assets: MediaAssetLibrary | None,
        *,
        document_id: str = '',
    ) -> WriterDocument:
        if isinstance(content, WriterDocument):
            return content.model_copy(deep=True)
        from ..utils import parse_document_markdown
        document = parse_document_markdown(
            content,
            document_id=document_id or f'writer-document-{uuid.uuid4()}',
            stage='final',
            media_assets=media_assets,
        )
        if media_assets is not None:
            for block in document.iter_blocks():
                if block.type != 'image':
                    continue
                for reference in block.references:
                    asset = media_assets.assets.get(reference.get('id'))
                    if asset is not None and asset.uri:
                        reference.setdefault('path', asset.uri)
        return document

    @staticmethod
    def _copyable_media_content(
        content: Any,
        media_assets: MediaAssetLibrary | None,
    ) -> Any:
        value = deepcopy(content)

        def visit(item: Any) -> None:
            if isinstance(item, list):
                for child in item:
                    visit(child)
                return
            if not isinstance(item, dict):
                return
            media = item.get('_media')
            if isinstance(media, dict):
                asset_id = str(media.get('media_asset_id') or '')
                asset = media_assets.assets.get(asset_id) if media_assets and asset_id else None
                uri = str(asset.uri or asset.local_path or '') if asset else ''
                media.pop('local_path', None)
                if uri:
                    media['uri'] = uri
            for child in item.values():
                visit(child)

        visit(value)
        return value

    @staticmethod
    def _writable_media_content(
        content: Any,
        media_assets: MediaAssetLibrary | None,
    ) -> Any:
        value = deepcopy(content)

        def visit(item: Any) -> None:
            if isinstance(item, list):
                for child in item:
                    visit(child)
                return
            if not isinstance(item, dict):
                return
            media = item.get('_media')
            if isinstance(media, Mapping):
                asset_id = str(media.get('media_asset_id') or '')
                asset = media_assets.assets.get(asset_id) if media_assets and asset_id else None
                path = Path(asset.local_path) if asset and asset.local_path else None
                if path is None or not path.is_file():
                    raise ValueError(f'Image media asset {asset_id!r} is unavailable for writing.')
                item['_media'] = {
                    'media_asset_id': asset_id,
                    'local_path': str(path),
                    'file_name': path.name,
                }
            for child in item.values():
                visit(child)

        visit(value)
        return value

    def convert_document(
        self,
        content: WriterDocument | str,
        *,
        target: TargetDocument | None = None,
        media_assets: MediaAssetLibrary | None = None,
        output_format: str = 'native',
    ) -> WriterProviderDocument:
        if output_format != 'native':
            return self.convert_common_document(content, output_format=output_format, media_assets=media_assets)
        return self._convert_native_document(content, target=target, media_assets=media_assets)

    @classmethod
    def convert_common_document(
        cls,
        content: WriterDocument | str,
        *,
        output_format: str,
        media_assets: MediaAssetLibrary | None = None,
    ) -> WriterProviderDocument:
        document = (writer_document_from_markdown(content) if isinstance(content, str)
                    else content.model_copy(deep=True))
        if media_assets:
            for block in document.iter_blocks():
                for reference in block.references:
                    asset = media_assets.assets.get(reference.get('id'))
                    if asset and (asset.uri or asset.local_path):
                        reference.setdefault('path', str(asset.uri or asset.local_path))
        return WriterProviderDocument(
            provider='',
            format=output_format,
            content=export_writer_document(
                document, output_format, markdown_source=content if isinstance(content, str) else None,
            ),
            source_document=document,
            media_references={
                key: str(asset.uri or asset.local_path or '')
                for key, asset in (media_assets.assets.items() if media_assets else [])
                if asset.uri or asset.local_path
            },
        )

    def _convert_native_document(
        self,
        content: WriterDocument | str,
        *,
        target: TargetDocument | None = None,
        media_assets: MediaAssetLibrary | None = None,
    ) -> WriterProviderDocument:
        raise NotImplementedError

    def convert_document_with_template(
        self,
        content: WriterDocument | str,
        *,
        target: TargetDocument | None = None,
        media_assets: MediaAssetLibrary | None = None,
        template: str | None = None,
    ) -> WriterProviderDocument:
        '''Convert content with an optional provider-specific presentation template.'''
        if str(template or '').strip():
            raise ValueError(
                f'Writer provider {self.provider!r} does not support templates.')
        return self.convert_document(
            content, target=target, media_assets=media_assets,
        )

    def write_document(
        self,
        document: WriterProviderDocument,
        target: TargetDocument,
        *,
        media_assets: MediaAssetLibrary | None = None,
        mode: WriterProviderWriteMode = 'replace',
    ) -> dict:
        '''Persist previously converted provider content.'''
        raise NotImplementedError(
            f'{self.provider or type(self).__name__} does not support write_document().')

    def prepare_markdown_for_editor(
        self,
        markdown: str,
        target: TargetDocument,
    ) -> str:
        return markdown

    @classmethod
    @abstractmethod
    def matches(cls, locator: str) -> bool:
        '''Return whether locator belongs to this provider.'''
        raise NotImplementedError

    @abstractmethod
    def resolve(self, locator: str) -> TargetDocument:
        '''Convert a provider locator into the existing target document model.'''
        raise NotImplementedError

    @abstractmethod
    def load_document(
        self,
        target: TargetDocument,
        *,
        stage: WriterStage = 'final',
    ) -> dict:
        '''Load a provider document and return its existing Writer representation.'''
        raise NotImplementedError

    def create_document(self, title: str, parent_uri: str = '') -> TargetDocument:
        '''Create an empty provider document.'''
        raise NotImplementedError(
            f'{self.provider or type(self).__name__} does not support create_document().')

    def document_image_resources(
        self,
        document: WriterDocument,
    ) -> tuple[list[InputResource], list[str]]:
        '''Return provider image resources and non-fatal discovery warnings.'''
        return [], []

    def download_document_image(
        self,
        document: WriterDocument,
        resource: InputResource,
    ) -> bytes | None:
        '''Return provider image bytes, or defer ordinary URIs to shared loading.'''
        # None delegates ordinary file/HTTP resources to the shared materializer.
        return None

    def replace_document(
        self,
        content: WriterDocument | str,
        target: TargetDocument,
        *,
        media_assets: MediaAssetLibrary | None = None,
    ) -> dict:
        '''Convert content and replace an existing provider document.'''
        self.require_capability('replace')
        converted = self.convert_document(
            content, target=target, media_assets=media_assets,
        )
        return self.write_document(
            converted, target, media_assets=media_assets, mode='replace',
        )

    def append_document(
        self,
        content: WriterDocument | str,
        target: TargetDocument,
        *,
        media_assets: MediaAssetLibrary | None = None,
    ) -> dict:
        '''Convert content and append it to an existing provider document.'''
        self.require_capability('append')
        converted = self.convert_document(
            content, target=target, media_assets=media_assets,
        )
        return self.write_document(
            converted, target, media_assets=media_assets, mode='append',
        )

    def apply_patch_to_document(
        self,
        patch_set: PatchSet,
        source_document: WriterDocument,
        target: TargetDocument,
        *,
        media_assets: MediaAssetLibrary | None = None,
    ) -> dict:
        '''Apply a structured patch to an existing provider document.'''
        raise NotImplementedError(
            f'{self.provider or type(self).__name__} does not support structured patches.')


__all__ = [
    'WriterProviderBase',
    'WriterProviderCapabilities',
    'WriterProviderCapability',
    'WriterProviderCapabilityError',
    'WriterProviderDocument',
    'WriterProviderRevisionError',
    'WriterProviderWriteMode',
    'WriterProviderWriteOutcomeError',
    'is_ambiguous_write_error',
]
