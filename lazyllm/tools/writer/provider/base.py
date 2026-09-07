from __future__ import annotations

from abc import ABC, abstractmethod

from ..data_models.multimodal import MediaAssetLibrary
from ..data_models.revision import PatchSet
from ..data_models.task import TargetDocument
from ..data_models.writer_ir import WriterDocument, WriterStage


class WriterProviderBase(ABC):
    '''Read and persist Writer content through one external document provider.'''

    provider: str = ''

    def __init__(self, adapters=None):
        self.adapters = adapters or {}

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

    @abstractmethod
    def replace_document(
        self,
        content: WriterDocument | str,
        target: TargetDocument,
        *,
        media_assets: MediaAssetLibrary | None = None,
    ) -> dict:
        '''Replace an existing provider document.'''
        raise NotImplementedError

    def append_document(
        self,
        content: WriterDocument | str,
        target: TargetDocument,
        *,
        media_assets: MediaAssetLibrary | None = None,
    ) -> dict:
        '''Append content to an existing provider document.'''
        raise NotImplementedError(
            f'{self.provider or type(self).__name__} does not support append_document().')

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


__all__ = ['WriterProviderBase']
