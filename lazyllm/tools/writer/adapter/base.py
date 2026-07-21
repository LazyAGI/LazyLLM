from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional
from uuid import NAMESPACE_URL, uuid5

from ..data_models.revision import PatchHunk
from ..data_models.writer_ir import WriterDocument, WriterStage


NativeBlock = Dict[str, Any]
NativePatchOperationType = Literal['create', 'update', 'delete', 'move']


@dataclass(frozen=True)
class NativePatchOperation:
    '''One provider-native operation produced from a Writer patch hunk.'''

    operation: NativePatchOperationType
    params: Dict[str, Any]

_WRITER_ID_NAMESPACE = uuid5(NAMESPACE_URL, 'https://lazyllm.ai/writer-ir')


class WriterAdapterBase(ABC):
    '''Convert between provider-native document blocks and Writer IR.'''

    provider: str = ''

    @classmethod
    def make_document_id(cls, external_document_id: str) -> str:
        '''Return a stable internal document ID for one provider document.'''
        provider = cls._provider_key()
        external_id = cls._require_identifier(external_document_id, 'external_document_id')
        value = uuid5(_WRITER_ID_NAMESPACE, f'document:{provider}:{external_id}')
        return f'writer-doc-{value}'

    @classmethod
    def make_node_id(cls, external_document_id: str, external_block_id: str) -> str:
        '''Return a stable internal node ID for one provider block.'''
        provider = cls._provider_key()
        document_id = cls._require_identifier(external_document_id, 'external_document_id')
        block_id = cls._require_identifier(external_block_id, 'external_block_id')
        value = uuid5(
            _WRITER_ID_NAMESPACE,
            f'node:{provider}:{document_id}:{block_id}',
        )
        return f'writer-node-{value}'

    @classmethod
    def _provider_key(cls) -> str:
        provider = cls.provider
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError(f'{cls.__name__}.provider must be a non-empty string.')
        return provider.strip().lower()

    @staticmethod
    def _require_identifier(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{name} must be a non-empty string.')
        return value.strip()

    @abstractmethod
    def blocks_to_ir(
        self,
        blocks: List[NativeBlock],
        *,
        external_document_id: str,
        stage: WriterStage = 'final',
        title: str = '',
        uri: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> WriterDocument:
        '''Convert provider-native blocks into a WriterDocument.'''
        raise NotImplementedError

    @abstractmethod
    def ir_to_blocks(self, document: WriterDocument) -> List[NativeBlock]:
        '''Convert a WriterDocument into provider-native blocks.'''
        raise NotImplementedError

    @abstractmethod
    def patch_to_operation(
        self,
        patch: PatchHunk,
        document: WriterDocument,
    ) -> NativePatchOperation:
        '''Convert one Writer patch hunk into a classified provider operation.'''
        raise NotImplementedError

__all__ = [
    'NativeBlock',
    'NativePatchOperation',
    'NativePatchOperationType',
    'WriterAdapterBase',
]
