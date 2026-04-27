from typing import Any, Dict, Optional

from lazyllm.common import LOG
from lazyllm.tracing.semantics import SemanticType

from ...datamodel.raw import RawSpanRecord
from .builtin import extract_embedding, extract_llm, extract_rerank, extract_retriever

_REGISTRY = {
    SemanticType.LLM: extract_llm,
    SemanticType.RETRIEVER: extract_retriever,
    SemanticType.EMBEDDING: extract_embedding,
    SemanticType.RERANK: extract_rerank,
}


def extract_semantic(span: RawSpanRecord) -> Optional[Dict[str, Any]]:
    sem = span.attributes.get('lazyllm.semantic_type')
    fn = _REGISTRY.get(sem)
    if fn is None:
        return None
    try:
        return fn(span)
    except Exception as exc:
        LOG.warning(f'extractor {sem} failed on span {span.span_id}: {exc}')
        return None


__all__ = ['_REGISTRY', 'extract_semantic']
