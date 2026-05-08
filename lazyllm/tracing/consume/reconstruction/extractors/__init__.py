from typing import Any, Dict, Optional

from lazyllm.common import LOG
from lazyllm.tracing.datamodel.raw import RawSpanRecord
from lazyllm.tracing.semantics import SemanticType

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
        LOG.warning(
            'extractor %s failed on span_id=%s span_name=%r: %s',
            sem,
            span.span_id,
            span.name,
            exc,
            exc_info=True,
        )
        return None


__all__ = ['extract_semantic']
