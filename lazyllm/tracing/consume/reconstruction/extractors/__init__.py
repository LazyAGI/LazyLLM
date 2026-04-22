from typing import Any, Dict, Optional

from lazyllm.common import LOG
from lazyllm.tracing.semantics import SemanticType

from ...datamodel.raw import RawSpanRecord
from . import embedding, llm, rerank, retriever

_REGISTRY = {
    SemanticType.LLM: llm.extract,
    SemanticType.RETRIEVER: retriever.extract,
    SemanticType.EMBEDDING: embedding.extract,
    SemanticType.RERANK: rerank.extract,
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
