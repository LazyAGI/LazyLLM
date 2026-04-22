from typing import Any, Dict, Optional

from ...datamodel.raw import RawSpanRecord
from .utils import (
    as_int,
    config_value,
    doc_node_ids,
    output_dim,
    parse_scores,
    query_from_input,
    sequence_len,
    span_input,
    span_output,
    summarize_input,
    usage,
)


def extract_llm(span: RawSpanRecord) -> Optional[Dict[str, Any]]:
    model_name = span.attributes.get('gen_ai.request.model') or config_value(span, 'model')
    return {
        'prompt': span_input(span),
        'answer': span_output(span),
        'model_name': model_name,
        'usage': usage(span),
    }


def extract_retriever(span: RawSpanRecord) -> Optional[Dict[str, Any]]:
    output = span_output(span)
    node_count = as_int(span.attributes.get('lazyllm.output.doc_count'))
    if node_count is None:
        node_count = sequence_len(output)

    return {
        'query': query_from_input(span_input(span)),
        'topk': as_int(config_value(span, 'topk')),
        'node_count': node_count,
        'returned_node_ids': doc_node_ids(output),
        'retrieve_scores': parse_scores(span.attributes.get('lazyllm.output.similarity_scores')),
    }


def extract_embedding(span: RawSpanRecord) -> Optional[Dict[str, Any]]:
    model_name = span.attributes.get('gen_ai.request.model') or config_value(span, 'model')
    return {
        'input_summary': summarize_input(span_input(span)),
        'model_name': model_name,
        'dim': output_dim(span_output(span)),
    }


def extract_rerank(span: RawSpanRecord) -> Optional[Dict[str, Any]]:
    input_value = span_input(span)
    args = input_value.get('args') if isinstance(input_value, dict) else None
    candidates = args[0] if isinstance(args, list) and args else None
    candidate_count = sequence_len(candidates)

    return {
        'query': query_from_input(input_value),
        'candidate_count': candidate_count,
        'topk': as_int(config_value(span, 'topk')),
        'reranked_scores': parse_scores(span.attributes.get('lazyllm.output.relevance_scores')),
        'output_count': sequence_len(span_output(span)),
    }


__all__ = [
    'extract_embedding',
    'extract_llm',
    'extract_rerank',
    'extract_retriever',
]
