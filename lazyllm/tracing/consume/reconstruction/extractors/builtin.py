from typing import Any, Dict, Optional

from ...datamodel.raw import RawSpanRecord
from .utils import (
    as_bool,
    as_finite_float,
    as_int,
    config_value,
    doc_node_ids,
    doc_node_summaries,
    find_first_key,
    input_count,
    input_filters,
    is_truncated,
    output_dim,
    parse_scores,
    prompt_messages,
    query_from_input,
    sequence_len,
    span_input,
    span_output,
    summarize_input,
    text_length,
    usage,
)


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def extract_llm(span: RawSpanRecord) -> Optional[Dict[str, Any]]:
    input_value = span_input(span)
    output_value = span_output(span)
    static_params = config_value(span, 'static_params')
    model_name = span.attributes.get('gen_ai.request.model') or config_value(span, 'model')
    return {
        'prompt': input_value,
        'answer': output_value,
        'model_name': model_name,
        'usage': usage(span),
        'base_url': config_value(span, 'base_url'),
        'stream': as_bool(config_value(span, 'stream')),
        'type': config_value(span, 'type'),
        'series': config_value(span, 'series'),
        'class': config_value(span, 'class'),
        'temperature': _first_not_none(
            as_finite_float(config_value(span, 'temperature')),
            as_finite_float(find_first_key(static_params, 'temperature')),
        ),
        'top_p': _first_not_none(
            as_finite_float(config_value(span, 'top_p')),
            as_finite_float(find_first_key(static_params, 'top_p')),
        ),
        'max_tokens': _first_not_none(
            as_int(config_value(span, 'max_tokens')),
            as_int(find_first_key(static_params, 'max_tokens')),
        ),
        'query': query_from_input(input_value),
        'context_str': find_first_key(input_value, 'context_str'),
        'prompt_messages': prompt_messages(input_value),
        'answer_length': text_length(output_value),
    }


def extract_retriever(span: RawSpanRecord) -> Optional[Dict[str, Any]]:
    output = span_output(span)
    similarity_cut_off = config_value(span, 'similarity_cut_off')
    node_count = as_int(span.attributes.get('lazyllm.output.doc_count'))
    if node_count is None:
        node_count = sequence_len(output)

    return {
        'query': query_from_input(span_input(span)),
        'filters': input_filters(span_input(span)),
        'topk': as_int(config_value(span, 'topk')),
        'node_count': node_count,
        'returned_node_ids': doc_node_ids(output),
        'scores': parse_scores(span.attributes.get('lazyllm.output.similarity_scores')),
        'group_name': config_value(span, 'group_name'),
        'similarity': config_value(span, 'similarity'),
        'similarity_cut_off': as_finite_float(similarity_cut_off),
        'index': config_value(span, 'index'),
        'mode': config_value(span, 'mode'),
        'output_format': config_value(span, 'output_format'),
        'join': config_value(span, 'join'),
        'priority': config_value(span, 'priority'),
        'target': config_value(span, 'target'),
        'weight': as_finite_float(config_value(span, 'weight')),
        'embed_keys': config_value(span, 'embed_keys'),
        'returned_nodes': doc_node_summaries(output),
    }


def extract_embedding(span: RawSpanRecord) -> Optional[Dict[str, Any]]:
    input_value = span_input(span)
    output_value = span_output(span)
    model_name = span.attributes.get('gen_ai.request.model') or config_value(span, 'model')
    return {
        'input_summary': summarize_input(input_value),
        'model_name': model_name,
        'dim': output_dim(output_value),
        'input_count': input_count(input_value),
        'base_url': config_value(span, 'base_url'),
        'batch_size': as_int(config_value(span, 'batch_size')),
        'num_worker': as_int(config_value(span, 'num_worker')),
        'timeout': as_finite_float(config_value(span, 'timeout')),
        'type': config_value(span, 'type'),
        'series': config_value(span, 'series'),
        'class': config_value(span, 'class'),
        'output_truncated': is_truncated(output_value),
    }


def extract_rerank(span: RawSpanRecord) -> Optional[Dict[str, Any]]:
    input_value = span_input(span)
    output_value = span_output(span)
    args = input_value.get('args') if isinstance(input_value, dict) else None
    candidates = args[0] if isinstance(args, list) and args else None
    rerank_model = config_value(span, 'model') or config_value(span, 'name')

    return {
        'query': query_from_input(input_value),
        'candidate_node_count': sequence_len(candidates),
        'topk': as_int(config_value(span, 'topk')),
        'scores': parse_scores(span.attributes.get('lazyllm.output.relevance_scores')),
        'node_count': sequence_len(output_value),
        'candidate_doc_ids': doc_node_ids(candidates),
        'ranked_doc_ids': doc_node_ids(output_value),
        'candidate_nodes': doc_node_summaries(candidates),
        'ranked_nodes': doc_node_summaries(output_value),
        'rerank_model': rerank_model,
        'output_format': config_value(span, 'output_format'),
        'join': config_value(span, 'join'),
    }


__all__ = [
    'extract_embedding',
    'extract_llm',
    'extract_rerank',
    'extract_retriever',
]
