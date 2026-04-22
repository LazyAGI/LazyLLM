import json
import math
import re
from typing import Any, Dict, List, Optional

from ...datamodel.raw import RawSpanRecord


_TRUNCATED_SUFFIX = '...<truncated>'
_DOC_NODE_ID_RE = re.compile(r'DocNode\(id\s*[:=]\s*([^,\s)]+)')
_DOC_NODE_GROUP_RE = re.compile(r'\bgroup\s*[:=]\s*([^,\s)]+)')
_DOC_NODE_CONTENT_RE = re.compile(r'\bcontent\s*[:=]\s*(.*?)(?:\)\s*parent:|\)\s*$)')


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text.endswith(_TRUNCATED_SUFFIX):
        return value
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return value


def span_input(span: RawSpanRecord) -> Any:
    if span.input is not None:
        return parse_jsonish(span.input)
    return parse_jsonish(span.attributes.get('lazyllm.io.input'))


def span_output(span: RawSpanRecord) -> Any:
    if span.output is not None:
        return parse_jsonish(span.output)
    return parse_jsonish(span.attributes.get('lazyllm.io.output'))


def config_value(span: RawSpanRecord, key: str) -> Any:
    return parse_jsonish(span.attributes.get(f'lazyllm.entity.config.{key}'))


def config_values(span: RawSpanRecord, *keys: str) -> Dict[str, Any]:
    return {key: config_value(span, key) for key in keys}


def as_int(value: Any) -> Optional[int]:
    if value is None or value == '':
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> Optional[float]:
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_finite_float(value: Any) -> Optional[float]:
    result = as_float(value)
    if result is None or not math.isfinite(result):
        return None
    return result


def as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None or value == '':
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ('true', '1', 'yes', 'y'):
            return True
        if lowered in ('false', '0', 'no', 'n'):
            return False
        if lowered in ('none', 'null'):
            return None
    return bool(value)


def parse_scores(value: Any) -> Optional[List[float]]:
    parsed = parse_jsonish(value)
    if not isinstance(parsed, list):
        return None
    out: List[float] = []
    for item in parsed:
        score = as_finite_float(item)
        if score is not None:
            out.append(score)
    return out or None


def usage(span: RawSpanRecord) -> Dict[str, int]:
    pairs = (
        ('input_tokens', 'gen_ai.usage.input_tokens'),
        ('output_tokens', 'gen_ai.usage.output_tokens'),
        ('total_tokens', 'gen_ai.usage.total_tokens'),
    )
    out: Dict[str, int] = {}
    for name, attr_key in pairs:
        value = as_int(span.attributes.get(attr_key))
        if value is not None:
            out[name] = value

    raw_usage = span.raw.get('usage')
    if isinstance(raw_usage, dict):
        for name, raw_key in (
            ('input_tokens', 'input'),
            ('output_tokens', 'output'),
            ('total_tokens', 'total'),
        ):
            if name not in out:
                value = as_int(raw_usage.get(raw_key))
                if value is not None:
                    out[name] = value
    return out


def find_first_key(value: Any, *keys: str) -> Any:
    if isinstance(value, dict):
        for key in keys:
            if key in value:
                return value[key]
        for item in value.values():
            found = find_first_key(item, *keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = find_first_key(item, *keys)
            if found is not None:
                return found
    return None


def kwargs_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict) and isinstance(value.get('kwargs'), dict):
        return value['kwargs']
    return {}


def args_list(value: Any) -> List[Any]:
    if isinstance(value, dict):
        args = value.get('args')
        if isinstance(args, list):
            return args
    return []


def query_from_input(value: Any) -> Any:
    found = find_first_key(value, 'query')
    if found not in (None, ''):
        return found
    args = args_list(value)
    for item in args:
        if isinstance(item, str):
            return item
        found = find_first_key(item, 'query')
        if found not in (None, ''):
            return found
    return None


def input_filters(value: Any) -> Any:
    return kwargs_dict(value).get('filters')


def prompt_messages(value: Any) -> Any:
    messages = find_first_key(value, 'messages', 'prompt_messages')
    if isinstance(messages, list):
        return messages
    return None


def summarize_input(value: Any, *, limit: int = 240) -> Optional[str]:
    if value is None:
        return None
    args = args_list(value)
    if args:
        value = args[0]
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            text = str(value)
    if len(text) > limit:
        return text[:limit] + _TRUNCATED_SUFFIX
    return text


def output_dim(value: Any) -> Optional[int]:
    parsed = parse_jsonish(value)
    if isinstance(parsed, list):
        if parsed and all(isinstance(item, (int, float)) for item in parsed):
            return len(parsed)
        if parsed and isinstance(parsed[0], list):
            return len(parsed[0])
    return None


def input_count(value: Any) -> Optional[int]:
    args = args_list(value)
    if not args:
        return sequence_len(value)
    first = parse_jsonish(args[0])
    if isinstance(first, (list, tuple)):
        return len(first)
    return len(args)


def is_truncated(value: Any) -> bool:
    if isinstance(value, str):
        return value.endswith(_TRUNCATED_SUFFIX)
    if isinstance(value, dict):
        return any(is_truncated(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(is_truncated(v) for v in value)
    return False


def text_length(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str))
    except TypeError:
        return len(str(value))


def sequence_len(value: Any) -> Optional[int]:
    parsed = parse_jsonish(value)
    if isinstance(parsed, (list, tuple)):
        return len(parsed)
    return None


def doc_node_ids(value: Any) -> Optional[List[str]]:
    summaries = doc_node_summaries(value, content_limit=0)
    if summaries:
        return [item['id'] for item in summaries if item.get('id')] or None
    return None


def doc_node_summaries(
    value: Any,
    *,
    limit: Optional[int] = None,
    content_limit: int = 240,
) -> Optional[List[Dict[str, Any]]]:
    parsed = parse_jsonish(value)
    items = parsed if isinstance(parsed, list) else [parsed]
    out: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            doc_id = item.get('id') or item.get('node_id') or item.get('doc_id')
            group = item.get('group') or item.get('group_name')
            content = item.get('content') or item.get('text')
        elif isinstance(item, str):
            doc_id_match = _DOC_NODE_ID_RE.search(item)
            if not doc_id_match:
                continue
            doc_id = doc_id_match.group(1)
            group_match = _DOC_NODE_GROUP_RE.search(item)
            content_match = _DOC_NODE_CONTENT_RE.search(item)
            group = group_match.group(1) if group_match else None
            content = content_match.group(1).strip() if content_match else None
        else:
            continue

        summary: Dict[str, Any] = {'id': doc_id}
        if group is not None:
            summary['group'] = group
        if content is not None and content_limit > 0:
            text = str(content)
            summary['content_preview'] = (
                text[:content_limit] + _TRUNCATED_SUFFIX if len(text) > content_limit else text
            )
        out.append(summary)
        if limit is not None and len(out) >= limit:
            break
    return out or None
