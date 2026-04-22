import json
import re
from typing import Any, Dict, List, Optional

from ...datamodel.raw import RawSpanRecord


_TRUNCATED_SUFFIX = '...<truncated>'
_DOC_NODE_ID_RE = re.compile(r'DocNode\(id:\s*([^,\s)]+)')


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


def parse_scores(value: Any) -> Optional[List[float]]:
    parsed = parse_jsonish(value)
    if not isinstance(parsed, list):
        return None
    out: List[float] = []
    for item in parsed:
        score = as_float(item)
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


def sequence_len(value: Any) -> Optional[int]:
    parsed = parse_jsonish(value)
    if isinstance(parsed, (list, tuple)):
        return len(parsed)
    return None


def doc_node_ids(value: Any) -> Optional[List[str]]:
    parsed = parse_jsonish(value)
    items = parsed if isinstance(parsed, list) else [parsed]
    out: List[str] = []
    for item in items:
        if isinstance(item, str):
            out.extend(_DOC_NODE_ID_RE.findall(item))
    return out or None
