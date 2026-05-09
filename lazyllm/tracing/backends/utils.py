from datetime import datetime, timezone
from typing import Any, Dict


_TRACE_METADATA_PREFIX = 'lazyllm.trace.metadata.'


def extract_trace_metadata(attrs: Dict[str, Any], target_prefix: str = '') -> Dict[str, Any]:
    return {
        f'{target_prefix}{key[len(_TRACE_METADATA_PREFIX):]}': value
        for key, value in attrs.items()
        if key.startswith(_TRACE_METADATA_PREFIX)
    }


def iso_to_epoch(value: str) -> float:
    if not isinstance(value, str) or not value:
        raise ValueError(f'invalid tracing backend timestamp: {value!r}')
    try:
        text = value.removesuffix('Z') + '+00:00' if value.endswith('Z') else value
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f'invalid tracing backend timestamp: {value!r}') from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


__all__ = ['extract_trace_metadata', 'iso_to_epoch']
