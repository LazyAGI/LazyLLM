from datetime import datetime, timezone
from typing import Any, Dict, Optional

from lazyllm.common import LOG


_TRACE_METADATA_PREFIX = 'lazyllm.trace.metadata.'


def extract_trace_metadata(attrs: Dict[str, Any], target_prefix: str = '') -> Dict[str, Any]:
    return {
        f'{target_prefix}{key[len(_TRACE_METADATA_PREFIX):]}': value
        for key, value in attrs.items()
        if key.startswith(_TRACE_METADATA_PREFIX)
    }


def iso_to_epoch(value: Optional[str]) -> Optional[float]:
    if not value or not isinstance(value, str):
        return None
    text = value.replace('Z', '+00:00')
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        LOG.warning(f'Failed to parse tracing backend timestamp: {value!r}')
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


__all__ = ['extract_trace_metadata', 'iso_to_epoch']
