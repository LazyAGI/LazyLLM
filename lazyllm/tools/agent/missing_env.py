from __future__ import annotations

import json
import re
from typing import Any, Iterable, List, Optional

from lazyllm.tools.tool_config_inject import effective_env_value

_VALID_ENV_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_HEURISTIC_NAME_RE = re.compile(r'\b([A-Z][A-Z0-9]*_[A-Z0-9_]+)\b')
_CONVENTION_RE = re.compile(r'(?im)MISSING_ENV\s*=\s*([A-Za-z_][A-Za-z0-9_]*)')
_HEURISTIC_KEYWORDS = ('not set', 'missing', 'undefined', 'required')
_SKIP_HEURISTIC_NAMES = {'MISSING_ENV'}


def normalize_required_env_names(raw: Any) -> List[str]:
    if isinstance(raw, str):
        items = [part.strip() for part in raw.split(',')]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        return []
    names: List[str] = []
    seen = set()
    for item in items:
        name = str(item or '').strip()
        if not _VALID_ENV_NAME_RE.fullmatch(name) or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def format_missing_env_message(reason: str, missing_env: Iterable[str]) -> str:
    names = [str(name) for name in missing_env if name]
    if not names:
        return reason
    return f'{reason}\nmissing_env: {json.dumps(names, ensure_ascii=False)}'


def collect_missing_env_hints(
    *texts: Any,
    declared_required: Optional[Iterable[str]] = None,
) -> List[str]:
    blob = '\n'.join(str(text or '') for text in texts)
    names: List[str] = []
    seen = set()

    def _add(name: str) -> None:
        cleaned = str(name or '').strip()
        if not cleaned or cleaned in seen or not _VALID_ENV_NAME_RE.fullmatch(cleaned):
            return
        seen.add(cleaned)
        names.append(cleaned)

    for match in _CONVENTION_RE.finditer(blob):
        _add(match.group(1))
    for name in normalize_required_env_names(declared_required):
        if not effective_env_value(name):
            _add(name)
    for line in blob.splitlines():
        lowered = line.lower()
        if not any(keyword in lowered for keyword in _HEURISTIC_KEYWORDS):
            continue
        for match in _HEURISTIC_NAME_RE.finditer(line):
            name = match.group(1)
            if name not in _SKIP_HEURISTIC_NAMES:
                _add(name)
    return names
