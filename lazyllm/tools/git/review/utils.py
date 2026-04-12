# Copyright (c) 2026 LazyAGI. All rights reserved.
import json
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import lazyllm

# ---------------------------------------------------------------------------
# Helpers: diff parsing
# ---------------------------------------------------------------------------

def _get_head_sha_from_pr(pr: Any) -> Optional[str]:
    raw = getattr(pr, 'raw', None) or {}
    if not isinstance(raw, dict):
        return None
    head = raw.get('head')
    if isinstance(head, dict) and head.get('sha'):
        return head['sha']
    diff_refs = raw.get('diff_refs', {})
    if isinstance(diff_refs, dict) and diff_refs.get('head_sha'):
        return diff_refs['head_sha']
    return None


def _parse_unified_diff(diff_text: str) -> List[Tuple[str, int, int, str]]:
    out: List[Tuple[str, int, int, str]] = []
    current_path: Optional[str] = None
    new_start, new_count = 0, 0
    hunk_lines: List[str] = []

    def flush_hunk():
        nonlocal hunk_lines
        if current_path and new_count > 0:
            content = '\n'.join(hunk_lines)
            if content.strip():
                out.append((current_path, new_start, new_count, content))
        hunk_lines = []

    for line in diff_text.splitlines():
        if line.startswith('diff --git '):
            flush_hunk()
            m = re.match(r'diff --git a/(.+) b/(.+)$', line)
            current_path = m.group(2) if m else None
            new_start, new_count = 0, 0
            continue
        if line.startswith('@@'):
            flush_hunk()
            mm = re.search(r'\+(\d+),(\d+)', line)
            if mm:
                new_start = int(mm.group(1))
                new_count = int(mm.group(2))
            continue
        if current_path is None:
            continue
        hunk_lines.append(line)
    flush_hunk()
    return out


def _truncate_hunk_content(content: str, max_lines: int) -> str:
    content_lines = content.splitlines()
    if len(content_lines) > max_lines:
        content_lines = content_lines[:max_lines]
        return '\n'.join(content_lines) + '\n... (truncated)'
    return '\n'.join(content_lines)


# ---------------------------------------------------------------------------
# Progress reporter
# ---------------------------------------------------------------------------

class _Progress:
    def __init__(self, stage: str, total: int = 0) -> None:
        self._stage = stage
        self._total = total
        self._current = 0
        self._emit(f'[START] {stage}' + (f' (0/{total})' if total else ''))

    def _emit(self, msg: str) -> None:
        sys.stderr.write(msg + '\n')
        sys.stderr.flush()

    def update(self, label: str = '', step: int = 1) -> None:
        self._current += step
        bar = f'{self._current}/{self._total}' if self._total else str(self._current)
        suffix = f' — {label}' if label else ''
        self._emit(f'  [{self._stage}] {bar}{suffix}')

    def done(self, msg: str = '') -> None:
        suffix = f': {msg}' if msg else ''
        self._emit(f'[DONE]  {self._stage}{suffix}')


# ---------------------------------------------------------------------------
# QPS retry wrapper
# ---------------------------------------------------------------------------

_QPS_PATTERNS = re.compile(
    r'rate.?limit|too.?many.?request|qps|quota|429|throttl', re.IGNORECASE
)
_RETRY_DELAYS = (3, 8, 20)  # seconds to wait before each retry attempt


def _extract_json_text(raw: str) -> str:
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
    if m:
        return m.group(1).strip()
    start = raw.find('[')
    end = raw.rfind(']')
    if start != -1 and end != -1 and end > start:
        return raw[start:end + 1]
    return raw.strip()


def _parse_json_with_repair(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        from json_repair import repair_json
        repaired = repair_json(text, return_objects=True)
        return repaired
    except Exception:
        pass
    return None


def _llm_call_with_retry(llm: Any, prompt: str, parse_json: bool = True) -> Any:
    last_llm_exc: Optional[Exception] = None
    raw_response: str = ''
    delays = list(_RETRY_DELAYS) + [None]
    for attempt, delay in enumerate(delays):
        try:
            resp = llm(prompt)
            if not resp or not isinstance(resp, str):
                return [] if parse_json else ''
            raw_response = resp.strip()
        except Exception as exc:
            last_llm_exc = exc
            err_str = str(exc)
            is_qps = bool(_QPS_PATTERNS.search(err_str))
            if delay is None:
                raise RuntimeError(f'LLM call failed after {len(delays) - 1} retries: {err_str}') from exc
            wait = delay * (3 if is_qps else 1)
            lazyllm.LOG.warning(
                f'LLM call failed (attempt {attempt + 1}/{len(delays) - 1}, {"QPS" if is_qps else "error"}): '
                f'{err_str[:120]}. Retrying in {wait}s...'
            )
            time.sleep(wait)
            continue

        if not parse_json:
            return raw_response
        json_text = _extract_json_text(raw_response)
        parsed = _parse_json_with_repair(json_text)
        if parsed is not None:
            if isinstance(parsed, list):
                # flatten one level of nesting in case LLM wraps the array: [[{...}]] → [{...}]
                flat = []
                for item in parsed:
                    if isinstance(item, list):
                        flat.extend(item)
                    else:
                        flat.append(item)
                return flat
            return [parsed] if isinstance(parsed, dict) else []
        lazyllm.LOG.warning(f'JSON parse/repair failed. Raw response snippet: {raw_response[:300]}')
        return []

    # unreachable, but satisfies type checker
    raise RuntimeError(f'LLM call gave up after retries: {last_llm_exc}')


# ---------------------------------------------------------------------------
# LLM utilities
# ---------------------------------------------------------------------------

def _get_default_llm() -> Any:
    try:
        return lazyllm.OnlineChatModule()
    except Exception as e:
        raise RuntimeError(
            'No llm provided and could not create default OnlineChatModule. Pass llm explicitly.'
        ) from e


def _ensure_non_streaming_llm(llm: Any) -> Any:
    if hasattr(llm, '_stream') and llm._stream and hasattr(llm, 'share'):
        return llm.share(stream=False)
    return llm


def _get_model_name(llm: Any) -> str:
    for attr in ('_model_name', 'model_name', '_model', 'model'):
        val = getattr(llm, attr, None)
        if val and isinstance(val, str):
            return val
    return 'unknown-model'


def _safe_llm_call(llm: Any, prompt: str) -> List[Dict[str, Any]]:
    return _llm_call_with_retry(llm, prompt, parse_json=True)


def _safe_llm_call_text(llm: Any, prompt: str) -> str:
    return _llm_call_with_retry(llm, prompt, parse_json=False)


# ---------------------------------------------------------------------------
# Review response parsing helpers
# ---------------------------------------------------------------------------

_VALID_CATEGORIES = {
    'logic', 'type', 'safety', 'exception', 'performance',
    'concurrency', 'design', 'style', 'maintainability',
}
_VALID_SEVERITIES = {'critical', 'medium', 'normal'}
_LANGUAGE_MAP = {'cn': 'Simplified Chinese(简体中文)', 'en': 'English'}


def _language_instruction(lang: str) -> str:
    lang_name = _LANGUAGE_MAP.get(lang, 'Simplified Chinese(简体中文)')
    return (
        f'Respond in {lang_name}. '
        'In the `suggestion` field, wrap ALL code snippets with markdown code fences using the correct language tag '
        r'(e.g., ```python\n...\n``` for Python, ```cpp\n...\n``` for C++). '
        r'When showing old and new code, use a unified diff block (```diff\n- old lines\n+ new lines\n```). '
        'Preserve exact indentation and formatting inside code blocks.'
    )


def _normalize_comment_item(
    item: Dict[str, Any], new_start: int = 0, end_line: Optional[int] = None,
    default_path: str = '', default_category: str = 'logic',
) -> Optional[Dict[str, Any]]:
    line = item.get('line')
    if line is None or item.get('problem') is None:
        return None
    try:
        line = int(line)
    except (TypeError, ValueError):
        return None
    if end_line is not None:
        if not (new_start <= line < end_line):
            return None
    elif line <= 0:
        return None
    category = item.get('bug_category') or default_category
    if category not in _VALID_CATEGORIES:
        category = default_category
    severity = item.get('severity') or 'normal'
    if severity not in _VALID_SEVERITIES:
        severity = 'normal'
    return {
        'path': item.get('path') or default_path,
        'line': line,
        'severity': severity,
        'bug_category': category,
        'problem': item.get('problem') or '',
        'suggestion': item.get('suggestion') or '',
    }


def _category_stats(comments: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in comments:
        cat = c.get('bug_category') or 'unknown'
        out[cat] = out.get(cat, 0) + 1
    return out


def _build_review_body(
    pr_summary: str,
    total: int,
    stats: Dict[str, int],
    model_name: str,
) -> str:
    stats_lines = '\n'.join(f'- {k}: {v}' for k, v in sorted(stats.items(), key=lambda kv: (-kv[1], kv[0])))
    return (
        f'PR Summary:\n{pr_summary}\n\n'
        f'Findings:\n- total_issues: {total}\n'
        f'{stats_lines}\n\n'
        f'---\n'
        f'auto reviewed by BOT ({model_name})'
    )
