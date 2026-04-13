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


def _annotate_diff_with_line_numbers(content: str, new_start: int) -> str:
    # Annotate each diff hunk line with [old_lineno|new_lineno] prefix so the LLM
    # can unambiguously identify the new-file line number to report.
    # Format:
    #   [  N|  M]   context line  (both old and new advance)
    #   [ --|  M] + added line    (only new advances)
    #   [  N|--]  - removed line  (only old advances)
    # new_start is the first new-file line number of this hunk.
    # old_start is derived by scanning the hunk header; since we receive the hunk body
    # without the @@ header, we approximate old_start = new_start (close enough for display).
    old_no = new_start
    new_no = new_start
    out_lines = []
    for raw_line in content.splitlines():
        if raw_line.startswith('+'):
            prefix = f'[--|{new_no:3d}]'
            new_no += 1
        elif raw_line.startswith('-'):
            prefix = f'[{old_no:3d}|--]'
            old_no += 1
        else:
            prefix = f'[{old_no:3d}|{new_no:3d}]'
            old_no += 1
            new_no += 1
        out_lines.append(f'{prefix} {raw_line}')
    return '\n'.join(out_lines)


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


JSON_START_MARKER = '<<<JSON_START>>>'
JSON_END_MARKER = '<<<JSON_END>>>'
JSON_OUTPUT_INSTRUCTION = (
    f'Wrap your JSON array with exactly these delimiters (no other text outside them):\n'
    f'{JSON_START_MARKER}\n'
    f'[ ... ]\n'
    f'{JSON_END_MARKER}'
)
JSON_OBJ_OUTPUT_INSTRUCTION = (
    f'Wrap your JSON object with exactly these delimiters (no other text outside them):\n'
    f'{JSON_START_MARKER}\n'
    f'{{ ... }}\n'
    f'{JSON_END_MARKER}'
)


def _extract_json_text(raw: str) -> str:
    # Priority 1: unambiguous delimiters that won't collide with code fences in suggestion fields.
    s = raw.find(JSON_START_MARKER)
    e = raw.find(JSON_END_MARKER)
    if s != -1 and e != -1 and e > s:
        return raw[s + len(JSON_START_MARKER):e].strip()
    # Priority 2: ```json ... ``` fence.
    # Use rfind for the closing ``` so we span the ENTIRE fence including any ``` inside suggestion fields.
    fence_open = raw.find('```')
    fence_close = raw.rfind('```')
    if fence_open != -1 and fence_close > fence_open:
        inner = raw[fence_open + 3:fence_close]
        inner = re.sub(r'^\s*json\s*', '', inner, count=1).strip()
        if inner:
            return inner
    # Priority 3: outermost bracket pair — try array first, then object.
    arr_start = raw.find('[')
    arr_end = raw.rfind(']')
    obj_start = raw.find('{')
    obj_end = raw.rfind('}')
    if arr_start != -1 and arr_end > arr_start:
        if obj_start == -1 or arr_start <= obj_start:
            return raw[arr_start:arr_end + 1]
    if obj_start != -1 and obj_end > obj_start:
        return raw[obj_start:obj_end + 1]
    if arr_start != -1 and arr_end > arr_start:
        return raw[arr_start:arr_end + 1]
    return raw.strip()


def _parse_json_with_repair(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Attempt to salvage a truncated JSON array: find the last complete object and close the array.
    last_obj_end = text.rfind('}')
    if last_obj_end != -1:
        candidate = text[:last_obj_end + 1]
        # strip trailing comma/whitespace and close the array
        candidate = candidate.rstrip().rstrip(',') + ']'
        # ensure it starts with '['
        if not candidate.lstrip().startswith('['):
            bracket = candidate.find('{')
            if bracket != -1:
                candidate = '[' + candidate[bracket:]
        try:
            return json.loads(candidate)
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
        # JSON parse failed — could be a truncated response; retry if budget allows
        if delay is None:
            lazyllm.LOG.warning(f'JSON parse/repair failed after all retries. Raw snippet: {raw_response[:300]}')
            return []
        lazyllm.LOG.warning(
            f'JSON parse failed (attempt {attempt + 1}/{len(delays) - 1}), retrying. '
            f'Raw snippet: {raw_response[:120]}'
        )
        time.sleep(delay)

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


_REVIEW_MAX_TOKENS = 32000


def _ensure_non_streaming_llm(llm: Any) -> Any:
    if hasattr(llm, '_stream') and llm._stream and hasattr(llm, 'share'):
        llm = llm.share(stream=False)
    # Ensure output is long enough to hold multi-issue JSON responses.
    if hasattr(llm, 'static_params') and hasattr(llm, '_static_params'):
        cur = llm._static_params.get('max_tokens', 0)
        if not cur or cur < _REVIEW_MAX_TOKENS:
            llm._static_params = dict(llm._static_params, max_tokens=_REVIEW_MAX_TOKENS)
    return llm


def _get_model_name(llm: Any) -> str:
    for attr in ('_model_name', 'model_name', '_model', 'model'):
        val = getattr(llm, attr, None)
        if val and isinstance(val, str):
            return val
    return 'unknown-model'


def _safe_llm_call(llm: Any, prompt: str, budget: Optional[Any] = None) -> List[Dict[str, Any]]:
    if budget is not None and not budget.consume_call():
        lazyllm.LOG.warning('LLM call budget exhausted, skipping JSON call')
        return []
    return _llm_call_with_retry(llm, prompt, parse_json=True)


def _safe_llm_call_text(llm: Any, prompt: str, budget: Optional[Any] = None) -> str:
    if budget is not None and not budget.consume_call():
        lazyllm.LOG.warning('LLM call budget exhausted, skipping text call')
        return ''
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
    # LLM sometimes outputs 'description' instead of 'problem' (especially in R4)
    if item.get('problem') is None and item.get('description') is not None:
        item = dict(item, problem=item['description'])
    if line is None or item.get('problem') is None:
        lazyllm.LOG.info(f'[NORMALIZE_SKIP] missing line or problem: {str(item)[:200]}')
        return None
    try:
        line = int(line)
    except (TypeError, ValueError):
        lazyllm.LOG.info(f'[NORMALIZE_SKIP] non-int line={line!r}: {str(item)[:200]}')
        return None
    if end_line is not None:
        # Allow a generous tolerance: LLM may reference lines slightly outside the hunk
        # (e.g. from file_context). Hard-reject only clearly out-of-range lines.
        hunk_size = max(end_line - new_start, 1)
        tolerance = max(50, hunk_size // 2)
        if not (new_start - tolerance <= line < end_line + tolerance):
            lazyllm.LOG.info(
                f'[NORMALIZE_SKIP] line={line} out of range [{new_start - tolerance}, {end_line + tolerance}): '
                f'{str(item)[:200]}'
            )
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
