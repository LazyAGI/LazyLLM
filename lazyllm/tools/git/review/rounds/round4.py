# Copyright (c) 2026 LazyAGI. All rights reserved.
# Round 4: merge and deduplicate all issues from R1, R2, R3, RMod, lint, dep_check.
# Entry point: _round4_merge_and_deduplicate

import json
import re
from typing import Any, Dict, List, Optional

import lazyllm

from ..utils import (
    _Progress, _language_instruction, _safe_llm_call,
    _VALID_CATEGORIES, _VALID_SEVERITIES,
)
from .common import (
    _safe_format, _compress_new_issues, _compress_existing_comments,
)
from .prompt import _ROUND4_DEDUP_PROMPT_TMPL


def _token_overlap(a: str, b: str, n: int = 3) -> float:
    '''Compute n-gram character Jaccard overlap between two strings.

    Works for both ASCII and CJK text (unlike word-tokenisation which produces
    one mega-token for Chinese sentences and yields 0 similarity).
    Threshold calibration: use >= 0.45 for "similar-enough" Chinese/English issue text.
    '''
    def _ngrams(s: str) -> set:
        s = re.sub(r'\s+', '', s.lower())
        return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}

    ta, tb = _ngrams(a), _ngrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _merge_similar_issues(group: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    _sev = {'critical': 0, 'medium': 1, 'normal': 2}
    kept: List[Dict[str, Any]] = []
    for c in sorted(group, key=lambda x: _sev.get(x.get('severity', 'normal'), 2)):
        for k in kept:
            if _token_overlap(c.get('problem', ''), k.get('problem', '')) >= threshold:
                extra = c.get('suggestion', '')
                if extra and extra not in k.get('suggestion', ''):
                    k['suggestion'] = k.get('suggestion', '') + '\n' + extra
                break
        else:
            kept.append(dict(c))
    return kept


def _deterministic_dedup(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    _sev_order = {'critical': 0, 'medium': 1, 'normal': 2}
    _src_order = {'r3': 0, 'r1': 1, 'r2': 2, 'rmod': 2, 'lint': 3, 'dep_check': 3}
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for c in issues:
        key = (c.get('path', ''), int(c.get('line') or 0), c.get('bug_category', ''))
        groups.setdefault(key, []).append(c)
    return [
        min(group, key=lambda c: (
            _sev_order.get(c.get('severity', 'normal'), 2),
            _src_order.get(c.get('source', ''), 9),
        ))
        for group in groups.values()
    ]


def _r4_restore_dropped_high_severity(
    deduped: List[Dict[str, Any]],
    result: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    '''Ensure no critical/medium issue is silently dropped by the LLM dedup step.

    For every critical or medium issue in `deduped`, check whether the result
    already contains an issue at the same (path, line) with equal or higher
    severity. If not, restore the original issue from `deduped`.
    '''
    _sev_order = {'critical': 0, 'medium': 1, 'normal': 2}
    result_by_pl: Dict[tuple, int] = {}
    for c in result:
        key = (c.get('path', ''), int(c.get('line') or 0))
        cur = result_by_pl.get(key, 99)
        result_by_pl[key] = min(cur, _sev_order.get(c.get('severity', 'normal'), 2))

    restored: List[Dict[str, Any]] = []
    for c in deduped:
        sev = c.get('severity', 'normal')
        if sev not in ('critical', 'medium'):
            continue
        key = (c.get('path', ''), int(c.get('line') or 0))
        best_in_result = result_by_pl.get(key, 99)
        if best_in_result > _sev_order.get(sev, 2):
            lazyllm.LOG.warning(
                f'Round 4: restoring dropped {sev} issue at '
                f'{c.get("path")}:{c.get("line")} [{c.get("bug_category")}]'
            )
            restored.append(c)
    return result + restored


def _r4_build_result_from_llm(
    items: Any, idx_map: Dict[int, Dict[str, Any]],
) -> tuple:
    result: List[Dict[str, Any]] = []
    kept_idxs: set = set()
    for item in (items if isinstance(items, list) else []):
        if not isinstance(item, dict) or item.get('problem') is None:
            continue
        try:
            idx = int(item.get('idx', -1))
        except (TypeError, ValueError):
            continue
        original = idx_map.get(idx)
        if original is None:
            continue
        kept_idxs.add(idx)
        category = item.get('bug_category') or 'logic'
        severity = item.get('severity') or 'normal'
        entry = {
            'path': original['path'], 'line': original['line'],
            'severity': severity if severity in _VALID_SEVERITIES else 'normal',
            'bug_category': category if category in _VALID_CATEGORIES else 'logic',
            'problem': item.get('problem') or '',
            'suggestion': original.get('suggestion') or '',
        }
        if original.get('source'):
            entry['source'] = original['source']
        result.append(entry)
    return result, kept_idxs


def _r4_fallback_dedup(deduped: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    '''Fallback when LLM returns no results: group by path+line and merge.'''
    _sev_order = {'critical': 0, 'medium': 1, 'normal': 2}
    by_pl: Dict[tuple, List[Dict[str, Any]]] = {}
    for c in sorted(deduped, key=lambda c: _sev_order.get(c.get('severity', 'normal'), 2)):
        key = (c.get('path', ''), int(c.get('line') or 0))
        by_pl.setdefault(key, []).append(c)
    result = []
    for group in by_pl.values():
        result.extend([group[0]] if len(group) == 1 else _merge_similar_issues(group, threshold=0.45))
    return result


def _round4_merge_and_deduplicate(
    llm: Any, all_comments: List[Dict[str, Any]],
    existing_comments: Optional[List[Dict[str, Any]]] = None, language: str = 'cn',
) -> List[Dict[str, Any]]:
    if not all_comments:
        return []
    prog = _Progress('Round 4: merge & deduplicate')
    valid = [c for c in all_comments if c.get('path')]
    if not valid:
        prog.done('no valid comments')
        return []

    deduped = _deterministic_dedup(valid)
    lazyllm.LOG.info(f'Round 4: deterministic dedup {len(valid)} -> {len(deduped)} issues')

    compressed_new = _compress_new_issues(llm, deduped)
    existing_json = json.dumps(_compress_existing_comments(llm, existing_comments), ensure_ascii=False, indent=2) \
        if existing_comments else '(none)'
    prompt = _safe_format(
        _ROUND4_DEDUP_PROMPT_TMPL,
        lang_instruction=_language_instruction(language),
        new_issues_json=json.dumps(compressed_new, ensure_ascii=False, indent=2),
        existing_json=existing_json,
    )
    items = _safe_llm_call(llm, prompt)
    idx_map = {i: c for i, c in enumerate(deduped)}
    result, kept_idxs = _r4_build_result_from_llm(items, idx_map)
    discarded_idxs = set(idx_map.keys()) - kept_idxs
    if discarded_idxs:
        lazyllm.LOG.info(
            f'Round 4: LLM discarded {len(discarded_idxs)} issues: '
            + ', '.join(
                f'#{i} {idx_map[i].get("path", "?")}:{idx_map[i].get("line", "?")} '
                f'[{idx_map[i].get("severity", "?")}][{idx_map[i].get("bug_category", "?")}]'
                for i in sorted(discarded_idxs)
            )
        )
    if not result:
        result = _r4_fallback_dedup(deduped)
    # Safety net: restore any critical/medium issue the LLM silently dropped
    result = _r4_restore_dropped_high_severity(deduped, result)
    prog.done(f'{len(result)} final issues')
    return result
