# Copyright (c) 2026 LazyAGI. All rights reserved.
# Post-merge dedup: cross-source dedup/merge for final (R1-R4) + rchain + rcov.
# Entry point: _post_merge_dedup

import json
from typing import Any, Dict, List, Optional

import lazyllm

from ..utils import (
    _Progress, _language_instruction, _safe_llm_call,
    _VALID_CATEGORIES, _VALID_SEVERITIES,
)
from .common import _safe_format, _compress_new_issues
from .prompt import _POST_MERGE_DEDUP_PROMPT_TMPL
from .rdedup_merge import _r4_restore_dropped_high_severity


def _post_merge_dedup(
    llm: Any,
    final_comments: List[Dict[str, Any]],
    rchain_issues: List[Dict[str, Any]],
    rcov_issues: List[Dict[str, Any]],
    existing_comments: Optional[List[Dict[str, Any]]] = None,
    language: str = 'cn',
) -> List[Dict[str, Any]]:
    '''Cross-source dedup/merge: final (R1-R4) + rchain + rcov → unified list.'''
    all_in = (
        [{**c, 'source': c.get('source') or 'r4'} for c in final_comments]
        + [{**c, 'source': c.get('source') or 'rchain'} for c in rchain_issues]
        + [{**c, 'source': c.get('source') or 'rcov'} for c in rcov_issues]
    )
    if not all_in:
        return []

    prog = _Progress('Post-merge dedup', len(all_in))

    if not rchain_issues and not rcov_issues:
        prog.done('no rchain/rcov issues, skipping LLM dedup')
        # Still run restore to ensure no high-severity issues were silently dropped upstream.
        result = _r4_restore_dropped_high_severity(all_in, all_in)
        return result

    if len(all_in) <= 1:
        prog.done(f'{len(all_in)} issue(s), nothing to dedup')
        return all_in

    compressed = _compress_new_issues(llm, all_in)
    prompt = _safe_format(
        _POST_MERGE_DEDUP_PROMPT_TMPL,
        lang_instruction=_language_instruction(language),
        issues_json=json.dumps(compressed, ensure_ascii=False, indent=2),
    )
    items = _safe_llm_call(llm, prompt)
    idx_map = {i: c for i, c in enumerate(all_in)}
    result: List[Dict[str, Any]] = []
    kept_idxs: set = set()
    for item in (items if isinstance(items, list) else []):
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get('idx', -1))
        except (TypeError, ValueError):
            continue
        original = idx_map.get(idx)
        if original is None or idx in kept_idxs:
            continue
        kept_idxs.add(idx)
        category = item.get('bug_category') or original.get('bug_category') or 'logic'
        severity = item.get('severity') or original.get('severity') or 'normal'
        entry: Dict[str, Any] = {
            'path': original['path'],
            'line': original['line'],
            'severity': severity if severity in _VALID_SEVERITIES else 'normal',
            'bug_category': category if category in _VALID_CATEGORIES else 'logic',
            'problem': item.get('problem') or original.get('problem') or '',
            'suggestion': (item.get('suggestion')
                           if item.get('suggestion') is not None else original.get('suggestion') or '')
        }
        if original.get('source'):
            entry['source'] = original['source']
        result.append(entry)

    dropped = len(all_in) - len(result)
    if dropped:
        lazyllm.LOG.info(f'Post-merge dedup: LLM dropped/merged {dropped} issue(s)')
    if not result:
        lazyllm.LOG.warning('Post-merge dedup: LLM returned empty, falling back to full input')
        result = all_in

    # Safety net: restore any critical/medium issue the LLM silently dropped.
    result = _r4_restore_dropped_high_severity(all_in, result)

    prog.done(f'{len(result)} issues after cross-source dedup')
    return result
