# Copyright (c) 2026 LazyAGI. All rights reserved.
# Round 2a (PR design document generation) + Round 2 (architect design review).
# Entry point: _round2_generate_pr_doc, _round2_architect_review

from typing import Any, Dict, List, Optional, Tuple

import lazyllm

from ..utils import (
    _Progress, _language_instruction, _safe_llm_call, _safe_llm_call_text,
    _annotate_full_diff, _normalize_comment_item,
)
from ..constants import (
    SINGLE_CALL_CONTEXT_BUDGET,
    max_issues_for_diff, cap_issues_by_severity, clip_text, clip_diff_by_hunk_budget,
    issue_density_rule,
)
from .common import _safe_format
from .prompt import _ROUND2_DOC_PROMPT_TMPL, _ROUND2_ARCHITECT_PROMPT_TMPL


def _round2_generate_pr_doc(
    llm: Any,
    diff_text: str,
    arch_doc: str,
    pr_summary: str = '',
    language: str = 'cn',
    agent_instructions: str = '',
) -> Tuple[str, List[str]]:
    prog = _Progress('Round 2a: generating PR design document')
    diff_use, dropped_files = clip_diff_by_hunk_budget(diff_text, SINGLE_CALL_CONTEXT_BUDGET - 22000)
    if dropped_files:
        lazyllm.LOG.warning(
            f'[R2a] Diff truncated: {len(dropped_files)} file(s) skipped due to context budget: '
            + ', '.join(dropped_files)
        )
    arch_use = clip_text(arch_doc or '', 12000) if arch_doc else '(not available)'
    prompt = _safe_format(
        _ROUND2_DOC_PROMPT_TMPL,
        lang_instruction=_language_instruction(language),
        arch_doc=arch_use,
        pr_summary=pr_summary[:800] if pr_summary else '(not available)',
        diff_text=diff_use,
    )
    result = _safe_llm_call_text(llm, prompt) or '(PR design document unavailable)'
    prog.done(f'{len(result)} chars')
    return result, dropped_files


def _round2_architect_review(
    llm: Any, diff_text: str, arch_doc: str,
    pr_summary: str = '', language: str = 'cn', agent_instructions: str = '',
    pr_design_doc: str = '', review_spec: str = '',
) -> Tuple[List[Dict[str, Any]], List[str]]:
    prog = _Progress('Round 2: architect design review')
    diff_use, dropped_files = clip_diff_by_hunk_budget(diff_text, SINGLE_CALL_CONTEXT_BUDGET - 38000)
    if dropped_files:
        lazyllm.LOG.warning(
            f'[R2] Diff truncated: {len(dropped_files)} file(s) skipped due to context budget: '
            + ', '.join(dropped_files)
        )
    arch_use = clip_text(arch_doc or '', 42000) if arch_doc else '(not available)'
    annotated_diff = _annotate_full_diff(diff_use)
    prompt = _safe_format(
        _ROUND2_ARCHITECT_PROMPT_TMPL,
        lang_instruction=_language_instruction(language),
        agent_instructions=agent_instructions or '(not available)',
        arch_doc=arch_use, review_spec=review_spec[:4000] if review_spec else '(not available)',
        pr_summary=pr_summary[:800] if pr_summary else '(not available)',
        pr_design_doc=clip_text(pr_design_doc, 12000) if pr_design_doc else '(not available)',
        diff_text=annotated_diff, density_rule=issue_density_rule(diff_use),
    )
    items = _safe_llm_call(llm, prompt)
    result = [n for item in (items if isinstance(items, list) else [])
              if (n := _normalize_comment_item(item, default_path='', default_category='design',
                                               demote_on_out_of_range=True)) is not None]
    result = cap_issues_by_severity(result, max_issues_for_diff(diff_use))
    prog.done(f'{len(result)} architect issues found')
    return result, dropped_files
