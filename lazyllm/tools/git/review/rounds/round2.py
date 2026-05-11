# Copyright (c) 2026 LazyAGI. All rights reserved.
# Round 2a (PR design document generation) + Round 2 (architect design review).
# Entry point: _round2_generate_pr_doc, _round2_architect_review

import re
from typing import Any, Dict, List, Tuple

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


def _split_diff_into_file_batches(
    diff_text: str, budget: int,
) -> List[Tuple[str, List[str]]]:
    '''Split a unified diff into batches that each fit within *budget* chars.

    Each batch is a (batch_diff_text, [file_paths]) tuple.  Files that are
    individually larger than *budget* are included alone in their own batch
    (the caller must handle oversized single-file diffs via skeleton fallback).
    '''
    _file_re = re.compile(r'^(diff --git .+?)(?=^diff --git |\Z)', re.MULTILINE | re.DOTALL)
    file_blocks: List[Tuple[str, str]] = []
    for m in _file_re.finditer(diff_text):
        block = m.group(0)
        path_m = re.search(r'^diff --git a/.+ b/(.+)$', block, re.MULTILINE)
        path = path_m.group(1) if path_m else '(unknown)'
        file_blocks.append((path, block))

    batches: List[Tuple[str, List[str]]] = []
    cur_text = ''
    cur_paths: List[str] = []
    for path, block in file_blocks:
        if cur_text and len(cur_text) + len(block) > budget:
            batches.append((cur_text, cur_paths))
            cur_text, cur_paths = '', []
        cur_text += block
        cur_paths.append(path)
    if cur_text:
        batches.append((cur_text, cur_paths))
    return batches


def _round2_generate_pr_doc(
    llm: Any,
    diff_text: str,
    arch_doc: str,
    pr_summary: str = '',
    language: str = 'cn',
    agent_instructions: str = '',
) -> Tuple[str, List[str]]:
    prog = _Progress('Round 2a: generating PR design document')
    budget = SINGLE_CALL_CONTEXT_BUDGET - 22000
    arch_use = clip_text(arch_doc or '', 12000) if arch_doc else '(not available)'

    if len(diff_text) <= budget:
        batches = [(diff_text, [])]
    else:
        batches = _split_diff_into_file_batches(diff_text, budget)

    doc_parts: List[str] = []
    for batch_diff, batch_paths in batches:
        prompt = _safe_format(
            _ROUND2_DOC_PROMPT_TMPL,
            lang_instruction=_language_instruction(language),
            arch_doc=arch_use,
            pr_summary=pr_summary[:800] if pr_summary else '(not available)',
            diff_text=batch_diff,
        )
        part = _safe_llm_call_text(llm, prompt) or ''
        if part:
            doc_parts.append(part)

    result = '\n\n'.join(doc_parts) if doc_parts else '(PR design document unavailable)'
    prog.done(f'{len(result)} chars')
    return result, []  # no files dropped


def _round2_architect_review(
    llm: Any, diff_text: str, arch_doc: str,
    pr_summary: str = '', language: str = 'cn', agent_instructions: str = '',
    pr_design_doc: str = '', review_spec: str = '',
) -> Tuple[List[Dict[str, Any]], List[str]]:
    prog = _Progress('Round 2: architect design review')
    budget = SINGLE_CALL_CONTEXT_BUDGET - 38000
    arch_use = clip_text(arch_doc or '', 42000) if arch_doc else '(not available)'

    if len(diff_text) <= budget:
        batches = [(diff_text, [])]
    else:
        batches = _split_diff_into_file_batches(diff_text, budget)

    all_results: List[Dict[str, Any]] = []
    for batch_diff, _batch_paths in batches:
        annotated_diff = _annotate_full_diff(batch_diff)
        prompt = _safe_format(
            _ROUND2_ARCHITECT_PROMPT_TMPL,
            lang_instruction=_language_instruction(language),
            agent_instructions=agent_instructions or '(not available)',
            arch_doc=arch_use, review_spec=review_spec[:4000] if review_spec else '(not available)',
            pr_summary=pr_summary[:800] if pr_summary else '(not available)',
            pr_design_doc=clip_text(pr_design_doc, 12000) if pr_design_doc else '(not available)',
            diff_text=annotated_diff, density_rule=issue_density_rule(batch_diff),
        )
        items = _safe_llm_call(llm, prompt)
        batch_result = [n for item in (items if isinstance(items, list) else [])
                        if (n := _normalize_comment_item(item, default_path='', default_category='design',
                                                         demote_on_out_of_range=True)) is not None]
        all_results.extend(batch_result)

    all_results = cap_issues_by_severity(all_results, max_issues_for_diff(diff_text))
    prog.done(f'{len(all_results)} architect issues found')
    return all_results, []  # no files dropped
