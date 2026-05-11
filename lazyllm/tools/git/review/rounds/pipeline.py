# Copyright (c) 2026 LazyAGI. All rights reserved.
# Pipeline: orchestrates the RHunkScan → RPrDoc → RArchReview/RMod → RAgentVerify → RDedupMerge chain.
# Each round's internal implementation lives in its own module.
# This file has no review logic — it only calls round entry points in order.

from collections import defaultdict as _dd
from typing import Any, Dict, List, Optional, Tuple

import lazyllm

from ..utils import _Progress
from ..pre_analysis import _get_symbol_index, _build_layered_agents_index
from ..constants import R1_WINDOW_MAX_HUNKS, R1_WINDOW_MAX_DIFF_CHARS
from ..checkpoint import ReviewStage
from .rhunk_scan import _rhunk_scan_analysis
from .rarch_review import _rpr_doc_generate, _rarch_review
from .ragent_verify import _ragent_verify
from .rdedup_merge import _rdedup_merge
from .rmod import _rmod_run


def _r1_dedup_and_filter(
    rhunk_scan_issues: List[Dict[str, Any]],
    discarded_prev_keys: set,
) -> List[Dict[str, Any]]:
    '''Filter RHunkScan issues using RAgentVerify disc_keys, then deduplicate within each group.

    Groups issues by (path, line, bug_category). For each group:
    - If the group key is in disc_keys → the whole group was explicitly discarded by RAgentVerify, skip it.
    - Otherwise → keep the group, but merge near-identical descriptions (n-gram >= 0.85).

    This prevents a single disc_key from silently dropping multiple distinct issues that
    happen to share the same path:line:category.
    '''
    from .rdedup_merge import _token_overlap

    groups: Dict[tuple, List[Dict[str, Any]]] = _dd(list)
    for c in rhunk_scan_issues:
        key = (c.get('path', ''), c.get('line'), c.get('bug_category', ''))
        groups[key].append(c)

    result: List[Dict[str, Any]] = []
    for key, group in groups.items():
        pl  = f'{key[0]}:{key[1]}'
        plc = f'{pl}:{key[2]}'
        if pl in discarded_prev_keys or plc in discarded_prev_keys:
            # Entire group explicitly discarded by R3 — skip
            continue
        # Deduplicate within the group: keep issues whose problem text is not
        # near-identical (>= 0.85 n-gram overlap) to an already-kept issue.
        kept: List[Dict[str, Any]] = []
        for c in group:
            prob = c.get('problem', '')
            if not any(_token_overlap(prob, k.get('problem', '')) >= 0.85 for k in kept):
                kept.append(c)
        result.extend(kept)
    return result


def _build_pr_file_summary(hunks: List[Tuple[str, int, int, str]], max_chars: int = 2000) -> str:
    added: Dict[str, int] = _dd(int)
    removed: Dict[str, int] = _dd(int)
    for path, _start, _count, content in hunks:
        for line in content.splitlines():
            if line.startswith('+') and not line.startswith('+++'):
                added[path] += 1
            elif line.startswith('-') and not line.startswith('---'):
                removed[path] += 1
    lines = [f'- {p}: +{added[p]}/-{removed[p]}' for p in sorted(set(added) | set(removed))]
    result = '\n'.join(lines)
    return result[:max_chars] if len(result) > max_chars else result


def _rebuild_diff_from_hunks(win_hunks: List[Tuple[str, int, int, str]]) -> str:
    by_path: Dict[str, List[str]] = {}
    for path, _s, _c, content in win_hunks:
        by_path.setdefault(path, []).append(content)
    return '\n'.join(
        f'--- a/{p}\n+++ b/{p}\n' + ''.join(cs) for p, cs in by_path.items()
    )


def _split_into_windows(
    all_hunks: List[Tuple[str, int, int, str]],
) -> List[Tuple[List[Tuple[str, int, int, str]], str]]:
    windows, cur, cur_sz = [], [], 0
    for h in all_hunks:
        sz = len(h[3])
        if cur and (len(cur) >= R1_WINDOW_MAX_HUNKS or cur_sz + sz > R1_WINDOW_MAX_DIFF_CHARS):
            windows.append((cur, _rebuild_diff_from_hunks(cur)))
            cur, cur_sz = [], 0
        cur.append(h)
        cur_sz += sz
    if cur:
        windows.append((cur, _rebuild_diff_from_hunks(cur)))
    return windows


def _run_review_pipeline(  # noqa: C901
    llm: Any,
    hunks: List[Tuple[str, int, int, str]],
    diff_text: str,
    arch_doc: str,
    review_spec: str,
    pr_summary: str,
    ckpt: Any,
    clone_dir: Optional[str] = None,
    existing_comments: Optional[List[Dict[str, Any]]] = None,
    language: str = 'cn',
    agent_instructions: str = '',
    strategy: Optional[Any] = None,
    lint_issues: Optional[List[Dict[str, Any]]] = None,
    dep_issues: Optional[List[Dict[str, Any]]] = None,
    owner_repo: str = '',
    arch_cache_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, Any]]:
    from ..constants import BudgetManager, TOTAL_CALL_BUDGET
    _budget = BudgetManager(total_calls=TOTAL_CALL_BUDGET)  # noqa: F841

    # report_data accumulates per-stage raw data for the final report
    report_data: Dict[str, Any] = {
        'rhunk_scan_issues': [],
        'rarch_review_issues': [],
        'rmod_issues': [],
        'ragent_verify_discarded_keys': set(),
        'ragent_verify_files_skipped': [],
        'rdedup_merge_input': [],
        'rdedup_merge_output': [],
    }

    agents_index: Dict[str, str] = {}
    if clone_dir:
        try:
            agents_index = _build_layered_agents_index(clone_dir)
            if agents_index:
                lazyllm.LOG.info(f'Layered agents index: {len(agents_index)} sub-directories with local rules')
        except Exception as e:
            lazyllm.LOG.warning(f'Failed to build layered agents index: {e}')

    # ── RHunkScan: hunk-level diff review ──
    windows = _split_into_windows(hunks)
    if len(windows) > 1:
        lazyllm.LOG.info(f'RHunkScan: processing {len(hunks)} hunks in {len(windows)} windows')

    rhunk_scan_all: List[Dict[str, Any]] = []
    sym_index = _get_symbol_index(arch_doc) if arch_doc else None
    pr_file_summary = _build_pr_file_summary(hunks)
    for win_idx, (win_hunks, _win_diff) in enumerate(windows):
        win_key = f'rhunk_scan_window_{win_idx}'
        use_win_cache = ckpt.should_use_cache(ReviewStage.RHunkScan)
        cached_win = ckpt.get(win_key) if use_win_cache else None
        if cached_win is not None:
            rhunk_scan_all.extend(cached_win)
            continue
        win_rhunk = _rhunk_scan_analysis(
            llm, win_hunks, arch_doc, review_spec, pr_summary=pr_summary,
            clone_dir=clone_dir, language=language,
            symbol_index=sym_index,
            ckpt=ckpt, agent_instructions=agent_instructions,
            pr_file_summary=pr_file_summary, agents_index=agents_index,
        )
        ckpt.save(win_key, win_rhunk)
        rhunk_scan_all.extend(win_rhunk)
    rhunk_scan = rhunk_scan_all
    ckpt.mark_stage_done(ReviewStage.RHunkScan)
    report_data['rhunk_scan_issues'] = list(rhunk_scan)

    rarch_review_meta_warnings: List[Dict[str, Any]] = []

    # ── RPrDoc: PR design document ──
    use_rpr_doc_cache = ckpt.should_use_cache(ReviewStage.RPrDoc)
    pr_design_doc = ckpt.get('pr_design_doc') if use_rpr_doc_cache else None
    if pr_design_doc is None:
        if not use_rpr_doc_cache:
            lazyllm.LOG.warning('RPrDoc: no cache found, re-computing')
        pr_design_doc, _rpr_doc_dropped = _rpr_doc_generate(
            llm, diff_text, arch_doc, pr_summary=pr_summary,
            language=language, agent_instructions=agent_instructions,
        )
        ckpt.save('pr_design_doc', pr_design_doc)
        ckpt.mark_stage_done(ReviewStage.RPrDoc)
    else:
        _Progress('RPrDoc: generating PR design document').done(
            f'loaded from checkpoint ({len(pr_design_doc)} chars)'
        )

    # ── RArchReview: architect design review ──
    use_rarch_cache = ckpt.should_use_cache(ReviewStage.RArchReview)
    rarch_review_issues = ckpt.get('rarch_review') if use_rarch_cache else None
    if rarch_review_issues is None:
        if not use_rarch_cache:
            lazyllm.LOG.warning('RArchReview: no cache found, re-computing')
        rarch_review_issues, _rarch_dropped = _rarch_review(
            llm, diff_text, arch_doc, pr_summary=pr_summary,
            language=language, agent_instructions=agent_instructions,
            pr_design_doc=pr_design_doc, review_spec=review_spec,
        )
        ckpt.save('rarch_review', rarch_review_issues)
        ckpt.mark_stage_done(ReviewStage.RArchReview)
    else:
        _Progress('RArchReview: architect design review').done(
            f'loaded from checkpoint ({len(rarch_review_issues)} issues)'
        )
    report_data['rarch_review_issues'] = list(rarch_review_issues)

    # ── RMod: modification necessity analysis (parallel with RArchReview, both depend on RPrDoc) ──
    use_rmod_cache = ckpt.should_use_cache(ReviewStage.RMod)
    rmod = ckpt.get('rmod') if use_rmod_cache else None
    if rmod is None:
        if not use_rmod_cache:
            lazyllm.LOG.warning('RMod: cache bypassed, re-computing')
        rmod = _rmod_run(
            llm, diff_text, arch_doc, pr_design_doc=pr_design_doc,
            pr_summary=pr_summary, clone_dir=clone_dir, language=language,
            agent_instructions=agent_instructions, owner_repo=owner_repo,
            arch_cache_path=arch_cache_path,
        )
        ckpt.save('rmod', rmod)
        ckpt.mark_stage_done(ReviewStage.RMod)
    else:
        _Progress('RMod: modification necessity analysis').done(
            f'loaded from checkpoint ({len(rmod)} issues)'
        )
    report_data['rmod_issues'] = list(rmod)

    # ── RAgentVerify: unified agent verification ──
    ragent_verify_issues, discarded_prev_keys, ragent_verify_metrics = _ragent_verify(
        llm, rhunk_scan, rarch_review_issues, diff_text, arch_doc, pr_summary=pr_summary,
        clone_dir=clone_dir, language=language, ckpt=ckpt,
        agent_instructions=agent_instructions, strategy=strategy,
        owner_repo=owner_repo, arch_cache_path=arch_cache_path,
        review_spec=review_spec, agents_index=agents_index,
    )
    ckpt.mark_stage_done(ReviewStage.RAgentVerify)
    report_data['ragent_verify_discarded_keys'] = discarded_prev_keys
    report_data['ragent_verify_files_skipped'] = ragent_verify_metrics.get('r3_skipped_files', [])

    # ── RDedupMerge: merge & deduplicate ──
    use_rdedup_cache = ckpt.should_use_cache(ReviewStage.RDedupMerge)
    final = ckpt.get('final') if use_rdedup_cache else None
    if final is not None:
        cached_rv = ckpt.get('_review_round_version')
        if cached_rv != ckpt._REVIEW_ROUND_VERSION:
            lazyllm.LOG.info(
                f'RDedupMerge: review_round_version mismatch '
                f'(cached={cached_rv}, expected={ckpt._REVIEW_ROUND_VERSION}), re-computing'
            )
            final = None
    if final is None:
        if not use_rdedup_cache:
            lazyllm.LOG.warning('RDedupMerge: no cache found, re-computing')

        def _tag(issues: List[Dict[str, Any]], src: str) -> List[Dict[str, Any]]:
            return [{**c, 'source': src} for c in issues]

        rhunk_passthrough = _r1_dedup_and_filter(rhunk_scan, discarded_prev_keys)
        lint_tagged = _tag(lint_issues or [], 'lint')
        dep_tagged = _tag(dep_issues or [], 'dep_check')
        rdedup_input_list = (
            _tag(rhunk_passthrough, 'rhunk_scan') + _tag(rarch_review_issues, 'rarch_review')
            + _tag(ragent_verify_issues, 'ragent_verify')
            + _tag(rmod, 'rmod') + lint_tagged + dep_tagged
        )
        final = _rdedup_merge(
            llm,
            rdedup_input_list,
            existing_comments=existing_comments, language=language,
        )
        report_data['rdedup_merge_input'] = rdedup_input_list
        report_data['rdedup_merge_output'] = list(final)
        ckpt.save('final', final)
        ckpt.save('_review_round_version', ckpt._REVIEW_ROUND_VERSION)
        ckpt.mark_stage_done(ReviewStage.RDedupMerge)
    else:
        _Progress('RDedupMerge: merge & deduplicate').done(f'loaded from checkpoint ({len(final)} issues)')
    return final, ragent_verify_metrics, report_data
