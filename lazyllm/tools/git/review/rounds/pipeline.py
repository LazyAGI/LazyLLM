# Copyright (c) 2026 LazyAGI. All rights reserved.
# Pipeline: orchestrates the R1 → R2a → R2/RMod → R3 → R4 chain.
# Each round's internal implementation lives in its own module.
# This file has no review logic — it only calls round entry points in order.

from collections import defaultdict as _dd
from typing import Any, Dict, List, Optional, Tuple

import lazyllm

from ..utils import _Progress
from ..pre_analysis import _get_symbol_index, _build_layered_agents_index
from ..constants import R1_WINDOW_MAX_HUNKS, R1_WINDOW_MAX_DIFF_CHARS
from ..checkpoint import ReviewStage
from .round1 import _round1_hunk_analysis
from .round2 import _round2_generate_pr_doc, _round2_architect_review
from .round3 import _round3_agent_verify
from .round4 import _round4_merge_and_deduplicate
from .rmod import _run_rmod_agent_round


def _r1_dedup_and_filter(
    r1_issues: List[Dict[str, Any]],
    discarded_prev_keys: set,
) -> List[Dict[str, Any]]:
    '''Filter R1 issues using R3 disc_keys, then deduplicate within each group.

    Groups issues by (path, line, bug_category). For each group:
    - If the group key is in disc_keys → the whole group was explicitly discarded by R3, skip it.
    - Otherwise → keep the group, but merge near-identical descriptions (n-gram >= 0.85).

    This prevents a single disc_key from silently dropping multiple distinct issues that
    happen to share the same path:line:category.
    '''
    from .round4 import _token_overlap

    groups: Dict[tuple, List[Dict[str, Any]]] = _dd(list)
    for c in r1_issues:
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


def _run_four_rounds(  # noqa: C901
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
        'r1_issues': [],
        'r2_issues': [],
        'rmod_issues': [],
        'r3_discarded_keys': set(),
        'r3_files_skipped': [],
        'r4_input': [],
        'r4_output': [],
    }

    agents_index: Dict[str, str] = {}
    if clone_dir:
        try:
            agents_index = _build_layered_agents_index(clone_dir)
            if agents_index:
                lazyllm.LOG.info(f'Layered agents index: {len(agents_index)} sub-directories with local rules')
        except Exception as e:
            lazyllm.LOG.warning(f'Failed to build layered agents index: {e}')

    # ── R1: hunk-level diff review ──
    windows = _split_into_windows(hunks)
    if len(windows) > 1:
        lazyllm.LOG.info(f'Round 1: processing {len(hunks)} hunks in {len(windows)} windows')

    r1_all: List[Dict[str, Any]] = []
    sym_index = _get_symbol_index(arch_doc) if arch_doc else None
    pr_file_summary = _build_pr_file_summary(hunks)
    for win_idx, (win_hunks, _win_diff) in enumerate(windows):
        win_key = f'r1_window_{win_idx}'
        use_win_cache = ckpt.should_use_cache(ReviewStage.R1)
        cached_win = ckpt.get(win_key) if use_win_cache else None
        if cached_win is not None:
            r1_all.extend(cached_win)
            continue
        win_r1 = _round1_hunk_analysis(
            llm, win_hunks, arch_doc, review_spec, pr_summary=pr_summary,
            clone_dir=clone_dir, language=language,
            symbol_index=sym_index,
            ckpt=ckpt, agent_instructions=agent_instructions,
            pr_file_summary=pr_file_summary, agents_index=agents_index,
        )
        ckpt.save(win_key, win_r1)
        r1_all.extend(win_r1)
    r1 = r1_all
    ckpt.mark_stage_done(ReviewStage.R1)
    report_data['r1_issues'] = list(r1)

    r2_meta_warnings: List[Dict[str, Any]] = []

    # ── R2a: PR design document ──
    use_r2a_cache = ckpt.should_use_cache(ReviewStage.R2A)
    pr_design_doc = ckpt.get('pr_design_doc') if use_r2a_cache else None
    if pr_design_doc is None:
        if not use_r2a_cache:
            lazyllm.LOG.warning('Round 2a: no cache found, re-computing')
        pr_design_doc, _r2a_dropped = _round2_generate_pr_doc(
            llm, diff_text, arch_doc, pr_summary=pr_summary,
            language=language, agent_instructions=agent_instructions,
        )
        ckpt.save('pr_design_doc', pr_design_doc)
        ckpt.mark_stage_done(ReviewStage.R2A)
    else:
        _Progress('Round 2a: generating PR design document').done(
            f'loaded from checkpoint ({len(pr_design_doc)} chars)'
        )

    # ── R2: architect design review ──
    use_r2_cache = ckpt.should_use_cache(ReviewStage.R2)
    r2 = ckpt.get('r2') if use_r2_cache else None
    if r2 is None:
        if not use_r2_cache:
            lazyllm.LOG.warning('Round 2: no cache found, re-computing')
        r2, _r2_dropped = _round2_architect_review(
            llm, diff_text, arch_doc, pr_summary=pr_summary,
            language=language, agent_instructions=agent_instructions,
            pr_design_doc=pr_design_doc, review_spec=review_spec,
        )
        ckpt.save('r2', r2)
        ckpt.mark_stage_done(ReviewStage.R2)
    else:
        _Progress('Round 2: architect design review').done(
            f'loaded from checkpoint ({len(r2)} issues)'
        )
    report_data['r2_issues'] = list(r2)

    # ── RMod: modification necessity analysis (parallel with R2, both depend on R2A) ──
    use_rmod_cache = ckpt.should_use_cache(ReviewStage.RMOD)
    rmod = ckpt.get('rmod') if use_rmod_cache else None
    if rmod is None:
        if not use_rmod_cache:
            lazyllm.LOG.warning('RMod: cache bypassed, re-computing')
        rmod = _run_rmod_agent_round(
            llm, diff_text, arch_doc, pr_design_doc=pr_design_doc,
            pr_summary=pr_summary, clone_dir=clone_dir, language=language,
            agent_instructions=agent_instructions, owner_repo=owner_repo,
            arch_cache_path=arch_cache_path,
        )
        ckpt.save('rmod', rmod)
        ckpt.mark_stage_done(ReviewStage.RMOD)
    else:
        _Progress('RMod: modification necessity analysis').done(
            f'loaded from checkpoint ({len(rmod)} issues)'
        )
    report_data['rmod_issues'] = list(rmod)

    # ── R3: unified agent verification ──
    r3, discarded_prev_keys, r3_metrics = _round3_agent_verify(
        llm, r1, r2, diff_text, arch_doc, pr_summary=pr_summary,
        clone_dir=clone_dir, language=language, ckpt=ckpt,
        agent_instructions=agent_instructions, strategy=strategy,
        owner_repo=owner_repo, arch_cache_path=arch_cache_path,
        review_spec=review_spec, agents_index=agents_index,
    )
    ckpt.mark_stage_done(ReviewStage.R3)
    report_data['r3_discarded_keys'] = discarded_prev_keys
    report_data['r3_files_skipped'] = r3_metrics.get('r3_skipped_files', [])

    # ── R4: merge & deduplicate ──
    use_final_cache = ckpt.should_use_cache(ReviewStage.FINAL)
    final = ckpt.get('final') if use_final_cache else None
    if final is not None:
        cached_rv = ckpt.get('_review_round_version')
        if cached_rv != ckpt._REVIEW_ROUND_VERSION:
            lazyllm.LOG.info(
                f'Round 4: review_round_version mismatch '
                f'(cached={cached_rv}, expected={ckpt._REVIEW_ROUND_VERSION}), re-computing'
            )
            final = None
    if final is None:
        if not use_final_cache:
            lazyllm.LOG.warning('Round 4: no cache found, re-computing')

        def _tag(issues: List[Dict[str, Any]], src: str) -> List[Dict[str, Any]]:
            return [{**c, 'source': src} for c in issues]

        r1_passthrough = _r1_dedup_and_filter(r1, discarded_prev_keys)
        lint_tagged = _tag(lint_issues or [], 'lint')
        dep_tagged = _tag(dep_issues or [], 'dep_check')
        r4_input_list = (
            _tag(r1_passthrough, 'r1') + _tag(r2, 'r2') + _tag(r3, 'r3')
            + _tag(rmod, 'rmod') + lint_tagged + dep_tagged
        )
        final = _round4_merge_and_deduplicate(
            llm,
            r4_input_list,
            existing_comments=existing_comments, language=language,
        )
        report_data['r4_input'] = r4_input_list
        report_data['r4_output'] = list(final)
        ckpt.save('final', final)
        ckpt.save('_review_round_version', ckpt._REVIEW_ROUND_VERSION)
        ckpt.mark_stage_done(ReviewStage.FINAL)
    else:
        _Progress('Round 4: merge & deduplicate').done(f'loaded from checkpoint ({len(final)} issues)')
    return final, r3_metrics, report_data
