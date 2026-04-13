# Copyright (c) 2026 LazyAGI. All rights reserved.
import dataclasses
import inspect
import os
import shutil
from typing import Any, Dict, List, Optional

from ..client import Git
from .checkpoint import _ReviewCheckpoint, ReviewStage
from .pre_analysis import _run_pre_analysis, _pre_round_pr_summary
from .rounds import _run_five_rounds
from .poster import _fetch_existing_pr_comments, _post_review_comments
from .utils import (
    _get_default_llm, _ensure_non_streaming_llm, _get_model_name,
    _get_head_sha_from_pr, _parse_unified_diff, _Progress,
    _category_stats, _build_review_body,
)
from .constants import R2_MAX_FILES, R2_MAX_CHUNKS_PER_FILE


@dataclasses.dataclass
class _DiffStats:
    diff_lines_total: int
    file_count: int
    file_diff_lines: Dict[str, int]  # path -> effective diff line count
    truncated_diff: bool
    truncated_hunks: bool


@dataclasses.dataclass
class _ReviewStrategy:
    enable_r2: bool
    large_file_threshold: int   # files with more diff lines than this → chunk mode
    max_files_for_r2: int       # total files R2 will process; excess → R1 passthrough
    max_chunks_per_file: int    # per-file chunk cap in chunk mode


def _compute_diff_stats(
    diff_text: str,
    hunks: list,
    truncated_diff: bool,
    truncated_hunks: bool,
) -> _DiffStats:
    from .constants import effective_diff_line_count
    file_diff_lines: Dict[str, int] = {}
    for path, _start, _count, content in hunks:
        file_diff_lines[path] = file_diff_lines.get(path, 0) + effective_diff_line_count(content)
    return _DiffStats(
        diff_lines_total=sum(file_diff_lines.values()),
        file_count=len(file_diff_lines),
        file_diff_lines=file_diff_lines,
        truncated_diff=truncated_diff,
        truncated_hunks=truncated_hunks,
    )


def _decide_review_strategy(stats: _DiffStats) -> _ReviewStrategy:
    # R2 is always enabled — large PRs get tighter limits so the most important files
    # (largest diff, sorted first by _classify_files_for_r2) are still deeply reviewed
    # while excess files fall back to R1 passthrough.
    if stats.diff_lines_total > 3000 or stats.file_count > 50:
        # very large PR: tight limits, only top files get chunk-mode agent review
        return _ReviewStrategy(
            enable_r2=True,
            large_file_threshold=100,
            max_files_for_r2=10,
            max_chunks_per_file=2,
        )
    if stats.diff_lines_total > 1000 or stats.file_count > 20:
        # large PR: moderate limits
        return _ReviewStrategy(
            enable_r2=True,
            large_file_threshold=150,
            max_files_for_r2=15,
            max_chunks_per_file=2,
        )
    # normal PR: full limits
    return _ReviewStrategy(
        enable_r2=True,
        large_file_threshold=200,
        max_files_for_r2=R2_MAX_FILES,
        max_chunks_per_file=R2_MAX_CHUNKS_PER_FILE,
    )


def _truncate_diff_at_file_boundary(diff_text: str, max_chars: int) -> tuple:
    # Truncate diff at a clean file boundary instead of mid-file.
    # Returns (truncated_diff, was_truncated).
    # Finds the last "diff --git" header that starts before max_chars,
    # and cuts just before it if including that file would exceed the limit.
    import re
    file_starts = [m.start() for m in re.finditer(r'^diff --git ', diff_text, re.MULTILINE)]
    if not file_starts:
        return diff_text[:max_chars] + '\n... [diff truncated]\n', True

    # Find the last file boundary we can include completely within max_chars
    cut_pos = file_starts[0]
    for pos in file_starts:
        if pos >= max_chars:
            break
        cut_pos = pos

    # If the last included file itself exceeds max_chars, cut at max_chars within it
    if cut_pos == file_starts[0] and file_starts[0] >= max_chars:
        return diff_text[:max_chars] + '\n... [diff truncated]\n', True

    # Find the start of the next file after cut_pos to get the full slice
    next_file_positions = [p for p in file_starts if p > cut_pos]
    if next_file_positions and next_file_positions[0] <= max_chars:
        # We can include everything up to the next file boundary
        end = next_file_positions[0]
        skipped = len([p for p in file_starts if p >= end])
        marker = f'\n... [diff truncated: {skipped} file(s) omitted]\n'
        return diff_text[:end] + marker, True
    elif not next_file_positions:
        # All files fit
        return diff_text, False
    else:
        # The file starting at cut_pos extends beyond max_chars; include it fully
        # only if it started before max_chars (partial is worse than full)
        end = next_file_positions[0]
        skipped = len([p for p in file_starts if p >= end])
        marker = f'\n... [diff truncated: {skipped} file(s) omitted]\n'
        return diff_text[:end] + marker, True


def review(  # noqa: C901
    pr_number: int,
    repo: str = 'LazyAGI/LazyLLM',
    token: Optional[str] = None,
    backend: Optional[str] = None,
    llm: Optional[Any] = None,
    api_base: Optional[str] = None,
    post_to_github: bool = True,
    max_diff_chars: Optional[int] = 120000,
    max_hunks: Optional[int] = 50,
    arch_cache_path: Optional[str] = None,
    review_spec_cache_path: Optional[str] = None,
    fetch_repo_code: bool = True,
    max_history_prs: int = 20,
    checkpoint_path: Optional[str] = None,
    clear_checkpoint: bool = False,
    resume_from: Optional[ReviewStage] = None,
    language: str = 'cn',
    keep_clone: bool = False,
) -> Dict[str, Any]:
    try:
        import lazyllm.tools.git.review as _self_mod
        original_review_code = inspect.getsource(_self_mod)
    except Exception:
        original_review_code = ''

    pr_dir = _ReviewCheckpoint.pr_dir(pr_number, repo)
    ckpt_path = checkpoint_path or _ReviewCheckpoint.default_path(pr_number, repo)
    if clear_checkpoint:
        # clear_checkpoint takes priority over resume_from
        ckpt = _ReviewCheckpoint(ckpt_path)
        ckpt.clear()
        if os.path.isdir(pr_dir):
            shutil.rmtree(pr_dir, ignore_errors=True)
        ckpt = _ReviewCheckpoint(ckpt_path)
    else:
        ckpt = _ReviewCheckpoint(ckpt_path, resume_from=resume_from)

    prog_main = _Progress(f'Review PR #{pr_number} @ {repo}')

    backend_inst = Git(backend=backend, token=token, repo=repo, api_base=api_base)

    pr_res = backend_inst.get_pull_request(pr_number)
    if not pr_res.get('success'):
        raise RuntimeError(f'Failed to get PR #{pr_number}: {pr_res.get("message", "unknown")}')
    pr = pr_res['pr']
    head_sha = _get_head_sha_from_pr(pr)
    if not head_sha and post_to_github:
        raise RuntimeError('Cannot get PR head sha; cannot post line-level comments')

    # diff: load from checkpoint if available, otherwise fetch and cache
    truncated_diff = False
    diff_text = ckpt.get('diff_text')
    if diff_text is None:
        diff_res = backend_inst.get_pr_diff(pr_number)
        if not diff_res.get('success'):
            raise RuntimeError(f'Failed to get PR #{pr_number} diff: {diff_res.get("message", "unknown")}')
        diff_text = diff_res.get('diff', '')
        if max_diff_chars and len(diff_text) > max_diff_chars:
            diff_text, truncated_diff = _truncate_diff_at_file_boundary(diff_text, max_diff_chars)
        ckpt.save('diff_text', diff_text)
    else:
        _Progress('Pre-round: fetching diff').done('loaded from checkpoint')

    hunks = _parse_unified_diff(diff_text)
    prog_main.update(f'diff parsed: {len(hunks)} hunks')

    # build diff stats and decide review strategy
    diff_stats = _compute_diff_stats(diff_text, hunks, truncated_diff, False)
    strategy = _decide_review_strategy(diff_stats)

    # build meta warning issues for truncation
    meta_warnings: List[Dict[str, Any]] = []
    if truncated_diff:
        meta_warnings.append({
            'type': 'meta',
            'severity': 'normal',
            'bug_category': 'maintainability',
            'path': '',
            'line': 0,
            'problem': (
                f'Review may be incomplete: diff exceeded {max_diff_chars} chars and was truncated '
                f'at a file boundary. Files beyond the limit were skipped entirely.'
            ),
            'suggestion': 'Consider increasing max_diff_chars or splitting the PR.',
            'source': 'meta',
        })

    llm = _get_default_llm() if llm is None else llm
    llm = _ensure_non_streaming_llm(llm)
    model_name = _get_model_name(llm)

    arch_doc, review_spec, clone_dir, agent_instructions = _run_pre_analysis(
        llm, backend_inst, repo, pr, fetch_repo_code,
        arch_cache_path, review_spec_cache_path, max_history_prs, ckpt,
        pr_dir=pr_dir,
    )

    pr_summary = ckpt.get('pr_summary')
    if pr_summary is None:
        pr_summary = _pre_round_pr_summary(
            llm,
            pr_title=getattr(pr, 'title', '') or '',
            pr_body=getattr(pr, 'body', '') or '',
            diff_text=diff_text,
            language=language,
        )
        ckpt.save('pr_summary', pr_summary)
        ckpt.mark_stage_done(ReviewStage.PR_SUMMARY)
    else:
        _Progress('Pre-round: summarizing PR changes').done('loaded from checkpoint')

    existing_comments = _fetch_existing_pr_comments(backend_inst, pr_number)
    prog_main.update(f'{len(existing_comments)} existing PR comments fetched')

    # run lint analysis (independent of LLM rounds)
    lint_issues: List[Dict[str, Any]] = []
    if clone_dir and os.path.isdir(clone_dir):
        try:
            from .lint_runner import _run_lint_analysis
            lint_issues = _run_lint_analysis(diff_text, clone_dir)
        except Exception as e:
            import lazyllm
            lazyllm.LOG.warning(f'Lint analysis failed: {e}')

    final_comments, r2_metrics = _run_five_rounds(
        llm, hunks, diff_text, arch_doc, review_spec, pr_summary, ckpt,
        clone_dir=clone_dir, existing_comments=existing_comments, language=language,
        agent_instructions=agent_instructions, strategy=strategy,
        lint_issues=lint_issues, owner_repo=repo, arch_cache_path=arch_cache_path,
    )
    final_comments = meta_warnings + final_comments

    # clean up only the clone subdirectory (not the whole pr_dir which contains checkpoint)
    clone_subdir = os.path.join(pr_dir, 'clone')
    if not keep_clone and os.path.isdir(clone_subdir):
        shutil.rmtree(clone_subdir, ignore_errors=True)

    posted = 0
    if post_to_github and head_sha:
        review_body = _build_review_body(
            pr_summary=pr_summary,
            total=len(final_comments),
            stats=_category_stats(final_comments),
            model_name=model_name,
        )
        posted = _post_review_comments(
            backend_inst, pr_number, head_sha, final_comments, model_name, review_body=review_body
        )

    n = len(final_comments)
    stats = _category_stats(final_comments)
    summary = (
        f'PR #{pr_number} "{getattr(pr, "title", "")}" — '
        f'{n} issue(s) found across 4 analysis rounds. '
        f'{posted} comment(s) posted to GitHub.'
    )
    prog_main.done(summary)

    metrics = {
        'r2_mode': 'skip' if not strategy.enable_r2 else 'mixed',
        'r2_files_chunk': r2_metrics.get('r2_files_chunk', 0),
        'r2_files_group': r2_metrics.get('r2_files_group', 0),
        'r2_files_skipped': r2_metrics.get('r2_files_skipped', 0),
        'r2_chunks_total': r2_metrics.get('r2_chunks_total', 0),
        'truncated_diff_flag': truncated_diff,
        'truncated_hunks_flag': False,  # hunks are now processed in windows, no truncation
        'lint_issues_count': len(lint_issues),
    }

    return {
        'summary': summary,
        'comments': final_comments,
        'comments_posted': posted,
        'comment_stats': stats,
        'pr_summary': pr_summary,
        'pr_design_doc': ckpt.get('pr_design_doc') or '',
        'original_review_code': original_review_code,
        'metrics': metrics,
    }
