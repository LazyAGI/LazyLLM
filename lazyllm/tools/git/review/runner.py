# Copyright (c) 2026 LazyAGI. All rights reserved.
import dataclasses
import inspect
import os
import shutil
from typing import Any, Dict, List, Optional

import lazyllm

from ..client import Git
from .checkpoint import _ReviewCheckpoint, ReviewStage
from .pre_analysis import _run_pre_analysis, _pre_round_pr_summary
from .rounds import _run_four_rounds
from .poster import _fetch_existing_pr_comments, _post_review_comments, _build_commentable_lines, _filter_commentable
from .output import write_review_json
from .utils import (
    _get_default_llm, _ensure_non_streaming_llm, _get_model_name,
    _get_head_sha_from_pr, _parse_unified_diff, _Progress,
    _category_stats, _build_review_body,
)
from .constants import R3_MAX_FILES, R3_MAX_CHUNKS_PER_FILE


@dataclasses.dataclass
class _DiffStats:
    diff_lines_total: int
    file_count: int
    file_diff_lines: Dict[str, int]  # path -> effective diff line count
    truncated_diff: bool
    truncated_hunks: bool


@dataclasses.dataclass
class _ReviewStrategy:
    enable_r3: bool             # always True for now; kept for future per-strategy toggle
    large_file_threshold: int   # files with more diff lines than this -> chunk mode
    max_files_for_r3: int       # total files R3 will process; excess -> R1 passthrough
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
    if stats.diff_lines_total > 3000 or stats.file_count > 50:
        return _ReviewStrategy(
            enable_r3=True,
            large_file_threshold=100,
            max_files_for_r3=10,
            max_chunks_per_file=2,
        )
    if stats.diff_lines_total > 1000 or stats.file_count > 20:
        return _ReviewStrategy(
            enable_r3=True,
            large_file_threshold=150,
            max_files_for_r3=15,
            max_chunks_per_file=2,
        )
    return _ReviewStrategy(
        enable_r3=True,
        large_file_threshold=200,
        max_files_for_r3=R3_MAX_FILES,
        max_chunks_per_file=R3_MAX_CHUNKS_PER_FILE,
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
    refresh_diff: bool = False,
    output_path: Optional[str] = None,
    repo_path: Optional[str] = None,
    base: Optional[str] = None,
    include_uncommitted: bool = True,
) -> Dict[str, Any]:
    try:
        import lazyllm.tools.git.review as _self_mod
        original_review_code = inspect.getsource(_self_mod)
    except Exception:
        original_review_code = ''

    backend_inst = Git(backend=backend, token=token, repo=repo, api_base=api_base,
                       repo_path=repo_path, base=base, include_uncommitted=include_uncommitted)

    # local mode: resolve checkpoint key from branch name before building paths
    from ..supplier.local import LocalGit
    _local_repo_path: Optional[str] = None
    _ckpt_key: Any = pr_number  # used as the "pr identifier" for checkpoint paths
    if isinstance(backend_inst, LocalGit):
        post_to_github = False
        fetch_repo_code = False
        _local_repo_path = backend_inst.local_repo_path
        # derive repo name from git remote origin so arch/spec cache is shared with cloud mode
        repo = backend_inst.get_origin_repo()
        branch = backend_inst._current_branch().replace('/', '_')
        _ckpt_key = f'local/{branch}'
        pr_dir = _ReviewCheckpoint.pr_dir(_ckpt_key, repo)
    ckpt_path = checkpoint_path or _ReviewCheckpoint.default_path(_ckpt_key, repo)
    if clear_checkpoint:
        # clear_checkpoint takes priority over resume_from
        ckpt = _ReviewCheckpoint(ckpt_path)
        ckpt.clear()
        if os.path.isdir(pr_dir):
            shutil.rmtree(pr_dir, ignore_errors=True)
        ckpt = _ReviewCheckpoint(ckpt_path)
    else:
        ckpt = _ReviewCheckpoint(ckpt_path, resume_from=resume_from)

    prog_main = _Progress(f'Review PR #{pr_number} @ {repo}' if not _local_repo_path
                          else f'Review local branch {_ckpt_key.split("/", 1)[-1]} @ {repo}')

    # always need PR metadata for title/body
    pr_res = backend_inst.get_pull_request(pr_number)
    if not pr_res.get('success'):
        raise RuntimeError(f'Failed to get PR #{pr_number}: {pr_res.get("message", "unknown")}')
    pr = pr_res['pr']

    # head_sha: when refresh_diff=False, prefer cached sha to avoid rotating checkpoint
    cached_head_sha = ckpt.get('head_sha')
    live_head_sha = _get_head_sha_from_pr(pr)
    if refresh_diff or not cached_head_sha:
        # use live sha; rotate if changed
        head_sha = live_head_sha
        if not head_sha and post_to_github:
            raise RuntimeError('Cannot get PR head sha; cannot post line-level comments')
        if cached_head_sha and head_sha and cached_head_sha != head_sha:
            ckpt.rotate_on_head_sha_change(head_sha)
        if head_sha:
            ckpt.save('head_sha', head_sha)
    else:
        head_sha = cached_head_sha

    # diff: load from checkpoint if available, otherwise fetch and cache
    truncated_diff = False
    diff_text = ckpt.get('diff_text')
    if diff_text is None:
        if not refresh_diff and cached_head_sha:
            # diff cache lost but head_sha is pinned — must refresh to stay consistent
            lazyllm.LOG.warning(
                'Diff cache missing but head_sha is pinned; auto-refreshing diff. '
                'If PR has new pushes, use --refresh_diff to rotate checkpoint.'
            )
            live_sha = _get_head_sha_from_pr(pr)
            if live_sha and cached_head_sha and live_sha != cached_head_sha:
                lazyllm.LOG.warning(
                    f'PR head changed ({cached_head_sha[:8]} → {live_sha[:8]}) since last run. '
                    f'Rotating checkpoint to stay consistent.'
                )
                ckpt.rotate_on_head_sha_change(live_sha)
                head_sha = live_sha
        diff_res = backend_inst.get_pr_diff(pr_number)
        if not diff_res.get('success'):
            raise RuntimeError(f'Failed to get PR #{pr_number} diff: {diff_res.get("message", "unknown")}')
        diff_text = diff_res.get('diff', '')
        if max_diff_chars and len(diff_text) > max_diff_chars:
            diff_text, truncated_diff = _truncate_diff_at_file_boundary(diff_text, max_diff_chars)
        ckpt.save('diff_text', diff_text)
        if head_sha:
            ckpt.save('head_sha', head_sha)
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
        pr_dir=pr_dir, head_sha=head_sha, local_repo_path=_local_repo_path,
    )

    # local mode: use the local repo path directly as clone_dir for file skeleton extraction
    if _local_repo_path and os.path.isdir(_local_repo_path):
        clone_dir = _local_repo_path

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
            lazyllm.LOG.warning(f'Lint analysis failed: {e}')

    cached_final = ckpt.get('final_comments')
    if cached_final is not None:
        final_comments = cached_final
        r3_metrics: Dict[str, Any] = {}
        _Progress('All review rounds').done('loaded from checkpoint')
    else:
        final_comments, r3_metrics = _run_four_rounds(
            llm, hunks, diff_text, arch_doc, review_spec, pr_summary, ckpt,
            clone_dir=clone_dir, existing_comments=existing_comments, language=language,
            agent_instructions=agent_instructions, strategy=strategy,
            lint_issues=lint_issues, owner_repo=repo, arch_cache_path=arch_cache_path,
        )
        final_comments = meta_warnings + final_comments
        ckpt.save('final_comments', final_comments)
        ckpt.mark_stage_done(ReviewStage.FINAL)

    # run RCov test coverage check (independent of R4, results appended directly)
    rcov_issues: List[Dict[str, Any]] = []
    if clone_dir and os.path.isdir(clone_dir):
        try:
            from .coverage_checker import _run_coverage_check
            rcov_issues = _run_coverage_check(llm, diff_text, pr_summary, clone_dir, language)
        except Exception as e:
            lazyllm.LOG.warning(f'RCov analysis failed: {e}')

    # merge: final_comments (through R4 dedup) + rcov_issues (bypass R4)
    all_comments = final_comments + rcov_issues

    # clean up only the clone subdirectory (not the whole pr_dir which contains checkpoint)
    # local mode: clone_dir IS the user's repo, never delete it
    clone_subdir = os.path.join(pr_dir, 'clone')
    if not keep_clone and not _local_repo_path and os.path.isdir(clone_subdir):
        shutil.rmtree(clone_subdir, ignore_errors=True)

    posted = 0
    if post_to_github and head_sha:
        if ckpt.should_use_cache(ReviewStage.UPLOAD):
            _Progress('Upload: posting review comments').done('loaded from checkpoint (already uploaded)')
        else:
            # clear stale batch tracking — if we're here, UPLOAD is not done,
            # so any previous batch markers are from an old run with different comments
            if ckpt:
                ckpt.save('upload_done_batches', [])
            review_body = _build_review_body(
                pr_summary=pr_summary,
                total=len(all_comments),
                stats=_category_stats(all_comments),
                model_name=model_name,
            )
            commentable = _build_commentable_lines(hunks)
            postable, n_dropped = _filter_commentable(all_comments, commentable)
            if n_dropped:
                lazyllm.LOG.warning(
                    f'{n_dropped} comment(s) dropped: line not in PR diff range '
                    f'(would cause GitHub 422)'
                )
            posted, upload_all_ok = _post_review_comments(
                backend_inst, pr_number, head_sha, postable, model_name, review_body=review_body, ckpt=ckpt
            )
            if upload_all_ok:
                ckpt.mark_stage_done(ReviewStage.UPLOAD)
            else:
                lazyllm.LOG.warning('Some upload batches failed; re-run with --resume_from=upload to retry')

    n = len(all_comments)
    stats = _category_stats(all_comments)
    summary = (
        f'PR #{pr_number} "{getattr(pr, "title", "")}" — '
        f'{n} issue(s) found across 4 analysis rounds. '
        f'{posted} comment(s) posted to GitHub.'
    )
    prog_main.done(summary)

    metrics = {
        'r3_mode': 'skip' if not strategy.enable_r3 else 'mixed',
        'r3_files_chunk': r3_metrics.get('r3_files_chunk', 0),
        'r3_files_group': r3_metrics.get('r3_files_group', 0),
        'r3_files_skipped': r3_metrics.get('r3_files_skipped', 0),
        'r3_chunks_total': r3_metrics.get('r3_chunks_total', 0),
        'truncated_diff_flag': truncated_diff,
        'truncated_hunks_flag': False,  # hunks are processed in sliding windows; per-hunk truncation is not applied
        'lint_issues_count': len(lint_issues),
        'rcov_issues_count': len(rcov_issues),
    }

    result = {
        'summary': summary,
        'comments': all_comments,
        'comments_posted': posted,
        'comment_stats': stats,
        'pr_summary': pr_summary,
        'pr_info': {'source_branch': getattr(pr, 'source_branch', ''),
                    'target_branch': getattr(pr, 'target_branch', '')},
        'pr_design_doc': ckpt.get('pr_design_doc') or '',
        'original_review_code': original_review_code,
        'metrics': metrics,
    }
    if output_path:
        write_review_json(result, output_path)
        lazyllm.LOG.info(f'Review result written to {output_path}')
    return result
