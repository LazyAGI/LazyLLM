# Copyright (c) 2026 LazyAGI. All rights reserved.
import inspect
import os
import shutil
from typing import Any, Dict, Optional

from ..client import Git
from .checkpoint import _ReviewCheckpoint
from .pre_analysis import _run_pre_analysis, _pre_round_pr_summary
from .rounds import _run_four_rounds
from .poster import _fetch_existing_pr_comments, _post_review_comments
from .utils import (
    _get_default_llm, _ensure_non_streaming_llm, _get_model_name,
    _get_head_sha_from_pr, _parse_unified_diff, _Progress,
    _category_stats, _build_review_body,
)


def review(
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
    language: str = 'cn',
) -> Dict[str, Any]:
    try:
        import lazyllm.tools.git.review as _self_mod
        original_review_code = inspect.getsource(_self_mod)
    except Exception:
        original_review_code = ''

    ckpt_path = checkpoint_path or _ReviewCheckpoint.default_path(pr_number, repo)
    ckpt = _ReviewCheckpoint(ckpt_path)
    if clear_checkpoint:
        ckpt.clear()
        ckpt = _ReviewCheckpoint(ckpt_path)

    prog_main = _Progress(f'Review PR #{pr_number} @ {repo}')

    backend_inst = Git(backend=backend, token=token, repo=repo, api_base=api_base)

    pr_res = backend_inst.get_pull_request(pr_number)
    if not pr_res.get('success'):
        raise RuntimeError(f'Failed to get PR #{pr_number}: {pr_res.get("message", "unknown")}')
    pr = pr_res['pr']
    head_sha = _get_head_sha_from_pr(pr)
    if not head_sha and post_to_github:
        raise RuntimeError('Cannot get PR head sha; cannot post line-level comments')

    diff_res = backend_inst.get_pr_diff(pr_number)
    if not diff_res.get('success'):
        raise RuntimeError(f'Failed to get PR #{pr_number} diff: {diff_res.get("message", "unknown")}')
    diff_text = diff_res.get('diff', '')
    if max_diff_chars and len(diff_text) > max_diff_chars:
        diff_text = diff_text[:max_diff_chars] + '\n... [diff truncated]\n'

    hunks = _parse_unified_diff(diff_text)
    if max_hunks and len(hunks) > max_hunks:
        hunks = hunks[:max_hunks]
    prog_main.update(f'diff parsed: {len(hunks)} hunks')

    llm = _get_default_llm() if llm is None else llm
    llm = _ensure_non_streaming_llm(llm)
    model_name = _get_model_name(llm)

    arch_doc, review_spec, clone_dir = _run_pre_analysis(
        llm, backend_inst, repo, pr, fetch_repo_code,
        arch_cache_path, review_spec_cache_path, max_history_prs, ckpt,
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
    else:
        _Progress('Pre-round: summarizing PR changes').done('loaded from checkpoint')

    try:
        existing_comments = _fetch_existing_pr_comments(backend_inst, pr_number)
        prog_main.update(f'{len(existing_comments)} existing PR comments fetched')
        final_comments = _run_four_rounds(
            llm, hunks, diff_text, arch_doc, review_spec, pr_summary, ckpt,
            clone_dir=clone_dir, existing_comments=existing_comments, language=language,
        )
    finally:
        if clone_dir:
            if os.path.isdir(clone_dir):
                shutil.rmtree(clone_dir, ignore_errors=True)

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
    ckpt.clear()

    return {
        'summary': summary,
        'comments': final_comments,
        'comments_posted': posted,
        'comment_stats': stats,
        'pr_summary': pr_summary,
        'original_review_code': original_review_code,
    }
