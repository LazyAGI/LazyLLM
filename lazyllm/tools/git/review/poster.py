# Copyright (c) 2026 LazyAGI. All rights reserved.
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import lazyllm

from ..base import LazyLLMGitBase
from .utils import _Progress

_BATCH_SIZE = 30          # comments per submit_review call (GitHub limit is ~50)
_BATCH_INTERVAL = 5.0     # seconds between batches to avoid secondary rate limit
_RATE_LIMIT_BACKOFF = [60, 120, 300]  # retry waits (seconds) on 403


def _build_commentable_lines(hunks: List[Tuple[str, int, int, str]]) -> Dict[str, Set[int]]:
    # Build a mapping of path -> set of new-file line numbers that GitHub will accept
    # for RIGHT-side review comments.  Only lines that actually appear in the new file
    # (i.e. context lines ' ' and added lines '+') are valid; deleted lines '-' are NOT
    # present in the new file and GitHub rejects them with 422.
    # When content is available, parse it precisely; otherwise fall back to new_count range.
    commentable: Dict[str, Set[int]] = {}
    for path, new_start, new_count, content in hunks:
        s = commentable.setdefault(path, set())
        lines = content.splitlines() if content else []
        if lines:
            new_no = new_start
            for raw_line in lines:
                if raw_line.startswith('-'):
                    continue
                s.add(new_no)
                new_no += 1
        else:
            s.update(range(new_start, new_start + new_count))
    return commentable


def _filter_commentable(
    comments: List[Dict[str, Any]],
    commentable: Dict[str, Set[int]],
) -> Tuple[List[Dict[str, Any]], int]:
    # Drop comments whose (path, line) is not in the PR diff.
    # Returns (kept, dropped_count).
    kept, dropped = [], 0
    for c in comments:
        path = c.get('path', '')
        line = c.get('line')
        if line is None:
            dropped += 1
            continue
        allowed = commentable.get(path)
        if allowed is None or int(line) not in allowed:
            lazyllm.LOG.error(
                f'[FILTER] dropping comment: {path}:{line} not in PR diff — '
                f'line does not correspond to any added/context line in the diff'
            )
            dropped += 1
        else:
            kept.append(c)
    return kept, dropped


def _suggestion_prefix(suggestion: str) -> str:
    return '\n' if suggestion.startswith('```') else ''


def _comment_body_text(c: Dict[str, Any], model_name: str) -> str:
    category_tag = f'[{c.get("bug_category", "logic")}]'
    severity_tag = f'[{c.get("severity", "normal")}]'
    return (
        '*This suggestion is AI-assisted and has been manually reviewed for relevance. If you disagree, '
        'provide concrete technical reasoning or counterexamples. Newly introduced architecture issues must '
        'be fixed before merging; pre-existing ones must be tracked via an issue (new or linked). '
        'Style issues must also be fixed; missing test coverage must be added. Responses '
        'without concrete actions are incomplete, and consensus-based arguments (e.g., “others are doing this”) '
        f'alone are not sufficient.*\n\n**{severity_tag} {category_tag}** {c.get("problem", "")}\n\n'
        f'**Suggestion:** {_suggestion_prefix(c.get("suggestion", ""))}{c.get("suggestion", "")}\n\n'
        f'---\nauto reviewed by BOT ({model_name})')


def _fetch_existing_pr_comments(backend: LazyLLMGitBase, pr_number: int) -> List[Dict[str, Any]]:
    res = backend.list_review_comments(pr_number)
    if not res.get('success'):
        lazyllm.LOG.warning(f'Failed to fetch existing PR comments: {res.get("message", "unknown")}')
        return []
    raw_comments = res.get('comments') or []
    result = []
    for c in raw_comments:
        body = (c.get('body') if isinstance(c, dict) else getattr(c, 'body', '')) or ''
        path = (c.get('path') if isinstance(c, dict) else getattr(c, 'path', '')) or ''
        line = (c.get('line') if isinstance(c, dict) else getattr(c, 'line', None))
        if not body.strip():
            continue
        entry: Dict[str, Any] = {'body': body}
        if path:
            entry['path'] = path
        if line is not None:
            try:
                entry['line'] = int(line)
            except (TypeError, ValueError):
                pass
        result.append(entry)
    return result


def _submit_with_retry(
    backend: LazyLLMGitBase,
    pr_number: int,
    head_sha: Optional[str],
    batch: List[Dict[str, Any]],
    body: str,
) -> bool:
    for wait in _RATE_LIMIT_BACKOFF + [None]:
        r = backend.submit_review(
            number=pr_number,
            event='COMMENT',
            body=body,
            comments=batch,
            commit_id=head_sha,
        )
        if r.get('success'):
            return True
        status = r.get('status_code', 0)
        if status == 403 and wait is not None:
            lazyllm.LOG.warning(f'Rate limited (403), retrying after {wait}s...')
            time.sleep(wait)
            continue
        lazyllm.LOG.warning(f'submit_review failed: {r.get("message", "unknown")[:200]}')
        return False
    return False


def _post_review_comments(
    backend: LazyLLMGitBase,
    pr_number: int,
    head_sha: Optional[str],
    all_comments: List[Dict[str, Any]],
    model_name: str = 'unknown-model',
    review_body: str = '',
    ckpt: Optional[Any] = None,
) -> tuple:
    comments_payload = [
        {
            'path': c['path'],
            'line': int(c['line']),
            'body': _comment_body_text(c, model_name),
            'side': 'RIGHT',
        }
        for c in all_comments if c.get('path') and c.get('line')
    ]
    dropped = len(all_comments) - len(comments_payload)
    if dropped:
        lazyllm.LOG.warning(f'{dropped} comment(s) dropped: missing path or line field')
    if not comments_payload:
        return 0, True

    batches = [comments_payload[i:i + _BATCH_SIZE] for i in range(0, len(comments_payload), _BATCH_SIZE)]
    # load already-completed batch indices from checkpoint (not subject to stage invalidation)
    done_batches: set = set(ckpt.get('upload_done_batches') or [] if ckpt else [])
    prog = _Progress(f'Posting review ({len(comments_payload)} comments, {len(batches)} batch(es))', len(batches))
    posted = sum(len(batches[i]) for i in done_batches if i < len(batches))
    all_ok = True
    for idx, batch in enumerate(batches):
        if idx in done_batches:
            prog.update(f'batch {idx + 1}/{len(batches)}: skipped (already posted)')
            continue
        # first non-skipped batch carries the review_body summary
        body = review_body if idx == 0 or not done_batches else ''
        ok = _submit_with_retry(backend, pr_number, head_sha, batch, body)
        if ok:
            posted += len(batch)
            done_batches.add(idx)
            if ckpt:
                ckpt.save('upload_done_batches', list(done_batches))
            prog.update(f'batch {idx + 1}/{len(batches)}: {len(batch)} comments ok')
        else:
            all_ok = False
            prog.update(f'batch {idx + 1}/{len(batches)}: FAILED, skipping')
        if idx < len(batches) - 1:
            time.sleep(_BATCH_INTERVAL)
    prog.done(f'{posted}/{len(comments_payload)} posted')
    return posted, all_ok
