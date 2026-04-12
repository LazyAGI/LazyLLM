# Copyright (c) 2026 LazyAGI. All rights reserved.
from typing import Any, Dict, List, Optional

import lazyllm

from ..base import LazyLLMGitBase
from .utils import _Progress


def _comment_body_text(c: Dict[str, Any], model_name: str) -> str:
    category_tag = f'[{c.get("bug_category", "logic")}]'
    severity_tag = f'[{c.get("severity", "normal")}]'
    return (
        f'*This is an AI-generated suggestion; please verify before applying.*\n\n'
        f'**{severity_tag} {category_tag}** {c.get("problem", "")}\n\n'
        f'**Suggestion:** {c.get("suggestion", "")}\n\n'
        f'---\n'
        f'auto reviewed by BOT ({model_name})'
    )


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


def _post_review_comments(
    backend: LazyLLMGitBase,
    pr_number: int,
    head_sha: Optional[str],
    all_comments: List[Dict[str, Any]],
    model_name: str = 'unknown-model',
    review_body: str = '',
) -> int:
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
        return 0

    prog = _Progress('Posting review (batch submit_review)', len(comments_payload))
    for c in comments_payload:
        prog.update(f'{c["path"]}:{c["line"]}')

    out = backend.submit_review(
        number=pr_number,
        event='COMMENT',
        body=review_body or '',
        comments=comments_payload,
        commit_id=head_sha,
    )
    if out.get('success'):
        prog.done(f'ok: {len(comments_payload)}/{len(comments_payload)}')
        return len(comments_payload)

    lazyllm.LOG.warning(
        f'Batch submit_review failed ({out.get("message", "unknown")}), '
        f'falling back to one-by-one create_review_comment'
    )
    prog2 = _Progress('Posting comments one-by-one (fallback)', len(comments_payload))
    posted = 0
    for c in comments_payload:
        r = backend.create_review_comment(
            pr_number, body=c['body'], path=c['path'], line=c['line'], commit_id=head_sha,
        )
        ok = r.get('success')
        if ok:
            posted += 1
        else:
            err_msg = r.get('message', 'fail')
            lazyllm.LOG.warning(
                f'Failed to post comment on {c["path"]}:{c["line"]}: {err_msg}'
            )
        prog2.update(f'{c["path"]}:{c["line"]} ({"ok" if ok else r.get("message", "fail")[:60]})')
    prog2.done(f'{posted}/{len(comments_payload)} posted')
    return posted
