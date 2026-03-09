# Copyright (c) 2026 LazyAGI. All rights reserved.
'''
Git PR code review: parses diff into hunks, calls the model per hunk, and posts line-level
review comments (issue + suggestion) on the code. No single large comment. Auth via token,
env vars, or gh CLI.
'''
import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union

from .supplier.github import GitHub


def _resolve_github_token(token: Optional[str] = None) -> str:
    if token and token.strip():
        return token.strip()
    env_token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    if env_token and env_token.strip():
        return env_token.strip()
    try:
        out = subprocess.run(
            ['gh', 'auth', 'token'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout and out.stdout.strip():
            return out.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        pass
    raise ValueError(
        'No GitHub token. Use: 1) review(..., token="..."); '
        '2) env GITHUB_TOKEN; 3) gh auth login then run again.'
    )


def _parse_unified_diff(diff_text: str) -> List[Tuple[str, int, int, str]]:
    '''
    Parse unified diff into [(path, new_start_line, new_line_count, hunk_content), ...].
    Each hunk is a contiguous range in the new file for line-level comments.
    '''
    out: List[Tuple[str, int, int, str]] = []
    current_path: Optional[str] = None
    new_start, new_count = 0, 0
    hunk_lines: List[str] = []

    def flush_hunk():
        nonlocal hunk_lines, new_start, new_count, current_path
        if current_path and new_count > 0:
            content = '\n'.join(hunk_lines)
            if content.strip():
                out.append((current_path, new_start, new_count, content))
        hunk_lines = []

    for line in diff_text.splitlines():
        if line.startswith('diff --git '):
            flush_hunk()
            m = re.match(r'diff --git a/(.+?) b/(.+?)(?:\s|$)', line)
            current_path = m.group(2) if m else None
            new_start, new_count = 0, 0
            continue
        if line.startswith('@@'):
            flush_hunk()
            # @@ -old_start,old_count +new_start,new_count @@
            mm = re.search(r'\+(\d+),(\d+)', line)
            if mm:
                new_start = int(mm.group(1))
                new_count = int(mm.group(2))
            continue
        if current_path is None:
            continue
        hunk_lines.append(line)
    flush_hunk()
    return out


def _truncate_hunk_content(content: str, max_lines: int) -> str:
    content_lines = content.splitlines()
    if len(content_lines) > max_lines:
        content_lines = content_lines[:max_lines]
        return '\n'.join(content_lines) + '\n... (truncated)'
    return '\n'.join(content_lines)


def _parse_llm_review_response(
    text: str, new_start: int, new_count: int
) -> List[Dict[str, Any]]:
    '''Parse LLM JSON response into list of review items; validate line in range.'''
    if '```' in text:
        m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if m:
            text = m.group(1).strip()
    arr = json.loads(text)
    if not isinstance(arr, list):
        return []
    result = []
    end_line = new_start + new_count
    for item in arr:
        if not isinstance(item, dict):
            continue
        line = item.get('line')
        if line is None or item.get('problem') is None:
            continue
        try:
            line = int(line)
        except (TypeError, ValueError):
            continue
        if not (new_start <= line < end_line):
            continue
        result.append({
            'line': line,
            'severity': item.get('severity') or 'normal',
            'problem': item.get('problem') or '',
            'suggestion': item.get('suggestion') or '',
        })
    return result


def _call_llm_for_hunk(
    llm: Any,
    path: str,
    new_start: int,
    new_count: int,
    content: str,
    max_content_lines: int = 80,
) -> List[Dict[str, Any]]:
    '''
    Call the model for one hunk; expect a JSON array [{line, severity, problem, suggestion}].
    line is the line number in the new file (between new_start and new_start+new_count-1).
    '''
    content = _truncate_hunk_content(content, max_content_lines)
    prompt = (
        f'You are a code review assistant. Below is a diff snippet from file `{path}`, '
        f'lines **{new_start}** to **{new_start + new_count - 1}** ({new_count} lines) in the new file.\n\n'
        'Inspect each line and list every issue. For **each issue** provide:\n'
        '- line: **line number** in the new file (integer in the range above)\n'
        '- severity: critical (security/serious) / medium / normal (suggestion)\n'
        '- problem: one-sentence description\n'
        '- suggestion: how to fix (concrete code or steps)\n\n'
        'If there are no issues, output an empty array [].\n'
        '**Output only a single JSON array**, no explanation or markdown. Format:\n'
        '[{"line": N, "severity": "critical|medium|normal", "problem": "...", "suggestion": "..."}, ...]\n\n'
        'Diff snippet:\n```\n' + content + '\n```'
    )
    try:
        resp = llm(prompt)
        if not resp or not isinstance(resp, str):
            return []
        return _parse_llm_review_response(resp.strip(), new_start, new_count)
    except (json.JSONDecodeError, Exception):
        return []


def _get_default_llm() -> Any:
    try:
        import lazyllm
        return lazyllm.OnlineChatModule()
    except Exception as e:
        raise RuntimeError(
            'No llm provided and could not create default OnlineChatModule. Pass llm explicitly.'
        ) from e


def _ensure_non_streaming_llm(llm: Any) -> Any:
    if hasattr(llm, '_stream') and llm._stream and hasattr(llm, 'share'):
        return llm.share(stream=False)
    return llm


def _collect_hunk_comments(
    llm: Any, hunks: List[Tuple[str, int, int, str]]
) -> List[Dict[str, Any]]:
    all_comments: List[Dict[str, Any]] = []
    for path, new_start, new_count, content in hunks:
        for it in _call_llm_for_hunk(llm, path, new_start, new_count, content):
            it['path'] = path
            all_comments.append(it)
    return all_comments


def _post_review_comments(
    gh: GitHub,
    pr_number: int,
    head_sha: str,
    all_comments: List[Dict[str, Any]],
) -> int:
    posted = 0
    for c in all_comments:
        body = (
            f'**[{c.get("severity", "normal")}]** {c.get("problem", "")}\n\n'
            f'Suggestion: {c.get("suggestion", "")}'
        )
        out = gh.create_review_comment(
            pr_number, body=body, path=c['path'], line=c['line'], commit_id=head_sha,
        )
        if out.get('success'):
            posted += 1
    return posted


def review(
    pr_number: int,
    repo: str = 'LazyAGI/LazyLLM',
    token: Optional[str] = None,
    llm: Optional[Any] = None,
    api_base: Optional[str] = None,
    post_to_github: bool = False,
    max_diff_chars: Optional[int] = 120000,
    max_hunks: Optional[int] = 50,
) -> Union[str, Dict[str, Any]]:
    '''
    Review a GitHub PR: parse diff, call the model per hunk, post line-level review comments
    (issue + suggestion) on the code. Does not post a single large conversation comment.

    Args:
        pr_number: PR number.
        repo: owner/repo.
        token: GitHub token; optional, resolved from env or gh.
        llm: LLM for inference; if None uses lazyllm.OnlineChatModule().
        api_base: GitHub API base URL.
        post_to_github: If True, post each issue as a line-level comment (visible in Files changed).
        max_diff_chars: Max diff length; None for no limit.
        max_hunks: Max hunks to process; None for no limit.

    Returns:
        dict: summary, comments_posted count, and comments list; no submit if post_to_github=False.
    '''
    token = _resolve_github_token(token)
    gh = GitHub(token=token, repo=repo, api_base=api_base or 'https://api.github.com')

    pr_res = gh.get_pull_request(pr_number)
    if not pr_res.get('success'):
        raise RuntimeError(f'Failed to get PR #{pr_number}: {pr_res.get("message", "unknown")}')
    pr = pr_res['pr']
    head_sha = (pr.raw or {}).get('head', {}).get('sha')
    if not head_sha and post_to_github:
        raise RuntimeError('Cannot get PR head sha; cannot post line-level comments')

    diff_res = gh.get_pr_diff(pr_number)
    if not diff_res.get('success'):
        raise RuntimeError(f'Failed to get PR #{pr_number} diff: {diff_res.get("message", "unknown")}')
    diff_text = diff_res.get('diff', '')
    if max_diff_chars and len(diff_text) > max_diff_chars:
        diff_text = diff_text[:max_diff_chars] + '\n... [diff truncated]\n'

    hunks = _parse_unified_diff(diff_text)
    if max_hunks and len(hunks) > max_hunks:
        hunks = hunks[:max_hunks]

    llm = _get_default_llm() if llm is None else llm
    llm = _ensure_non_streaming_llm(llm)
    all_comments = _collect_hunk_comments(llm, hunks)

    if post_to_github and head_sha:
        posted = _post_review_comments(gh, pr_number, head_sha, all_comments)
        return {
            'summary': f'PR #{pr_number}: {len(all_comments)} review comment(s), {posted} line-level comment(s) posted.',
            'comments_posted': posted,
            'comments': all_comments,
        }
    return {
        'summary': f'PR #{pr_number}: {len(all_comments)} review comment(s) (not posted to GitHub).',
        'comments_posted': 0,
        'comments': all_comments,
    }
