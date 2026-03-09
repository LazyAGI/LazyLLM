#!/usr/bin/env python3
# Test lazyllm.tools.git.review and GitHub integration (e.g. PR #1053).
# Usage (token resolved inside review; if gh is only in zshrc, prepend PATH):
#   python scripts/test_git_review.py
#   PATH="$HOME/gh/bin:$PATH" python scripts/test_git_review.py
# Full review (model called per hunk, line-level comments; default sensenova):
#   LAZYLLM_RUN_FULL_REVIEW=1 python scripts/test_git_review.py
# Review and post to PR (line comments visible in Files changed):
#   LAZYLLM_RUN_FULL_REVIEW=1 LAZYLLM_POST_REVIEW_TO_GITHUB=1 python scripts/test_git_review.py

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# If gh lives in ~/gh/bin and is only in zshrc, add it to PATH
_gh_bin = os.path.expanduser('~/gh/bin')
if os.path.isdir(_gh_bin):
    os.environ['PATH'] = _gh_bin + os.pathsep + os.environ.get('PATH', '')


def main():
    repo = 'LazyAGI/LazyLLM'
    pr_number = 1053

    from lazyllm.tools.git import Git, review

    print('1. Creating Git backend (token from env or gh CLI)...')
    try:
        backend = Git(backend='github', repo=repo)
        print('   Backend ready.')
    except ValueError as e:
        print(f'   Failed: {e}')
        return 1

    print('2. Fetching PR and diff via API...')
    pr_res = backend.get_pull_request(pr_number)
    if not pr_res.get('success'):
        print(f'   Failed to get PR: {pr_res.get("message")}')
        return 1
    pr = pr_res['pr']
    print(f'   PR #{pr.number}: {pr.title}')
    print(f'   {pr.source_branch} -> {pr.target_branch}')

    diff_res = backend.get_pr_diff(pr_number)
    if not diff_res.get('success'):
        print(f'   Failed to get diff: {diff_res.get("message")}')
        return 1
    diff_len = len(diff_res.get('diff', ''))
    print(f'   Diff length: {diff_len} chars')

    if os.environ.get('LAZYLLM_RUN_FULL_REVIEW') != '1':
        print('3. Skipping full review (set LAZYLLM_RUN_FULL_REVIEW=1 and configure LLM to run).')
        print('Done.')
        return 0

    # Use sensenova for testing (real model output, no bypass)
    import lazyllm
    llm = lazyllm.OnlineChatModule(source='sensenova')
    post_to_github = os.environ.get('LAZYLLM_POST_REVIEW_TO_GITHUB') == '1'

    print('3. Running code review (model per hunk, line-level comments)...')
    if post_to_github:
        print('   Will post line-level comments to PR Files changed.')
    try:
        out = review(
            pr_number,
            repo=repo,
            backend='github',
            llm=llm,
            post_to_github=post_to_github,
        )
        print('--- Review result ---')
        print(out.get('summary', out))
        print(f'   Comments: {len(out.get("comments", []))}, posted: {out.get("comments_posted", 0)}')
        for i, c in enumerate(out.get('comments', [])[:5]):
            print(f'   [{i+1}] {c.get("path")} L{c.get("line")} [{c.get("severity")}] {c.get("problem", "")[:60]}...')
        if len(out.get('comments', [])) > 5:
            print('   ...')
        if post_to_github and out.get('comments_posted', 0) > 0:
            print('\n[Posted] Line-level comments posted to PR; check Files changed for inline comments.')
    except Exception as e:
        print(f'   Review failed: {e}')
        return 1

    print('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
