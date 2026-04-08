#!/usr/bin/env python3
# Review PR #1068 (LazyAGI/LazyLLM) using the multi-round intelligent review system.
# Usage:
#   python examples/git_review.py
#   LAZYLLM_RUN_FULL_REVIEW=1 python examples/git_review.py
#   LAZYLLM_RUN_FULL_REVIEW=1 LAZYLLM_POST_REVIEW_TO_GITHUB=1 python examples/git_review.py
#
# Cache files (optional, skip re-analysis on repeated runs):
#   LAZYLLM_ARCH_CACHE=/tmp/lazyllm_arch.json
#   LAZYLLM_SPEC_CACHE=/tmp/lazyllm_spec.json

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_gh_bin = os.path.expanduser('~/gh/bin')
if os.path.isdir(_gh_bin):
    os.environ['PATH'] = _gh_bin + os.pathsep + os.environ.get('PATH', '')


def main():
    repo = 'LazyAGI/LazyLLM'
    pr_number = 1083

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

    import lazyllm
    llm = lazyllm.OnlineChatModule(source='claude', model='claude-opus-4-6', base_url='https://cld.ppapi.vip/v1/')
    post_to_github = os.environ.get('LAZYLLM_POST_REVIEW_TO_GITHUB') == '1'
    arch_cache = os.environ.get('LAZYLLM_ARCH_CACHE') or None
    spec_cache = os.environ.get('LAZYLLM_SPEC_CACHE') or None

    print('3. Running multi-round code review (pre-analysis + 4 rounds)...')
    if post_to_github:
        print('   Will post line-level comments to PR Files changed.')
    try:
        out = review(
            pr_number,
            repo=repo,
            backend='github',
            llm=llm,
            post_to_github=post_to_github,
            arch_cache_path=arch_cache,
            review_spec_cache_path=spec_cache,
            fetch_repo_code=True,
            max_history_prs=400,
        )
        print('--- Review result ---')
        print(out.get('summary', out))
        comments = out.get('comments', [])
        print(f'   Total issues: {len(comments)}, posted: {out.get("comments_posted", 0)}')
        for i, c in enumerate(comments[:10]):
            cat = c.get('bug_category', '')
            sev = c.get('severity', '')
            prob = c.get('problem', '')[:60]
            print(f'   [{i+1}] {c.get("path")} L{c.get("line")} [{sev}][{cat}] {prob}...')
        if len(comments) > 10:
            print(f'   ... and {len(comments) - 10} more')
        if post_to_github and out.get('comments_posted', 0) > 0:
            print('\n[Posted] Line-level comments posted to PR; check Files changed for inline comments.')
    except Exception as e:
        print(f'   Review failed: {e}')
        return 1

    print('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
