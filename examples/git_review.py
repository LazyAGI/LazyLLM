#!/usr/bin/env python3
# Multi-round intelligent code review for GitHub PRs.
# Usage:
#   python examples/git_review.py --pr_number=1083
#   LAZYLLM_RUN_FULL_REVIEW=1 python examples/git_review.py --pr_number=1083 --model=claude-opus-4-6
#   LAZYLLM_RUN_FULL_REVIEW=1 LAZYLLM_POST_REVIEW_TO_GITHUB=1 \
#       python examples/git_review.py --pr_number=1083 --model=claude-opus-4.6
#
# Required:
#   --pr_number     PR number to review (no default, must be provided)
#
# Optional:
#   --repo          GitHub repo in "owner/name" format (default: LazyAGI/LazyLLM)
#   --model         LLM model name (default: claude-opus-4.6)
#   --base_url      LLM API base URL, e.g. a proxy URL (default: None, use official endpoint)
#   --language      Review language: cn or en (default: cn)
#   --max_history_prs  Max historical PRs to analyze for review spec (default: 400)
#   --keep_clone    Keep cloned repo after review (flag, default: off)
#   --clear_checkpoint  Clear checkpoint and start fresh (flag, default: off)
#
# Environment variables (still supported):
#   LAZYLLM_RUN_FULL_REVIEW=1       actually run the LLM review (otherwise just fetch PR info)
#   LAZYLLM_POST_REVIEW_TO_GITHUB=1 post comments to GitHub
#   LAZYLLM_ARCH_CACHE              path to arch doc cache JSON
#   LAZYLLM_SPEC_CACHE              path to review spec cache JSON

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_gh_bin = os.path.expanduser('~/gh/bin')
if os.path.isdir(_gh_bin):
    os.environ['PATH'] = _gh_bin + os.pathsep + os.environ.get('PATH', '')


def _parse_args():
    parser = argparse.ArgumentParser(description='Multi-round PR code review')
    parser.add_argument('--pr_number', type=int, required=True,
                        help='PR number to review (required)')
    parser.add_argument('--repo', default='LazyAGI/LazyLLM',
                        help='GitHub repo in owner/name format (default: LazyAGI/LazyLLM)')
    parser.add_argument('--model', default='claude-opus-4-6',
                        help='LLM model name (default: claude-opus-4-6)')
    parser.add_argument('--base_url', default=None,
                        help='LLM API base URL, e.g. a proxy URL (default: None, use official endpoint)')
    parser.add_argument('--language', default='cn', choices=['cn', 'en'],
                        help='Review output language (default: cn)')
    parser.add_argument('--max_history_prs', type=int, default=50,
                        help='Max historical PRs to analyze for review spec (default: 400)')
    parser.add_argument('--keep_clone', action='store_true',
                        help='Keep cloned repo directory after review')
    parser.add_argument('--clear_checkpoint', action='store_true',
                        help='Clear checkpoint and start fresh')
    return parser.parse_args()


def main():  # noqa C901
    args = _parse_args()
    repo = args.repo
    pr_number = args.pr_number

    from lazyllm.tools.git import Git, review

    print(f'1. Creating Git backend for {repo} (token from env or gh CLI)...')
    try:
        backend = Git(backend='github', repo=repo)
        print('   Backend ready.')
    except ValueError as e:
        print(f'   Failed: {e}')
        return 1

    print(f'2. Fetching PR #{pr_number} and diff via API...')
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
        print('3. Skipping full review (set LAZYLLM_RUN_FULL_REVIEW=1 to run).')
        print('Done.')
        return 0

    import lazyllm
    llm_kwargs = dict(source='claude', model=args.model)
    if args.base_url:
        llm_kwargs['base_url'] = args.base_url
    llm = lazyllm.OnlineChatModule(**llm_kwargs)
    post_to_github = os.environ.get('LAZYLLM_POST_REVIEW_TO_GITHUB') == '1'
    arch_cache = os.environ.get('LAZYLLM_ARCH_CACHE') or None
    spec_cache = os.environ.get('LAZYLLM_SPEC_CACHE') or None

    print(f'3. Running multi-round review (model={args.model}, language={args.language})...')
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
            max_history_prs=args.max_history_prs,
            language=args.language,
            keep_clone=args.keep_clone,
            clear_checkpoint=args.clear_checkpoint,
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
