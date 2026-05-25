#!/usr/bin/env python3
# Multi-round intelligent code review — Python API example.
#
# For most use cases, prefer the CLI commands:
#
#   Cloud PR review (GitHub):
#     lazyllm review --pr 1083 --repo LazyAGI/LazyLLM --model claude-opus-4-6
#     lazyllm review --pr 1083 --post          # also post comments to PR
#
#   Local git review:
#     lazyllm review-local                     # review current branch vs main, output review.json
#     lazyllm review-local --base develop --output my-review.json
#     lazyllm review-local --no-uncommitted    # only committed changes
#
# This file demonstrates the Python API directly.
#
# Usage (cloud PR):
#   python examples/git_review.py --pr_number=1083
#   python examples/git_review.py --pr_number=1083 --model=claude-opus-4-6 --post
#
# Usage (local):
#   python examples/git_review.py --local
#   python examples/git_review.py --local --base develop --output my-review.json
#
# Required (cloud mode):
#   --pr_number     PR number to review
#
# Optional:
#   --repo          GitHub repo in "owner/name" format (default: LazyAGI/LazyLLM)
#   --model         LLM model name (default: claude-opus-4-6)
#   --source        LLM source (default: claude)
#   --base_url      LLM API base URL (default: None)
#   --language      Review language: cn or en (default: cn)
#   --post          Post comments to GitHub PR (flag)
#   --keep_clone    Keep cloned repo after review (flag)
#   --clear_checkpoint  Clear checkpoint and start fresh (flag)
#   --refresh_diff  Fetch latest PR head SHA and diff (flag)
#   --resume_from   Resume from a specific stage
#   --local         Use local git mode instead of cloud PR
#   --repo_path     Local repo path (default: .)
#   --base          Base branch for local diff (default: main)
#   --no_uncommitted  Exclude uncommitted changes (flag)
#   --output        Output JSON path for local mode (default: review.json)

import argparse
import os
import sys
import traceback

from lazyllm.tools.git.review.checkpoint import ReviewStage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_gh_bin = os.path.expanduser('~/gh/bin')
if os.path.isdir(_gh_bin):
    os.environ['PATH'] = _gh_bin + os.pathsep + os.environ.get('PATH', '')


def _parse_args():
    parser = argparse.ArgumentParser(description='Multi-round PR/local code review (Python API example)')
    # cloud mode
    parser.add_argument('--pr_number', type=int, default=None,
                        help='PR number to review (required for cloud mode)')
    parser.add_argument('--source', default='claude',
                        help='LLM source (default: claude)')
    parser.add_argument('--repo', default='LazyAGI/LazyLLM',
                        help='GitHub repo in owner/name format (default: LazyAGI/LazyLLM)')
    parser.add_argument('--model', default='claude-opus-4-6',
                        help='LLM model name (default: claude-opus-4-6)')
    parser.add_argument('--base_url', default=None,
                        help='LLM API base URL (default: None)')
    parser.add_argument('--language', default='cn', choices=['cn', 'en'],
                        help='Review output language (default: cn)')
    parser.add_argument('--max_history_prs', type=int, default=None,
                        help='Max historical PRs to analyze for review spec')
    parser.add_argument('--keep_clone', action='store_true',
                        help='Keep cloned repo directory after review')
    parser.add_argument('--clear_checkpoint', action='store_true',
                        help='Clear checkpoint and start fresh')
    parser.add_argument('--refresh_diff', action='store_true',
                        help='Fetch latest PR head SHA and diff')
    parser.add_argument('--resume_from', default=None,
                        choices=['clone', 'arch', 'spec', 'pr_summary',
                                 'r1', 'r2', 'r3', 'r4a', 'r4', 'final', 'upload'],
                        help='Resume from a specific stage')
    parser.add_argument('--post', action='store_true',
                        help='Post review comments to GitHub PR')
    # local mode
    parser.add_argument('--local', action='store_true',
                        help='Use local git mode instead of cloud PR')
    parser.add_argument('--repo_path', default='.',
                        help='Local repo path (default: current directory)')
    parser.add_argument('--base', default='main',
                        help='Base branch for local diff (default: main)')
    parser.add_argument('--no_uncommitted', action='store_true',
                        help='Exclude uncommitted changes from diff')
    parser.add_argument('--output', default='review.json',
                        help='Output JSON path for local mode (default: review.json)')
    return parser.parse_args()


def main():  # noqa C901
    args = _parse_args()

    from lazyllm.tools.git import Git, review

    import lazyllm
    llm_kwargs = dict(source=args.source, model=args.model)
    if args.base_url:
        llm_kwargs['base_url'] = args.base_url
    llm = lazyllm.OnlineChatModule(**llm_kwargs)

    if args.local:
        # --- local git mode ---
        print(f'Running local review: {args.repo_path!r}, base={args.base}, '
              f'uncommitted={"no" if args.no_uncommitted else "yes"}')
        try:
            out = review(
                pr_number=0,
                backend='local',
                llm=llm,
                post_to_github=False,
                fetch_repo_code=False,
                language=args.language,
                clear_checkpoint=args.clear_checkpoint,
                output_path=args.output,
                repo_path=args.repo_path,
                base=args.base,
                include_uncommitted=not args.no_uncommitted,
            )
            print('--- Review result ---')
            print(out.get('summary', out))
            comments = out.get('comments', [])
            print(f'Total issues: {len(comments)}')
            print(f'Results written to: {args.output}')
        except Exception as e:
            print(f'Review failed: {e}, backtrace: {traceback.format_exc()}')
            return 1
    else:
        # --- cloud PR mode ---
        if not args.pr_number:
            print('Error: --pr_number is required for cloud mode (or use --local for local git review)')
            return 1
        repo = args.repo
        pr_number = args.pr_number

        print(f'1. Creating Git backend for {repo}...')
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

        arch_cache = os.environ.get('LAZYLLM_ARCH_CACHE') or None
        spec_cache = os.environ.get('LAZYLLM_SPEC_CACHE') or None

        print(f'3. Running multi-round review (model={args.model}, language={args.language})...')
        if args.post:
            print('   Will post line-level comments to PR Files changed.')
        try:
            resume_from = ReviewStage(args.resume_from) if args.resume_from else None
            out = review(
                pr_number,
                repo=repo,
                backend='github',
                llm=llm,
                post_to_github=args.post,
                arch_cache_path=arch_cache,
                review_spec_cache_path=spec_cache,
                fetch_repo_code=True,
                language=args.language,
                keep_clone=args.keep_clone,
                clear_checkpoint=args.clear_checkpoint,
                resume_from=resume_from,
                refresh_diff=args.refresh_diff,
                **({'max_history_prs': args.max_history_prs} if args.max_history_prs is not None else {}),
            )
            print('--- Review result ---')
            print(out.get('summary', out))
            comments = out.get('comments', [])
            print(f'   Total issues: {len(comments)}, posted: {out.get("comments_posted", 0)}')
            for i, c in enumerate(comments[:10]):
                cat = c.get('bug_category', '')
                sev = c.get('severity', '')
                prob = (c.get('problem') or '')[:60]
                print(f'   [{i + 1}] {c.get("path")} L{c.get("line")} [{sev}][{cat}] {prob}...')
            if len(comments) > 10:
                print(f'   ... and {len(comments) - 10} more')
        except Exception as e:
            print(f'   Review failed: {e}, backtrace: {traceback.format_exc()}')
            return 1

    print('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
