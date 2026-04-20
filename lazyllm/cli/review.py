# Copyright (c) 2026 LazyAGI. All rights reserved.
import argparse
import sys
import traceback


def _build_review_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='lazyllm review',
        description='Multi-round AI code review for cloud-hosted PRs (GitHub, GitLab, Gitee, GitCode).',
    )
    p.add_argument('--pr', type=int, required=True, metavar='NUMBER',
                   help='PR number to review (required)')
    p.add_argument('--repo', default='LazyAGI/LazyLLM', metavar='OWNER/NAME',
                   help='Repository in owner/name format (default: LazyAGI/LazyLLM)')
    p.add_argument('--backend', default=None,
                   choices=['github', 'gitlab', 'gitee', 'gitcode'],
                   help='Git backend (default: auto-detect from repo or env)')
    p.add_argument('--source', default='claude',
                   help='LLM source, e.g. claude, openai, qwen (default: claude)')
    p.add_argument('--model', default='claude-opus-4-6',
                   help='LLM model name (default: claude-opus-4-6)')
    p.add_argument('--base-url', default=None, metavar='URL',
                   help='LLM API base URL, e.g. a proxy URL (default: official endpoint)')
    p.add_argument('--language', default='cn', choices=['cn', 'en'],
                   help='Review output language (default: cn)')
    p.add_argument('--post', action='store_true',
                   help='Post review comments back to the PR (default: off)')
    p.add_argument('--keep-clone', action='store_true',
                   help='Keep cloned repo directory after review')
    p.add_argument('--clear-checkpoint', action='store_true',
                   help='Clear checkpoint and start fresh')
    p.add_argument('--refresh-diff', action='store_true',
                   help='Fetch latest PR head SHA and diff; rotate checkpoint if changed')
    p.add_argument('--resume-from', default=None,
                   choices=['clone', 'arch', 'spec', 'pr_summary', 'r1', 'r2', 'r3', 'r4a', 'r4', 'final', 'upload'],
                   metavar='STAGE',
                   help='Resume from a specific stage: clone|arch|spec|pr_summary|r1|r2|r3|final|upload')
    p.add_argument('--max-history-prs', type=int, default=None, metavar='N',
                   help='Max historical PRs to analyze for review spec (default: 20)')
    p.add_argument('--arch-cache', default=None, metavar='PATH',
                   help='Path to architecture doc cache JSON')
    p.add_argument('--spec-cache', default=None, metavar='PATH',
                   help='Path to review spec cache JSON')
    return p


def _build_review_local_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='lazyllm review-local',
        description='Multi-round AI code review for a local git repository. '
                    'Diffs current branch against base using git merge-base, '
                    'so only changes introduced by this branch are reviewed. '
                    'Results are written to a JSON file.',
    )
    p.add_argument('--repo-path', default='.', metavar='PATH',
                   help='Path to local git repository (default: current directory)')
    p.add_argument('--base', default='main', metavar='BRANCH',
                   help='Base branch to diff against (default: main)')
    p.add_argument('--no-uncommitted', action='store_true',
                   help='Exclude uncommitted (staged + working tree) changes from diff (default: include)')
    p.add_argument('--source', default='claude',
                   help='LLM source, e.g. claude, openai, qwen (default: claude)')
    p.add_argument('--model', default='claude-opus-4-6',
                   help='LLM model name (default: claude-opus-4-6)')
    p.add_argument('--base-url', default=None, metavar='URL',
                   help='LLM API base URL, e.g. a proxy URL (default: official endpoint)')
    p.add_argument('--language', default='cn', choices=['cn', 'en'],
                   help='Review output language (default: cn)')
    p.add_argument('--output', default='review.json', metavar='PATH',
                   help='Output JSON file path (default: review.json)')
    p.add_argument('--clear-checkpoint', action='store_true',
                   help='Clear checkpoint and start fresh')
    return p


def review(commands):
    parser = _build_review_parser()
    args = parser.parse_args(commands)

    import lazyllm
    from lazyllm.tools.git.review.checkpoint import ReviewStage

    llm_kwargs = dict(source=args.source, model=args.model)
    if args.base_url:
        llm_kwargs['base_url'] = args.base_url
    llm = lazyllm.OnlineChatModule(**llm_kwargs)

    resume_from = ReviewStage(args.resume_from) if args.resume_from else None

    print(f'Reviewing PR #{args.pr} on {args.repo} (model={args.model}, language={args.language})...')  # noqa print
    try:
        from lazyllm.tools.git import review as _review
        out = _review(
            pr_number=args.pr,
            repo=args.repo,
            backend=args.backend,
            llm=llm,
            post_to_github=args.post,
            arch_cache_path=args.arch_cache,
            review_spec_cache_path=args.spec_cache,
            fetch_repo_code=True,
            language=args.language,
            keep_clone=args.keep_clone,
            clear_checkpoint=args.clear_checkpoint,
            resume_from=resume_from,
            refresh_diff=args.refresh_diff,
            **({'max_history_prs': args.max_history_prs} if args.max_history_prs is not None else {}),
        )
        print('--- Review result ---')  # noqa print
        print(out.get('summary', ''))  # noqa print
        comments = out.get('comments', [])
        print(f'Total issues: {len(comments)}, posted: {out.get("comments_posted", 0)}')  # noqa print
    except Exception as e:
        print(f'Review failed: {e}\n{traceback.format_exc()}', file=sys.stderr)  # noqa print
        sys.exit(1)


def review_local(commands):
    parser = _build_review_local_parser()
    args = parser.parse_args(commands)

    import lazyllm

    llm_kwargs = dict(source=args.source, model=args.model)
    if args.base_url:
        llm_kwargs['base_url'] = args.base_url
    llm = lazyllm.OnlineChatModule(**llm_kwargs)

    print(f'Reviewing local repo at {args.repo_path!r} '  # noqa print
          f'(base={args.base}, uncommitted={"no" if args.no_uncommitted else "yes"}, '
          f'model={args.model}, language={args.language})...')
    try:
        from lazyllm.tools.git import review as _review
        out = _review(
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
        print('--- Review result ---')  # noqa print
        print(out.get('summary', ''))  # noqa print
        comments = out.get('comments', [])
        print(f'Total issues: {len(comments)}')  # noqa print
        print(f'Results written to: {args.output}')  # noqa print
    except Exception as e:
        print(f'Review failed: {e}\n{traceback.format_exc()}', file=sys.stderr)  # noqa print
        sys.exit(1)
