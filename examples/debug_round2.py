#!/usr/bin/env python3
# Debug script for Round 2 — runs only the FIRST changed file to fail fast.
# Usage:
#   python examples/debug_round2.py
#
# Environment variables:
#   LAZYLLM_CKPT_PATH   path to checkpoint JSON (default: auto-detect from PR)
#   LAZYLLM_CLONE_DIR   pre-cloned repo dir (skip clone step)
#   R2_FILE_IDX         which file index to test (default: 0 = first file)

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_gh_bin = os.path.expanduser('~/gh/bin')
if os.path.isdir(_gh_bin):
    os.environ['PATH'] = _gh_bin + os.pathsep + os.environ.get('PATH', '')

REPO = 'LazyAGI/LazyLLM'
PR_NUMBER = 1083
FILE_IDX = int(os.environ.get('R2_FILE_IDX', '0'))


def main():  # noqa: C901
    import lazyllm
    from lazyllm.tools.git import Git
    from lazyllm.tools.git.review.checkpoint import _ReviewCheckpoint
    from lazyllm.tools.git.review.pre_analysis import (
        _resolve_clone_target, _fetch_repo_code,
        _build_scoped_agent_tools_with_cache,
    )
    from lazyllm.tools.git.review.rounds import (
        _r2_build_shared_context, _r2_build_file_context, _r2_extract_issues,
        _R2_FILE_AGENT_RETRIES, _R2_FILE_TIMEOUT_SECS,
        _R2_DIFF_CHUNK, _R2_R1_BUDGET,
        _split_file_diff_into_chunks,
    )
    from lazyllm.tools.git.review.utils import (
        _ensure_non_streaming_llm, _parse_unified_diff,
    )

    llm = lazyllm.OnlineChatModule(source='claude', model='claude-opus-4-6', base_url='https://cld.ppapi.vip/v1/')
    llm = _ensure_non_streaming_llm(llm)

    # ── 1. checkpoint ────────────────────────────────────────────────────────
    ckpt_path = os.environ.get('LAZYLLM_CKPT_PATH') or _ReviewCheckpoint.default_path(PR_NUMBER, REPO)
    print(f'[ckpt] {ckpt_path}')
    ckpt = _ReviewCheckpoint(ckpt_path)

    r1 = ckpt.get('r1')
    if r1 is None:
        print('[FAIL] No r1 in checkpoint. Run full review first.')
        return 1
    print(f'[r1] {len(r1)} issues')

    arch_doc = ckpt.get('arch_doc') or ''
    pr_summary = ckpt.get('pr_summary') or ''

    # ── 2. diff ──────────────────────────────────────────────────────────────
    diff_cache = '/tmp/r2debug_diff.txt'
    if os.path.isfile(diff_cache):
        with open(diff_cache, 'r', encoding='utf-8') as f:
            diff_text = f.read()
        print(f'[diff] loaded from cache: {len(diff_text)} chars')
        backend = None
    else:
        print(f'[diff] fetching PR #{PR_NUMBER}...')
        backend = Git(backend='github', repo=REPO)
        diff_res = backend.get_pr_diff(PR_NUMBER)
        if not diff_res.get('success'):
            print(f'[FAIL] get_pr_diff: {diff_res.get("message")}')
            return 1
        diff_text = diff_res.get('diff', '')
        with open(diff_cache, 'w', encoding='utf-8') as f:
            f.write(diff_text)
        print(f'[diff] fetched and cached: {len(diff_text)} chars')

    file_diffs = {}
    for path, _s, _c, content in _parse_unified_diff(diff_text):
        file_diffs[path] = file_diffs.get(path, '') + content + '\n'
    print(f'[diff] {len(diff_text)} chars, {len(file_diffs)} files')

    # ── 3. clone ─────────────────────────────────────────────────────────────
    clone_dir = os.environ.get('LAZYLLM_CLONE_DIR') or ''
    if clone_dir and os.path.isdir(clone_dir):
        print(f'[clone] reusing {clone_dir}')
    else:
        clone_cache_file = '/tmp/r2debug_clone_dir.txt'
        if os.path.isfile(clone_cache_file):
            with open(clone_cache_file) as f:
                cached = f.read().strip()
            if os.path.isdir(cached):
                clone_dir = cached
                print(f'[clone] reusing cached: {clone_dir}')
        if not clone_dir:
            if backend is None:
                backend = Git(backend='github', repo=REPO)
            pr_res = backend.get_pull_request(PR_NUMBER)
            if not pr_res.get('success'):
                print(f'[FAIL] get_pull_request: {pr_res.get("message")}')
                return 1
            clone_url, branch = _resolve_clone_target(pr_res['pr'], REPO)
            print(f'[clone] {clone_url} @ {branch}')
            try:
                clone_dir, _ = _fetch_repo_code(clone_url, branch)
                with open(clone_cache_file, 'w') as f:
                    f.write(clone_dir)
                print(f'[clone] done: {clone_dir}')
            except Exception as e:
                print(f'[FAIL] clone: {e}')
                traceback.print_exc()
                return 1

    # ── 4. shared context ────────────────────────────────────────────────────
    shared_context = ckpt.get('r2_shared_context') or ''
    if shared_context:
        print(f'[shared_ctx] loaded from ckpt: {len(shared_context)} chars')
    else:
        print('[shared_ctx] building...')
        try:
            shared_context = _r2_build_shared_context(llm, diff_text, arch_doc, clone_dir)
            print(f'[shared_ctx] built: {len(shared_context)} chars')
        except Exception as e:
            print(f'[FAIL] shared_ctx: {e}')
            traceback.print_exc()
            return 1

    # ── 5. pick target file ──────────────────────────────────────────────────
    files = list(file_diffs.keys())
    if FILE_IDX >= len(files):
        print(f'[FAIL] R2_FILE_IDX={FILE_IDX} out of range (0..{len(files)-1})')
        return 1
    target_path = files[FILE_IDX]
    fdiff = file_diffs[target_path]
    print(f'\n[target] file [{FILE_IDX}/{len(files)-1}]: {target_path}')

    r1_by_file = {}
    for c in r1:
        p = c.get('path') or ''
        r1_by_file.setdefault(p, []).append(
            f'line {c.get("line")}: [{c.get("severity")}] {c.get("problem", "")[:100]}'
        )
    r1_lines = r1_by_file.get(target_path, [])
    r1_text = '\n'.join(r1_lines) if r1_lines else '(none)'
    if len(r1_text) > _R2_R1_BUDGET:
        r1_text = r1_text[:_R2_R1_BUDGET] + '\n...(truncated)'

    # ── 6. build symbol cache + tools ────────────────────────────────────────
    symbol_cache = {}
    tools = _build_scoped_agent_tools_with_cache(clone_dir, llm, symbol_cache)
    print(f'[agent] max_retries={_R2_FILE_AGENT_RETRIES}, timeout={_R2_FILE_TIMEOUT_SECS}s')

    # ── 7. per-chunk: stage1 (context) + stage2 (issues) ────────────────────
    all_issues = []
    chunks = list(_split_file_diff_into_chunks(fdiff, _R2_DIFF_CHUNK))
    print(f'[chunks] {len(chunks)} chunk(s) for {target_path}')

    for chunk_idx, (hunk_range, diff_chunk) in enumerate(chunks):
        print(f'\n[chunk {chunk_idx+1}/{len(chunks)}] {hunk_range}')

        # Stage 1: agent explores context
        print('[stage1] collecting symbol context...')
        try:
            context_str = _r2_build_file_context(
                llm, target_path, diff_chunk, clone_dir, tools, language='cn',
            )
            print(f'[stage1] done: {len(context_str)} chars')
            print('--- context preview (first 600 chars) ---')
            print(context_str[:600])
            print('--- end ---\n')
        except Exception as e:
            print(f'[FAIL] stage1 context collection: {e}')
            traceback.print_exc()
            return 1

        # Stage 2: LLM extracts issues
        print('[stage2] extracting issues...')
        try:
            items = _r2_extract_issues(
                llm, target_path, diff_chunk, hunk_range,
                context_str, shared_context, r1_text,
                arch_doc, pr_summary, language='cn',
            )
            print(f'[stage2] {len(items)} issues')
            for i, c in enumerate(items):
                print(f'  [{i+1}] L{c.get("line")} [{c.get("severity")}] {c.get("problem","")[:80]}')
            all_issues.extend(items)
        except Exception as e:
            print(f'[FAIL] stage2 issue extraction: {e}')
            traceback.print_exc()
            return 1

    print(f'\n[SUCCESS] total {len(all_issues)} issues across {len(chunks)} chunk(s)')
    print(f'[symbol_cache] {len(symbol_cache)} symbols analyzed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
