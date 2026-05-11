# Copyright (c) 2026 LazyAGI. All rights reserved.
# RMod: modification necessity analysis (ReactAgent, per-file parallel).
# Entry point: _rmod_run

import os
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from typing import Any, Dict, List, Optional

import lazyllm

from ..utils import (
    _Progress, _language_instruction,
    _annotate_full_diff, _normalize_comment_item, _parse_unified_diff,
    _extract_json_text, _parse_json_with_repair,
)
from ..pre_analysis import _build_scoped_agent_tools_with_cache
from ..constants import clip_text
from lazyllm.tools.agent import ReactAgent
from .common import _safe_format, _make_traced_tool, _rmod_new_file_paths
from .prompt import _RMOD_PROMPT_TMPL

_RMOD_AGENT_TIMEOUT_SECS = 180
_RMOD_AGENT_RETRIES = 6
_RMOD_MAX_PARALLEL_FILES = 4


def _rmod_file_diff(diff_text: str, file_path: str) -> str:
    lines = diff_text.splitlines(keepends=True)
    result: List[str] = []
    in_file = False
    for line in lines:
        if line.startswith('diff --git '):
            if in_file:
                break
            continue
        if line.startswith('--- a/') or line.startswith('--- '):
            candidate = line[6:].strip() if line.startswith('--- a/') else line[4:].strip()
            in_file = (candidate == file_path
                       or os.path.realpath(candidate) == os.path.realpath(file_path))
        if in_file:
            result.append(line)
    return ''.join(result) if result else diff_text


def _rmod_run_single_file(
    llm: Any,
    file_path: str,
    file_diff: str,
    arch_doc: str,
    pr_design_doc: str,
    pr_summary: str,
    clone_dir: str,
    language: str,
    agent_instructions: str,
    tools: List[Any],
) -> List[Dict[str, Any]]:
    annotated = _annotate_full_diff(file_diff)
    arch_use = clip_text(arch_doc or '', 12000) if arch_doc else '(not available)'
    prompt = _safe_format(
        _RMOD_PROMPT_TMPL,
        lang_instruction=_language_instruction(language),
        pr_design_doc=clip_text(pr_design_doc, 8000) if pr_design_doc else '(not available)',
        arch_doc=arch_use,
        agent_instructions=agent_instructions or '(not available)',
        file_path=file_path,
        diff_text=annotated,
    )
    step_counter = [0]
    exploration_log: List[str] = []
    traced_tools = [_make_traced_tool(t, step_counter, file_path, exploration_log, round_name='RMod') for t in tools]
    try:
        agent = ReactAgent(
            llm, tools=traced_tools, max_retries=_RMOD_AGENT_RETRIES,
            workspace=clone_dir, force_summarize=True,
            force_summarize_context=(
                f'Analyzing modification necessity for {file_path}:\n{pr_summary[:400]}\n\n'
                f'Key framework conventions:\n{agent_instructions[:300]}'
            ),
            keep_full_turns=2,
        )
    except Exception as e:
        lazyllm.LOG.warning(f'  [RMod] ReactAgent init failed for {file_path}: {e}')
        return []
    lazyllm.LOG.info(f'  [RMod] Analyzing {file_path} ...')
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(agent, prompt)
        try:
            raw = fut.result(timeout=_RMOD_AGENT_TIMEOUT_SECS)
        except FuturesTimeoutError:
            lazyllm.LOG.warning(f'  [RMod] Timed out for {file_path} after {_RMOD_AGENT_TIMEOUT_SECS}s')
            return []
        except Exception as e:
            lazyllm.LOG.warning(f'  [RMod] Failed for {file_path}: {e}')
            return []
    raw_str = raw if isinstance(raw, str) else str(raw)
    json_text = _extract_json_text(raw_str)
    items = _parse_json_with_repair(json_text) if json_text else []
    issues = [n for item in (items if isinstance(items, list) else [])
              if (n := _normalize_comment_item(item, default_path=file_path,
                                               default_category='design',
                                               demote_on_out_of_range=True)) is not None]
    lazyllm.LOG.info(f'  [RMod] Done {file_path}: {len(issues)} issue(s)')
    return issues


def _rmod_collect_file_diffs(diff_text: str) -> Dict[str, str]:
    new_paths = _rmod_new_file_paths(diff_text)
    hunks = _parse_unified_diff(diff_text)
    file_diffs: Dict[str, str] = {}
    for path, start, count, content in hunks:
        if path not in new_paths:
            if path not in file_diffs:
                file_diffs[path] = f'--- a/{path}\n+++ b/{path}\n'
            file_diffs[path] += f'@@ -{start},{count} +{start},{count} @@\n{content}\n'
    return file_diffs


def _rmod_run(
    llm: Any,
    diff_text: str,
    arch_doc: str,
    pr_design_doc: str,
    pr_summary: str = '',
    clone_dir: Optional[str] = None,
    language: str = 'cn',
    agent_instructions: str = '',
    owner_repo: str = '',
    arch_cache_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    prog = _Progress('RMod: modification necessity analysis')
    if not clone_dir or not os.path.isdir(clone_dir):
        prog.done('skipped (no clone_dir)')
        return []

    file_diffs = _rmod_collect_file_diffs(diff_text)
    if not file_diffs:
        prog.done('no pre-existing modified files found')
        return []

    lazyllm.LOG.info(f'  [RMod] Analyzing {len(file_diffs)} modified file(s) in parallel')

    # symbol_cache is shared across all per-file workers so that symbols referenced by
    # multiple files (e.g. a shared base class) are only fetched and summarised once.
    # Individual dict read/write operations are GIL-protected, and the TOCTOU race is
    # benign here because the same key always maps to the same deterministic value.
    symbol_cache: Dict[str, Any] = {}
    shared_tools = _build_scoped_agent_tools_with_cache(clone_dir, llm, symbol_cache, owner_repo, arch_cache_path)
    for tool in shared_tools:
        if hasattr(tool, 'execute_in_sandbox'):
            tool.execute_in_sandbox = False

    all_results: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def _run_file(file_path: str, file_diff: str) -> None:
        issues = _rmod_run_single_file(
            llm, file_path, file_diff, arch_doc, pr_design_doc, pr_summary,
            clone_dir, language, agent_instructions, shared_tools,
        )
        with lock:
            all_results.extend(issues)

    with ThreadPoolExecutor(max_workers=min(_RMOD_MAX_PARALLEL_FILES, len(file_diffs))) as ex:
        futs = {ex.submit(_run_file, fp, fd): fp for fp, fd in file_diffs.items()}
        for fut in as_completed(futs):
            fp = futs[fut]
            try:
                fut.result()
            except Exception as e:
                lazyllm.LOG.warning(f'  [RMod] Unexpected error for {fp}: {e}')

    prog.done(f'{len(all_results)} modification necessity issues found across {len(file_diffs)} file(s)')
    return all_results
