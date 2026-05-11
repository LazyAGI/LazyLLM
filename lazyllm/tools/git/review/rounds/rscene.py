# Copyright (c) 2026 LazyAGI. All rights reserved.
# RScene: usage scenario inference for modified public APIs.
# Entry point: infer_usage_scenarios

import re
import concurrent.futures as _cf
from typing import Any, Dict, List, Optional

import lazyllm

from ..utils import (
    _Progress, _language_instruction,
    _extract_json_text, _parse_json_with_repair,
)
from ..pre_analysis import _extract_arch_for_file, _build_scoped_agent_tools_with_cache
from ..constants import compress_diff_for_agent_heuristic
from ..checkpoint import ReviewStage
from lazyllm.tools.agent import ReactAgent
from .common import (
    _safe_format, _make_traced_tool,
    _rmod_new_file_paths, _collect_all_file_diffs, _build_review_units,
)
from .prompt import _RSCENE_PROMPT_TMPL

_RSCENE_AGENT_TIMEOUT_SECS = 180
_RSCENE_AGENT_RETRIES = 10
_RSCENE_DIFF_BUDGET = 6000


def _rscene_run_single_group(
    llm: Any,
    anchor: Optional[str],
    files: List[str],
    diff_text: str,
    arch_doc: str,
    pr_summary: str,
    clone_dir: str,
    language: str,
    symbol_cache: Dict[str, Any],
    tools: Any,
    ckpt: Optional[Any],
    use_cache: bool,
) -> List[Dict[str, Any]]:
    safe = re.sub(r'[^a-zA-Z0-9_]', '_', anchor or '_'.join(sorted(files))[:60])
    ckpt_key = f'rscene_group_{safe}'
    if ckpt and use_cache:
        cached = ckpt.get(ckpt_key)
        if cached is not None:
            return cached

    compressed_diff = compress_diff_for_agent_heuristic(diff_text, _RSCENE_DIFF_BUDGET)
    arch_snippet = _extract_arch_for_file(arch_doc, anchor or files[0], max_chars=4000) if arch_doc else ''

    prompt = _safe_format(
        _RSCENE_PROMPT_TMPL,
        lang_instruction=_language_instruction(language),
        pr_summary=pr_summary[:600] if pr_summary else '(not available)',
        compressed_diff=compressed_diff,
        arch_doc=arch_snippet or '(not available)',
    )

    step_counter = [0]
    exploration_log: List[str] = []
    traced_tools = [_make_traced_tool(t, step_counter, anchor or (files[0] if files else 'rscene'), exploration_log,
                                      round_name='RScene')
                    for t in tools]

    try:
        agent = ReactAgent(
            llm, tools=traced_tools,
            max_retries=_RSCENE_AGENT_RETRIES,
            workspace=clone_dir,
            force_summarize=True,
            force_summarize_context=(
                'Output a JSON array wrapped with <<<JSON_START>>> and <<<JSON_END>>> markers. '
                'Each element must have: title, description, api_sequence, state_changes, '
                'entry_point, call_chain, edge_cases.'
            ),
            keep_full_turns=2,
        )
        with _cf.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(agent, prompt)
            try:
                raw = fut.result(timeout=_RSCENE_AGENT_TIMEOUT_SECS)
            except _cf.TimeoutError:
                lazyllm.LOG.warning(f'  [RScene] Timed out for {anchor or files} after {_RSCENE_AGENT_TIMEOUT_SECS}s')
                return []
    except Exception as e:
        lazyllm.LOG.warning(f'  [RScene] Agent failed for {anchor or files}: {e}')
        return []

    json_text = _extract_json_text(raw or '')
    scenarios = _parse_json_with_repair(json_text) if json_text else None
    if not isinstance(scenarios, list):
        lazyllm.LOG.warning(f'  [RScene] Could not parse scenarios JSON for {anchor or files}')
        return []

    result = [s for s in scenarios if isinstance(s, dict) and s.get('title')]
    lazyllm.LOG.info(f'  [RScene] {len(result)} scenarios inferred for {anchor or files}')
    if ckpt and result:
        ckpt.save(ckpt_key, result)
    return result


def _rscene_collect_modified_file_diffs(diff_text: str) -> Dict[str, str]:
    '''Collect per-file diffs for modified (non-new) files only, preserving original @@ headers.'''
    new_paths = _rmod_new_file_paths(diff_text)
    file_diffs: Dict[str, str] = {}
    current_path: Optional[str] = None
    current_lines: List[str] = []

    def _flush():
        if current_path and current_lines:
            file_diffs.setdefault(current_path, '')
            file_diffs[current_path] += ''.join(current_lines)
        current_lines.clear()

    for line in diff_text.splitlines(keepends=True):
        if line.startswith('diff --git '):
            _flush()
            m = re.match(r'diff --git a/(.+) b/(.+)$', line.rstrip())
            current_path = m.group(2) if m else None
            current_lines.clear()
        elif current_path is not None and current_path not in new_paths:
            current_lines.append(line)
    _flush()
    cleaned: Dict[str, str] = {}
    for path, raw in file_diffs.items():
        hunk_lines = [line for line in raw.splitlines(keepends=True)
                      if not line.startswith(('index ', '--- ', '+++ '))]
        cleaned[path] = ''.join(hunk_lines)
    return cleaned


def _rscene_dedup_scenarios(scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    '''Deduplicate scenarios by exact title match (case-insensitive).'''
    seen_titles: set = set()
    deduped: List[Dict[str, Any]] = []
    for s in scenarios:
        t = s.get('title', '').strip().lower()
        if t and t not in seen_titles:
            seen_titles.add(t)
            deduped.append(s)
    return deduped


def infer_usage_scenarios(
    llm: Any,
    diff_text: str,
    arch_doc: str,
    pr_summary: str,
    clone_dir: str,
    ckpt: Optional[Any] = None,
    language: str = 'cn',
    strategy: Optional[Any] = None,
    owner_repo: str = '',
    arch_cache_path: Optional[str] = None,
    symbol_cache: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    '''RScene: infer typical usage scenarios for modified public APIs.
    Returns a list of scenario dicts (title, description, api_sequence, call_chain, edge_cases, ...).
    '''
    use_cache = ckpt.should_use_cache(ReviewStage.RScene) if ckpt else True
    if ckpt and use_cache:
        cached_all = ckpt.get('rscene_all')
        if cached_all is not None:
            lazyllm.LOG.info(f'[RScene] Using cached scenarios ({len(cached_all)} total)')
            return cached_all

    file_diffs = _collect_all_file_diffs(diff_text)

    if not file_diffs:
        lazyllm.LOG.info('[RScene] No changed files found, skipping')
        return []

    large_threshold = strategy.large_file_threshold if strategy else 200
    max_files = strategy.max_files_for_r3 if strategy else 20
    units, skipped = _build_review_units(file_diffs, large_threshold, max_files)
    if skipped:
        lazyllm.LOG.info(f'[RScene] Skipped {len(skipped)} files (over budget)')

    prog = _Progress('RScene: inferring usage scenarios', len(units))
    if symbol_cache is None:
        symbol_cache = {}
    tools = _build_scoped_agent_tools_with_cache(clone_dir, llm, symbol_cache, owner_repo, arch_cache_path)

    all_scenarios: List[Dict[str, Any]] = []
    _RSCENE_MAX_PARALLEL = 3
    with _cf.ThreadPoolExecutor(max_workers=min(_RSCENE_MAX_PARALLEL, len(units))) as ex:
        futs = {
            ex.submit(
                _rscene_run_single_group,
                llm, unit['anchor'], unit['files'], unit['diff'],
                arch_doc, pr_summary, clone_dir, language,
                symbol_cache, tools, ckpt, use_cache,
            ): unit
            for unit in units
        }
        for fut in _cf.as_completed(futs):
            unit = futs[fut]
            try:
                scenarios = fut.result()
                all_scenarios.extend(scenarios)
            except Exception as e:
                lazyllm.LOG.warning(f'  [RScene] Group {unit["anchor"] or unit["files"]} failed: {e}')
            prog.update(str(unit['anchor'] or unit['files']))

    deduped = _rscene_dedup_scenarios(all_scenarios)

    prog.done(f'{len(deduped)} unique scenarios inferred')
    if ckpt and deduped:
        ckpt.save('rscene_all', deduped)
        ckpt.mark_stage_done(ReviewStage.RScene)
    return deduped
