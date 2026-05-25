# Copyright (c) 2026 LazyAGI. All rights reserved.
# RChain: scenario-driven call chain review (bug + usability issues).
# Entry point: _rscenario_call_chain

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from typing import Any, Dict, List, Optional, Tuple

import lazyllm

from ..utils import (
    _Progress, _language_instruction,
    _annotate_full_diff, _normalize_comment_item,
    _extract_json_text, _parse_json_with_repair,
)
from ..pre_analysis import _build_scoped_agent_tools_with_cache
from ..constants import SINGLE_CALL_CONTEXT_BUDGET
from ..checkpoint import ReviewStage
from lazyllm.tools.agent import ReactAgent
from .common import _safe_format, _make_traced_tool, _collect_all_file_diffs, _split_file_diff_into_chunks
from .prompt import _RCHAIN_PROMPT_TMPL

_RCHAIN_AGENT_TIMEOUT_SECS = 240
_RCHAIN_AGENT_RETRIES = 12
_RCHAIN_MAX_PARALLEL_SCENARIOS = 3
_RCHAIN_FIXED_OVERHEAD = 20000


def _rchain_parse_issues(raw: Optional[str], scenario_title: str = '') -> List[Dict[str, Any]]:
    '''Parse and normalize RChain agent output into issue dicts.'''
    json_text = _extract_json_text(raw or '')
    items = _parse_json_with_repair(json_text) if json_text else None
    result: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return result
    for item in items:
        if not isinstance(item, dict):
            continue
        item['source'] = 'rchain'
        if scenario_title and not item.get('scenario'):
            item['scenario'] = scenario_title
        normalized = _normalize_comment_item(item, default_category='design')
        if normalized is None:
            normalized = _normalize_comment_item(
                item, default_category='design', allow_null_line=True,
            )
        if normalized:
            normalized['source'] = 'rchain'
            if scenario_title:
                normalized['scenario'] = item.get('scenario') or scenario_title
            result.append(normalized)
    return result


def _rchain_dedup_issues(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    '''Deduplicate issues: same path+line keeps highest severity.'''
    seen: Dict[Tuple[str, int], Dict[str, Any]] = {}
    _sev = {'critical': 0, 'medium': 1, 'normal': 2}
    for issue in issues:
        key = (issue.get('path', ''), issue.get('line', 0))
        if key not in seen or _sev.get(issue.get('severity', 'normal'), 2) < _sev.get(
                seen[key].get('severity', 'normal'), 2):
            seen[key] = issue
    return list(seen.values())


def _rchain_run_single_scenario(  # noqa: C901
    llm: Any,
    scenario: Dict[str, Any],
    scene_idx: int,
    file_diffs: Dict[str, str],
    arch_doc: str,
    clone_dir: str,
    language: str,
    symbol_cache: Dict[str, Any],
    tools: Any,
    ckpt: Optional[Any],
    use_cache: bool,
) -> List[Dict[str, Any]]:
    title = scenario.get('title', f'scenario_{scene_idx}')
    safe_title = re.sub(r'[^a-zA-Z0-9_]', '_', title)[:50]
    ckpt_key = f'rchain_scene_{scene_idx}_{safe_title}'
    if ckpt and use_cache:
        cached = ckpt.get(ckpt_key)
        if cached is not None:
            if isinstance(cached, list):
                return cached
            if isinstance(cached, dict):
                return cached.get('issues') or []

    call_chain = scenario.get('call_chain', [])

    # Use relevant_diff_files from the scenario if available; fall back to all diff files.
    # This scopes the diff to only the files involved in this scenario's call chain,
    # avoiding context overflow from unrelated files.
    scenario_relevant = scenario.get('relevant_diff_files') or []
    if scenario_relevant:
        relevant_files = [f for f in scenario_relevant if f in file_diffs]
        if not relevant_files:
            # Fallback: scenario listed files not in diff — use all diff files
            relevant_files = list(file_diffs.keys())
    else:
        relevant_files = list(file_diffs.keys())

    combined_diff = '\n'.join(file_diffs[f] for f in relevant_files if f in file_diffs)
    arch_snippet = arch_doc[:4000] if arch_doc else '(not available)'

    diff_cap = SINGLE_CALL_CONTEXT_BUDGET - _RCHAIN_FIXED_OVERHEAD - len(arch_snippet)
    all_chunks = _split_file_diff_into_chunks(combined_diff, diff_cap)
    # No hard cap on chunks — process all of them to avoid missing issues

    all_issues: List[Dict[str, Any]] = []
    _timeout_occurred = False
    _agent_error: Optional[str] = None

    for chunk_label, diff_chunk in all_chunks:
        annotated = _annotate_full_diff(diff_chunk)
        prompt = _safe_format(
            _RCHAIN_PROMPT_TMPL,
            lang_instruction=_language_instruction(language),
            scenario_title=title,
            scenario_description=scenario.get('description', ''),
            api_sequence=json.dumps(scenario.get('api_sequence', []), ensure_ascii=False),
            call_chain=json.dumps(call_chain, ensure_ascii=False),
            edge_cases=json.dumps(scenario.get('edge_cases', []), ensure_ascii=False),
            arch_doc=arch_snippet,
            diff_text=annotated[:diff_cap],
        )

        step_counter = [0]
        exploration_log: List[str] = []
        traced_tools = [_make_traced_tool(t, step_counter, f'rchain_{safe_title}', exploration_log,
                                          round_name='RChain')
                        for t in tools]

        try:
            agent = ReactAgent(
                llm, tools=traced_tools,
                max_retries=_RCHAIN_AGENT_RETRIES,
                workspace=clone_dir,
                force_summarize=True,
                keep_full_turns=2,
            )
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(agent, prompt)
                try:
                    raw = fut.result(timeout=_RCHAIN_AGENT_TIMEOUT_SECS)
                except FuturesTimeoutError:
                    _timeout_occurred = True
                    _agent_error = (
                        f'Timed out after {_RCHAIN_AGENT_TIMEOUT_SECS}s '
                        f'for scenario "{title}" chunk {chunk_label}'
                    )
                    lazyllm.LOG.warning(f'  [RChain] {_agent_error}')
                    continue
        except Exception as e:
            _agent_error = f'Agent failed for scenario "{title}" chunk {chunk_label}: {e}'
            lazyllm.LOG.warning(f'  [RChain] {_agent_error}')
            continue

        all_issues.extend(_rchain_parse_issues(raw, scenario_title=title))

    result = _rchain_dedup_issues(all_issues)

    lazyllm.LOG.info(f'  [RChain] Scenario "{title}": {len(result)} issues found')
    if ckpt:
        status = 'ok' if result or not _agent_error else ('timeout' if _timeout_occurred else 'error')
        ckpt.save(ckpt_key, {'status': status, 'issues': result, 'error': _agent_error or ''})
    return result


def _rchain_collect_scenario_status(ckpt: Optional[Any], ckpt_key: str) -> str:
    '''Read the saved status for a scenario from checkpoint.'''
    if not ckpt:
        return 'ok'
    saved = ckpt.get(ckpt_key)
    if isinstance(saved, dict):
        return saved.get('status', 'ok')
    return 'ok'


def _rchain_run_and_track(
    llm: Any, scenario: Dict[str, Any], idx: int,
    file_diffs: Dict[str, str], arch_doc: str, clone_dir: str,
    language: str, symbol_cache: Dict[str, Any], tools: Any,
    ckpt: Optional[Any], use_cache: bool,
    counters: Dict[str, int], lock: threading.Lock,
) -> List[Dict[str, Any]]:
    '''Run a single scenario and update ok/timeout/error counters.'''
    title = scenario.get('title', f'scenario_{idx}')
    safe_title = re.sub(r'[^a-zA-Z0-9_]', '_', title)[:50]
    ckpt_key = f'rchain_scene_{idx}_{safe_title}'
    result = _rchain_run_single_scenario(
        llm, scenario, idx, file_diffs, arch_doc, clone_dir, language,
        symbol_cache, tools, ckpt, use_cache,
    )
    status = _rchain_collect_scenario_status(ckpt, ckpt_key)
    with lock:
        if status == 'timeout':
            counters['timeout'] += 1
        elif status == 'error':
            counters['error'] += 1
        else:
            counters['ok'] += 1
    return result


def _rscenario_call_chain(
    llm: Any,
    usage_scenarios: List[Dict[str, Any]],
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
    '''RChain: scenario-driven call chain review producing bug + usability issues.'''
    if not usage_scenarios:
        lazyllm.LOG.info('[RChain] No scenarios to review, skipping')
        return []

    use_cache = ckpt.should_use_cache(ReviewStage.RChain) if ckpt else True
    if ckpt and use_cache:
        cached_all = ckpt.get('rchain_all')
        if cached_all is not None:
            lazyllm.LOG.info(f'[RChain] Using cached issues ({len(cached_all)} total)')
            return cached_all

    file_diffs = _collect_all_file_diffs(diff_text)

    if not file_diffs:
        lazyllm.LOG.info('[RChain] No changed files found, skipping')
        return []

    prog = _Progress('RChain: call chain review', len(usage_scenarios))
    if symbol_cache is None:
        symbol_cache = {}
    tools = _build_scoped_agent_tools_with_cache(clone_dir, llm, symbol_cache, owner_repo, arch_cache_path)

    all_issues: List[Dict[str, Any]] = []
    counters: Dict[str, int] = {'ok': 0, 'timeout': 0, 'error': 0}
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=min(_RCHAIN_MAX_PARALLEL_SCENARIOS, len(usage_scenarios))) as ex:
        futs = {
            ex.submit(
                _rchain_run_and_track,
                llm, scenario, idx, file_diffs, arch_doc, clone_dir,
                language, symbol_cache, tools, ckpt, use_cache, counters, lock,
            ): scenario
            for idx, scenario in enumerate(usage_scenarios)
        }
        for fut in as_completed(futs):
            scenario = futs[fut]
            try:
                all_issues.extend(fut.result())
            except Exception as e:
                lazyllm.LOG.warning(f'  [RChain] Scenario "{scenario.get("title")}" failed: {e}')
                with lock:
                    counters['error'] += 1
            prog.update(scenario.get('title', ''))

    prog.done(
        f'{len(all_issues)} issues found across {len(usage_scenarios)} scenarios '
        f'(ok={counters["ok"]}, timeout={counters["timeout"]}, error={counters["error"]})'
    )
    if counters['timeout'] or counters['error']:
        lazyllm.LOG.warning(
            f'[RChain] Completed with issues: '
            f'{counters["timeout"]} timeout(s), {counters["error"]} error(s) out of {len(usage_scenarios)} scenarios.'
        )
    if ckpt:
        ckpt.save('rchain_all', all_issues)
        ckpt.save('rchain_metrics', counters)
        ckpt.mark_stage_done(ReviewStage.RChain)
    return all_issues
