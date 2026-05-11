# Copyright (c) 2026 LazyAGI. All rights reserved.
# Round 3: unified agent verification pass (context-enriched, cross-file).
# Entry point: _round3_agent_verify

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from typing import Any, Dict, List, Optional, Tuple

import lazyllm

from ..utils import (
    _Progress, _language_instruction, _safe_llm_call, _safe_llm_call_text,
    _annotate_full_diff, _normalize_comment_item, _parse_unified_diff,
    _extract_json_text, _parse_json_with_repair,
)
from ..pre_analysis import (
    _extract_arch_for_file, _extract_file_skeleton,
    _build_scoped_agent_tools_with_cache, _lookup_relevant_rules,
)
from ..constants import (
    SINGLE_CALL_CONTEXT_BUDGET,
    R3_MAX_CHUNKS_HARD, compress_diff_for_agent_heuristic, issue_density_rule,
)
from ..checkpoint import ReviewStage
from lazyllm.tools.agent import ReactAgent
from .common import (
    _safe_format, _make_traced_tool, _inject_local_agent_instructions,
    _split_file_diff_into_chunks, _r3_build_shared_context, _build_review_units,
    _R3_SHARED_CTX_BUDGET,
)
from .prompt import (
    _ROUND3_GROUP_PROMPT_TMPL, _R3_CONTEXT_COLLECT_PROMPT_TMPL, _R3_ISSUE_EXTRACT_PROMPT_TMPL,
)

_R3_R1_BUDGET = 8000
_R3_ARCH_BUDGET = 6000
_R3_SUMMARY_BUDGET = 600
_R3_AGENT_DIFF_BUDGET = SINGLE_CALL_CONTEXT_BUDGET - 14000
_R3_FILE_AGENT_RETRIES = 8
_R3_FILE_TIMEOUT_SECS = 300

_R3_RICH_CONTEXT_BUDGET = 12000
_R3_SYMBOL_CONTEXT_MAX = 12000
_R3_FILE_SKELETON_MAX = 8000
_R3_FIXED_OVERHEAD = 26000


def _r3_diff_budget(symbol_context: str, file_skeleton: str) -> int:
    sym_actual = min(len(symbol_context), _R3_SYMBOL_CONTEXT_MAX)
    skel_actual = min(len(file_skeleton), _R3_FILE_SKELETON_MAX)
    return SINGLE_CALL_CONTEXT_BUDGET - _R3_FIXED_OVERHEAD - sym_actual - skel_actual


def _r3_trim_rich_context(context: str, budget: int) -> str:
    if len(context) <= budget:
        return context
    sections = re.split(r'\n(?=# )', context)
    tier1 = [s for s in sections if 'Framework Notes' in s or 'Exploration Log' in s]
    tier2 = [s for s in sections if 'skeleton' in s.split('\n')[0] and s not in tier1]
    tier3 = [s for s in sections if s not in tier1 and s not in tier2]
    result_parts: List[str] = []
    used = 0
    for section in tier1 + tier2 + tier3:
        if used + len(section) + 2 <= budget:
            result_parts.append(section)
            used += len(section) + 2
        else:
            remaining = budget - used - 2
            if remaining > 300:
                result_parts.append(section[:remaining] + '\n...(trimmed)')
            break
    return '\n'.join(result_parts)


def _r3_trim_skeleton(skeleton: str, diff_chunk: str, budget: int) -> str:
    if len(skeleton) <= budget:
        return skeleton
    diff_symbols = set(re.findall(r'\b([A-Za-z_][A-Za-z0-9_]{2,})\b', diff_chunk))
    lines = skeleton.splitlines()
    kept = [ln for ln in lines
            if ln.strip().startswith(('import ', 'from '))
            or ln.strip().startswith('class ')
            or any(s in ln for s in diff_symbols)]
    result = '\n'.join(kept)
    if len(result) <= budget:
        return result
    kept_no_import = [ln for ln in kept if not ln.strip().startswith(('import ', 'from '))]
    result = '\n'.join(kept_no_import)
    if len(result) <= budget:
        return result
    kept_core = [ln for ln in kept_no_import
                 if ln.strip().startswith('class ') or any(s in ln for s in diff_symbols)]
    return '\n'.join(kept_core)[:budget]


def _r3_trim_shared_context(context: str, budget: int) -> str:
    if len(context) <= budget:
        return context
    sections = re.split(r'\n(?=\[)', context)
    priority = {'Changed Interfaces': 0, 'Shared Symbols': 1, 'Intra-PR': 2}

    def _sort_key(s: str) -> int:
        for k, v in priority.items():
            if k in s:
                return v
        return 9

    sections.sort(key=_sort_key)
    result_parts: List[str] = []
    used = 0
    for section in sections:
        if used + len(section) + 1 <= budget:
            result_parts.append(section)
            used += len(section) + 1
        else:
            remaining = budget - used - 1
            if remaining > 200:
                result_parts.append(section[:remaining] + '\n...(trimmed)')
            break
    return '\n'.join(result_parts)


def _r3_split_group_if_needed(
    group_paths: List[str], file_diffs: Dict[str, str], budget: int = 40000,
) -> List[List[str]]:
    sub_groups: List[List[str]] = []
    current_group: List[str] = []
    current_size = 0
    for path in group_paths:
        fdiff = file_diffs.get(path, '')
        if current_size + len(fdiff) > budget and current_group:
            sub_groups.append(current_group)
            current_group = []
            current_size = 0
        current_group.append(path)
        current_size += len(fdiff)
    if current_group:
        sub_groups.append(current_group)
    return sub_groups


def _r3_parse_exploration_json(raw: str) -> dict:
    try:
        text = _extract_json_text(raw)
        if text:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    try:
        parsed = _parse_json_with_repair(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def _r3_read_lines(clone_dir: str, rel_path: str, start: int, end: int) -> str:
    abs_path = os.path.join(clone_dir, rel_path)
    try:
        with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        start_idx = max(0, start - 1)
        end_idx = min(len(lines), end)
        numbered = []
        for i, line in enumerate(lines[start_idx:end_idx], start=start_idx + 1):
            numbered.append(f'{i:>6}| {line}')
        return ''.join(numbered)
    except OSError:
        return ''


def _r3_grep_relevant_lines(clone_dir: str, rel_path: str, symbols: set, context: int = 10) -> str:
    abs_path = os.path.join(clone_dir, rel_path)
    try:
        with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except OSError:
        return ''
    hit_indices: set = set()
    for i, line in enumerate(lines):
        if any(s in line for s in symbols):
            for j in range(max(0, i - context), min(len(lines), i + context + 1)):
                hit_indices.add(j)
    if not hit_indices:
        return ''
    return ''.join(f'{i + 1:>6}| {lines[i]}' for i in sorted(hit_indices))


def _r3_collect_related_file_parts(
    clone_dir: str, exploration: dict, diff_symbols: set, seen_skeletons: set,
) -> List[str]:
    parts: List[str] = []
    for item in exploration.get('related_files', [])[:5]:
        path = item.get('path', '')
        if not path:
            continue
        lines = item.get('lines', [1, 50])
        start = lines[0] if len(lines) > 0 else 1
        end = lines[1] if len(lines) > 1 else start + 50
        if (end - start) > 200 and diff_symbols:
            content = _r3_grep_relevant_lines(clone_dir, path, diff_symbols)
        else:
            content = _r3_read_lines(clone_dir, path, start, end)
        if content:
            parts.append(f'# {path}:{start}-{end} ({item.get("reason", "")})\n{content}')
        if path not in seen_skeletons:
            skeleton = _extract_file_skeleton(clone_dir, path)
            if skeleton:
                parts.append(f'# {path} (file skeleton)\n{skeleton}')
            seen_skeletons.add(path)
    for bc in exploration.get('base_classes', []):
        bc_path = bc.get('file', '')
        if bc_path and bc_path not in seen_skeletons:
            skeleton = _extract_file_skeleton(clone_dir, bc_path)
            if skeleton:
                parts.append(f'# {bc_path} (base class {bc.get("symbol", "")} skeleton)\n{skeleton}')
            seen_skeletons.add(bc_path)
    return parts


def _r3_build_rich_context(clone_dir: str, raw_agent_output: str, primary_path: str,
                           exploration_log: Optional[List[str]] = None,
                           primary_diff: str = '') -> str:
    exploration = _r3_parse_exploration_json(raw_agent_output)
    if not exploration:
        return raw_agent_output[:_R3_RICH_CONTEXT_BUDGET] if raw_agent_output else ''

    seen_skeletons: set = set()
    diff_symbols = set(re.findall(r'\b([A-Za-z_][A-Za-z0-9_]{2,})\b', primary_diff)) if primary_diff else set()
    parts = _r3_collect_related_file_parts(clone_dir, exploration, diff_symbols, seen_skeletons)

    framework_notes = exploration.get('framework_notes', [])
    if framework_notes:
        parts.append('# Framework Notes\n' + '\n'.join(f'- {n}' for n in framework_notes[:3]))

    if exploration_log:
        parts.append('# Exploration Log\n' + '\n'.join(exploration_log[-20:]))

    result = '\n\n'.join(parts)
    return _r3_trim_rich_context(result, _R3_RICH_CONTEXT_BUDGET)


def _r3_build_file_context(
    llm: Any, path: str, diff_chunk: str, clone_dir: str, tools: List[Any],
    language: str = 'cn', agent_instructions: str = '',
) -> str:
    prompt = _safe_format(
        _R3_CONTEXT_COLLECT_PROMPT_TMPL,
        path=path, diff_chunk=diff_chunk[:8000],
        agent_instructions=agent_instructions or '(not available)',
        lang_instruction=_language_instruction(language),
    )
    step_counter = [0]
    exploration_log: List[str] = []
    traced_tools = [_make_traced_tool(t, step_counter, path, exploration_log, round_name='R3') for t in tools]
    try:
        agent = ReactAgent(
            llm, tools=traced_tools, max_retries=_R3_FILE_AGENT_RETRIES,
            workspace=clone_dir, force_summarize=True,
            force_summarize_context=(
                f'Exploring context for {path}:\n{diff_chunk[:800]}\n\n'
                f'Key framework conventions:\n{agent_instructions[:400]}'
            ),
            keep_full_turns=2,
        )
    except Exception as e:
        lazyllm.LOG.warning(f'  [Agent] ReactAgent init failed for {path}: {e}; skipping context collection')
        return ''
    for tool in agent._tools_manager.all_tools:
        tool.execute_in_sandbox = False
    lazyllm.LOG.info(f'  [Agent] Analyzing {path} ...')
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(agent, prompt)
        try:
            raw = fut.result(timeout=_R3_FILE_TIMEOUT_SECS)
        except FuturesTimeoutError:
            raise RuntimeError(f'Round 3 context collection timed out for {path} after {_R3_FILE_TIMEOUT_SECS}s')
    lazyllm.LOG.info(f'  [Agent] Done {path}')
    raw_str = raw if isinstance(raw, str) else str(raw)
    return _r3_build_rich_context(clone_dir, raw_str, path, exploration_log=exploration_log)


def _r3_parse_item(item: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], Optional[int]]]:
    if not isinstance(item, dict) or item.get('problem') is None:
        return None
    if not item.get('path'):
        return None
    normalized = _normalize_comment_item(item, default_path='', default_category='design')
    if normalized is None:
        return None
    r1_idx = item.get('r1_idx')
    try:
        r1_idx = int(r1_idx) if r1_idx is not None else None
    except (TypeError, ValueError):
        r1_idx = None
    return normalized, r1_idx


def _r3_extract_issues(
    llm: Any, path: str, diff_chunk: str, hunk_range: str, symbol_context: str,
    shared_context: str, r1_issues: List[Dict[str, Any]], arch_doc: str,
    pr_summary: str, language: str = 'cn', agent_instructions: str = '',
    file_skeleton: str = '', all_paths: Optional[List[str]] = None,
    review_spec: str = '',
) -> Tuple[List[Dict[str, Any]], set]:
    arch_snippet = _extract_arch_for_file(arch_doc, path, max_chars=_R3_ARCH_BUDGET)
    r1_indexed = [{**c, 'r1_idx': i, 'problem': (c.get('problem') or '')[:120]} for i, c in enumerate(r1_issues)]
    r1_text = json.dumps(r1_indexed, ensure_ascii=False, indent=2) if r1_indexed else '(none)'
    if len(r1_text) > _R3_R1_BUDGET:
        r1_text = r1_text[:_R3_R1_BUDGET] + '\n...(truncated)'
    paths_str = ', '.join(f'`{p}`' for p in (all_paths or [path]))
    spec_snippet = _lookup_relevant_rules(review_spec, diff_chunk, max_detail=10) if review_spec else '(not available)'
    sym_trimmed = _r3_trim_rich_context(symbol_context, _R3_SYMBOL_CONTEXT_MAX) if symbol_context else ''
    skel_trimmed = _r3_trim_skeleton(file_skeleton, diff_chunk, _R3_FILE_SKELETON_MAX) if file_skeleton else ''
    shared_trimmed = _r3_trim_shared_context(shared_context, _R3_SHARED_CTX_BUDGET) if shared_context else ''
    prompt = _safe_format(
        _R3_ISSUE_EXTRACT_PROMPT_TMPL,
        lang_instruction=_language_instruction(language),
        pr_summary=(pr_summary or '')[:_R3_SUMMARY_BUDGET],
        agent_instructions=agent_instructions or '(not available)',
        arch_doc=arch_snippet,
        review_spec=spec_snippet,
        shared_context=shared_trimmed or '(none)',
        file_skeleton=skel_trimmed or '(not available)',
        symbol_context=sym_trimmed or '(none)',
        round1_json=r1_text, path=path, hunk_range=hunk_range, diff_text=diff_chunk,
        density_rule=issue_density_rule(diff_chunk),
        all_paths=paths_str,
    )
    items = _safe_llm_call(llm, prompt)
    result: List[Dict[str, Any]] = []
    kept_r1_idxs: set = set()
    for item in (items if isinstance(items, list) else []):
        parsed = _r3_parse_item(item)
        if parsed is None:
            continue
        entry, r1_idx = parsed
        if r1_idx is not None:
            kept_r1_idxs.add(r1_idx)
        result.append(entry)
    discarded_keys = {
        f'{c.get("path", path)}:{c.get("line")}:{c.get("bug_category", "")}'
        for i, c in enumerate(r1_issues) if i not in kept_r1_idxs
    }
    return result, discarded_keys


def _r3_dedupe_issues_by_line(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    return [it for it in issues if (k := f'{it.get("path")}:{it.get("line")}') not in seen and not seen.add(k)]


def _filter_symbol_context_for_chunk(symbol_context: str, diff_chunk: str) -> str:
    chunk_symbols = set(re.findall(r'\b([A-Za-z_][A-Za-z0-9_]{2,})\b', diff_chunk))
    if not chunk_symbols:
        return _r3_trim_rich_context(symbol_context, _R3_SYMBOL_CONTEXT_MAX)
    lines = symbol_context.splitlines()
    relevant: List[str] = []
    prev_matched = False
    for line in lines:
        matched = any(s in line for s in chunk_symbols)
        if matched or prev_matched:
            relevant.append(line)
        prev_matched = matched
    filtered = '\n'.join(relevant)
    return _r3_trim_rich_context(filtered, _R3_SYMBOL_CONTEXT_MAX) if filtered \
        else _r3_trim_rich_context(symbol_context, _R3_SYMBOL_CONTEXT_MAX)


def _trim_r1_for_group(all_r1: List[Dict[str, Any]], budget: int = 4000) -> str:
    trimmed: List[Dict[str, Any]] = []
    chars = 0
    for item in all_r1:
        s = json.dumps(item, ensure_ascii=False)
        if chars + len(s) > budget:
            break
        trimmed.append(item)
        chars += len(s)
    return json.dumps(trimmed, ensure_ascii=False, indent=2) if trimmed else '[]'


def _r3_group_review(
    llm: Any,
    group_paths: List[str],
    file_diffs: Dict[str, str],
    r1_by_file: Dict[str, List[Dict[str, Any]]],
    shared_context: str,
    arch_doc: str,
    pr_summary: str,
    language: str,
    ckpt: Optional[Any],
    all_results: List[Dict[str, Any]],
    all_discarded: set,
    use_cache: bool = True,
    agent_instructions: str = '',
) -> None:
    group_key = 'r3_group_' + re.sub(r'[^a-zA-Z0-9]', '_', '_'.join(sorted(group_paths)))[:80]
    cached = ckpt.get(group_key) if ckpt else None
    if cached is not None and use_cache:
        all_results.extend(cached)
        lazyllm.LOG.info(f'  [R3-group] {group_paths} loaded from cache ({len(cached)} issues)')
        return

    files_block_parts: List[str] = []
    all_r1: List[Dict[str, Any]] = []
    for path in group_paths:
        fdiff = file_diffs.get(path, '')
        files_block_parts.append(f'### File: {path}\n```diff\n{fdiff}\n```')
        all_r1.extend(r1_by_file.get(path, []))

    files_block = '\n\n'.join(files_block_parts)
    if len(files_block) > 40000:
        sub_groups = _r3_split_group_if_needed(group_paths, file_diffs, budget=40000)
        if len(sub_groups) > 1:
            lazyllm.LOG.info(
                f'  [R3-group] Split {len(group_paths)} files into {len(sub_groups)} sub-groups'
            )
            for sg in sub_groups:
                _r3_group_review(
                    llm, sg, file_diffs, r1_by_file, shared_context, arch_doc,
                    pr_summary, language, ckpt, all_results, all_discarded,
                    use_cache=use_cache, agent_instructions=agent_instructions,
                )
            return
    round1_json = _trim_r1_for_group(all_r1)
    arch_snippet = arch_doc[:4000] if arch_doc else ''  # simple clip for group mode
    density_rule = issue_density_rule(files_block)

    prompt = _safe_format(
        _ROUND3_GROUP_PROMPT_TMPL,
        lang_instruction=_language_instruction(language),
        pr_summary=pr_summary[:600] if pr_summary else '(not available)',
        agent_instructions=agent_instructions[:400] if agent_instructions else '',
        arch_doc=arch_snippet,
        shared_context=_r3_trim_shared_context(shared_context, _R3_SHARED_CTX_BUDGET),
        round1_json=round1_json,
        files_block=files_block,
        density_rule=density_rule,
    )

    from ..utils import _extract_json_text, _parse_json_with_repair
    raw = _safe_llm_call_text(llm, prompt)
    items: List[Dict[str, Any]] = []
    try:
        parsed = _parse_json_with_repair(_extract_json_text(raw))
        if isinstance(parsed, list):
            for it in parsed:
                norm = _normalize_comment_item(it, group_paths[0] if group_paths else '')
                if norm:
                    items.append(norm)
    except Exception as e:
        lazyllm.LOG.warning(f'Round 3 group review parse failed for {group_paths}: {e}')

    if ckpt:
        ckpt.save(group_key, items)
    all_results.extend(items)


def _r3_unit_agent_verify(
    llm: Any,
    unit: Dict[str, Any],
    r1_by_file: Dict[str, List[Dict[str, Any]]],
    shared_context: str,
    arch_doc: str,
    pr_summary: str,
    clone_dir: str,
    symbol_cache: Dict[str, Any],
    tools: Any,
    language: str,
    ckpt: Optional[Any],
    all_results: List[Dict[str, Any]],
    all_discarded: set,
    use_cache: bool,
    agent_instructions: str,
    max_chunks: int,
    review_spec: str = '',
    agents_index: Optional[Dict[str, str]] = None,
) -> None:
    files = unit['files']
    anchor = unit['anchor']
    unit_diff = unit['diff']

    primary_file = anchor or (files[0] if files else '')
    effective_agent_instructions = _inject_local_agent_instructions(
        agent_instructions, agents_index, primary_file,
    )

    if anchor:
        safe = re.sub(r'[^a-zA-Z0-9_]', '_', anchor)
        r3_key = f'r3_file_{safe}'
        r3_disc_key = f'r3_disc_{safe}'
    else:
        safe = re.sub(r'[^a-zA-Z0-9]', '_', '_'.join(sorted(files)))[:80]
        r3_key = f'r3_group_{safe}'
        r3_disc_key = f'r3_disc_group_{safe}'

    cached = ckpt.get(r3_key) if ckpt else None
    if cached is not None and use_cache:
        all_results.extend(cached)
        cached_disc = (ckpt.get(r3_disc_key) if ckpt else None) or []
        all_discarded.update(cached_disc)
        lazyllm.LOG.info(f'  [R3] {files} loaded from cache ({len(cached)} issues)')
        return

    primary = anchor or files[0]
    agent_diff = compress_diff_for_agent_heuristic(unit_diff, _R3_AGENT_DIFF_BUDGET)
    try:
        symbol_context = _r3_build_file_context(llm, primary, agent_diff, clone_dir, tools, language,
                                                agent_instructions=effective_agent_instructions)
    except Exception as e:
        if 'timed out' in str(e):
            raise
        lazyllm.LOG.warning(f'Round 3 unit context failed for {files}: {e}')
        symbol_context = ''

    skeleton = _extract_file_skeleton(clone_dir, anchor) if anchor else ''
    if skeleton:
        lazyllm.LOG.info(f'  [R3] File skeleton extracted for {anchor} ({len(skeleton)} chars)')
    diff_cap = _r3_diff_budget(symbol_context, skeleton)
    all_chunks = _split_file_diff_into_chunks(unit_diff, diff_cap)
    if len(all_chunks) > R3_MAX_CHUNKS_HARD:
        lazyllm.LOG.warning(
            f'Round 3: {anchor or files} has {len(all_chunks)} chunks, '
            f'capping at {R3_MAX_CHUNKS_HARD} (R3_MAX_CHUNKS_HARD)'
        )
        all_chunks = all_chunks[:R3_MAX_CHUNKS_HARD]
    r1_issues = [c for f in files for c in r1_by_file.get(f, [])]
    items: List[Dict[str, Any]] = []
    discarded: set = set()
    for hunk_range, diff_chunk in all_chunks:
        filtered_ctx = _filter_symbol_context_for_chunk(symbol_context, diff_chunk)
        annotated_chunk = _annotate_full_diff(diff_chunk)
        new_items, new_disc = _r3_extract_issues(
            llm, primary, annotated_chunk, hunk_range,
            filtered_ctx, shared_context, r1_issues, arch_doc, pr_summary,
            language, effective_agent_instructions, file_skeleton=skeleton,
            all_paths=files, review_spec=review_spec,
        )
        items.extend(new_items)
        discarded.update(new_disc)

    if ckpt:
        ckpt.save(r3_key, items)
        ckpt.save(r3_disc_key, list(discarded))
    all_results.extend(items)
    all_discarded.update(discarded)


def _r3_run_unit(
    llm: Any, unit: Dict[str, Any],
    r1_by_file: Dict[str, List[Dict[str, Any]]],
    shared_context: str, arch_doc: str, pr_summary: str,
    clone_dir: str, symbol_cache: Dict[str, Any], tools: Any,
    language: str, ckpt: Optional[Any], use_cache: bool,
    agent_instructions: str, max_chunks: int,
    review_spec: str, agents_index: Optional[Dict[str, str]],
    all_results: List[Dict[str, Any]], all_discarded: set,
    r3_metrics: Dict[str, int], lock: threading.Lock,
    prog: Any,
) -> None:
    '''Run a single R3 unit and merge results into shared lists under lock.'''
    unit_results: List[Dict[str, Any]] = []
    unit_discarded: set = set()
    try:
        _r3_unit_agent_verify(
            llm, unit, r1_by_file, shared_context, arch_doc, pr_summary,
            clone_dir, symbol_cache, tools, language, ckpt, unit_results, unit_discarded,
            use_cache=use_cache, agent_instructions=agent_instructions,
            max_chunks=max_chunks, review_spec=review_spec, agents_index=agents_index,
        )
    except Exception as exc:
        lazyllm.LOG.warning(f'[R3] Unit {unit.get("anchor") or unit.get("files")} failed: {exc}')
    with lock:
        all_results.extend(unit_results)
        all_discarded.update(unit_discarded)
        if unit['anchor']:
            r3_metrics['r3_files_chunk'] += 1
        else:
            r3_metrics['r3_files_group'] += len(unit['files'])
    prog.update(
        f'{unit["anchor"]} [anchor+{len(unit["files"]) - 1} related]'
        if unit['anchor']
        else f'group {unit["files"]} [{len(unit["files"])} files]'
    )


def _r3_prepare_context(
    diff_text: str, round1: List[Dict[str, Any]], round2: List[Dict[str, Any]],
    ckpt: Optional[Any], use_cache: bool,
) -> tuple:
    '''Build shared_context, file_diffs, and r1_by_file from diff and prior rounds.'''
    shared_context = (ckpt.get('r3_shared_context') if ckpt and use_cache else None) or ''
    if not shared_context:
        shared_context = _r3_build_shared_context(diff_text)
        if ckpt and shared_context:
            ckpt.save('r3_shared_context', shared_context)

    file_diffs: Dict[str, str] = {}
    for path, new_start, new_count, content in _parse_unified_diff(diff_text):
        hunk_header = f'@@ -{new_start},{new_count} +{new_start},{new_count} @@\n'
        file_diffs[path] = file_diffs.get(path, '') + hunk_header + content + '\n'

    r1_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for c in list(round1) + list(round2):
        r1_by_file.setdefault(c.get('path') or '', []).append(c)

    return shared_context, file_diffs, r1_by_file


def _r3_dispatch_units(
    units: List[Dict[str, Any]], run_unit_kwargs: Dict[str, Any],
) -> None:
    '''Run all R3 units, using a thread pool when there are multiple units.'''
    max_workers = min(3, len(units))
    if max_workers <= 1:
        for unit in units:
            _r3_run_unit(unit=unit, **run_unit_kwargs)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as r3_pool:
            futs = [r3_pool.submit(_r3_run_unit, unit=unit, **run_unit_kwargs) for unit in units]
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as exc:
                    lazyllm.LOG.warning(f'[R3] Unit future raised: {exc}')


def _round3_agent_verify(
    llm: Any,
    round1: List[Dict[str, Any]],
    round2: List[Dict[str, Any]],
    diff_text: str,
    arch_doc: str,
    pr_summary: str = '',
    clone_dir: Optional[str] = None,
    language: str = 'cn',
    ckpt: Optional[Any] = None,
    agent_instructions: str = '',
    strategy: Optional[Any] = None,
    owner_repo: str = '',
    arch_cache_path: Optional[str] = None,
    review_spec: str = '',
    agents_index: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], set, Dict[str, int]]:
    r3_metrics: Dict[str, int] = {
        'r3_files_chunk': 0, 'r3_files_group': 0,
        'r3_files_skipped': 0, 'r3_chunks_total': 0,
    }

    if strategy is not None and not strategy.enable_r3:
        lazyllm.LOG.warning('Round 3 agent: skipped by strategy (enable_r3=False)')
        return [], set(), r3_metrics

    if clone_dir is None or not os.path.isdir(clone_dir):
        lazyllm.LOG.warning('Round 3 agent: clone_dir not available, skipping agent verify')
        return [], set(), r3_metrics

    max_files = strategy.max_files_for_r3 if strategy else 20
    large_threshold = strategy.large_file_threshold if strategy else 200
    max_chunks = strategy.max_chunks_per_file if strategy else 3

    use_cache = ckpt.should_use_cache(ReviewStage.R3) if ckpt else True
    shared_context, file_diffs, r1_by_file = _r3_prepare_context(
        diff_text, round1, round2, ckpt, use_cache,
    )

    units, skipped_files = _build_review_units(file_diffs, large_threshold, max_files)
    r3_metrics['r3_files_skipped'] = len(skipped_files)
    if skipped_files:
        lazyllm.LOG.warning(
            f'Round 3: {len(skipped_files)} files skipped due to max_files_for_r3={max_files}: '
            + ', '.join(skipped_files[:5])
        )

    symbol_cache: Dict[str, Any] = {}
    tools = _build_scoped_agent_tools_with_cache(clone_dir, llm, symbol_cache, owner_repo, arch_cache_path)

    prog = _Progress('Round 3: unified agent verify', len(units))
    all_results: List[Dict[str, Any]] = []
    all_discarded: set = set()

    _r3_dispatch_units(units, dict(
        llm=llm, r1_by_file=r1_by_file, shared_context=shared_context,
        arch_doc=arch_doc, pr_summary=pr_summary, clone_dir=clone_dir,
        symbol_cache=symbol_cache, tools=tools, language=language,
        ckpt=ckpt, use_cache=use_cache, agent_instructions=agent_instructions,
        max_chunks=max_chunks, review_spec=review_spec, agents_index=agents_index,
        all_results=all_results, all_discarded=all_discarded,
        r3_metrics=r3_metrics, lock=threading.Lock(), prog=prog,
    ))

    prog.done(f'{len(all_results)} issues from agent; {len(all_discarded)} prev issues discarded')
    return all_results, all_discarded, r3_metrics
