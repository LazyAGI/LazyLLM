# Copyright (c) 2026 LazyAGI. All rights reserved.
# Round 1: hunk-level diff analysis.
# Entry point: _round1_hunk_analysis

import math
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import lazyllm

from ..utils import (
    _Progress, _language_instruction, _safe_llm_call,
    _truncate_hunk_content, _annotate_diff_with_line_numbers,
    _normalize_comment_item,
)
from ..pre_analysis import (
    _read_file_context, _extract_arch_for_file,
    _extract_abstract_method_names, _find_subclass_implementations,
    _extract_file_skeleton, _lookup_relevant_rules,
)
from ..constants import (
    SINGLE_CALL_CONTEXT_BUDGET, R1_DIFF_BUDGET,
    max_issues_for_diff, cap_issues_by_severity, issue_density_rule,
)
from ..checkpoint import ReviewStage
from .common import (
    _lookup_relevant_symbols, _safe_format,
    _extract_code_tags, _format_code_profile, _format_review_focus,
    _inject_local_agent_instructions,
)
from .prompt import _ROUND1_PROMPT_TMPL, _ROUND1_BATCH_PROMPT_TMPL

_R1_LARGE_HUNK_OVERLAP = 30
_R1_PROMPT_OVERHEAD = 28000


def _r1_diff_budget(arch_snippet: str, spec_snippet: str, summary_snippet: str,
                    agent_instructions: str, file_context: str,
                    file_skeleton: str = '', code_profile: str = '', review_focus: str = '') -> int:
    overhead = (len(arch_snippet or '') + len(spec_snippet or '') + len(summary_snippet or '')
                + len(agent_instructions or '') + len(file_context or '')
                + len(file_skeleton or '') + len(code_profile or '') + len(review_focus or '')
                + _R1_PROMPT_OVERHEAD)
    return max(8000, SINGLE_CALL_CONTEXT_BUDGET - overhead)


def _analyze_single_hunk(
    llm: Any, path: str, new_start: int, new_count: int, content: str,
    arch_snippet: str, spec_snippet: str, summary_snippet: str,
    clone_dir: Optional[str] = None, language: str = 'cn',
    symbol_index: Optional[Dict[str, str]] = None, agent_instructions: str = '',
    file_skeleton: str = '', code_tags: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    file_context = _read_file_context(clone_dir, path, new_start, new_start + new_count) if clone_dir else ''
    context_end_line: Optional[int] = None
    if file_context:
        first_line = file_context.split('\n', 1)[0]
        m = re.search(r'full file,\s*(\d+)\s*lines', first_line)
        if m:
            context_end_line = int(m.group(1))
        else:
            m = re.search(r'excerpt lines\s*\d+\s*[–-]\s*(\d+)', first_line)
            if m:
                context_end_line = int(m.group(1))
    code_profile = _format_code_profile(code_tags) if code_tags else '(not available)'
    review_focus_block = _format_review_focus(code_tags) if code_tags else ''
    diff_budget = _r1_diff_budget(arch_snippet, spec_snippet, summary_snippet, agent_instructions,
                                  file_context, file_skeleton, code_profile, review_focus_block)
    window_lines = max(80, diff_budget // 50)
    lines = content.splitlines(keepends=True)
    if len(lines) > window_lines:
        return _analyze_large_hunk(
            llm, path, new_start, new_count, lines, window_lines,
            arch_snippet, spec_snippet, summary_snippet,
            clone_dir, language, symbol_index, agent_instructions,
            file_skeleton=file_skeleton, code_tags=code_tags,
        )
    content = _truncate_hunk_content(content, window_lines)
    actual_count = sum(1 for ln in content.splitlines() if not ln.startswith('-'))
    annotated_content = _annotate_diff_with_line_numbers(content, new_start)
    effective_arch = arch_snippet
    if symbol_index:
        sym_notes = _lookup_relevant_symbols(annotated_content, symbol_index)
        if sym_notes:
            effective_arch = f'{arch_snippet}\n\nKey utilities in this diff:\n{sym_notes}'
    prompt = _safe_format(
        _ROUND1_PROMPT_TMPL,
        lang_instruction=_language_instruction(language),
        pr_summary=summary_snippet, agent_instructions=agent_instructions or '(not available)',
        arch_doc=effective_arch, review_spec=spec_snippet,
        file_skeleton=file_skeleton or '(not available)',
        code_profile=code_profile, review_focus_block=review_focus_block,
        file_context=file_context or '(not available)',
        path=path, start=new_start, end=new_start + actual_count, content=annotated_content,
        density_rule=issue_density_rule(annotated_content),
    )
    items = _safe_llm_call(llm, prompt)
    effective_end = max(new_start + actual_count, context_end_line or 0)
    return [n for item in items if (n := _normalize_comment_item(
        item, new_start=new_start, end_line=effective_end, default_path=path,
    )) is not None]


def _analyze_large_hunk(
    llm: Any, path: str, new_start: int, new_count: int, lines: List[str], window_lines: int,
    arch_snippet: str, spec_snippet: str, summary_snippet: str,
    clone_dir: Optional[str], language: str,
    symbol_index: Optional[Dict[str, str]], agent_instructions: str,
    file_skeleton: str = '', code_tags: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    step = max(1, window_lines - _R1_LARGE_HUNK_OVERLAP)
    all_items: List[Dict[str, Any]] = []
    seen_keys: set = set()
    total = len(lines)
    code_profile = _format_code_profile(code_tags) if code_tags else '(not available)'
    review_focus_block = _format_review_focus(code_tags) if code_tags else ''
    win_idx = 0
    while win_idx * step < total:
        start_offset = win_idx * step
        end_offset = min(start_offset + window_lines, total)
        win_lines = lines[start_offset:end_offset]
        win_content = ''.join(win_lines)
        new_file_before = sum(1 for ln in lines[:start_offset] if not ln.startswith('-'))
        win_start = new_start + new_file_before
        win_count = sum(1 for ln in win_lines if not ln.startswith('-'))
        win_content_trunc = _truncate_hunk_content(win_content, window_lines)
        file_context = _read_file_context(clone_dir, path, win_start, win_start + win_count) if clone_dir else ''
        actual_budget = _r1_diff_budget(arch_snippet, spec_snippet, summary_snippet, agent_instructions,
                                        file_context, file_skeleton, code_profile, review_focus_block)
        actual_window = max(80, actual_budget // 50)
        if actual_window < window_lines:
            win_content_trunc = _truncate_hunk_content(win_content, actual_window)
        win_annotated = _annotate_diff_with_line_numbers(win_content_trunc, win_start)
        effective_arch = arch_snippet
        if symbol_index:
            sym_notes = _lookup_relevant_symbols(win_annotated, symbol_index)
            if sym_notes:
                effective_arch = f'{arch_snippet}\n\nKey utilities in this diff:\n{sym_notes}'
        prompt = _safe_format(
            _ROUND1_PROMPT_TMPL,
            lang_instruction=_language_instruction(language),
            pr_summary=summary_snippet, agent_instructions=agent_instructions or '(not available)',
            arch_doc=effective_arch, review_spec=spec_snippet,
            file_skeleton=file_skeleton or '(not available)',
            code_profile=code_profile, review_focus_block=review_focus_block,
            file_context=file_context or '(not available)',
            path=path, start=win_start, end=win_start + win_count, content=win_annotated,
            density_rule=issue_density_rule(win_annotated),
        )
        items = _safe_llm_call(llm, prompt)
        for item in items:
            n = _normalize_comment_item(
                item, new_start=win_start, end_line=win_start + win_count, default_path=path,
            )
            if n is None:
                continue
            dedup_key = (n.get('path'), n.get('line'), n.get('bug_category'), (n.get('problem') or '')[:60])
            if dedup_key not in seen_keys:
                seen_keys.add(dedup_key)
                all_items.append(n)
        win_idx += 1
        lazyllm.LOG.info(
            f'  [R1] Large hunk window {win_idx}/{math.ceil(total / step)}: '
            f'{path}:{win_start}-{win_start + win_count} ({len(all_items)} issues so far)'
        )
    return all_items


def _assign_batch_item(
    item: Dict[str, Any],
    hunks: List[Tuple[int, int, str]],
    path: str,
    context_end_line: Optional[int] = None,
) -> Optional[Tuple[int, Dict[str, Any]]]:
    for new_start, new_count, _ in hunks:
        effective_end = max(new_start + new_count, context_end_line or 0)
        normalized = _normalize_comment_item(
            item, new_start=new_start, end_line=effective_end, default_path=path,
        )
        if normalized is not None:
            return new_start, normalized
    return None


def _analyze_hunk_batch(
    llm: Any, path: str, hunks: List[Tuple[int, int, str]],
    arch_snippet: str, spec_snippet: str, summary_snippet: str,
    clone_dir: Optional[str] = None, language: str = 'cn',
    symbol_index: Optional[Dict[str, str]] = None, agent_instructions: str = '',
    file_skeleton: str = '', code_tags: Optional[Dict[str, Any]] = None,
) -> Dict[int, List[Dict[str, Any]]]:
    min_start = min(s for s, _, _ in hunks) if hunks else 1
    max_end = max(s + c for s, c, _ in hunks) if hunks else 1
    file_context = _read_file_context(clone_dir, path, min_start, max_end) if clone_dir else ''
    all_content = '\n'.join(cnt for _, _, cnt in hunks)
    abstract_methods = _extract_abstract_method_names(all_content)
    if abstract_methods and clone_dir:
        subclass_sigs = _find_subclass_implementations(clone_dir, abstract_methods)
        if subclass_sigs:
            file_context = (
                file_context
                + f'\n\n[Subclass implementations of changed abstract methods '
                f'({", ".join(abstract_methods)})]\n{subclass_sigs}'
            )
    effective_arch = arch_snippet
    if symbol_index:
        sym_notes = _lookup_relevant_symbols(all_content, symbol_index)
        if sym_notes:
            effective_arch = f'{arch_snippet}\n\nKey utilities in this diff:\n{sym_notes}'
    code_profile = _format_code_profile(code_tags) if code_tags else '(not available)'
    review_focus_block = _format_review_focus(code_tags) if code_tags else ''
    hunk_budget = max(500, _r1_diff_budget(arch_snippet, spec_snippet, summary_snippet,
                                           agent_instructions, file_context,
                                           file_skeleton, code_profile, review_focus_block
                                           ) // max(1, len(hunks)))
    hunk_budget_lines = max(80, hunk_budget // 50)
    hunk_blocks = [
        f'<hunk path="{path}" start={s} end={s + c}>\n'
        f'{_annotate_diff_with_line_numbers(_truncate_hunk_content(cnt, hunk_budget_lines), s)}\n</hunk>'
        for s, c, cnt in hunks
    ]
    prompt = _safe_format(
        _ROUND1_BATCH_PROMPT_TMPL,
        lang_instruction=_language_instruction(language),
        pr_summary=summary_snippet, agent_instructions=agent_instructions or '(not available)',
        arch_doc=effective_arch, review_spec=spec_snippet,
        file_skeleton=file_skeleton or '(not available)',
        code_profile=code_profile, review_focus_block=review_focus_block,
        file_context=file_context or '(not available)',
        path=path, hunks_content='\n\n'.join(hunk_blocks), density_rule=issue_density_rule('\n'.join(hunk_blocks)),
    )
    items = _safe_llm_call(llm, prompt)
    context_end_line: Optional[int] = None
    if file_context:
        first_line = file_context.split('\n', 1)[0]
        m = re.search(r'full file,\s*(\d+)\s*lines', first_line)
        if m:
            context_end_line = int(m.group(1))
        else:
            m = re.search(r'excerpt lines\s*\d+\s*[–-]\s*(\d+)', first_line)
            if m:
                context_end_line = int(m.group(1))
    results: Dict[int, List[Dict[str, Any]]] = {s: [] for s, _, _ in hunks}
    for item in (items if isinstance(items, list) else []):
        if not isinstance(item, dict) or item.get('problem') is None:
            continue
        assigned = _assign_batch_item(item, hunks, path, context_end_line=context_end_line)
        if assigned is not None:
            results[assigned[0]].append(assigned[1])
    return results


def _r1_build_batches(
    hunks: List[Tuple[str, int, int, str]], uncached_idxs: List[int],
) -> List[List[int]]:
    batches: List[List[int]] = []
    cur: List[int] = []
    cur_sz = 0
    for idx in uncached_idxs:
        content = hunks[idx][3]
        if cur and cur_sz + len(content) > R1_DIFF_BUDGET:
            batches.append(cur)
            cur, cur_sz = [idx], len(content)
        else:
            cur.append(idx)
            cur_sz += len(content)
    if cur:
        batches.append(cur)
    return batches


def _r1_run_batch(
    llm: Any, path: str, batch_idxs: List[int], hunks: List[Tuple[str, int, int, str]],
    arch_snippet: str, spec_snippet: str, summary_snippet: str,
    clone_dir: Optional[str], language: str, symbol_index: Optional[Dict[str, str]],
    lock: threading.Lock, results_by_idx: Dict[int, List[Dict[str, Any]]],
    ckpt: Optional[Any], prog: Any, cache_key_fn: Any, agent_instructions: str = '',
    file_skeleton: str = '', code_tags: Optional[Dict[str, Any]] = None,
) -> None:
    if len(batch_idxs) == 1:
        idx = batch_idxs[0]
        _, new_start, new_count, content = hunks[idx]
        items = _analyze_single_hunk(
            llm, path, new_start, new_count, content,
            arch_snippet, spec_snippet, summary_snippet,
            clone_dir, language, symbol_index, agent_instructions,
            file_skeleton=file_skeleton, code_tags=code_tags,
        )
        with lock:
            results_by_idx[idx] = items
            if ckpt:
                ckpt.save(cache_key_fn(path, new_start), items)
            prog.update(f'{path}:{new_start}-{new_start + new_count - 1} ({len(items)} issues)')
    else:
        batch_hunks = [(hunks[i][1], hunks[i][2], hunks[i][3]) for i in batch_idxs]
        batch_results = _analyze_hunk_batch(
            llm, path, batch_hunks,
            arch_snippet, spec_snippet, summary_snippet,
            clone_dir, language, symbol_index, agent_instructions,
            file_skeleton=file_skeleton, code_tags=code_tags,
        )
        with lock:
            for idx in batch_idxs:
                _, new_start, _, _ = hunks[idx]
                items = batch_results.get(new_start, [])
                results_by_idx[idx] = items
                if ckpt:
                    ckpt.save(cache_key_fn(path, new_start), items)
                _, new_start, new_count, _ = hunks[idx]
                prog.update(f'{path}:{new_start}-{new_start + new_count - 1} ({len(items)} issues)')


def _r1_cache_key(path: str, new_start: int) -> str:
    return f'r1_hunk_{re.sub(r"[^a-zA-Z0-9_]", "_", path)}_{new_start}'


def _r1_task_batch(
    path: str, idxs: List[int], hunks: List[Tuple[str, int, int, str]],
    arch_doc: str, spec_snippet: str, summary_snippet: str,
    clone_dir: Optional[str], language: str, symbol_index: Optional[Dict[str, str]],
    lock: threading.Lock, results_by_idx: Dict[int, List[Dict[str, Any]]],
    ckpt: Optional[Any], prog: Any, use_cache: bool, llm: Any, agent_instructions: str = '',
    pr_file_summary: str = '', agents_index: Optional[Dict[str, str]] = None,
) -> None:
    arch_snippet = _extract_arch_for_file(arch_doc, path, max_chars=3000)
    if pr_file_summary:
        arch_snippet = f'{arch_snippet}\n\n## PR Changed Files\n{pr_file_summary}' if arch_snippet else \
            f'## PR Changed Files\n{pr_file_summary}'
    effective_agent_instructions = _inject_local_agent_instructions(agent_instructions, agents_index, path)
    file_skeleton = _extract_file_skeleton(clone_dir, path) if clone_dir else ''
    first_hunk_content = hunks[idxs[0]][3] if idxs else ''
    code_tags: Dict[str, Any] = {}
    if file_skeleton and len(file_skeleton.strip()) >= 50:
        try:
            code_tags = _extract_code_tags(llm, file_skeleton, first_hunk_content[:1500])
        except Exception as e:
            lazyllm.LOG.warning(f'Round 1: code tag extraction failed for {path}: {e}')
    uncached_idxs: List[int] = []
    for idx in idxs:
        _, new_start, new_count, _ = hunks[idx]
        cached = ckpt.get(_r1_cache_key(path, new_start)) if ckpt else None
        if cached is not None and use_cache:
            with lock:
                results_by_idx[idx] = cached
                prog.update(f'{path}:{new_start}-{new_start + new_count - 1} (cached)')
        else:
            if cached is None and not use_cache:
                lazyllm.LOG.warning(f'Round 1: no cache for {path}:{new_start}, re-computing')
            uncached_idxs.append(idx)
    if not uncached_idxs:
        return
    for batch_idxs in _r1_build_batches(hunks, uncached_idxs):
        _r1_run_batch(
            llm, path, batch_idxs, hunks,
            arch_snippet, spec_snippet, summary_snippet,
            clone_dir, language, symbol_index,
            lock, results_by_idx, ckpt, prog, _r1_cache_key, effective_agent_instructions,
            file_skeleton=file_skeleton, code_tags=code_tags,
        )


def _round1_hunk_analysis(
    llm: Any,
    hunks: List[Tuple[str, int, int, str]],
    arch_doc: str,
    review_spec: str,
    pr_summary: str = '',
    max_workers: int = 4,
    clone_dir: Optional[str] = None,
    language: str = 'cn',
    symbol_index: Optional[Dict[str, str]] = None,
    ckpt: Optional[Any] = None,
    agent_instructions: str = '',
    pr_file_summary: str = '',
    agents_index: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    all_diff = '\n'.join(h[3] for h in hunks) if hunks else ''
    spec_snippet = _lookup_relevant_rules(review_spec, all_diff, max_detail=8) if review_spec else '(not available)'
    summary_snippet = pr_summary[:600] if pr_summary else '(not available)'
    prog = _Progress('Round 1: hunk analysis', len(hunks))
    lock = threading.Lock()
    results_by_idx: Dict[int, List[Dict[str, Any]]] = {}
    use_cache = ckpt.should_use_cache(ReviewStage.R1) if ckpt else True

    file_to_idxs: Dict[str, List[int]] = {}
    for idx, (path, _, _, _) in enumerate(hunks):
        file_to_idxs.setdefault(path, []).append(idx)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _r1_task_batch, path, idxs, hunks,
                arch_doc, spec_snippet, summary_snippet,
                clone_dir, language, symbol_index,
                lock, results_by_idx, ckpt, prog, use_cache, llm, agent_instructions,
                pr_file_summary, agents_index,
            ): path
            for path, idxs in file_to_idxs.items()
        }
        failed = 0
        for f in as_completed(futures):
            exc = f.exception()
            if exc is not None:
                failed += 1
                lazyllm.LOG.warning(
                    f'Round 1 file task failed ({futures[f]}): {exc}\n'
                    + ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                )
        if failed > 0 and len(file_to_idxs) > 0 and failed / len(file_to_idxs) > 0.5:
            raise RuntimeError(f'Round 1 failed on {failed}/{len(file_to_idxs)} files (>{50}%); aborting.')

    all_comments = [c for i in range(len(hunks)) for c in results_by_idx.get(i, [])]
    _seen_r1: set = set()
    deduped_comments: List[Dict[str, Any]] = []
    for c in all_comments:
        key = (
            c.get('path', ''),
            int(c.get('line') or 0),
            c.get('bug_category', ''),
            (c.get('problem') or '')[:60],
        )
        if key not in _seen_r1:
            _seen_r1.add(key)
            deduped_comments.append(c)
    all_comments = deduped_comments
    cap = max_issues_for_diff('\n'.join(h[3] for h in hunks))
    all_comments = cap_issues_by_severity(all_comments, cap)
    prog.done(f'{len(all_comments)} issues total')
    return all_comments
