# Copyright (c) 2026 LazyAGI. All rights reserved.
# Shared utility functions used by multiple review rounds.
# No round-specific logic here — only cross-round helpers.

import inspect
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import lazyllm

from ..utils import (
    _safe_llm_call, _safe_llm_call_text,
    _parse_json_with_repair, _extract_json_text, _parse_unified_diff,
    _safe_format,
)
from ..pre_analysis import _get_local_agent_instructions
from ..constants import R3_UNIT_DIFF_BUDGET
from .prompt import _CODE_TAG_PROMPT_TMPL, _COMPRESS_COMMENTS_PROMPT_TMPL

# ── String template utilities ─────────────────────────────────────────────────

def _lookup_relevant_symbols(diff_content: str, symbol_index: Dict[str, str]) -> str:
    hits = [f'{sym}: {desc}' for sym, desc in symbol_index.items() if sym in diff_content]
    return '\n'.join(hits[:5])


def _sample_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    third = max_chars // 3
    mid = len(text) // 2
    return (
        text[:third]
        + '\n...\n'
        + text[mid - third // 2: mid + third // 2]
        + '\n...\n'
        + text[-third:]
    )


# ── Code tag utilities (R1) ───────────────────────────────────────────────────

def _extract_code_tags(
    llm: Any, skeleton: str, diff_excerpt: str, max_focus: int = 5,
) -> Dict[str, Any]:
    prompt = _safe_format(
        _CODE_TAG_PROMPT_TMPL,
        skeleton=skeleton[:2000] if skeleton else '(not available)',
        diff_excerpt=diff_excerpt[:1500] if diff_excerpt else '(not available)',
        max_focus=max_focus,
    )
    raw = _safe_llm_call_text(llm, prompt)
    if not raw or not raw.strip():
        return {}
    parsed = _parse_json_with_repair(_extract_json_text(raw))
    return parsed if isinstance(parsed, dict) else {}


def _format_code_profile(tags: Dict[str, Any]) -> str:
    if not tags:
        return '(not available)'
    parts = []
    if tags.get('module_type'):
        parts.append(f'Module: {tags["module_type"]}')
    if tags.get('concurrency_model') and tags['concurrency_model'] != 'none':
        parts.append(f'Concurrency: {tags["concurrency_model"]}')
    if tags.get('io_profile') and tags['io_profile'] != 'minimal':
        parts.append(f'IO: {tags["io_profile"]}')
    ext = tags.get('external_systems')
    if ext:
        parts.append(f'External: {", ".join(ext) if isinstance(ext, list) else ext}')
    if tags.get('stateful'):
        parts.append('Stateful: yes')
    hints = tags.get('scope_hints')
    if hints and isinstance(hints, list):
        for h in hints[:3]:
            if isinstance(h, dict):
                parts.append(f'  {h.get("scope", "?")}: {", ".join(h.get("traits", []))}')
    return ' | '.join(parts[:4]) + ('\n' + '\n'.join(parts[4:]) if len(parts) > 4 else '')


def _format_review_focus(tags: Dict[str, Any]) -> str:
    focus = tags.get('review_focus')
    if not focus or not isinstance(focus, list):
        return ''
    lines = [f'{i + 1}. {f}' for i, f in enumerate(focus[:5]) if isinstance(f, str)]
    return 'Review focus for this file (prioritize these checks):\n' + '\n'.join(lines) if lines else ''


# ── Diff utilities ────────────────────────────────────────────────────────────

def _split_file_diff_into_chunks(diff_text: str, max_chars: int) -> List[Tuple[str, str]]:
    if len(diff_text) <= max_chars:
        return [('all hunks', diff_text)]
    chunks = []
    lines = diff_text.splitlines(keepends=True)
    current: List[str] = []
    current_len = 0
    chunk_start_line: Optional[int] = None
    chunk_end_line: Optional[int] = None
    last_hunk_header: Optional[str] = None

    def _flush(start: Optional[int], end: Optional[int], buf: List[str]) -> None:
        if buf:
            label = f'lines {start}-{end}' if start and end else 'partial'
            chunks.append((label, ''.join(buf)))

    for line in lines:
        mm = re.search(r'\+(\d+)', line) if line.startswith('@@') else None
        if mm:
            ln = int(mm.group(1))
            if chunk_start_line is None:
                chunk_start_line = ln
            chunk_end_line = ln
            last_hunk_header = line
        if current_len + len(line) > max_chars and current:
            _flush(chunk_start_line, chunk_end_line, current)
            current = []
            current_len = 0
            chunk_start_line = chunk_end_line
            if not line.startswith('@@'):
                if last_hunk_header is None:
                    raise ValueError(
                        '_split_file_diff_into_chunks: need to split diff but no @@ '
                        'hunk header has been seen yet — diff_text is missing hunk '
                        f'headers. First 120 chars: {diff_text[:120]!r}'
                    )
                current.append(last_hunk_header)
                current_len += len(last_hunk_header)
        current.append(line)
        current_len += len(line)
    _flush(chunk_start_line, chunk_end_line, current)
    return chunks or [('all hunks', diff_text[:max_chars])]


def _rmod_new_file_paths(diff_text: str) -> set:
    '''Return the set of file paths that are newly created in this diff (--- /dev/null).'''
    new_paths: set = set()
    is_new_file: bool = False
    for line in diff_text.splitlines():
        if line.startswith('--- /dev/null'):
            is_new_file = True
        elif line.startswith('--- '):
            is_new_file = False
        elif line.startswith('+++ b/') and is_new_file:
            new_paths.add(line[6:].strip())
            is_new_file = False
    return new_paths


def _collect_all_file_diffs(diff_text: str) -> Dict[str, str]:
    '''Collect per-file diffs for ALL changed files (including new files), preserving @@ headers.'''
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
        elif current_path is not None:
            current_lines.append(line)
    _flush()
    cleaned: Dict[str, str] = {}
    for path, raw in file_diffs.items():
        hunk_lines = [line for line in raw.splitlines(keepends=True)
                      if not line.startswith(('index ', '--- ', '+++ '))]
        cleaned[path] = ''.join(hunk_lines)
    return cleaned


# ── R3 shared context ─────────────────────────────────────────────────────────

_R3_SHARED_CTX_BUDGET = 4000
_R3_RELATED_DIFF_BUDGET = 4000


def _r3_parse_diff_imports(diff_text: str) -> Tuple[Dict[str, set], Dict[str, str], Dict[str, str]]:
    file_imports: Dict[str, set] = {}
    old_sigs: Dict[str, str] = {}
    new_sigs: Dict[str, str] = {}
    current_file = ''
    _import_re = re.compile(r'\s*(?:from\s+(\S+)\s+import\s+(.+)|import\s+(\S+))')
    _def_re = re.compile(r'\s*def\s+(\w+)\s*\(([^)]*)\)')
    for line in diff_text.splitlines():
        if line.startswith('+++ b/'):
            current_file = line[6:].strip()
        elif line.startswith('+') and not line.startswith('+++'):
            body = line[1:]
            if m := _import_re.match(body):
                if current_file:
                    syms = (s.strip().split(' as ')[0].strip() for s in re.split(r',\s*', m.group(2))) \
                        if m.group(2) else [m.group(3).strip()]
                    file_imports.setdefault(current_file, set()).update(s for s in syms if s)
            if m := _def_re.match(body):
                new_sigs[m.group(1)] = f'def {m.group(1)}({m.group(2)[:80]})'
        elif line.startswith('-') and not line.startswith('---'):
            if m := _def_re.match(line[1:]):
                old_sigs[m.group(1)] = f'def {m.group(1)}({m.group(2)[:80]})'
    return file_imports, old_sigs, new_sigs


def _r3_build_shared_context(diff_text: str) -> str:
    changed_files = list({path for path, _, _, _ in _parse_unified_diff(diff_text)})
    if len(changed_files) < 2:
        return ''

    file_imports, old_sigs, new_sigs = _r3_parse_diff_imports(diff_text)
    changed_interfaces = {sym: [old_sigs[sym], new_sigs[sym]] for sym in new_sigs
                          if sym in old_sigs and old_sigs[sym] != new_sigs[sym]}

    all_symbols: Dict[str, List[str]] = {}
    for fpath, syms in file_imports.items():
        for sym in syms:
            all_symbols.setdefault(sym, []).append(fpath)
    shared = {sym: files for sym, files in all_symbols.items() if len(files) >= 2}

    changed_file_set = set(changed_files)
    intra_deps = [
        f'{fpath} → {other} (imports {sym})'
        for fpath, syms in file_imports.items()
        for sym in syms
        for other in changed_file_set
        if other != fpath and sym in (file_imports.get(other) or set())
    ]

    shared_lines = '\n'.join(f'{sym} (in {", ".join(files[:3])})'
                             for sym, files in list(shared.items())[:10]) or '(none)'
    iface_lines = (
        '\n'.join(f'{sym}: {old} → {new}' for sym, (old, new) in list(changed_interfaces.items())[:8])
        if changed_interfaces else '(none)')
    parts = [
        '[Shared Symbols]\n' + shared_lines,
        '[Intra-PR Dependencies]\n' + ('\n'.join(intra_deps[:10]) if intra_deps else '(none)'),
        '[Changed Interfaces]\n' + iface_lines,
    ]
    result = '\n\n'.join(parts)[:_R3_SHARED_CTX_BUDGET]
    lazyllm.LOG.info(f'Round 3 shared context built (static): {len(result)} chars')
    return result


def _find_related_small_files(
    large_diff: str,
    small_files: List[str],
    file_diffs: Dict[str, str],
) -> List[str]:
    '''Find small files imported (directly or via relative import) by the large file.

    Handles:
    - ``import foo`` / ``from foo import X`` → last component of ``foo``
    - ``from .bar import X`` / ``from ..pkg.bar import X`` → ``bar`` (basename of relative module)
    '''
    abs_import_re = re.compile(r'^\+\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))', re.MULTILINE)
    rel_import_re = re.compile(r'^\+\s*from\s+(\.+[\w.]*)\s+import', re.MULTILINE)

    imported_modules: set = set()
    for m in abs_import_re.finditer(large_diff):
        mod = (m.group(1) or m.group(2) or '').split('.')[-1]
        if mod:
            imported_modules.add(mod)
    for m in rel_import_re.finditer(large_diff):
        rel = m.group(1).lstrip('.')
        if rel:
            mod = rel.split('.')[-1]
            if mod:
                imported_modules.add(mod)

    if not imported_modules:
        return []
    related: List[str] = []
    for path in small_files:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem in imported_modules:
            related.append(path)
    return related


# ── Review unit builder (shared by R3 and RScene) ────────────────────────────

def _build_review_units(
    file_diffs: Dict[str, str],
    large_file_threshold: int,
    max_files: int,
    unit_diff_budget: int = R3_UNIT_DIFF_BUDGET,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    '''Returns (units, skipped_files).
    Each unit: {'anchor': path_or_None, 'files': [path, ...], 'diff': combined_diff_str}
    anchor units: large file + absorbed related small files (within budget)
    group units: remaining small files grouped by directory (within budget)
    '''
    def diff_chars(p: str) -> int:
        return len(file_diffs.get(p, ''))

    sorted_files = sorted(file_diffs.keys(), key=diff_chars, reverse=True)
    anchor_files = [p for p in sorted_files if diff_chars(p) > large_file_threshold]
    small_files = [p for p in sorted_files if diff_chars(p) <= large_file_threshold]

    total = anchor_files + small_files
    if len(total) > max_files:
        skipped = set(total[max_files:])
        anchor_files = [p for p in anchor_files if p not in skipped]
        small_files = [p for p in small_files if p not in skipped]
    else:
        skipped = set()

    absorbed: set = set()
    units: List[Dict[str, Any]] = []

    for anchor in anchor_files:
        related = _find_related_small_files(
            file_diffs[anchor], [p for p in small_files if p not in absorbed], file_diffs
        )
        unit_files = [anchor]
        unit_diff = file_diffs[anchor]
        for rel in related:
            candidate = unit_diff + '\n' + file_diffs[rel]
            if len(candidate) <= unit_diff_budget:
                unit_files.append(rel)
                unit_diff = candidate
                absorbed.add(rel)
        units.append({'anchor': anchor, 'files': unit_files, 'diff': unit_diff})

    remaining = [p for p in small_files if p not in absorbed]
    dir_map: Dict[str, List[str]] = {}
    for p in remaining:
        dir_map.setdefault(os.path.dirname(p) or '.', []).append(p)

    for dir_files in dir_map.values():
        cur_files: List[str] = []
        cur_diff = ''
        for p in dir_files:
            candidate = (cur_diff + '\n' + file_diffs[p]) if cur_diff else file_diffs[p]
            if cur_files and len(candidate) > unit_diff_budget:
                units.append({'anchor': None, 'files': cur_files, 'diff': cur_diff})
                cur_files, cur_diff = [p], file_diffs[p]
            else:
                cur_files.append(p)
                cur_diff = candidate
        if cur_files:
            units.append({'anchor': None, 'files': cur_files, 'diff': cur_diff})

    return units, list(skipped)


# ── Agent instruction injection (shared by R1 and R3) ────────────────────────

def _inject_local_agent_instructions(
    agent_instructions: str, agents_index: Optional[Dict[str, str]], file_path: str,
) -> str:
    '''Merge global agent_instructions with local AGENTS.md rules for the given file path.'''
    if not agents_index:
        return agent_instructions
    local_rules = _get_local_agent_instructions(agents_index, file_path)
    if not local_rules:
        return agent_instructions
    return (agent_instructions + '\n\n## Local Rules\n' + local_rules
            if agent_instructions else local_rules)


# ── Tool tracing wrapper (shared by R3, RMod, RScene, RChain) ────────────────

_r3_agent_instance_counter = [0]


def _make_traced_tool(tool: Any, step_counter: List[int], path: str,
                      log_list: Optional[List[str]] = None,
                      round_name: str = 'Agent') -> Any:
    sig = inspect.signature(tool)
    params = list(sig.parameters.keys())
    _r3_agent_instance_counter[0] += 1
    unique_name = f'{tool.__name__}_r3_{_r3_agent_instance_counter[0]}'

    def traced(*args, **kwargs):
        step_counter[0] += 1
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        arg_parts = []
        for k, v in bound.arguments.items():
            v_str = repr(v) if not isinstance(v, str) else v
            if len(v_str) > 60:
                v_str = v_str[:57] + '...'
            arg_parts.append(f'{k}={v_str}' if k != params[0] else v_str)
        call_desc = f'{tool.__name__}({", ".join(arg_parts)})'
        lazyllm.LOG.info(f'  [{round_name} Step {step_counter[0]}] {call_desc}')
        result = tool(*args, **kwargs)
        if log_list is not None:
            status = 'ok'
            if isinstance(result, dict):
                if result.get('status') == 'error' or result.get('error'):
                    status = 'error'
                elif not result.get('status'):
                    status = 'empty' if not result else 'ok'
            elif isinstance(result, str) and ('not found' in result or not result):
                status = 'empty'
            log_list.append(f'- {call_desc} -> {status}')
        return result

    traced.__name__ = unique_name
    traced.__doc__ = tool.__doc__
    traced.__annotations__ = tool.__annotations__
    if not traced.__doc__:
        lazyllm.LOG.warning(f'Tool {tool.__name__!r} has no docstring; ReactAgent will fail to init')
    return traced


# ── Compression utilities (shared by R4 and post_merge) ──────────────────────

_BODY_COMPRESS_THRESHOLD = 200
_NEW_ISSUE_COMPRESS_THRESHOLD = 300


def _batch_llm_summarize(
    llm: Any, items: List[Dict[str, Any]], body_key: str, label: str
) -> Dict[int, str]:
    batch_input = [{'idx': item['idx'], body_key: item[body_key][:800]} for item in items]
    prompt = _safe_format(
        _COMPRESS_COMMENTS_PROMPT_TMPL,
        items_json=json.dumps(batch_input, ensure_ascii=False, indent=2)
    )
    summaries: Dict[int, str] = {}
    try:
        result = _safe_llm_call(llm, prompt)
        for r in (result if isinstance(result, list) else []):
            if isinstance(r, dict) and 'idx' in r and 'summary' in r:
                summaries[int(r['idx'])] = str(r['summary'])
    except Exception as e:
        raise RuntimeError(f'{label} compression failed: {e}') from e
    return summaries


def _compress_items(
    llm: Any, items: List[Dict[str, Any]], threshold: int, body_fn: Any, extra_fields_fn: Any,
) -> List[Dict[str, Any]]:
    long_items = [{'idx': i, 'body': body_fn(c)} for i, c in enumerate(items) if len(body_fn(c)) > threshold]
    summaries = _batch_llm_summarize(llm, long_items, 'body', 'Item') if long_items else {}
    long_idx_set = {item['idx'] for item in long_items}
    return [
        {'idx': i, **extra_fields_fn(c, summaries.get(i) if i in long_idx_set else None)}
        for i, c in enumerate(items)
    ]


def _compress_existing_comments(llm: Any, comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def body_fn(c: Dict[str, Any]) -> str:
        return c.get('body', '')

    def extra_fn(c: Dict[str, Any], summary: Optional[str]) -> Dict[str, Any]:
        return {
            'path': c.get('path', ''), 'line': c.get('line', ''),
            'summary': summary or c.get('body', '')[:_BODY_COMPRESS_THRESHOLD],
        }
    return _compress_items(llm, comments, _BODY_COMPRESS_THRESHOLD, body_fn, extra_fn)


def _compress_new_issues(llm: Any, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def body_fn(c: Dict[str, Any]) -> str:
        problem = c.get('problem') or ''
        suggestion = c.get('suggestion') or ''
        if isinstance(problem, list):
            problem = ' '.join(str(x) for x in problem)
        if isinstance(suggestion, list):
            suggestion = ' '.join(str(x) for x in suggestion)
        return problem + ' ' + suggestion

    def extra_fn(c: Dict[str, Any], summary: Optional[str]) -> Dict[str, Any]:
        return {
            'path': c.get('path', ''), 'line': c.get('line', ''),
            'severity': c.get('severity', 'normal'), 'bug_category': c.get('bug_category', 'logic'),
            'source': c.get('source', ''),
            'summary': summary or (c.get('problem') or '')[:120],
        }
    return _compress_items(llm, issues, _NEW_ISSUE_COMPRESS_THRESHOLD, body_fn, extra_fn)
