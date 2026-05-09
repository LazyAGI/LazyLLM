# Copyright (c) 2026 LazyAGI. All rights reserved.
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import lazyllm

from ..checkpoint import _load_cache, _save_cache, _save_cache_multi
from ..constants import SINGLE_CALL_CONTEXT_BUDGET
from ..utils import _Progress, _safe_llm_call, _safe_llm_call_text, _extract_json_text, _parse_json_with_repair
from .agent_instructions import _AGENT_INSTRUCTION_FILES
from .deepwiki import _fetch_deepwiki_summary, _deepwiki_ask_cached, _parse_owner_repo
from .file_context import (
    _build_dir_tree, _read_file_head,
    _extract_class_method_signatures, _SKIP_DIRS, _SKIP_EXTS,
)
from .prompt import (
    _ARCH_OUTLINE_PROMPT, _ARCH_GOTCHAS_INSTRUCTION, _ARCH_HAS_AGENT_INSTRUCTION,
    _ARCH_SECTION_PROMPT, _ARCH_BATCH_SECTIONS_PROMPT,
    _PUBLIC_API_FILES_PROMPT_TMPL,
)

_ARCH_SNAPSHOT_BUDGET = 6000
_ARCH_OUTLINE_MAX_SECTIONS = 13
_ARCH_OUTLINE_MAX_SECTIONS_WITH_AGENT = 10
_ARCH_PREV_SUMMARY_BUDGET = 1500

_PUBLIC_SYM_EXTS = {'.py', '.go', '.ts', '.js', '.tsx', '.jsx', '.java', '.rs', '.cpp', '.cc', '.h', '.hpp'}

_JAVA_PUB_PAT = (
    r'^\s*(?:public\s+|protected\s+)(?:static\s+)?'
    r'(?:class|interface|enum|\w[\w<>\[\]]*)\s+([A-Z][A-Za-z0-9_]*)'
)
_PUBLIC_SYM_PATTERNS: List[Tuple[frozenset, str, str]] = [
    (frozenset({'.py'}), r'^(?:def|class)\s+([A-Za-z][A-Za-z0-9_]*)\s*[\(:]', 'name'),
    (frozenset({'.go'}), r'^func\s+(?:\(\w+\s+\*?\w+\)\s+)?([A-Z][A-Za-z0-9_]*)\s*\(', 'name'),
    (frozenset({'.ts', '.tsx', '.js', '.jsx'}),
     r'^export\s+(?:async\s+)?(?:function|class|const|let|var)\s+([A-Za-z][A-Za-z0-9_]*)', 'name'),
    (frozenset({'.java'}), _JAVA_PUB_PAT, 'name'),
    (frozenset({'.rs'}), r'^pub\s+(?:async\s+)?(?:fn|struct|enum|trait)\s+([a-z_][A-Za-z0-9_]*)', 'name'),
    (frozenset({'.cpp', '.cc', '.h', '.hpp'}),
     r'^(?:class|struct|enum)\s+([A-Z][A-Za-z0-9_]*)', 'name'),
]

_PUBLIC_API_MAX_ENTRIES_PER_FILE = 40
_PUBLIC_API_MAX_FILES = 30

_ARCH_ALWAYS_INJECT = frozenset({
    'module hierarchy',
    'module ownership rules',
    'environment & dependencies',
    'non-obvious behaviors & gotchas',
    'non-obvious behaviors',
    'gotchas',
    'key utilities',
    'key utilities & usage notes',
    'typical usage patterns',
    'concurrency & multi-user conventions',
})


def _collect_structured_snapshot(clone_dir: str) -> str:  # noqa: C901
    parts: List[str] = []
    budget = _ARCH_SNAPSHOT_BUDGET

    def _add(label: str, text: str) -> None:
        nonlocal budget
        if budget <= 0 or not text:
            return
        block = f'## {label}\n{text[:budget]}'
        parts.append(block)
        budget -= len(block)

    _add('Directory Tree', _build_dir_tree(clone_dir, max_depth=2)[:800])
    _add('__init__.py', _read_file_head(os.path.join(clone_dir, '__init__.py'), 1500))

    try:
        sub_pkgs = sorted(
            d for d in os.listdir(clone_dir)
            if os.path.isdir(os.path.join(clone_dir, d)) and d not in _SKIP_DIRS and not d.startswith('.')
        )
    except OSError:
        sub_pkgs = []
    for pkg in sub_pkgs:
        _add(f'{pkg}/__init__.py', _read_file_head(os.path.join(clone_dir, pkg, '__init__.py'), 300))

    for rel in ['module/module.py', 'flow/flow.py', 'components/core.py', 'common/common.py']:
        _add(rel, _read_file_head(os.path.join(clone_dir, rel), 400))

    for rel, limit in [('setup.py', 500), ('pyproject.toml', 500), ('requirements.txt', 500),
                       ('requirements-dev.txt', 300), ('CMakeLists.txt', 300), ('Makefile', 200)]:
        _add(rel, _read_file_head(os.path.join(clone_dir, rel), limit))

    for rel in _AGENT_INSTRUCTION_FILES:
        _add(f'{rel} (agent instructions)', _read_file_head(os.path.join(clone_dir, rel), 2000))

    return '\n\n'.join(parts)


def _arch_generate_outline(
    llm: Any, snapshot: str, agent_instructions: str = '',
) -> List[Dict[str, Any]]:
    has_agent = bool(agent_instructions)
    max_sections = _ARCH_OUTLINE_MAX_SECTIONS_WITH_AGENT if has_agent else _ARCH_OUTLINE_MAX_SECTIONS
    gotchas_instruction = _ARCH_HAS_AGENT_INSTRUCTION if has_agent else _ARCH_GOTCHAS_INSTRUCTION
    snap_budget = max(4000, SINGLE_CALL_CONTEXT_BUDGET - 6000)
    prompt = _ARCH_OUTLINE_PROMPT.format(
        max_sections=max_sections,
        gotchas_instruction=gotchas_instruction,
        snapshot=snapshot[:snap_budget],
    )
    result = _safe_llm_call(llm, prompt)
    if isinstance(result, list) and result:
        return result[:max_sections]
    raise ValueError(f'Arch outline generation returned invalid result: {result!r}')


def _arch_collect_snippets(  # noqa: C901
    clone_dir: str, section: Dict[str, Any], max_chars: int = 6000,
    extra_globs: Optional[List[str]] = None, max_results_per_pattern: int = 8,
) -> str:
    from lazyllm.tools.agent.file_tool import search_in_files, read_file
    hints = section.get('search_hints', [])
    parts: List[str] = []
    seen_paths: set = set()
    globs_to_search = ['*.py'] + (extra_globs or [])
    for pattern in hints:
        for glob_pat in globs_to_search:
            try:
                result = search_in_files(
                    pattern, path=clone_dir, glob=glob_pat,
                    max_results=max_results_per_pattern, root=clone_dir,
                )
                matches = result.get('results', []) if isinstance(result, dict) else []
            except Exception:
                matches = []
            for m in matches:
                path = m.get('path', '')
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                try:
                    line = int(m.get('line', 1))
                    match_text = m.get('text', '')
                    if re.match(r'\s*class\s+\w+', match_text):
                        fc = read_file(path, start_line=max(1, line - 1), end_line=line + 2, root=clone_dir)
                        class_def = fc.get('content', '') if isinstance(fc, dict) else ''
                        try:
                            with open(path, 'r', encoding='utf-8', errors='replace') as _f:
                                all_lines = _f.readlines()
                            sigs = _extract_class_method_signatures(all_lines, line - 1)
                            snippet = class_def + ('\n' + '\n'.join(sigs[:12]) if sigs else '')
                        except Exception:
                            snippet = class_def
                    else:
                        end = line + 10 if re.match(r'\s*def\s+\w+', match_text) else line + 20
                        fc = read_file(path, start_line=max(1, line - 1), end_line=end, root=clone_dir)
                        snippet = fc.get('content', '') if isinstance(fc, dict) else ''
                except Exception:
                    snippet = m.get('text', '')
                if snippet:
                    parts.append(f'# {os.path.relpath(path, clone_dir)}\n{snippet}')
            if sum(len(p) for p in parts) >= max_chars:
                break
        if sum(len(p) for p in parts) >= max_chars:
            break
    combined = '\n\n'.join(parts)
    return combined[:max_chars] if combined else '(no relevant snippets found)'


def _arch_collect_env_deps(clone_dir: str) -> str:
    parts = []
    for fname in ('pyproject.toml', 'setup.py', 'setup.cfg',
                  'requirements.txt', 'requirements-dev.txt', 'requirements_dev.txt'):
        fpath = os.path.join(clone_dir, fname)
        if os.path.isfile(fpath):
            try:
                with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read(3000)
                parts.append(f'# {fname}\n{content}')
            except OSError:
                lazyllm.LOG.debug(f'Failed to read dependency file {fpath}, skipping')
    return '\n\n'.join(parts)[:6000] if parts else '(no dependency files found)'


def _arch_collect_snippets_for_section(clone_dir: str, section: Dict[str, Any]) -> str:
    from .agent_tools import _CONCURRENCY_MAX_RESULTS
    title_lower = section.get('title', '').lower()
    if 'environment' in title_lower or 'dependenc' in title_lower:
        return _arch_collect_env_deps(clone_dir)
    if 'typical usage' in title_lower or 'usage pattern' in title_lower:
        return _arch_collect_snippets(
            clone_dir, section, max_chars=6000,
            extra_globs=['*.md', '*.rst', 'test_*.py', '*_test.py'],
        )
    if 'concurrency' in title_lower or 'multi-user' in title_lower:
        return _arch_collect_snippets(
            clone_dir, section, max_chars=6000, max_results_per_pattern=_CONCURRENCY_MAX_RESULTS,
        )
    return _arch_collect_snippets(clone_dir, section)


def _arch_fill_section(
    llm: Any, clone_dir: str, section: Dict[str, Any],
    dir_tree_1level: str, prev_summaries: List[str],
) -> str:
    prev_text = ('\n'.join(prev_summaries) or '(none yet)')[-_ARCH_PREV_SUMMARY_BUDGET:]
    snippets = _arch_collect_snippets_for_section(clone_dir, section)
    prompt = _ARCH_SECTION_PROMPT.format(
        section_title=section.get('title', ''), section_focus=section.get('focus', ''),
        dir_tree=dir_tree_1level[:400], prev_summaries=prev_text, code_snippets=snippets[:6000],
    )
    raw = _safe_llm_call_text(llm, prompt)
    if not raw:
        raise ValueError(f'LLM returned empty result for section "{section.get("title")}"')
    return raw[:3500]


def _arch_pack_pairs(
    pairs: List[Tuple[Dict[str, Any], str]], budget: int,
) -> List[List[Tuple[Dict[str, Any], str]]]:
    batches: List[List[Tuple[Dict[str, Any], str]]] = []
    cur: List[Tuple[Dict[str, Any], str]] = []
    cur_sz = 0
    for sec, snip in pairs:
        need = len(snip) + len(sec.get('title', '')) + len(sec.get('focus', '')) + 200
        if cur and cur_sz + need > budget:
            batches.append(cur)
            cur, cur_sz = [(sec, snip)], need
        else:
            cur.append((sec, snip))
            cur_sz += need
    if cur:
        batches.append(cur)
    return batches


def _arch_fill_batch_llm(
    llm: Any, batch: List[Tuple[Dict[str, Any], str]],
    dir_tree_1level: str, prev_summaries: List[str],
    owner_repo: str = '', cache_path: Optional[str] = None,
) -> Dict[str, str]:
    def _deepwiki_block(sec: Dict[str, Any]) -> str:
        if not owner_repo:
            return ''
        title, focus = sec.get('title', ''), sec.get('focus', '')
        question = f'Explain the "{title}" aspect of this project: {focus}'
        answer = _deepwiki_ask_cached(owner_repo, question, max_chars=1200)
        if not answer:
            return ''
        return (
            f'\n\nDeepWiki background reference (may be 1-3 months stale — use as context only, '
            f'verify against local code):\n{answer}'
        )

    sections_block = '\n---\n'.join(
        f'### Section: {sec.get("title", "")}\nFocus: {sec.get("focus", "")}'
        f'\n\nSnippets:\n{snippets}{_deepwiki_block(sec)}'
        for sec, snippets in batch
    )
    prev_text = ('\n'.join(prev_summaries) or '(none yet)')[-_ARCH_PREV_SUMMARY_BUDGET:]
    prompt = _ARCH_BATCH_SECTIONS_PROMPT.format(
        dir_tree=dir_tree_1level[:400], prev_summaries=prev_text, sections_block=sections_block,
    )
    if len(prompt) > SINGLE_CALL_CONTEXT_BUDGET - 4000:
        sections_block = sections_block[:SINGLE_CALL_CONTEXT_BUDGET - 14000] + '\n...(truncated)'
        prompt = _ARCH_BATCH_SECTIONS_PROMPT.format(
            dir_tree=dir_tree_1level[:400], prev_summaries=prev_text, sections_block=sections_block,
        )
    raw = _safe_llm_call_text(llm, prompt)
    parsed = _parse_json_with_repair(_extract_json_text(raw) if raw else '')
    return {
        str(item['title']).strip(): str(item['content'])[:3500]
        for item in (parsed if isinstance(parsed, list) else [])
        if isinstance(item, dict) and item.get('title') is not None and item.get('content') is not None
    }


def _section_cache_key(title: str) -> str:
    return f'arch_section_{re.sub(r"[^a-zA-Z0-9]", "_", title).lower()}'


def _arch_fill_all_sections(
    llm: Any, clone_dir: str, outline: List[Dict[str, Any]], dir_tree_1level: str,
    cache_path: Optional[str] = None, owner_repo: str = '',
) -> str:
    pairs = [(sec, _arch_collect_snippets_for_section(clone_dir, sec)) for sec in outline]
    batch_budget = max(28000, SINGLE_CALL_CONTEXT_BUDGET - 18000)
    batches = _arch_pack_pairs(pairs, batch_budget)
    sections: List[str] = []
    prev_summaries: List[str] = []
    prog = _Progress('Arch: filling sections', len(outline))
    for batch in batches:
        batch_content: Dict[str, str] = {}
        missing: List[Tuple[Dict[str, Any], str]] = []
        for sec, snip in batch:
            title = sec.get('title', 'Section')
            hit = _load_cache(cache_path, _section_cache_key(title))
            if hit:
                batch_content[title] = hit
                prog.update(f'{title} (cached)')
            else:
                missing.append((sec, snip))
        if missing:
            got = _arch_fill_batch_llm(llm, missing, dir_tree_1level, prev_summaries, owner_repo, cache_path)
            for sec, _snip in missing:
                title = sec.get('title', 'Section')
                content = got[title][:3500] if title in got and got[title].strip() else \
                    _arch_fill_section(llm, clone_dir, sec, dir_tree_1level, prev_summaries)
                batch_content[title] = content
                _save_cache(cache_path, _section_cache_key(title), content)
                prog.update(title)
        for sec, _ in batch:
            title = sec.get('title', 'Section')
            content = batch_content[title]
            sections.append(f'[{title}]\n{content}')
            prev_summaries.append(f'{title}: {content[:200]}')
            while sum(len(s) for s in prev_summaries) > _ARCH_PREV_SUMMARY_BUDGET:
                prev_summaries.pop(0)
    prog.done(f'{len(sections)} sections filled')
    return '\n\n'.join(sections)


def _get_sym_pattern(ext: str) -> Optional[re.Pattern]:
    for ext_set, pat, _ in _PUBLIC_SYM_PATTERNS:
        if ext in ext_set:
            return re.compile(pat)
    return None


def _extract_sym_desc(lines: List[str], idx: int) -> str:
    if idx + 1 >= len(lines):
        return ''
    nxt = lines[idx + 1].strip()
    if nxt.startswith('//') or nxt.startswith('#') or nxt.startswith('/*'):
        return re.sub(r'^[/#*\s]+', '', nxt).strip()[:300]
    if nxt.startswith('"""') or nxt.startswith("'''"):
        quote = nxt[:3]
        rest = nxt[3:]
        if quote in rest:
            return rest[:rest.index(quote)].strip()[:300]
        parts = [rest]
        for j in range(idx + 2, min(idx + 10, len(lines))):
            ln = lines[j]
            if quote in ln:
                parts.append(ln[:ln.index(quote)])
                break
            parts.append(ln.rstrip())
        return ' '.join(p.strip() for p in parts if p.strip())[:300]
    return ''


def _scan_file_symbols(lines: List[str], pattern: re.Pattern) -> List[str]:
    entries: List[str] = []
    for i, line in enumerate(lines):
        m = pattern.match(line)
        if not m or m.group(1).startswith('_'):
            continue
        desc = _extract_sym_desc(lines, i)
        sig = line.rstrip()[:120]
        entries.append(f'{sig}: {desc}' if desc else sig)
        if len(entries) >= _PUBLIC_API_MAX_ENTRIES_PER_FILE:
            break
    return entries


def _extract_public_symbols(clone_dir: str, file_entries: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    catalog: Dict[str, List[str]] = {}
    for entry in file_entries[:_PUBLIC_API_MAX_FILES]:
        fpath, scope = entry.get('file', ''), entry.get('scope', 'global') or 'global'
        abs_path = os.path.join(clone_dir, fpath)
        ext = os.path.splitext(fpath)[1].lower()
        if not os.path.isfile(abs_path) or ext not in _PUBLIC_SYM_EXTS:
            continue
        pattern = _get_sym_pattern(ext)
        if pattern is None:
            continue
        try:
            with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        except OSError:
            continue
        if syms := _scan_file_symbols(lines, pattern):
            catalog.setdefault(scope, []).extend(syms)
    return catalog


def _build_public_api_catalog(llm: Any, clone_dir: str, cache_path: Optional[str] = None) -> str:
    cached = _load_cache(cache_path, 'public_api_catalog')
    if cached:
        return cached
    file_entries = _safe_llm_call(llm, _PUBLIC_API_FILES_PROMPT_TMPL.format(
        dir_tree=_build_dir_tree(clone_dir, max_depth=3)[:4000]
    ))
    catalog = _extract_public_symbols(clone_dir, file_entries if isinstance(file_entries, list) else [])
    result = json.dumps(catalog, ensure_ascii=False)
    _save_cache(cache_path, 'public_api_catalog', result)
    return result


def _parse_arch_sections(arch_doc: str) -> List[Tuple[str, str]]:
    sections: List[Tuple[str, str]] = []
    current_title = ''
    current_lines: List[str] = []
    for line in arch_doc.splitlines():
        m = re.match(r'^\[(.+?)\]', line)
        if m:
            if current_title:
                sections.append((current_title, '\n'.join(current_lines).strip()))
            current_title = m.group(1)
            rest = line[m.end():].strip()
            current_lines = [rest] if rest else []
        else:
            current_lines.append(line)
    if current_title:
        sections.append((current_title, '\n'.join(current_lines).strip()))
    return sections


def _build_arch_index(arch_doc: str) -> str:
    sections = _parse_arch_sections(arch_doc)
    if not sections:
        lines = [para.strip().splitlines()[0][:100] for para in arch_doc.split('\n\n') if para.strip()]
        return '\n'.join(lines[:20])
    lines = []
    for title, content in sections:
        for line in content.splitlines():
            if line.strip():
                lines.append(f'[{title}] {line.strip()[:80]}')
                break
    return '\n'.join(lines[:20])


def _get_arch_index(arch_doc: str) -> str:
    return _build_arch_index(arch_doc)[:400]


def _candidate_scopes(file_path: str) -> List[str]:
    parts = file_path.replace('\\', '/').split('/')
    scopes = ['global']
    for i in range(1, len(parts)):
        scopes.append('/'.join(parts[:i]))
    return scopes


def _format_catalog_for_file(catalog_json: str, file_path: str, max_chars: int = 1500) -> str:
    try:
        catalog: Dict[str, List[str]] = json.loads(catalog_json)
    except (json.JSONDecodeError, ValueError):
        return catalog_json[:max_chars]
    candidate = set(_candidate_scopes(file_path))
    lines: List[str] = []
    for scope, entries in catalog.items():
        if scope not in candidate:
            continue
        lines.append(f'[scope: {scope}]')
        lines.extend(f'  {e}' for e in entries[:30])
    result = '\n'.join(lines)
    return result[:max_chars] if result else '(no matching public APIs for this file)'


def _extract_arch_for_file(arch_doc: str, file_path: str, max_chars: int = 3000) -> str:
    if not arch_doc:
        return '(not available)'
    sections = _parse_arch_sections(arch_doc)
    if not sections:
        return arch_doc[:max_chars]

    path_keywords = {p for p in re.split(r'[/\\._]', file_path.lower()) if len(p) > 2}

    def _score(title: str, content: str) -> int:
        t_lower = title.lower()
        return (
            (100 if any(p in t_lower for p in _ARCH_ALWAYS_INJECT) else 0)
            + sum(10 for kw in path_keywords if kw in t_lower + ' ' + content.lower())
        )

    parts: List[str] = []
    remaining = max_chars
    for title, content in sorted(sections, key=lambda sc: _score(sc[0], sc[1]), reverse=True):
        if remaining <= 0:
            break
        if re.match(r'public api catalog', title, re.IGNORECASE):
            filtered = _format_catalog_for_file(content, file_path, max_chars=min(remaining, 1200))
            if filtered and not filtered.startswith('(no matching'):
                block = f'[{title}]\n{filtered}'
                parts.append(block[:remaining])
                remaining -= len(parts[-1]) + 2
            continue
        block = f'[{title}]\n{content}'
        parts.append(block[:remaining])
        remaining -= len(parts[-1]) + 2
    return '\n\n'.join(parts)


def _build_symbol_index(arch_doc: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    sections = _parse_arch_sections(arch_doc)
    for title, content in sections:
        if re.match(r'key utilities', title, re.IGNORECASE):
            for line in content.splitlines():
                m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*[:(]\s*(.+)', line.strip())
                if m:
                    index[m.group(1)] = m.group(2)[:150]
            break
    return index


def _get_symbol_index(arch_doc: str) -> Dict[str, str]:
    return _build_symbol_index(arch_doc)


def analyze_repo_architecture(
    llm: Any, clone_dir: str, cache_path: Optional[str] = None, agent_instructions: str = '',
    clone_url: str = '', base_repo: str = '',
) -> str:
    cached = _load_cache(cache_path, 'arch_doc')
    if cached:
        return cached

    snapshot = _collect_structured_snapshot(clone_dir)

    owner_repo = base_repo or _parse_owner_repo(clone_url)
    if owner_repo:
        lazyllm.LOG.info(f'Fetching DeepWiki summary for {owner_repo}...')
        deepwiki_text = _fetch_deepwiki_summary(owner_repo)
        if deepwiki_text:
            stale_notice = (
                '\n\n> NOTE: DeepWiki data may be 1-3 months stale. '
                'Use as background context only; verify against local code before drawing conclusions.'
            )
            header = '\n\n## DeepWiki Pre-indexed Summary (background reference — may be stale)'
            snapshot = snapshot + header + stale_notice + '\n' + deepwiki_text
            lazyllm.LOG.info(f'DeepWiki summary injected ({len(deepwiki_text)} chars)')
        else:
            lazyllm.LOG.info(f'DeepWiki: no summary available for {owner_repo}')
    dir_tree_1 = _build_dir_tree(clone_dir, max_depth=1)

    outline_cached = _load_cache(cache_path, 'arch_outline')
    try:
        outline = json.loads(outline_cached) if outline_cached else None
    except (json.JSONDecodeError, TypeError):
        outline = None
    if not outline:
        outline = _arch_generate_outline(llm, snapshot, agent_instructions)
        if not outline:
            raise ValueError('Arch outline generation returned empty result')
        _save_cache(cache_path, 'arch_outline', json.dumps(outline, ensure_ascii=False))
    arch_doc = _arch_fill_all_sections(llm, clone_dir, outline, dir_tree_1, cache_path, owner_repo or '') or \
        '(architecture analysis unavailable)'

    try:
        public_api_json = _build_public_api_catalog(llm, clone_dir, cache_path)
        if public_api_json and not public_api_json.startswith('('):
            arch_doc = arch_doc + f'\n\n[Public API Catalog]\n{public_api_json}'
    except Exception as e:
        lazyllm.LOG.warning(f'Public API Catalog generation failed: {e}')

    _save_cache_multi(cache_path, {
        'arch_doc': arch_doc,
        'arch_index': _build_arch_index(arch_doc),
        'arch_symbol_index': _build_symbol_index(arch_doc),
    })
    return arch_doc
