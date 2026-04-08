# Copyright (c) 2026 LazyAGI. All rights reserved.
import json
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import lazyllm

from ..base import LazyLLMGitBase
from .checkpoint import _load_cache, _save_cache, _save_cache_multi
from .utils import _Progress, _safe_llm_call, _safe_llm_call_text

# ---------------------------------------------------------------------------
# Repo code fetching
# ---------------------------------------------------------------------------

_SKIP_DIRS = {'.git', '__pycache__', '.cache', '.tox', 'node_modules', '.mypy_cache', '.pytest_cache', 'dist', 'build'}
_SKIP_EXTS = {'.pyc', '.pyo', '.so', '.egg', '.egg-info'}

# arch analysis budgets (chars)
_ARCH_SNAPSHOT_BUDGET = 6000
_ARCH_OUTLINE_MAX_SECTIONS = 7
_ARCH_SECTION_PROMPT_BUDGET = 3500
_ARCH_SECTION_MAX_RETRIES = 6
_ARCH_PREV_SUMMARY_BUDGET = 600

# bot user filter for review spec
_BOT_USER_PATTERNS = re.compile(
    r'bot$|robot$|\[bot\]|-bot-|autobot|github-actions|dependabot', re.IGNORECASE
)

_LARGE_FILE_THRESHOLD = 600
_CONTEXT_LINES = 300


def _fetch_repo_code(repo_url: str, branch: str, work_dir: Optional[str] = None) -> Tuple[str, str]:
    clone_dir = work_dir or tempfile.mkdtemp(prefix='lazyllm_review_')
    try:
        subprocess.run(
            ['git', 'clone', '--single-branch', '--branch', branch, '--depth', '1', repo_url, clone_dir],
            capture_output=True, text=True, timeout=120, check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'git clone failed: {e.stderr or e.stdout}') from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError('git clone timed out') from e

    tree_lines: List[str] = []
    for root, dirs, files in os.walk(clone_dir):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        rel_root = os.path.relpath(root, clone_dir)
        for fname in sorted(files):
            if any(fname.endswith(ext) for ext in _SKIP_EXTS):
                continue
            rel_path = os.path.join(rel_root, fname) if rel_root != '.' else fname
            tree_lines.append(rel_path)

    return clone_dir, '\n'.join(tree_lines)


def _read_file_context(
    clone_dir: str,
    path: str,
    hunk_start: int,
    hunk_end: int,
) -> str:
    abs_path = os.path.join(clone_dir, path)
    if not os.path.isfile(abs_path):
        return ''
    try:
        with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except OSError:
        return ''
    total = len(lines)
    if total <= _LARGE_FILE_THRESHOLD:
        numbered = ''.join(f'{i + 1:4d} | {ln}' for i, ln in enumerate(lines))
        return f'(full file, {total} lines)\n{numbered}'
    start = max(0, hunk_start - 1 - _CONTEXT_LINES)
    end = min(total, hunk_end + _CONTEXT_LINES)
    numbered = ''.join(f'{start + i + 1:4d} | {ln}' for i, ln in enumerate(lines[start:end]))
    return f'(excerpt lines {start + 1}–{end} of {total})\n{numbered}'


def _resolve_clone_target(pr: Any, base_repo: str) -> Tuple[str, str]:
    raw = getattr(pr, 'raw', None) or {}

    head = raw.get('head') or {}
    head_repo = head.get('repo') or {}
    head_clone_url = (head_repo.get('clone_url')
                      or head_repo.get('http_url_to_repo')
                      or head_repo.get('web_url')
                      or '')
    head_branch = head.get('ref') or head.get('branch') or ''

    base = raw.get('base') or {}
    base_repo_info = base.get('repo') or {}
    base_full_name = base_repo_info.get('full_name') or base_repo
    head_full_name = head_repo.get('full_name') or ''

    def _default_url(r: str) -> str:
        if r.startswith('http'):
            return r if r.endswith('.git') else r + '.git'
        return f'https://github.com/{r}.git'

    if head_clone_url and head_full_name and head_full_name != base_full_name:
        if not head_clone_url.endswith('.git'):
            head_clone_url += '.git'
        branch = head_branch or 'main'
        return head_clone_url, branch

    base_branch = base.get('ref') or getattr(pr, 'target_branch', '') or 'main'
    return _default_url(base_repo), base_branch


# ---------------------------------------------------------------------------
# Architecture analysis — stage 1: structured snapshot
# ---------------------------------------------------------------------------

def _build_dir_tree(clone_dir: str, max_depth: int = 2) -> str:
    lines = []
    base_depth = clone_dir.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(clone_dir):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        depth = root.count(os.sep) - base_depth
        if depth >= max_depth:
            dirs[:] = []
            continue
        indent = '  ' * depth
        # use '.' for root so LLM doesn't construct wrong relative paths
        label = '.' if depth == 0 else os.path.basename(root)
        lines.append(f'{indent}{label}/')
        for fname in sorted(files):
            if not any(fname.endswith(ext) for ext in _SKIP_EXTS):
                lines.append(f'{indent}  {fname}')
    return '\n'.join(lines)


def _read_file_head(path: str, max_bytes: int) -> str:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read(max_bytes)
    except OSError:
        return ''


def _collect_structured_snapshot(clone_dir: str) -> str:
    parts: List[str] = []
    budget = _ARCH_SNAPSHOT_BUDGET

    # 1. two-level directory tree
    tree = _build_dir_tree(clone_dir, max_depth=2)[:800]
    parts.append(f'## Directory Tree\n{tree}')
    budget -= len(parts[-1])

    # 2. top-level __init__.py
    top_init = os.path.join(clone_dir, '__init__.py')
    content = _read_file_head(top_init, 1500)
    if content and budget > 0:
        parts.append(f'## __init__.py\n{content[:min(1500, budget)]}')
        budget -= len(parts[-1])

    # 3. each top-level sub-package __init__.py (300 bytes each)
    try:
        sub_pkgs = sorted(
            d for d in os.listdir(clone_dir)
            if os.path.isdir(os.path.join(clone_dir, d)) and d not in _SKIP_DIRS and not d.startswith('.')
        )
    except OSError:
        sub_pkgs = []
    for pkg in sub_pkgs:
        if budget <= 0:
            break
        init_path = os.path.join(clone_dir, pkg, '__init__.py')
        content = _read_file_head(init_path, 300)
        if content:
            snippet = content[:min(300, budget)]
            parts.append(f'## {pkg}/__init__.py\n{snippet}')
            budget -= len(parts[-1])

    # 4. key base files (400 bytes each)
    key_files = ['module/module.py', 'flow/flow.py', 'components/core.py', 'common/common.py']
    for rel in key_files:
        if budget <= 0:
            break
        fpath = os.path.join(clone_dir, rel)
        content = _read_file_head(fpath, 400)
        if content:
            snippet = content[:min(400, budget)]
            parts.append(f'## {rel}\n{snippet}')
            budget -= len(parts[-1])

    return '\n\n'.join(parts)


# ---------------------------------------------------------------------------
# Architecture analysis — stage 2: outline generation
# ---------------------------------------------------------------------------

_ARCH_OUTLINE_PROMPT = '''\
You are a senior software architect. Based on the project snapshot below, generate an outline \
for an architecture document with {max_sections} sections.

For each section output a JSON object with:
- "title": section name (e.g. "Module Responsibilities")
- "focus": one sentence describing what to cover
- "search_hints": list of 2-3 regex patterns for search_in_files to find relevant code

The LAST section MUST be:
{{"title": "Key Utilities & Usage Notes", "focus": "关键辅助函数、数据结构的典型用法和注意事项", \
"search_hints": ["def _[a-z]", "class.*Dict", "ArgsDict|LazyLLMCMD"]}}

Output ONLY a JSON array. No explanation.

<snapshot>
{snapshot}
</snapshot>
'''


def _arch_generate_outline(llm: Any, snapshot: str) -> List[Dict[str, Any]]:
    prompt = _ARCH_OUTLINE_PROMPT.format(
        max_sections=_ARCH_OUTLINE_MAX_SECTIONS,
        snapshot=snapshot[:4000],
    )
    result = _safe_llm_call(llm, prompt)
    if isinstance(result, list) and result:
        return result[:_ARCH_OUTLINE_MAX_SECTIONS]
    raise ValueError(f'Arch outline generation returned invalid result: {result!r}')


# ---------------------------------------------------------------------------
# Architecture analysis — stage 3: per-section agent fill
# ---------------------------------------------------------------------------

_ARCH_SECTION_PROMPT = '''\
You are analyzing a Python project. Fill in ONE section of the architecture document.

## Current Section
Title: {section_title}
Focus: {section_focus}

## Directory Overview
{dir_tree}

## Already Documented (brief)
{prev_summaries}

## Relevant Code Snippets
{code_snippets}

Based on the code snippets above, write the section content.
- Max 150 words for this section
- Plain text, no markdown headers
- Focus only on: {section_focus}

Output ONLY the section content text.
'''

_ARCH_STATIC_PROMPT_TMPL = '''\
You are a senior software architect. Analyze the following project snapshot and produce a concise \
architecture document covering: module responsibilities, class hierarchies, design patterns, \
public API conventions, key utilities usage, and notable constraints.
Be concise (max 800 words). Output plain text, no markdown headers.

<snapshot>
{snapshot}
</snapshot>
'''


def _build_arch_agent_tools(clone_dir: str) -> list:
    from lazyllm.tools.agent.file_tool import read_file, list_dir, search_in_files
    from lazyllm.tools.agent.shell_tool import shell_tool

    def read_file_scoped(path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> dict:
        '''Read a source file from the cloned repository, with optional line range.

        Args:
            path (str): File path relative to repo root, or absolute path inside clone_dir.
            start_line (int, optional): 1-based start line (inclusive).
            end_line (int, optional): 1-based end line (inclusive).

        Returns:
            dict: File content and metadata.
        '''
        abs_path = path if os.path.isabs(path) else os.path.join(clone_dir, path)
        return read_file(abs_path, start_line=start_line, end_line=end_line, root=clone_dir)

    def search_scoped(pattern: str, glob: Optional[str] = None, max_results: int = 30) -> dict:
        '''Search files in the cloned repository for a regex pattern.

        Args:
            pattern (str): Regex pattern to search for.
            glob (str, optional): Filename glob filter (e.g., "*.py").
            max_results (int, optional): Max number of matches to return. Defaults to 30.

        Returns:
            dict: List of matches with file path and line number.
        '''
        return search_in_files(pattern, path=clone_dir, glob=glob, max_results=max_results, root=clone_dir)

    def list_dir_scoped(path: str = '.', recursive: bool = False) -> dict:
        '''List directory entries inside the cloned repository.

        Args:
            path (str, optional): Path relative to repo root. Defaults to repo root.
            recursive (bool, optional): Whether to walk recursively.

        Returns:
            dict: List of entries.
        '''
        abs_path = path if os.path.isabs(path) else os.path.join(clone_dir, path)
        return list_dir(abs_path, recursive=recursive, root=clone_dir)

    def shell_scoped(cmd: str, timeout: int = 30) -> dict:
        '''Run a read-only shell command (grep, find, git log, etc.) in the cloned repository.
        Do NOT use write commands (rm, mv, git commit, etc.).

        Args:
            cmd (str): The shell command to execute.
            timeout (int, optional): Timeout in seconds. Defaults to 30.

        Returns:
            dict: Execution result including stdout, stderr, exit_code.
        '''
        return shell_tool(cmd, cwd=clone_dir, timeout=timeout, allow_unsafe=False)

    return [read_file_scoped, search_scoped, list_dir_scoped, shell_scoped]


def _summarize_section(llm: Any, title: str, content: str) -> str:
    prompt = f'Summarize in 1-2 sentences (max 200 chars):\n[{title}]\n{content[:800]}'
    result = _safe_llm_call_text(llm, prompt) or content[:100]
    return result[:200]


def _arch_collect_snippets(clone_dir: str, section: Dict[str, Any], max_chars: int = 2000) -> str:
    from lazyllm.tools.agent.file_tool import search_in_files, read_file
    hints = section.get('search_hints', [])
    parts: List[str] = []
    seen_paths: set = set()
    for pattern in hints:
        try:
            result = search_in_files(pattern, path=clone_dir, glob='*.py', max_results=8, root=clone_dir)
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
                fc = read_file(path, start_line=max(1, line - 2), end_line=line + 20, root=clone_dir)
                snippet = fc.get('content', '') if isinstance(fc, dict) else ''
            except Exception:
                snippet = m.get('text', '')
            if snippet:
                rel = os.path.relpath(path, clone_dir)
                parts.append(f'# {rel}\n{snippet}')
        if sum(len(p) for p in parts) >= max_chars:
            break
    combined = '\n\n'.join(parts)
    return combined[:max_chars] if combined else '(no relevant snippets found)'


def _arch_fill_section(
    llm: Any, clone_dir: str, section: Dict[str, Any],
    dir_tree_1level: str, prev_summaries: List[str],
) -> str:
    prev_text = '\n'.join(prev_summaries) if prev_summaries else '(none yet)'
    if len(prev_text) > _ARCH_PREV_SUMMARY_BUDGET:
        prev_text = prev_text[-_ARCH_PREV_SUMMARY_BUDGET:]
    snippets = _arch_collect_snippets(clone_dir, section)
    prompt = _ARCH_SECTION_PROMPT.format(
        section_title=section.get('title', ''),
        section_focus=section.get('focus', ''),
        dir_tree=dir_tree_1level[:400],
        prev_summaries=prev_text,
        code_snippets=snippets[:2000],
    )
    raw = _safe_llm_call_text(llm, prompt)
    if not raw:
        raise ValueError(f'LLM returned empty result for section "{section.get("title")}"')
    return raw[:1200]


def _arch_fill_all_sections(
    llm: Any, clone_dir: str, outline: List[Dict[str, Any]], dir_tree_1level: str
) -> str:
    sections: List[str] = []
    prev_summaries: List[str] = []
    prog = _Progress('Arch: filling sections', len(outline))
    for sec in outline:
        content = _arch_fill_section(llm, clone_dir, sec, dir_tree_1level, prev_summaries)
        title = sec.get('title', 'Section')
        sections.append(f'[{title}]\n{content}')
        summary = _summarize_section(llm, title, content)
        prev_summaries.append(f'{title}: {summary}')
        # trim oldest summaries when over budget
        while sum(len(s) for s in prev_summaries) > _ARCH_PREV_SUMMARY_BUDGET:
            prev_summaries.pop(0)
        prog.update(title)
    prog.done(f'{len(sections)} sections filled')
    return '\n\n'.join(sections)


# ---------------------------------------------------------------------------
# Architecture analysis — index & symbol extraction
# ---------------------------------------------------------------------------

def _build_arch_index(arch_doc: str) -> str:
    lines = []
    current_title = ''
    for line in arch_doc.splitlines():
        m = re.match(r'^\[(.+?)\]', line)
        if m:
            current_title = m.group(1)
            rest = line[m.end():].strip()
            if rest:
                lines.append(f'[{current_title}] {rest[:80]}')
        elif current_title and line.strip() and not lines or (lines and not lines[-1].startswith(f'[{current_title}]')):
            lines.append(f'[{current_title}] {line.strip()[:80]}')
    if not lines:
        # fallback: take first sentence of each paragraph
        for para in arch_doc.split('\n\n'):
            first = para.strip().splitlines()[0] if para.strip() else ''
            if first:
                lines.append(first[:100])
    return '\n'.join(lines[:20])


def _get_arch_index(arch_doc: str) -> str:
    return _build_arch_index(arch_doc)[:400]


def _build_symbol_index(arch_doc: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    in_utilities = False
    for line in arch_doc.splitlines():
        if re.match(r'^\[Key Utilities', line, re.IGNORECASE):
            in_utilities = True
            continue
        if re.match(r'^\[', line) and in_utilities:
            break
        if in_utilities and line.strip():
            # expect lines like "symbol_name: description"
            m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*[:(]\s*(.+)', line.strip())
            if m:
                index[m.group(1)] = m.group(2)[:150]
    return index


def _get_symbol_index(arch_doc: str) -> Dict[str, str]:
    return _build_symbol_index(arch_doc)


def analyze_repo_architecture(
    llm: Any, clone_dir: str, cache_path: Optional[str] = None
) -> str:
    cached = _load_cache(cache_path, 'arch_doc')
    if cached:
        return cached

    snapshot = _collect_structured_snapshot(clone_dir)
    dir_tree_1 = _build_dir_tree(clone_dir, max_depth=1)

    outline = _arch_generate_outline(llm, snapshot)
    if not outline:
        raise ValueError('Arch outline generation returned empty result')
    arch_doc = _arch_fill_all_sections(llm, clone_dir, outline, dir_tree_1)

    arch_doc = arch_doc or '(architecture analysis unavailable)'
    # save full doc + index + symbol index in same cache file
    _save_cache_multi(cache_path, {
        'arch_doc': arch_doc,
        'arch_index': _build_arch_index(arch_doc),
        'arch_symbol_index': _build_symbol_index(arch_doc),
    })
    return arch_doc


# ---------------------------------------------------------------------------
# Historical review spec analysis
# ---------------------------------------------------------------------------

# max comments per user group kept after compression
_MAX_COMMENTS_PER_USER = 15
# compress individual comment bodies longer than this
_COMMENT_COMPRESS_THRESHOLD = 150
# max rules kept in final spec
_SPEC_MAX_RULES = 15


def _is_merged_pr(pr: Any) -> bool:
    raw = getattr(pr, 'raw', None) or (pr if isinstance(pr, dict) else {})
    # GitHub / Gitee: merged_at; GitLab: merged_at or state == 'merged'
    if raw.get('merged_at'):
        return True
    if raw.get('state') == 'merged':
        return True
    return False


_EXTRACT_RULES_PROMPT = '''\
You are a code review expert. The following are HUMAN review comments from a single pull request.
Extract concrete, actionable review rules from these comments.

For each rule found, output a JSON object with these fields:
- "rule_id": string like "ERR001", "STY002", "DSN003" (ERR=error/exception, STY=style, DSN=design, \
PERF=performance, SEC=security, XFILE=cross-file consistency)
- "title": short title (max 8 words)
- "severity": "P0" | "P1" | "P2"  (P0=must fix, P1=should fix, P2=nice to have)
- "detect": list of strings describing how to detect this issue (patterns, keywords, conditions)
- "bad_example": short code snippet showing the bad pattern (or "" if not applicable)
- "good_example": short code snippet showing the correct pattern (or "" if not applicable)
- "fix": one-sentence fix suggestion

Pay special attention to cross-file consistency issues (use rule_id prefix "XFILE"):
- Interface changed but callers not updated
- Symmetric methods (encode/decode, open/close) only one side updated
- Registry/factory pattern: new entry added but docs/tests not updated
- Abstract method added to base class but not implemented in subclasses

Output ONLY a JSON array. If no clear rules can be extracted: output [].

<review_comments>
{comments_text}
</review_comments>
'''

_MERGE_RULES_PROMPT = '''\
You are a code review expert. Below are rule cards extracted from multiple pull requests.
Merge duplicate or highly similar rules into one (keep the most informative example).
Remove rules that are too vague or project-unspecific.
Sort by severity (P0 first), then by frequency (most common first).
Keep at most {max_rules} rules total.

For each final rule, output a JSON object with:
- "rule_id", "title", "severity", "detect" (list), "bad_example", "good_example", "fix"

Output ONLY a JSON array.

<rule_cards>
{rules_json}
</rule_cards>
'''

_RULE_CARD_TEMPLATE = '''\
[Rule ID] {rule_id}
[Title] {title}
[Severity] {severity}

[Detect]
{detect_bullets}

[Bad Example]
{bad_example}

[Good Example]
{good_example}

[Auto Fix Suggestion]
- {fix}'''


def _format_rule_card(rule: Dict[str, Any]) -> str:
    detect = rule.get('detect') or []
    detect_bullets = '\n'.join(f'- {d}' for d in detect) if detect else '- (see bad example)'
    return _RULE_CARD_TEMPLATE.format(
        rule_id=rule.get('rule_id', 'RULE000'),
        title=rule.get('title', ''),
        severity=rule.get('severity', 'P2'),
        detect_bullets=detect_bullets,
        bad_example=rule.get('bad_example') or '(n/a)',
        good_example=rule.get('good_example') or '(n/a)',
        fix=rule.get('fix') or '',
    )


def _compress_comments_for_pr(
    llm: Any, comments: List[Dict[str, Any]]
) -> str:
    # compress long comments to one sentence each
    indexed = [{'idx': i, 'body': c['body']} for i, c in enumerate(comments)]
    long_items = [it for it in indexed if len(it['body']) > _COMMENT_COMPRESS_THRESHOLD]
    summaries: Dict[int, str] = {}
    if long_items:
        batch_input = [{'idx': it['idx'], 'body': it['body'][:800]} for it in long_items]
        compress_prompt = (
            'Summarize each review comment into ONE sentence (max 20 words). '
            'Preserve: what is wrong and why.\n'
            'Output a JSON array, each item: {"idx": <same>, "summary": "<one sentence>"}.\n'
            'Output ONLY the JSON array.\n\n'
            + json.dumps(batch_input, ensure_ascii=False, indent=2)
        )
        try:
            result = _safe_llm_call(llm, compress_prompt)
            for r in (result if isinstance(result, list) else []):
                if isinstance(r, dict) and 'idx' in r and 'summary' in r:
                    summaries[int(r['idx'])] = str(r['summary'])
        except Exception as e:
            lazyllm.LOG.warning(f'Comment compression failed: {e}')
    lines = []
    for item in indexed:
        body = summaries.get(item['idx']) or item['body'][:_COMMENT_COMPRESS_THRESHOLD]
        user = comments[item['idx']].get('user', '')
        lines.append(f'[{user}]: {body}' if user else body)
    return '\n'.join(lines)


def _extract_rules_from_pr_comments(
    llm: Any, pr_num: int, comments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not comments:
        return []
    comments_text = _compress_comments_for_pr(llm, comments)
    prompt = _EXTRACT_RULES_PROMPT.format(comments_text=comments_text)
    try:
        result = _safe_llm_call(llm, prompt)
        return [r for r in (result if isinstance(result, list) else []) if isinstance(r, dict)]
    except Exception as e:
        lazyllm.LOG.warning(f'Rule extraction for PR #{pr_num} failed: {e}')
        return []


def _merge_rule_cards(
    llm: Any, all_rules: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not all_rules:
        return []
    rules_json = json.dumps(all_rules, ensure_ascii=False, indent=2)[:8000]
    prompt = _MERGE_RULES_PROMPT.format(rules_json=rules_json, max_rules=_SPEC_MAX_RULES)
    try:
        result = _safe_llm_call(llm, prompt)
        return [r for r in (result if isinstance(result, list) else []) if isinstance(r, dict)]
    except Exception as e:
        lazyllm.LOG.warning(f'Rule merge failed: {e}')
        # fallback: deduplicate by title, keep first occurrence
        seen: set = set()
        deduped = []
        for r in all_rules:
            key = r.get('title', '')
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        return deduped[:_SPEC_MAX_RULES]


def analyze_historical_reviews(
    backend: LazyLLMGitBase, llm: Any, cache_path: Optional[str] = None, max_prs: int = 200
) -> str:
    cached = _load_cache(cache_path, 'review_spec')
    if cached:
        return cached

    pr_list_res = backend.list_pull_requests(state='closed')
    if not pr_list_res.get('success'):
        return '(historical review analysis unavailable)'
    prs = pr_list_res.get('list') or []

    # filter to merged PRs first
    merged = [p for p in prs if _is_merged_pr(p)]
    target = merged[:max_prs] if merged else prs[:max_prs]
    total = len(target)

    if not target:
        return '(no historical review comments found)'

    prog = _Progress('Spec: extracting rules from historical PRs', total)
    all_rules: List[Dict[str, Any]] = []

    for idx, pr in enumerate(target, 1):
        pr_num = getattr(pr, 'number', None) or (pr.get('number') if isinstance(pr, dict) else None)
        if pr_num is None:
            prog.update(f'[{idx}/{total}] skipped (no number)')
            continue
        res = backend.list_review_comments(pr_num)
        if not res.get('success'):
            prog.update(f'[{idx}/{total}] PR #{pr_num} (fetch failed)')
            continue
        comments = []
        for c in (res.get('comments') or []):
            body = (c.get('body') if isinstance(c, dict) else getattr(c, 'body', '')) or ''
            user = (c.get('user') if isinstance(c, dict) else getattr(c, 'user', '')) or ''
            if not body.strip() or _BOT_USER_PATTERNS.search(user):
                continue
            comments.append({'user': user, 'body': body})
        if not comments:
            prog.update(f'[{idx}/{total}] PR #{pr_num} — 0 human comments → skipped')
            continue
        rules = _extract_rules_from_pr_comments(llm, pr_num, comments)
        all_rules.extend(rules)
        prog.update(f'[{idx}/{total}] PR #{pr_num} — {len(comments)} comments → {len(rules)} rules extracted')

    prog.done(f'{len(all_rules)} raw rules from {total} PRs, merging...')

    if not all_rules:
        return '(no historical review comments found)'

    merged_rules = _merge_rule_cards(llm, all_rules)
    review_spec = '\n\n'.join(_format_rule_card(r) for r in merged_rules)
    review_spec = review_spec or '(review spec analysis unavailable)'
    _save_cache(cache_path, 'review_spec', review_spec)
    return review_spec


# ---------------------------------------------------------------------------
# Pre-round: PR change summary
# ---------------------------------------------------------------------------

_PRE_ROUND_PROMPT_TMPL = '''\
You are a senior code reviewer. Summarize the following pull request diff concisely.
{lang_instruction}

## PR Title
{pr_title}

## PR Description
{pr_body}

## Diff (may be truncated)
{diff_text}

Produce a structured summary covering:
1. What is the purpose of this PR? (1-2 sentences)
2. Which files/modules are changed and why?
3. Key design decisions or trade-offs visible in the diff
4. Potential risk areas that deserve extra scrutiny

Be concise (max 400 words). Output plain text, no markdown headers.
'''


def _pre_round_pr_summary(
    llm: Any,
    pr_title: str,
    pr_body: str,
    diff_text: str,
    language: str = 'cn',
) -> str:
    from .utils import _language_instruction
    prog = _Progress('Pre-round: summarizing PR changes')
    prompt = _PRE_ROUND_PROMPT_TMPL.format(
        lang_instruction=_language_instruction(language),
        pr_title=pr_title or '(no title)',
        pr_body=(pr_body or '(no description)')[:800],
        diff_text=diff_text[:5000] if diff_text else '',
    )
    summary = _safe_llm_call_text(llm, prompt) or '(PR summary unavailable)'
    prog.done(f'{len(summary)} chars')
    return summary


# ---------------------------------------------------------------------------
# Orchestration: run pre-analysis phase
# ---------------------------------------------------------------------------

def _run_arch_analysis(
    llm: Any, pr: Any, repo: str, arch_cache_path: str, ckpt: Any
) -> Tuple[str, Optional[str]]:
    arch_doc = ckpt.get('arch_doc') or ''
    clone_dir: Optional[str] = None
    prog = _Progress('Pre-analysis: fetch repo & analyze architecture')
    try:
        clone_url, branch = _resolve_clone_target(pr, repo)
        lazyllm.LOG.info(f'Cloning {clone_url} @ {branch}')
        clone_dir, _file_tree = _fetch_repo_code(clone_url, branch)
        prog.update('cloned, analyzing...')
        arch_doc = analyze_repo_architecture(llm, clone_dir, arch_cache_path)
        ckpt.save('arch_doc', arch_doc)
        if arch_doc and arch_cache_path:
            lazyllm.LOG.success(f'Architecture doc saved to: {arch_cache_path}')
    except Exception as e:
        lazyllm.LOG.warning(f'Pre-analysis fetch/arch failed: {e}')
        if clone_dir and os.path.isdir(clone_dir):
            import shutil
            shutil.rmtree(clone_dir, ignore_errors=True)
        clone_dir = None
    prog.done('architecture doc ready' if arch_doc else 'skipped (error)')
    return arch_doc, clone_dir


def _run_spec_analysis(
    backend_inst: LazyLLMGitBase, llm: Any,
    review_spec_cache_path: str, max_history_prs: int, ckpt: Any
) -> str:
    review_spec = ckpt.get('review_spec') or ''
    if not review_spec:
        try:
            review_spec = analyze_historical_reviews(backend_inst, llm, review_spec_cache_path, max_history_prs)
            ckpt.save('review_spec', review_spec)
            if review_spec and review_spec_cache_path:
                lazyllm.LOG.success(f'Review spec saved to: {review_spec_cache_path}')
        except Exception as e:
            lazyllm.LOG.warning(f'Historical review analysis failed: {e}')
    else:
        _save_cache(review_spec_cache_path, 'review_spec', review_spec)
        _Progress('Pre-analysis: review spec').done('loaded from checkpoint')
    return review_spec


def _run_pre_analysis(
    llm: Any,
    backend_inst: LazyLLMGitBase,
    repo: str,
    pr: Any,
    fetch_repo_code: bool,
    arch_cache_path: Optional[str],
    review_spec_cache_path: Optional[str],
    max_history_prs: int,
    ckpt: Any,
) -> Tuple[str, str, Optional[str]]:
    from .checkpoint import _ReviewCheckpoint
    safe_repo = re.sub(r'[^a-zA-Z0-9_-]', '_', repo)
    cache_dir = _ReviewCheckpoint.review_cache_dir()
    if arch_cache_path is None:
        arch_cache_path = os.path.join(cache_dir, f'arch_{safe_repo}.json')
    if review_spec_cache_path is None:
        review_spec_cache_path = os.path.join(cache_dir, f'spec_{safe_repo}.json')

    arch_doc = ckpt.get('arch_doc') or ''
    clone_dir: Optional[str] = None

    if fetch_repo_code and not arch_doc:
        arch_doc, clone_dir = _run_arch_analysis(llm, pr, repo, arch_cache_path, ckpt)
    elif fetch_repo_code:
        # arch_doc already cached — still clone so Round 2 agent can read source files
        _save_cache(arch_cache_path, 'arch_doc', arch_doc)
        _Progress('Pre-analysis: architecture').done('loaded from checkpoint')
        try:
            clone_url, branch = _resolve_clone_target(pr, repo)
            lazyllm.LOG.info(f'Cloning {clone_url} @ {branch} for agent file access')
            clone_dir, _ = _fetch_repo_code(clone_url, branch)
        except Exception as e:
            lazyllm.LOG.warning(f'Clone for agent failed: {e}')
    else:
        _save_cache(arch_cache_path, 'arch_doc', arch_doc)
        _Progress('Pre-analysis: architecture').done('loaded from checkpoint')

    review_spec = _run_spec_analysis(backend_inst, llm, review_spec_cache_path, max_history_prs, ckpt)
    return arch_doc, review_spec, clone_dir
