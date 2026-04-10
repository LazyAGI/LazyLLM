# Copyright (c) 2026 LazyAGI. All rights reserved.
import json
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import lazyllm

from ..base import LazyLLMGitBase
from .checkpoint import _load_cache, _save_cache, _save_cache_multi, ReviewStage
from .utils import _Progress, _safe_llm_call, _safe_llm_call_text

# ---------------------------------------------------------------------------
# Repo code fetching
# ---------------------------------------------------------------------------

_SKIP_DIRS = {'.git', '__pycache__', '.cache', '.tox', 'node_modules', '.mypy_cache', '.pytest_cache', 'dist', 'build'}
_SKIP_EXTS = {'.pyc', '.pyo', '.so', '.egg', '.egg-info'}

# arch analysis budgets (chars)
_ARCH_SNAPSHOT_BUDGET = 6000
_ARCH_OUTLINE_MAX_SECTIONS = 12
_ARCH_OUTLINE_MAX_SECTIONS_WITH_AGENT = 9
_ARCH_PREV_SUMMARY_BUDGET = 1500

# agent instruction files to scan (in priority order)
_AGENT_INSTRUCTION_FILES = [
    'AGENTS.md', 'AGENTS.override.md', 'CLAUDE.md',
    'GEMINI.md', '.cursorrules', 'CONTRIBUTING.md',
]
_AGENT_INSTRUCTIONS_MAX_CHARS = 8000

# bot user filter for review spec
_BOT_USER_PATTERNS = re.compile(
    r'bot$|robot$|\[bot\]|-bot-|autobot|github-actions|dependabot', re.IGNORECASE
)

_LARGE_FILE_THRESHOLD = 600
_CONTEXT_LINES = 50


def _read_agent_instructions(clone_dir: str) -> str:
    parts = []
    for fname in _AGENT_INSTRUCTION_FILES:
        fpath = os.path.join(clone_dir, fname)
        content = _read_file_head(fpath, _AGENT_INSTRUCTIONS_MAX_CHARS)
        if content:
            parts.append(f'### {fname}\n{content}')
    combined = '\n\n'.join(parts)
    return combined[:_AGENT_INSTRUCTIONS_MAX_CHARS]


def _is_complete_clone(clone_dir: str) -> bool:
    try:
        result = subprocess.run(
            ['git', '-C', clone_dir, 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _try_pull_if_outdated(clone_dir: str, branch: str) -> bool:
    '''Fetch remote and fast-forward if the local HEAD is behind. Returns True if updated.'''
    try:
        fetch = subprocess.run(
            ['git', '-C', clone_dir, 'fetch', '--depth', '1', 'origin', branch],
            capture_output=True, text=True, timeout=60,
        )
        if fetch.returncode != 0:
            lazyllm.LOG.warning(f'git fetch failed: {fetch.stderr.strip()}')
            return False
        local = subprocess.run(
            ['git', '-C', clone_dir, 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        remote = subprocess.run(
            ['git', '-C', clone_dir, 'rev-parse', 'FETCH_HEAD'],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if local == remote:
            lazyllm.LOG.info(f'Clone is up-to-date at {local[:8]}')
            return False
        subprocess.run(
            ['git', '-C', clone_dir, 'reset', '--hard', 'FETCH_HEAD'],
            capture_output=True, text=True, timeout=30, check=True,
        )
        lazyllm.LOG.info(f'Clone updated: {local[:8]} → {remote[:8]}')
        return True
    except Exception as e:
        lazyllm.LOG.warning(f'Failed to pull latest changes: {e}')
        return False


def _fetch_repo_code(repo_url: str, branch: str, work_dir: Optional[str] = None) -> Tuple[str, str]:
    import shutil
    clone_dir = work_dir or tempfile.mkdtemp(prefix='lazyllm_review_')
    if os.path.isdir(clone_dir):
        if _is_complete_clone(clone_dir):
            lazyllm.LOG.info(f'Reusing existing clone at {clone_dir}, checking for updates...')
            _try_pull_if_outdated(clone_dir, branch)
        else:
            # incomplete or broken clone — wipe and retry
            shutil.rmtree(clone_dir, ignore_errors=True)
    try:
        if not _is_complete_clone(clone_dir):
            subprocess.run(
                ['git', 'clone', '--single-branch', '--branch', branch, '--depth', '1', '--', repo_url, clone_dir],
                capture_output=True, text=True, timeout=300, check=True,
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


def _find_enclosing_scope(lines: List[str], hunk_start: int) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    '''Scan upward from hunk_start to find the enclosing class or function.
    Returns (scope_line_idx, scope_kind, scope_name), scope_line_idx is 0-based.
    '''
    for i in range(min(hunk_start - 1, len(lines) - 1), -1, -1):
        m = re.match(r'^(\s*)(class|def)\s+(\w+)', lines[i])
        if m:
            return i, m.group(2), m.group(3)
    return None, None, None


def _find_enclosing_class(lines: List[str], from_idx: int) -> Optional[int]:
    '''Walk upward from from_idx to find the nearest class definition at a lower indent.'''
    if from_idx <= 0:
        return None
    ref_indent = len(lines[from_idx]) - len(lines[from_idx].lstrip())
    for i in range(from_idx - 1, -1, -1):
        m = re.match(r'^(\s*)class\s+(\w+)', lines[i])
        if m:
            if len(m.group(1)) < ref_indent:
                return i
    return None


def _extract_class_method_signatures(lines: List[str], class_line_idx: int) -> List[str]:
    '''Extract all direct method signatures of the class at class_line_idx.'''
    if class_line_idx < 0 or class_line_idx >= len(lines):
        return []
    class_indent = len(lines[class_line_idx]) - len(lines[class_line_idx].lstrip())
    sigs: List[str] = []
    for i in range(class_line_idx + 1, len(lines)):
        line = lines[i]
        stripped = line.lstrip()
        if not stripped:
            continue
        indent = len(line) - len(stripped)
        if indent <= class_indent and not stripped.startswith('#'):
            break
        if re.match(r'def\s+\w+', stripped) and indent in (class_indent + 4, class_indent + 2):
            sig = stripped.rstrip()
            if ')' not in sig:
                for j in range(i + 1, min(i + 5, len(lines))):
                    sig += ' ' + lines[j].strip()
                    if ')' in lines[j]:
                        break
            sigs.append(f'  {sig.split(chr(10))[0][:120]}')
    return sigs


def _extract_module_function_signatures(lines: List[str]) -> List[str]:
    '''Extract top-level function signatures (def at indent 0).'''
    return [line.rstrip()[:120] for line in lines if re.match(r'^def\s+\w+', line)]


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
        base = f'(full file, {total} lines)\n{numbered}'
    else:
        start = max(0, hunk_start - 1 - _CONTEXT_LINES)
        end = min(total, hunk_end + _CONTEXT_LINES)
        numbered = ''.join(f'{start + i + 1:4d} | {ln}' for i, ln in enumerate(lines[start:end]))
        base = f'(excerpt lines {start + 1}–{end} of {total})\n{numbered}'

    # --- enclosing scope annotation ---
    scope_idx, scope_kind, scope_name = _find_enclosing_scope(lines, hunk_start)
    if scope_idx is None:
        return base

    scope_line_no = scope_idx + 1
    extras: List[str] = [f'\n[Enclosing scope: {scope_kind} {scope_name} (line {scope_line_no})]']

    if scope_kind == 'def':
        class_idx = _find_enclosing_class(lines, scope_idx)
        if class_idx is not None:
            cm = re.match(r'^\s*class\s+(\w+)', lines[class_idx])
            class_name = cm.group(1) if cm else '?'
            extras[0] += f' inside class {class_name} (line {class_idx + 1})'
            sigs = _extract_class_method_signatures(lines, class_idx)
            if sigs:
                extras.append('[Sibling method signatures of enclosing class]')
                extras.extend(sigs)
        else:
            sigs = _extract_module_function_signatures(lines)
            sigs = [s for s in sigs if not re.match(rf'^def\s+{re.escape(scope_name)}\s*\(', s)]
            if sigs:
                extras.append('[Other top-level function signatures in this file]')
                extras.extend(sigs[:20])
    else:
        # enclosing scope is a class
        sigs = _extract_class_method_signatures(lines, scope_idx)
        if sigs:
            extras.append('[Method signatures of enclosing class]')
            extras.extend(sigs)

    return base + '\n' + '\n'.join(extras)


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


def _collect_structured_snapshot(clone_dir: str) -> str:  # noqa: C901
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

    # 5. dependency / build files (500 bytes each)
    dep_files = [
        ('setup.py', 500), ('pyproject.toml', 500),
        ('requirements.txt', 500), ('requirements-dev.txt', 300),
        ('CMakeLists.txt', 300), ('Makefile', 200),
    ]
    for rel, limit in dep_files:
        if budget <= 0:
            break
        fpath = os.path.join(clone_dir, rel)
        content = _read_file_head(fpath, limit)
        if content:
            snippet = content[:min(limit, budget)]
            parts.append(f'## {rel}\n{snippet}')
            budget -= len(parts[-1])

    # 6. agent instruction files (up to 2000 chars each, highest priority for outline generation)
    for rel in _AGENT_INSTRUCTION_FILES:
        if budget <= 0:
            break
        fpath = os.path.join(clone_dir, rel)
        content = _read_file_head(fpath, 2000)
        if content:
            snippet = content[:min(2000, budget)]
            parts.append(f'## {rel} (agent instructions)\n{snippet}')
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

The FIRST section MUST be:
{{"title": "Module Hierarchy", "focus": "模块分层结构：底层基础模块、中间层、上层业务模块，以及明确禁止的依赖方向（底层不得感知上层）", \
"search_hints": ["^from lazyllm", "^import lazyllm", "from \\."]}}

The SECOND section MUST be:
{{"title": "Environment & Dependencies",
 "focus": "Python/compiler version requirements, key dependency packages and version constraints",
 "search_hints": ["python_requires", "install_requires", "cmake_minimum_required"]}}

{gotchas_instruction}

The LAST section MUST be:
{{"title": "Key Utilities & Usage Notes", "focus": "关键辅助函数、数据结构的典型用法和注意事项", \
"search_hints": ["def _[a-z]", "class.*Dict", "ArgsDict|LazyLLMCMD"]}}

Output ONLY a JSON array. No explanation.

<snapshot>
{snapshot}
</snapshot>
'''

_ARCH_GOTCHAS_INSTRUCTION = '''\
The SECOND-TO-LAST section MUST be:
{{"title": "Non-Obvious Behaviors & Gotchas", \
"focus": "初始值、全局状态、线程安全约定、注册系统行为、容易被误解的设计决策（如某字段永不为 None 的保证）", \
"search_hints": ["__global_attrs__", "ThreadSafeDict", "once_wrapper", "LazyLLMRegisterMeta"]}}'''

_ARCH_HAS_AGENT_INSTRUCTION = '''\
NOTE: This project already has an AGENTS.md (or equivalent) covering conventions and gotchas. \
Focus sections on module structure, class hierarchies, cross-module dependency rules, and design \
patterns instead. Do NOT generate a "Gotchas" section — that is already covered by AGENTS.md.'''


def _arch_generate_outline(
    llm: Any, snapshot: str, agent_instructions: str = '',
) -> List[Dict[str, Any]]:
    has_agent = bool(agent_instructions)
    max_sections = _ARCH_OUTLINE_MAX_SECTIONS_WITH_AGENT if has_agent else _ARCH_OUTLINE_MAX_SECTIONS
    gotchas_instruction = _ARCH_HAS_AGENT_INSTRUCTION if has_agent else _ARCH_GOTCHAS_INSTRUCTION
    prompt = _ARCH_OUTLINE_PROMPT.format(
        max_sections=max_sections,
        gotchas_instruction=gotchas_instruction,
        snapshot=snapshot[:4000],
    )
    result = _safe_llm_call(llm, prompt)
    if isinstance(result, list) and result:
        return result[:max_sections]
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
- Max 500 words for this section
- Plain text, no markdown headers
- Focus only on: {section_focus}
- Include key class names, function signatures, and usage patterns where relevant

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


def _build_scoped_agent_tools(clone_dir: str) -> list:  # noqa: C901
    from lazyllm.tools.agent.file_tool import read_file, list_dir, search_in_files
    from lazyllm.tools.agent.shell_tool import shell_tool

    def read_file_scoped(path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> dict:
        '''Read a source file from the cloned repository.

        Args:
            path (str): File path relative to repo root.
            start_line (int, optional): 1-based start line (inclusive).
            end_line (int, optional): 1-based end line (inclusive).

        Returns:
            dict: File content and metadata.
        '''
        abs_path = path if os.path.isabs(path) else os.path.join(clone_dir, path)
        line_info = f':{start_line}-{end_line}' if (start_line or end_line) else ''
        lazyllm.LOG.info(f'  [Agent] Read {path}{line_info}')
        return read_file(abs_path, start_line=start_line, end_line=end_line, root=clone_dir)

    def search_scoped(pattern: str, glob: Optional[str] = None, max_results: int = 40) -> dict:
        '''Search files in the cloned repository for a regex pattern.

        Args:
            pattern (str): Regex pattern to search for.
            glob (str, optional): Filename glob filter (e.g., "*.py").
            max_results (int, optional): Max number of matches to return. Defaults to 40.

        Returns:
            dict: List of matches with file path and line number.
        '''
        lazyllm.LOG.info(f'  [Agent] Search {pattern!r}' + (f' in {glob}' if glob else ''))
        return search_in_files(pattern, path=clone_dir, glob=glob, max_results=max_results, root=clone_dir)

    def list_dir_scoped(path: str = '.', recursive: bool = False) -> dict:
        '''List directory entries inside the cloned repository.

        Args:
            path (str, optional): Path relative to repo root. Defaults to repo root.
            recursive (bool, optional): Whether to walk recursively.

        Returns:
            dict: List of entries.
        '''
        lazyllm.LOG.info(f'  [Agent] ListDir {path}' + (' (recursive)' if recursive else ''))
        abs_path = path if os.path.isabs(path) else os.path.join(clone_dir, path)
        return list_dir(abs_path, recursive=recursive, root=clone_dir)

    def shell_scoped(cmd: str, timeout: int = 30) -> dict:
        '''Run a read-only shell command (grep, find, git log, etc.) in the cloned repository.

        Args:
            cmd (str): The shell command to execute.
            timeout (int, optional): Timeout in seconds. Defaults to 30.

        Returns:
            dict: Execution result including stdout, stderr, exit_code.
        '''
        lazyllm.LOG.info(f'  [Agent] Shell {cmd!r}')
        return shell_tool(cmd, cwd=clone_dir, timeout=timeout, allow_unsafe=False)

    def read_files_batch(paths: str) -> dict:
        '''Read multiple source files at once. Pass a comma-separated list of relative paths.
        More efficient than calling read_file_scoped multiple times.

        Args:
            paths (str): Comma-separated file paths relative to repo root (e.g. "a.py,b.py,c.py").

        Returns:
            dict: Mapping of path -> content (truncated to 800 chars each). Missing files are noted.
        '''
        path_list = [p.strip() for p in paths.split(',') if p.strip()]
        lazyllm.LOG.info(f'  [Agent] ReadBatch {path_list}')
        results = {}
        for p in path_list[:6]:  # cap at 6 files to avoid context explosion
            abs_path = p if os.path.isabs(p) else os.path.join(clone_dir, p)
            if not os.path.isfile(abs_path):
                results[p] = '(not found)'
                continue
            try:
                with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read(1600)
                results[p] = content
            except OSError:
                results[p] = '(read error)'
        return {'files': results, 'count': len(results)}

    def grep_callers(symbol: str, max_results: int = 20) -> dict:
        '''Find all call sites of a function or class in the repository.
        Faster and more precise than search_scoped for finding callers.
        Searches all source files regardless of language.

        Args:
            symbol (str): Function or class name to search for (e.g. "MyClass" or "my_func").
            max_results (int, optional): Max number of results. Defaults to 20.

        Returns:
            dict: List of matches with file, line, and context snippet.
        '''
        lazyllm.LOG.info(f'  [Agent] GrepCallers {symbol!r}')
        pattern = rf'\b{re.escape(symbol)}\s*[\(\.]'
        return search_in_files(pattern, path=clone_dir, max_results=max_results, root=clone_dir)

    return [read_file_scoped, read_files_batch, grep_callers, search_scoped, list_dir_scoped, shell_scoped]


# ---------------------------------------------------------------------------
# Symbol Knowledge Cache — shared across all files in a Round 2 pass
# ---------------------------------------------------------------------------

# Per-language patterns for locating symbol definitions.
# Each entry: (glob, def_pattern, kind_group_idx)
# def_pattern must have a named group `name` matching the symbol name.
_LANG_SYMBOL_PATTERNS: List[Tuple[str, str, str]] = [
    # Python: class Foo / def foo
    ('*.py', r'^\s*(?P<kw>class|def)\s+(?P<name>\w+)\s*[\(:]', 'kw'),
    # C/C++: RetType ClassName::method( or just FuncName(
    ('*.cpp', r'^\s*(?:[\w:<>*&]+\s+)+(?P<name>\w+)\s*\(', ''),
    ('*.cc', r'^\s*(?:[\w:<>*&]+\s+)+(?P<name>\w+)\s*\(', ''),
    ('*.cxx', r'^\s*(?:[\w:<>*&]+\s+)+(?P<name>\w+)\s*\(', ''),
    ('*.h', r'^\s*(?:class|struct|enum)\s+(?P<name>\w+)', ''),
    ('*.hpp', r'^\s*(?:class|struct|enum)\s+(?P<name>\w+)', ''),
    # Go: func (recv) FuncName( or func FuncName(
    ('*.go', r'^\s*func\s+(?:\(\w+\s+\*?\w+\)\s+)?(?P<name>\w+)\s*\(', ''),
    # Rust: fn func_name / struct Name / impl Name / trait Name
    ('*.rs', r'^\s*(?:pub\s+)?(?:async\s+)?(?P<kw>fn|struct|impl|trait|enum)\s+(?P<name>\w+)', 'kw'),
    # Java: class/interface/enum or method
    ('*.java', r'^\s*(?:public\s+|private\s+|protected\s+|static\s+|final\s+)*'
               r'(?P<kw>class|interface|enum)\s+(?P<name>\w+)', 'kw'),
    ('*.java', r'^\s*(?:public\s+|private\s+|protected\s+|static\s+|final\s+)*'
               r'[\w<>\[\]]+\s+(?P<name>\w+)\s*\(', ''),
    # JavaScript / TypeScript
    ('*.js', r'^\s*(?:export\s+)?(?:async\s+)?(?:function\s+(?P<name>\w+)|'
             r'(?:class|const|let|var)\s+(?P<name2>\w+))', ''),
    ('*.ts', r'^\s*(?:export\s+)?(?:async\s+)?(?:function\s+(?P<name>\w+)|'
             r'(?:class|const|let|var|interface|type)\s+(?P<name2>\w+))', ''),
    ('*.jsx', r'^\s*(?:export\s+)?(?:async\s+)?(?:function\s+(?P<name>\w+)|'
              r'(?:class|const|let|var)\s+(?P<name2>\w+))', ''),
    ('*.tsx', r'^\s*(?:export\s+)?(?:async\s+)?(?:function\s+(?P<name>\w+)|'
              r'(?:class|const|let|var|interface|type)\s+(?P<name2>\w+))', ''),
]

# Globs for source files to search when no file_path is given
_SOURCE_GLOBS = [
    '*.py', '*.cpp', '*.cc', '*.cxx', '*.h', '*.hpp',
    '*.go', '*.rs', '*.java', '*.js', '*.ts', '*.jsx', '*.tsx',
]

_SYMBOL_SUMMARY_PROMPT_TMPL = '''\
You are a code analyst. Given the following class or function definition, write a ONE-sentence summary \
(max 100 words) covering: purpose, key constraints, and any important usage notes.

File: {file_path}
Symbol: {symbol_name}

```
{code_snippet}
```

Output ONLY the one-sentence summary.
'''


def _build_analyze_symbol_tool(llm: Any, clone_dir: str, symbol_cache: Dict[str, Any]) -> Any:  # noqa: C901
    '''Build the analyze_symbol tool, bound to the given llm, clone_dir, and shared symbol_cache.'''
    from lazyllm.tools.agent.file_tool import search_in_files

    def analyze_symbol(symbol_name: str, file_path: str = '', max_depth: int = 2) -> dict:
        '''Read and analyze a class or function, store result in the shared symbol cache.
        Returns the cached entry immediately if already analyzed.
        Recursively analyzes direct dependencies up to max_depth=2.

        Args:
            symbol_name (str): Class or function name to analyze (e.g. "TrainableModule").
            file_path (str, optional): File path relative to repo root. If empty, will search.
            max_depth (int, optional): Recursion depth limit. Defaults to 2.

        Returns:
            dict: SymbolEntry with keys: key, kind, file, line_start, signature, docstring, summary, deps.
        '''
        # normalize file_path
        if file_path and os.path.isabs(file_path):
            file_path = os.path.relpath(file_path, clone_dir)

        cache_key = f'{file_path}::{symbol_name}' if file_path else f'::{symbol_name}'

        # check cache first
        if cache_key in symbol_cache:
            return {'cached': True, 'entry': symbol_cache[cache_key]}

        # search for definition if file_path not given or not found
        abs_file = os.path.join(clone_dir, file_path) if file_path else ''
        if not abs_file or not os.path.isfile(abs_file):
            try:
                # try each language pattern until a match is found
                for glob_pat, def_pat, _ in _LANG_SYMBOL_PATTERNS:
                    res = search_in_files(
                        def_pat.replace('(?P<name>', f'(?P<name>{re.escape(symbol_name)}')
                        if '(?P<name>' in def_pat else
                        def_pat.replace('(?P<name2>', f'(?P<name2>{re.escape(symbol_name)}'),
                        path=clone_dir, glob=glob_pat, max_results=3, root=clone_dir,
                    )
                    matches = res.get('results', []) if isinstance(res, dict) else []
                    if matches:
                        abs_file = matches[0].get('path', '')
                        file_path = os.path.relpath(abs_file, clone_dir) if abs_file else file_path
                        cache_key = f'{file_path}::{symbol_name}'
                        if cache_key in symbol_cache:
                            return {'cached': True, 'entry': symbol_cache[cache_key]}
                        break
            except Exception:
                pass

        if not abs_file or not os.path.isfile(abs_file):
            return {'cached': False, 'entry': None, 'error': f'Cannot locate {symbol_name}'}

        # read definition + docstring + method signatures
        try:
            with open(abs_file, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        except OSError:
            return {'cached': False, 'entry': None, 'error': f'Cannot read {abs_file}'}

        # find the definition line (language-agnostic: search for symbol name near line start)
        def_line_idx = None
        kind = 'function'
        ext = os.path.splitext(abs_file)[1].lower()
        # build a broad pattern: symbol name preceded by keyword or type tokens
        broad_pat = re.compile(
            rf'(?:^|\s)(?P<kw>class|struct|interface|trait|enum|def|fn|func|function)'
            rf'\s+{re.escape(symbol_name)}\s*[\(\[:<{{]'
            rf'|(?:^|\s){re.escape(symbol_name)}\s*[\(\[:]',
            re.MULTILINE,
        )
        for i, line in enumerate(lines):
            m = broad_pat.search(line)
            if m:
                def_line_idx = i
                kw = (m.group('kw') if 'kw' in m.groupdict() and m.group('kw') else '').lower()
                kind = 'class' if kw in ('class', 'struct', 'interface', 'trait', 'enum') else 'function'
                break

        if def_line_idx is None:
            return {'cached': False, 'entry': None, 'error': f'{symbol_name} not found in {file_path}'}

        # extract signature: grab lines until closing ')' or '{' or ':'
        sig_lines = [lines[def_line_idx].rstrip()]
        if not any(c in sig_lines[0] for c in ('{', ':', ')')):
            for j in range(def_line_idx + 1, min(def_line_idx + 8, len(lines))):
                sig_lines.append(lines[j].rstrip())
                if any(c in lines[j] for c in ('{', ':', ')')):
                    break
        signature = ' '.join(s.strip() for s in sig_lines)[:200]

        # extract leading comment/docstring (language-agnostic heuristic)
        docstring = ''
        ds_start = def_line_idx + 1
        if ds_start < len(lines):
            stripped = lines[ds_start].strip()
            # Python-style
            if stripped.startswith('"""') or stripped.startswith("'''"):
                quote = stripped[:3]
                ds_lines = [stripped]
                if not stripped.endswith(quote) or len(stripped) == 3:
                    for j in range(ds_start + 1, min(ds_start + 15, len(lines))):
                        ds_lines.append(lines[j].rstrip())
                        if quote in lines[j] and j > ds_start:
                            break
                docstring = '\n'.join(ds_lines)[:500]
            # C/C++/Java/Go/Rust block comment /** ... */ or /* ... */
            elif stripped.startswith('/*') or stripped.startswith('//'):
                ds_lines = [stripped]
                for j in range(ds_start + 1, min(ds_start + 15, len(lines))):
                    ds_lines.append(lines[j].rstrip())
                    if '*/' in lines[j] or (not lines[j].strip().startswith('//') and j > ds_start):
                        break
                docstring = '\n'.join(ds_lines)[:500]

        # extract method/function signatures if class-like (language-agnostic)
        method_sigs: List[str] = []
        if kind == 'class':
            # Python: look for `def ` at one indent level deeper
            if ext == '.py':
                class_indent = len(lines[def_line_idx]) - len(lines[def_line_idx].lstrip())
                for i in range(def_line_idx + 1, len(lines)):
                    ln = lines[i]
                    stripped_ln = ln.lstrip()
                    if not stripped_ln:
                        continue
                    indent = len(ln) - len(stripped_ln)
                    if indent <= class_indent and not stripped_ln.startswith('#'):
                        break
                    if re.match(r'def\s+\w+', stripped_ln) and indent in (class_indent + 4, class_indent + 2):
                        method_sigs.append(f'  {stripped_ln.rstrip()[:120]}')
            else:
                # Generic: look for function/method keywords within next 60 lines
                kw_pat = re.compile(r'(?:func|fn|def|function|void|public|private|protected)\s+\w+\s*\(')
                for i in range(def_line_idx + 1, min(def_line_idx + 60, len(lines))):
                    if kw_pat.search(lines[i]):
                        method_sigs.append(f'  {lines[i].rstrip()[:120]}')

        # build code snippet for LLM summary
        end_idx = min(def_line_idx + 40, len(lines))
        code_snippet = ''.join(lines[def_line_idx:end_idx])
        if method_sigs:
            code_snippet += '\n  # ... methods:\n' + '\n'.join(method_sigs[:10])

        # LLM summary
        summary = ''
        try:
            summary_prompt = _SYMBOL_SUMMARY_PROMPT_TMPL.format(
                file_path=file_path, symbol_name=symbol_name, code_snippet=code_snippet[:1500],
            )
            summary = _safe_llm_call_text(llm, summary_prompt) or ''
        except Exception:
            summary = signature

        # extract deps from imports/includes in the file (language-agnostic)
        deps: List[str] = []
        # Python: from X import Y / import X
        py_import = re.compile(r'^\s*(?:from\s+(\S+)\s+import\s+(.+)|import\s+(.+))')
        # C/C++: #include "foo.h" or <foo>
        c_include = re.compile(r'^\s*#include\s+[<"]([^>"]+)[>"]')
        # Go: import "pkg" or import ( "pkg" )
        go_import = re.compile(r'^\s*import\s+"([^"]+)"')
        # Rust: use crate::module::Type
        rs_use = re.compile(r'^\s*use\s+([\w:]+)(?:::\{([^}]+)\})?')
        # Java: import com.example.Class
        java_import = re.compile(r'^\s*import\s+([\w.]+);')
        # JS/TS: import { X } from 'y' or require('y')
        js_import = re.compile(
            r'''^\s*(?:import\s+.*from\s+['"]([^'"]+)['"]'''
            r'''|(?:const|let|var)\s+\w+\s*=\s*require\(['"]([^'"]+)['"]\))'''
        )

        for line in lines[:80]:
            if ext == '.py':
                m = py_import.match(line)
                if m:
                    if m.group(2):
                        for sym in re.split(r',\s*', m.group(2)):
                            sym = sym.strip().split(' as ')[0].strip()
                            if sym and sym[0].isupper():
                                deps.append(f'{m.group(1)}::{sym}')
                    elif m.group(3):
                        deps.append(m.group(3).strip())
            elif ext in ('.cpp', '.cc', '.cxx', '.h', '.hpp'):
                m = c_include.match(line)
                if m:
                    deps.append(m.group(1))
            elif ext == '.go':
                m = go_import.match(line)
                if m:
                    deps.append(m.group(1))
            elif ext == '.rs':
                m = rs_use.match(line)
                if m:
                    deps.append(m.group(1) + (f'::{{{m.group(2)}}}' if m.group(2) else ''))
            elif ext == '.java':
                m = java_import.match(line)
                if m:
                    deps.append(m.group(1))
            elif ext in ('.js', '.ts', '.jsx', '.tsx'):
                m = js_import.match(line)
                if m:
                    deps.append(m.group(1) or m.group(2) or '')

        entry: Dict[str, Any] = {
            'key': cache_key,
            'kind': kind,
            'file': file_path,
            'line_start': def_line_idx + 1,
            'signature': signature,
            'docstring': docstring,
            'summary': summary,
            'method_signatures': method_sigs[:15],
            'deps': deps[:10],
        }
        symbol_cache[cache_key] = entry

        # recursively analyze direct deps (depth - 1)
        if max_depth > 1:
            for dep_key in deps[:5]:
                parts = dep_key.split('::')
                if len(parts) == 2 and parts[1] and parts[1][0].isupper():
                    dep_sym, dep_file = parts[1], ''
                    dep_cache_key = f'::{dep_sym}'
                    if dep_cache_key not in symbol_cache and dep_sym != symbol_name:
                        try:
                            analyze_symbol(dep_sym, dep_file, max_depth=max_depth - 1)
                        except Exception:
                            pass

        return {'cached': False, 'entry': entry}

    return analyze_symbol


def _build_scoped_agent_tools_with_cache(
    clone_dir: str, llm: Any, symbol_cache: Dict[str, Any]
) -> list:
    '''Build agent tools including analyze_symbol bound to the shared symbol_cache.'''
    tools = _build_scoped_agent_tools(clone_dir)
    analyze_symbol = _build_analyze_symbol_tool(llm, clone_dir, symbol_cache)
    return tools + [analyze_symbol]


def _summarize_section(llm: Any, title: str, content: str) -> str:
    prompt = f'Summarize in 1-2 sentences (max 200 chars):\n[{title}]\n{content[:800]}'
    result = _safe_llm_call_text(llm, prompt) or content[:100]
    return result[:200]


def _arch_collect_snippets(clone_dir: str, section: Dict[str, Any], max_chars: int = 6000) -> str:  # noqa: C901
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
                match_text = m.get('text', '')
                # for class definitions: read definition line + method signatures only
                if re.match(r'\s*class\s+\w+', match_text):
                    fc = read_file(path, start_line=max(1, line - 1), end_line=line + 2, root=clone_dir)
                    class_def = fc.get('content', '') if isinstance(fc, dict) else ''
                    # extract method signatures from the class body
                    try:
                        with open(path, 'r', encoding='utf-8', errors='replace') as _f:
                            all_lines = _f.readlines()
                        sigs = _extract_class_method_signatures(all_lines, line - 1)
                        snippet = class_def + ('\n' + '\n'.join(sigs[:12]) if sigs else '')
                    except Exception:
                        snippet = class_def
                # for function definitions: read signature + docstring
                elif re.match(r'\s*def\s+\w+', match_text):
                    fc = read_file(path, start_line=max(1, line - 1), end_line=line + 10, root=clone_dir)
                    snippet = fc.get('content', '') if isinstance(fc, dict) else ''
                else:
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
        code_snippets=snippets[:6000],
    )
    raw = _safe_llm_call_text(llm, prompt)
    if not raw:
        raise ValueError(f'LLM returned empty result for section "{section.get("title")}"')
    return raw[:3500]


def _arch_fill_all_sections(
    llm: Any, clone_dir: str, outline: List[Dict[str, Any]], dir_tree_1level: str,
    cache_path: Optional[str] = None,
) -> str:
    sections: List[str] = []
    prev_summaries: List[str] = []
    prog = _Progress('Arch: filling sections', len(outline))
    for sec in outline:
        title = sec.get('title', 'Section')
        cache_key = f'arch_section_{re.sub(r"[^a-zA-Z0-9]", "_", title).lower()}'
        content = _load_cache(cache_path, cache_key)
        if content:
            prog.update(f'{title} (cached)')
        else:
            content = _arch_fill_section(llm, clone_dir, sec, dir_tree_1level, prev_summaries)
            _save_cache(cache_path, cache_key, content)
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
# Public API Catalog — LLM identifies public files, regex extracts symbols
# ---------------------------------------------------------------------------

_PUBLIC_API_FILES_PROMPT_TMPL = '''\
You are analyzing a software project. Based on the directory tree below, identify files that \
serve as shared/public utility or base-class libraries — files whose functions and classes are \
intended to be reused by other modules.

For each such file output a JSON object with:
- "file": path relative to repo root (e.g. "common/utils.py")
- "scope": the path prefix under which this file is relevant.
  Use "global" if it is a top-level shared library usable by the entire project.
  Otherwise use the directory path of the module it belongs to (e.g. "tools/agent").
- "reason": one short phrase explaining why it is public (e.g. "top-level utils", "agent helpers")

Rules:
- Only include files that are clearly utility/helper/base libraries, NOT application logic files.
- Do NOT include test files, example files, migration scripts, or generated files.
- Limit to at most 30 files.
- Output ONLY a JSON array. No explanation.

<directory_tree>
{dir_tree}
</directory_tree>
'''

# source file extensions to scan for public symbols
_PUBLIC_SYM_EXTS = {'.py', '.go', '.ts', '.js', '.tsx', '.jsx', '.java', '.rs', '.cpp', '.cc', '.h', '.hpp'}

# per-language patterns: (ext_set, pattern, name_group)
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


def _get_sym_pattern(ext: str) -> Optional[re.Pattern]:
    for ext_set, pat, _ in _PUBLIC_SYM_PATTERNS:
        if ext in ext_set:
            return re.compile(pat)
    return None


def _extract_sym_desc(lines: List[str], idx: int) -> str:
    if idx + 1 >= len(lines):
        return ''
    nxt = lines[idx + 1].strip()
    if nxt.startswith('"""') or nxt.startswith("'''"):
        return nxt.strip('"\'').strip()[:80]
    if nxt.startswith('//') or nxt.startswith('#') or nxt.startswith('/*'):
        return re.sub(r'^[/#*\s]+', '', nxt).strip()[:80]
    return ''


def _scan_file_symbols(lines: List[str], pattern: re.Pattern) -> List[str]:
    entries: List[str] = []
    for i, line in enumerate(lines):
        m = pattern.match(line)
        if not m:
            continue
        sym_name = m.group(1)
        if sym_name.startswith('_'):
            continue
        sig = line.rstrip()[:120]
        desc = _extract_sym_desc(lines, i)
        entries.append(f'{sig}: {desc}' if desc else sig)
        if len(entries) >= _PUBLIC_API_MAX_ENTRIES_PER_FILE:
            break
    return entries


def _extract_public_symbols(clone_dir: str, file_entries: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    catalog: Dict[str, List[str]] = {}
    for entry in file_entries[:_PUBLIC_API_MAX_FILES]:
        fpath = entry.get('file', '')
        scope = entry.get('scope', 'global') or 'global'
        abs_path = os.path.join(clone_dir, fpath)
        if not os.path.isfile(abs_path):
            continue
        ext = os.path.splitext(fpath)[1].lower()
        if ext not in _PUBLIC_SYM_EXTS:
            continue
        pattern = _get_sym_pattern(ext)
        if pattern is None:
            continue
        try:
            with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        except OSError:
            continue
        syms = _scan_file_symbols(lines, pattern)
        if syms:
            catalog.setdefault(scope, []).extend(syms)
    return catalog


def _build_public_api_catalog(
    llm: Any, clone_dir: str, cache_path: Optional[str] = None,
) -> str:
    cached = _load_cache(cache_path, 'public_api_catalog')
    if cached:
        return cached

    dir_tree = _build_dir_tree(clone_dir, max_depth=3)
    prompt = _PUBLIC_API_FILES_PROMPT_TMPL.format(dir_tree=dir_tree[:4000])
    file_entries = _safe_llm_call(llm, prompt)
    if not isinstance(file_entries, list):
        file_entries = []

    catalog = _extract_public_symbols(clone_dir, file_entries)
    result = json.dumps(catalog, ensure_ascii=False)
    _save_cache(cache_path, 'public_api_catalog', result)
    return result


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
        elif current_title and line.strip() and (not lines or not lines[-1].startswith(f'[{current_title}]')):
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


def _parse_arch_sections(arch_doc: str) -> List[Tuple[str, str]]:
    '''Parse arch_doc into (title, content) pairs. Sections start with [Title].'''
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


# sections always injected regardless of file path (high value for all reviews)
_ARCH_ALWAYS_INJECT = frozenset({
    'module hierarchy',
    'non-obvious behaviors & gotchas',
    'non-obvious behaviors',
    'gotchas',
    'key utilities',
    'key utilities & usage notes',
})


def _candidate_scopes(file_path: str) -> List[str]:
    parts = file_path.replace('\\', '/').split('/')
    scopes = ['global']
    for i in range(1, len(parts)):  # exclude filename itself
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
    '''Return the most relevant arch_doc sections for a given file path.

    Strategy:
    1. Always include Module Hierarchy, Gotchas, and Key Utilities sections.
    2. For Public API Catalog: filter by candidate scopes derived from file_path (pure prefix match).
    3. Score remaining sections by keyword overlap with the file path components.
    4. Fill up to max_chars, prioritising higher-scored sections.
    '''
    if not arch_doc:
        return '(not available)'
    sections = _parse_arch_sections(arch_doc)
    if not sections:
        return arch_doc[:max_chars]

    path_keywords = {p for p in re.split(r'[/\\._]', file_path.lower()) if len(p) > 2}

    def _score(title: str, content: str) -> int:
        t_lower = title.lower()
        base = 100 if any(p in t_lower for p in _ARCH_ALWAYS_INJECT) else 0
        combined = t_lower + ' ' + content.lower()
        overlap = sum(1 for kw in path_keywords if kw in combined)
        return base + overlap * 10

    scored = sorted(sections, key=lambda sc: _score(sc[0], sc[1]), reverse=True)

    parts: List[str] = []
    remaining = max_chars
    for title, content in scored:
        if remaining <= 0:
            break
        # Public API Catalog: filter by candidate scopes, not full content
        if re.match(r'public api catalog', title, re.IGNORECASE):
            filtered = _format_catalog_for_file(content, file_path, max_chars=min(remaining, 1200))
            if filtered and not filtered.startswith('(no matching'):
                block = f'[{title}]\n{filtered}'
                parts.append(block[:remaining])
                remaining -= len(parts[-1]) + 2
            continue
        block = f'[{title}]\n{content}'
        parts.append(block[:remaining])
        remaining -= len(parts[-1]) + 2  # +2 for '\n\n' separator

    return '\n\n'.join(parts)


def _build_symbol_index(arch_doc: str) -> Dict[str, str]:
    '''Extract symbol→description pairs from the Key Utilities section of arch_doc.
    Parses the full arch_doc (not a truncated slice) so the index is always populated.
    '''
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
) -> str:
    cached = _load_cache(cache_path, 'arch_doc')
    if cached:
        return cached

    snapshot = _collect_structured_snapshot(clone_dir)
    dir_tree_1 = _build_dir_tree(clone_dir, max_depth=1)

    outline_cached = _load_cache(cache_path, 'arch_outline')
    if outline_cached:
        try:
            outline = json.loads(outline_cached)
        except (json.JSONDecodeError, TypeError):
            outline = None
    else:
        outline = None
    if not outline:
        outline = _arch_generate_outline(llm, snapshot, agent_instructions)
        if not outline:
            raise ValueError('Arch outline generation returned empty result')
        _save_cache(cache_path, 'arch_outline', json.dumps(outline, ensure_ascii=False))
    arch_doc = _arch_fill_all_sections(llm, clone_dir, outline, dir_tree_1, cache_path)

    arch_doc = arch_doc or '(architecture analysis unavailable)'

    # append Public API Catalog section (LLM identifies files, regex extracts symbols)
    try:
        public_api_json = _build_public_api_catalog(llm, clone_dir, cache_path)
        if public_api_json and not public_api_json.startswith('('):
            arch_doc = arch_doc + f'\n\n[Public API Catalog]\n{public_api_json}'
    except Exception as e:
        lazyllm.LOG.warning(f'Public API Catalog generation failed: {e}')

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
_SPEC_MAX_RULES = 50


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
- "rule_id": string like "PR{pr_num}_ERR001", "PR{pr_num}_STY002" (ERR=error/exception, STY=style, DSN=design, \
PERF=performance, SEC=security, XFILE=cross-file consistency)
- "title": short title (max 8 words)
- "severity": "P0" | "P1" | "P2"  (P0=must fix, P1=should fix, P2=nice to have)
- "detect": list of strings describing how to detect this issue (patterns, keywords, conditions)
- "bad_example": short code snippet showing the bad pattern (or "" if not applicable)
- "good_example": short code snippet showing the correct pattern (or "" if not applicable)
- "fix": one-sentence fix suggestion

Pay special attention to cross-file consistency issues (use rule_id prefix "PR{pr_num}_XFILE"):
- Interface changed but callers not updated
- Symmetric methods (encode/decode, open/close) only one side updated
- Registry/factory pattern: new entry added but docs/tests not updated
- Abstract method added to base class but not implemented in subclasses

Output ONLY a JSON array. If no clear rules can be extracted: output [].

<review_comments>
{{comments_text}}
</review_comments>
'''

_MERGE_RULES_PROMPT = '''\
You are a code review expert. Below are rule cards extracted from multiple pull requests.
Your task:
1. Merge duplicate or highly similar rules into one (keep the most informative example and detect patterns).
2. Remove rules that are too vague, trivial, or project-unspecific.
3. Re-assign clean rule_ids using standard prefixes: ERR, STY, DSN, PERF, SEC, XFILE (e.g. ERR001, STY002).
4. Sort by severity (P0 first), then by frequency/importance.
5. Keep at most {max_rules} rules total.

For each final rule, output a JSON object with:
- "rule_id": clean id like "ERR001" (no PR prefix)
- "title", "severity", "detect" (list), "bad_example", "good_example", "fix"

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
            raise RuntimeError(f'Comment compression failed: {e}') from e
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
    prompt = _EXTRACT_RULES_PROMPT.format(pr_num=pr_num).replace('{{comments_text}}', comments_text)
    try:
        result = _safe_llm_call(llm, prompt)
        return [r for r in (result if isinstance(result, list) else []) if isinstance(r, dict)]
    except Exception as e:
        raise RuntimeError(f'Rule extraction for PR #{pr_num} failed: {e}') from e


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
        raise RuntimeError(f'Rule merge failed: {e}') from e


def _collect_rules_for_pr(
    backend: LazyLLMGitBase, llm: Any, pr: Any,
    idx: int, total: int, cache_path: Optional[str],
    prog: Any, all_rules: List[Dict[str, Any]],
) -> None:
    pr_num = getattr(pr, 'number', None) or (pr.get('number') if isinstance(pr, dict) else None)
    if pr_num is None:
        prog.update(f'[{idx}/{total}] skipped (no number)')
        return
    pr_cache_key = f'spec_pr_{pr_num}_rules'
    cached_rules_str = _load_cache(cache_path, pr_cache_key)
    if cached_rules_str:
        try:
            rules = json.loads(cached_rules_str)
            all_rules.extend(rules)
            prog.update(f'[{idx}/{total}] PR #{pr_num} — {len(rules)} rules (cached)')
            return
        except (json.JSONDecodeError, TypeError):
            pass
    try:
        res = backend.list_review_comments(pr_num)
    except Exception as e:
        lazyllm.LOG.warning(f'PR #{pr_num} comments fetch error (skipped): {e}')
        prog.update(f'[{idx}/{total}] PR #{pr_num} (network error, skipped)')
        return
    if not res.get('success'):
        prog.update(f'[{idx}/{total}] PR #{pr_num} (fetch failed)')
        return
    comments = [
        {'user': (c.get('user') if isinstance(c, dict) else getattr(c, 'user', '')) or '',
         'body': (c.get('body') if isinstance(c, dict) else getattr(c, 'body', '')) or ''}
        for c in (res.get('comments') or [])
        if ((c.get('body') if isinstance(c, dict) else getattr(c, 'body', '')) or '').strip()
        and not _BOT_USER_PATTERNS.search(
            (c.get('user') if isinstance(c, dict) else getattr(c, 'user', '')) or '')
    ]
    if not comments:
        prog.update(f'[{idx}/{total}] PR #{pr_num} — 0 human comments → skipped')
        return
    rules = _extract_rules_from_pr_comments(llm, pr_num, comments)
    if rules:
        _save_cache(cache_path, pr_cache_key, json.dumps(rules, ensure_ascii=False))
    all_rules.extend(rules)
    prog.update(f'[{idx}/{total}] PR #{pr_num} — {len(comments)} comments → {len(rules)} rules extracted')


def analyze_historical_reviews(
    backend: LazyLLMGitBase, llm: Any, cache_path: Optional[str] = None, max_prs: int = 200
) -> str:
    cached = _load_cache(cache_path, 'review_spec')
    cached_max_prs_str = _load_cache(cache_path, 'review_spec_max_prs')
    cached_max_prs = int(cached_max_prs_str) if cached_max_prs_str and cached_max_prs_str.isdigit() else 0
    # reuse cache only when max_prs has not grown
    if cached and cached_max_prs >= max_prs:
        return cached

    # fetch in batches until we have enough merged PRs or exhaust the API
    merged: List[Any] = []
    fetch_size = max_prs
    while len(merged) < max_prs:
        pr_list_res = backend.list_pull_requests(state='closed', max_results=fetch_size)
        if not pr_list_res.get('success'):
            return '(historical review analysis unavailable)'
        prs = pr_list_res.get('list') or []
        if not prs:
            break
        merged = [p for p in prs if _is_merged_pr(p)]
        if len(merged) >= max_prs or len(prs) < fetch_size:
            # enough merged found, or no more pages available
            break
        # not enough merged — fetch more closed PRs (2x each round, capped at 1000)
        fetch_size = min(fetch_size * 2, 1000)
    target = merged[:max_prs]
    total = len(target)

    if not target:
        _save_cache_multi(cache_path, {'review_spec': '(no historical review comments found)',
                                       'review_spec_max_prs': str(max_prs)})
        return '(no historical review comments found)'

    prog = _Progress('Spec: extracting rules from historical PRs', total)
    all_rules: List[Dict[str, Any]] = []

    for idx, pr in enumerate(target, 1):
        _collect_rules_for_pr(backend, llm, pr, idx, total, cache_path, prog, all_rules)

    prog.done(f'{len(all_rules)} raw rules from {total} PRs, merging...')

    if not all_rules:
        _save_cache_multi(cache_path, {'review_spec': '(no historical review comments found)',
                                       'review_spec_max_prs': str(max_prs)})
        return '(no historical review comments found)'

    merged_rules = _merge_rule_cards(llm, all_rules)
    # two-level storage: summaries (lightweight index) + details (full rule cards)
    summaries = [{'rule_id': r.get('rule_id', ''), 'title': r.get('title', '')} for r in merged_rules]
    details = {r.get('rule_id', f'R{i:03d}'): r for i, r in enumerate(merged_rules)}
    review_spec_obj = {'summaries': summaries, 'details': details}
    review_spec = json.dumps(review_spec_obj, ensure_ascii=False)
    review_spec = review_spec or '(review spec analysis unavailable)'
    _save_cache_multi(cache_path, {'review_spec': review_spec, 'review_spec_max_prs': str(max_prs)})
    return review_spec


def _lookup_relevant_rules(review_spec: str, diff_content: str, max_detail: int = 10) -> str:  # noqa: C901
    '''Two-level rule lookup: match summaries by keywords, then load full detail cards.

    Returns a formatted string with:
    - Full rule cards for matched rules (up to max_detail)
    - Title-only list for unmatched rules
    '''
    if not review_spec or review_spec.startswith('('):
        return review_spec or ''
    try:
        spec_obj = json.loads(review_spec)
    except (json.JSONDecodeError, ValueError):
        # legacy plain-text format — return as-is (truncated)
        return review_spec[:2000]

    summaries = spec_obj.get('summaries', [])
    details = spec_obj.get('details', {})

    # extract keywords from diff: file names, class names, function names
    keywords: set = set()
    for line in diff_content.splitlines()[:200]:
        # file paths
        if line.startswith('+++ ') or line.startswith('--- '):
            fname = line.split('/')[-1].replace('.py', '').lower()
            if fname:
                keywords.add(fname)
        # class/function names
        for m in re.finditer(r'\b([A-Z][a-zA-Z0-9]+|[a-z_][a-z_0-9]{3,})\b', line):
            keywords.add(m.group(1).lower())

    matched_ids: List[str] = []
    unmatched_titles: List[str] = []
    for s in summaries:
        rule_id = s.get('rule_id', '')
        title = s.get('title', '')
        title_lower = title.lower()
        if any(kw in title_lower for kw in keywords if len(kw) > 3):
            matched_ids.append(rule_id)
        else:
            unmatched_titles.append(f'[{rule_id}] {title}')

    parts: List[str] = []
    for rule_id in matched_ids[:max_detail]:
        rule = details.get(rule_id)
        if rule:
            parts.append(_format_rule_card(rule))

    if unmatched_titles:
        parts.append('## Other rules (title only)\n' + '\n'.join(unmatched_titles))

    return '\n\n'.join(parts) if parts else '(no matching rules found)'


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
    llm: Any, pr: Any, repo: str, arch_cache_path: str, ckpt: Any,
    clone_target_dir: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    arch_doc = ckpt.get('arch_doc') or ''
    clone_dir: Optional[str] = None
    prog = _Progress('Pre-analysis: fetch repo & analyze architecture')
    clone_url, branch = _resolve_clone_target(pr, repo)
    lazyllm.LOG.info(f'Cloning {clone_url} @ {branch}')
    try:
        clone_dir, _file_tree = _fetch_repo_code(clone_url, branch, work_dir=clone_target_dir)
    except Exception as e:
        raise RuntimeError(f'Failed to clone repo {clone_url} @ {branch}: {e}') from e
    ckpt.save('clone_dir', clone_dir)
    ckpt.mark_stage_done(ReviewStage.CLONE)
    prog.update('cloned, analyzing...')
    agent_instructions = _read_agent_instructions(clone_dir)
    if agent_instructions:
        _save_cache(arch_cache_path, 'agent_instructions', agent_instructions)
        lazyllm.LOG.info(f'Found agent instructions ({len(agent_instructions)} chars)')
    try:
        arch_doc = analyze_repo_architecture(llm, clone_dir, arch_cache_path, agent_instructions)
    except Exception:
        import shutil
        shutil.rmtree(clone_dir, ignore_errors=True)
        raise
    ckpt.save('arch_doc', arch_doc)
    ckpt.mark_stage_done(ReviewStage.ARCH)
    if arch_doc and arch_cache_path:
        lazyllm.LOG.success(f'Architecture doc saved to: {arch_cache_path}')
    prog.done('architecture doc ready')
    return arch_doc, clone_dir, agent_instructions


def _run_spec_analysis(
    backend_inst: LazyLLMGitBase, llm: Any,
    review_spec_cache_path: str, max_history_prs: int, ckpt: Any
) -> str:
    review_spec = ckpt.get('review_spec') or ''
    if not review_spec:
        try:
            review_spec = analyze_historical_reviews(backend_inst, llm, review_spec_cache_path, max_history_prs)
            ckpt.save('review_spec', review_spec)
            ckpt.mark_stage_done(ReviewStage.SPEC)
            if review_spec and review_spec_cache_path:
                if review_spec.startswith('('):
                    lazyllm.LOG.warning(f'Review spec not generated: {review_spec}')
                else:
                    lazyllm.LOG.success(f'Review spec saved to: {review_spec_cache_path}')
        except Exception as e:
            # distinguish: no history PRs (acceptable) vs API/LLM failure (fatal)
            if 'no review comments' in str(e).lower() or 'not found' in str(e).lower():
                lazyllm.LOG.warning(f'Historical review analysis: {e}')
            else:
                raise
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
    pr_dir: Optional[str] = None,
) -> Tuple[str, str, Optional[str], str]:
    # returns (arch_doc, review_spec, clone_dir, agent_instructions)
    from .checkpoint import _ReviewCheckpoint
    repo_cache_dir = _ReviewCheckpoint.repo_cache_dir(repo)
    if arch_cache_path is None:
        arch_cache_path = os.path.join(repo_cache_dir, 'arch.json')
    if review_spec_cache_path is None:
        review_spec_cache_path = os.path.join(repo_cache_dir, 'spec.json')

    arch_doc = ckpt.get('arch_doc') or ''
    # restore clone_dir from checkpoint if it still exists on disk
    clone_dir: Optional[str] = ckpt.get('clone_dir') or None
    if clone_dir and not os.path.isdir(clone_dir):
        clone_dir = None

    clone_target_dir = os.path.join(pr_dir, 'clone') if pr_dir else None
    agent_instructions = _load_cache(arch_cache_path, 'agent_instructions') or ''

    if fetch_repo_code and not arch_doc:
        arch_doc, clone_dir, agent_instructions = _run_arch_analysis(
            llm, pr, repo, arch_cache_path, ckpt, clone_target_dir,
        )
    elif fetch_repo_code:
        # arch_doc already cached — still need clone for Round 2 agent
        _save_cache(arch_cache_path, 'arch_doc', arch_doc)
        _Progress('Pre-analysis: architecture').done('loaded from checkpoint')
        if not clone_dir:
            try:
                clone_url, branch = _resolve_clone_target(pr, repo)
                lazyllm.LOG.info(f'Cloning {clone_url} @ {branch} for agent file access')
                clone_dir, _ = _fetch_repo_code(clone_url, branch, work_dir=clone_target_dir)
                ckpt.save('clone_dir', clone_dir)
            except Exception as e:
                raise RuntimeError(f'Clone for agent failed: {e}') from e
        else:
            lazyllm.LOG.info(f'Reusing cached clone at {clone_dir}')
        # read agent instructions from the (possibly freshly cloned) repo
        if not agent_instructions and clone_dir:
            agent_instructions = _read_agent_instructions(clone_dir)
            if agent_instructions:
                _save_cache(arch_cache_path, 'agent_instructions', agent_instructions)
    else:
        _save_cache(arch_cache_path, 'arch_doc', arch_doc)
        _Progress('Pre-analysis: architecture').done('loaded from checkpoint')

    review_spec = _run_spec_analysis(backend_inst, llm, review_spec_cache_path, max_history_prs, ckpt)
    return arch_doc, review_spec, clone_dir, agent_instructions
