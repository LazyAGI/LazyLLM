# Copyright (c) 2026 LazyAGI. All rights reserved.
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import lazyllm

from ..utils import _safe_llm_call_text
from .deepwiki import _deepwiki_ask_cached
from .file_context import _extract_file_skeleton
from .prompt import _SYMBOL_SUMMARY_PROMPT_TMPL

_CONCURRENCY_MAX_RESULTS = 20

_LANG_SYMBOL_PATTERNS: List[Tuple[str, str, str]] = [
    ('*.py', r'^\s*(?P<kw>class|def)\s+(?P<name>\w+)\s*[\(:]', 'kw'),
    ('*.cpp', r'^\s*(?:[\w:<>*&]+\s+)+(?P<name>\w+)\s*\(', ''),
    ('*.cc', r'^\s*(?:[\w:<>*&]+\s+)+(?P<name>\w+)\s*\(', ''),
    ('*.cxx', r'^\s*(?:[\w:<>*&]+\s+)+(?P<name>\w+)\s*\(', ''),
    ('*.h', r'^\s*(?:class|struct|enum)\s+(?P<name>\w+)', ''),
    ('*.hpp', r'^\s*(?:class|struct|enum)\s+(?P<name>\w+)', ''),
    ('*.go', r'^\s*func\s+(?:\(\w+\s+\*?\w+\)\s+)?(?P<name>\w+)\s*\(', ''),
    ('*.rs', r'^\s*(?:pub\s+)?(?:async\s+)?(?P<kw>fn|struct|impl|trait|enum)\s+(?P<name>\w+)', 'kw'),
    ('*.java', r'^\s*(?:public\s+|private\s+|protected\s+|static\s+|final\s+)*'
               r'(?P<kw>class|interface|enum)\s+(?P<name>\w+)', 'kw'),
    ('*.java', r'^\s*(?:public\s+|private\s+|protected\s+|static\s+|final\s+)*'
               r'[\w<>\[\]]+\s+(?P<name>\w+)\s*\(', ''),
    ('*.js', r'^\s*(?:export\s+)?(?:async\s+)?(?:function\s+(?P<name>\w+)|'
             r'(?:class|const|let|var)\s+(?P<name2>\w+))', ''),
    ('*.ts', r'^\s*(?:export\s+)?(?:async\s+)?(?:function\s+(?P<name>\w+)|'
             r'(?:class|const|let|var|interface|type)\s+(?P<name2>\w+))', ''),
    ('*.jsx', r'^\s*(?:export\s+)?(?:async\s+)?(?:function\s+(?P<name>\w+)|'
              r'(?:class|const|let|var)\s+(?P<name2>\w+))', ''),
    ('*.tsx', r'^\s*(?:export\s+)?(?:async\s+)?(?:function\s+(?P<name>\w+)|'
              r'(?:class|const|let|var|interface|type)\s+(?P<name2>\w+))', ''),
]

_SOURCE_GLOBS = [
    '*.py', '*.cpp', '*.cc', '*.cxx', '*.h', '*.hpp',
    '*.go', '*.rs', '*.java', '*.js', '*.ts', '*.jsx', '*.tsx',
]

_LANG_IMPORT_PATTERNS: Dict[str, re.Pattern] = {
    '.py': re.compile(r'^\s*(?:from\s+(\S+)\s+import\s+(.+)|import\s+(.+))'),
    '.go': re.compile(r'^\s*import\s+"([^"]+)"'),
    '.rs': re.compile(r'^\s*use\s+([\w:]+)(?:::\{([^}]+)\})?'),
    '.java': re.compile(r'^\s*import\s+([\w.]+);'),
}
for _ext in ('.cpp', '.cc', '.cxx', '.h', '.hpp'):
    _LANG_IMPORT_PATTERNS[_ext] = re.compile(r'^\s*#include\s+[<"]([^>"]+)[>"]')
for _ext in ('.js', '.ts', '.jsx', '.tsx'):
    _LANG_IMPORT_PATTERNS[_ext] = re.compile(
        r'''^\s*(?:import\s+.*from\s+['"]([^'"]+)['"]|(?:const|let|var)\s+\w+\s*=\s*require\(['"]([^'"]+)['"]\))'''
    )


def _build_scoped_agent_tools(  # noqa: C901
    clone_dir: str, owner_repo: str = '', cache_path: Optional[str] = None,
) -> list:
    from lazyllm.tools.agent.file_tool import read_file, list_dir, search_in_files
    from lazyllm.tools.agent.shell_tool import shell_tool

    def read_file_scoped(path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> dict:
        '''Read a source file from the repository, with optional line range.

        Args:
            path (str): Relative or absolute path to the file.
            start_line (int, optional): 1-based start line (inclusive).
            end_line (int, optional): 1-based end line (inclusive).
        '''
        abs_path = path if os.path.isabs(path) else os.path.join(clone_dir, path)
        line_info = f':{start_line}-{end_line}' if (start_line or end_line) else ''
        lazyllm.LOG.info(f'  [Agent] Read {path}{line_info}')
        return read_file(abs_path, start_line=start_line, end_line=end_line, root=clone_dir)

    def read_file_skeleton_scoped(path: str) -> dict:
        '''Get the structural skeleton of a file: imports, class definitions, function signatures.
        Use this FIRST to understand file structure before reading specific sections.

        Args:
            path (str): Relative or absolute path to the file.
        '''
        lazyllm.LOG.info(f'  [Agent] Skeleton {path}')
        skeleton = _extract_file_skeleton(clone_dir, path)
        return {'status': 'ok', 'path': path, 'skeleton': skeleton or '(empty or not found)'}

    def search_scoped(pattern: str, glob: Optional[str] = None, max_results: int = 40) -> dict:
        '''Search files in the repository for a regex pattern.

        Args:
            pattern (str): Regex pattern to search for.
            glob (str, optional): Filename glob filter, e.g. "*.py".
            max_results (int, optional): Max number of matches to return.
        '''
        lazyllm.LOG.info(f'  [Agent] Search {pattern!r}' + (f' in {glob}' if glob else ''))
        return search_in_files(pattern, path=clone_dir, glob=glob, max_results=max_results, root=clone_dir)

    def list_dir_scoped(path: str = '.', recursive: bool = False) -> dict:
        '''List directory entries in the repository.

        Args:
            path (str, optional): Directory path relative to repo root. Defaults to root.
            recursive (bool, optional): Whether to walk recursively.
        '''
        lazyllm.LOG.info(f'  [Agent] ListDir {path}' + (' (recursive)' if recursive else ''))
        abs_path = path if os.path.isabs(path) else os.path.join(clone_dir, path)
        return list_dir(abs_path, recursive=recursive, root=clone_dir)

    def shell_scoped(cmd: str, timeout: int = 30) -> dict:
        '''Run a read-only shell command inside the repository directory.

        Args:
            cmd (str): Shell command to execute (unsafe commands are blocked).
            timeout (int, optional): Timeout in seconds. Defaults to 30.
        '''
        lazyllm.LOG.info(f'  [Agent] Shell {cmd!r}')
        return shell_tool(cmd, cwd=clone_dir, timeout=timeout, allow_unsafe=False)

    def read_files_batch(paths: str) -> dict:
        '''Read multiple source files at once.

        Args:
            paths (str): Comma-separated list of relative file paths to read.
        '''
        path_list = [p.strip() for p in paths.split(',') if p.strip()]
        lazyllm.LOG.info(f'  [Agent] ReadBatch {path_list}')
        results = {}
        for p in path_list[:6]:
            abs_path = p if os.path.isabs(p) else os.path.join(clone_dir, p)
            try:
                with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                    results[p] = f.read(1600)
            except OSError:
                results[p] = '(not found)' if not os.path.isfile(abs_path) else '(read error)'
        return {'files': results, 'count': len(results)}

    def grep_callers(symbol: str, max_results: int = 20) -> dict:
        '''Find all call sites of a function or class in the repository.

        Args:
            symbol (str): Function or class name to search for.
            max_results (int, optional): Max number of matches to return.
        '''
        lazyllm.LOG.info(f'  [Agent] GrepCallers {symbol!r}')
        pattern = rf'\b{re.escape(symbol)}\s*[\(\.]'
        return search_in_files(pattern, path=clone_dir, max_results=max_results, root=clone_dir)

    def ask_deepwiki(question: str) -> str:
        '''Ask DeepWiki a background question about this repository's architecture or design.

        IMPORTANT — data may be 1-3 months stale. Use ONLY as background knowledge, NOT as
        source of truth for current code. Treat answers as "possibly outdated context" and
        always cross-verify against the actual diff and local code.

        USE for:
        - Understanding overall system architecture (module boundaries, layering, responsibilities)
        - Learning design conventions or usage patterns of public/infrastructure modules
        - Identifying potential architectural issues (wrong dependency direction, misuse of abstractions)
        - Supplementing cross-module context not visible in the diff

        DO NOT USE for:
        - Verifying whether new/modified code in the diff is correct
        - Determining current function/interface behavior, parameters, or implementation details
        - Drawing definitive conclusions that depend on the latest code state

        When using the answer, treat it as a hypothesis ("based on background knowledge, this may
        violate the existing design") rather than a fact. Always verify with local code tools.

        Args:
            question (str): Architecture or design question about the repository (not about specific
                diff correctness). Prefer questions about module roles, design patterns, or
                cross-module relationships.
        '''
        if not owner_repo:
            return '(DeepWiki not configured for this repo)'
        lazyllm.LOG.info(f'  [Agent] DeepWiki ask: {question!r}')
        answer = _deepwiki_ask_cached(owner_repo, question, max_chars=2000)
        if not answer:
            return '(no answer from DeepWiki)'
        return f'[DeepWiki background knowledge — may be 1-3 months stale, verify with local code]\n{answer}'

    tools = [
        read_file_scoped, read_file_skeleton_scoped, read_files_batch,
        grep_callers, search_scoped, list_dir_scoped, shell_scoped,
    ]
    if owner_repo:
        tools.append(ask_deepwiki)
    return tools


def _build_analyze_symbol_tool(  # noqa: C901
    llm: Any, clone_dir: str, symbol_cache: Dict[str, Any],
    cache_lock: Optional[Any] = None,
) -> Any:
    from lazyllm.tools.agent.file_tool import search_in_files

    def _cache_get(key: str) -> Optional[Any]:
        if cache_lock:
            with cache_lock:
                return symbol_cache.get(key)
        return symbol_cache.get(key)

    def _cache_set(key: str, value: Any) -> None:
        if cache_lock:
            with cache_lock:
                symbol_cache[key] = value
        else:
            symbol_cache[key] = value

    def analyze_symbol(symbol_name: str, file_path: str = '', max_depth: int = 2) -> dict:
        '''Read and analyze a class or function definition, with its dependencies.

        Args:
            symbol_name (str): Name of the class or function to analyze.
            file_path (str, optional): Relative path to the file containing the symbol.
            max_depth (int, optional): How many levels of dependencies to follow.
        '''
        if file_path and os.path.isabs(file_path):
            file_path = os.path.relpath(file_path, clone_dir)

        cache_key = f'{file_path}::{symbol_name}' if file_path else f'::{symbol_name}'

        cached = _cache_get(cache_key)
        if cached is not None:
            return {'cached': True, 'entry': cached}

        abs_file = os.path.join(clone_dir, file_path) if file_path else ''
        if not abs_file or not os.path.isfile(abs_file):
            try:
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
                        cached = _cache_get(cache_key)
                        if cached is not None:
                            return {'cached': True, 'entry': cached}
                        break
            except Exception as _e:
                lazyllm.LOG.warning(f'[analyze_symbol] Symbol search failed for {symbol_name!r}: {_e}')

        if not abs_file or not os.path.isfile(abs_file):
            return {'cached': False, 'entry': None, 'error': f'Cannot locate {symbol_name}'}

        try:
            with open(abs_file, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        except OSError:
            return {'cached': False, 'entry': None, 'error': f'Cannot read {abs_file}'}

        def_line_idx = None
        kind = 'function'
        ext = os.path.splitext(abs_file)[1].lower()
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

        sig_lines = [lines[def_line_idx].rstrip()]
        if not any(c in sig_lines[0] for c in ('{', ':', ')')):
            for j in range(def_line_idx + 1, min(def_line_idx + 8, len(lines))):
                sig_lines.append(lines[j].rstrip())
                if any(c in lines[j] for c in ('{', ':', ')')):
                    break
        signature = ' '.join(s.strip() for s in sig_lines)[:200]

        docstring = ''
        ds_start = def_line_idx + 1
        if ds_start < len(lines):
            stripped = lines[ds_start].strip()
            if stripped.startswith(('"""', "'''")):
                quote = stripped[:3]
                ds_lines = [stripped]
                if not stripped.endswith(quote) or len(stripped) == 3:
                    for j in range(ds_start + 1, min(ds_start + 15, len(lines))):
                        ds_lines.append(lines[j].rstrip())
                        if quote in lines[j] and j > ds_start:
                            break
                docstring = '\n'.join(ds_lines)[:500]
            elif stripped.startswith(('/*', '//')):
                ds_lines = [stripped]
                for j in range(ds_start + 1, min(ds_start + 15, len(lines))):
                    ds_lines.append(lines[j].rstrip())
                    if '*/' in lines[j] or (not lines[j].strip().startswith('//') and j > ds_start):
                        break
                docstring = '\n'.join(ds_lines)[:500]

        method_sigs: List[str] = []
        if kind == 'class':
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
                kw_pat = re.compile(r'(?:func|fn|def|function|void|public|private|protected)\s+\w+\s*\(')
                for i in range(def_line_idx + 1, min(def_line_idx + 60, len(lines))):
                    if kw_pat.search(lines[i]):
                        method_sigs.append(f'  {lines[i].rstrip()[:120]}')

        end_idx = min(def_line_idx + 40, len(lines))
        code_snippet = ''.join(lines[def_line_idx:end_idx])
        if method_sigs:
            code_snippet += '\n  # ... methods:\n' + '\n'.join(method_sigs[:10])

        summary = ''
        try:
            summary_prompt = _SYMBOL_SUMMARY_PROMPT_TMPL.format(
                file_path=file_path, symbol_name=symbol_name, code_snippet=code_snippet[:1500],
            )
            summary = _safe_llm_call_text(llm, summary_prompt) or ''
        except Exception:
            summary = signature

        deps: List[str] = []
        import_pat = _LANG_IMPORT_PATTERNS.get(ext)
        for line in lines[:80]:
            if not import_pat:
                break
            m = import_pat.match(line)
            if not m:
                continue
            if ext == '.py':
                if m.group(2):
                    for sym in re.split(r',\s*', m.group(2)):
                        sym = sym.strip().split(' as ')[0].strip()
                        if sym and sym[0].isupper():
                            deps.append(f'{m.group(1)}::{sym}')
                elif m.group(3):
                    deps.append(m.group(3).strip())
            elif ext == '.rs':
                deps.append(m.group(1) + (f'::{{{m.group(2)}}}' if m.group(2) else ''))
            else:
                deps.append(next(g for g in m.groups() if g) or '')

        entry: Dict[str, Any] = {
            'key': cache_key, 'kind': kind, 'file': file_path, 'line_start': def_line_idx + 1,
            'signature': signature, 'docstring': docstring, 'summary': summary,
            'method_signatures': method_sigs[:15], 'deps': deps[:10],
        }
        _cache_set(cache_key, entry)

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
    clone_dir: str, llm: Any, symbol_cache: Dict[str, Any],
    owner_repo: str = '', cache_path: Optional[str] = None,
    cache_lock: Optional[Any] = None,
) -> list:
    tools = _build_scoped_agent_tools(clone_dir, owner_repo=owner_repo, cache_path=cache_path)
    analyze_symbol = _build_analyze_symbol_tool(llm, clone_dir, symbol_cache, cache_lock=cache_lock)
    all_tools = tools + [analyze_symbol]
    missing = [fn.__name__ for fn in all_tools if callable(fn) and not getattr(fn, '__doc__', None)]
    if missing:
        lazyllm.LOG.warning(
            f'Agent tools missing docstring (will cause ReactAgent init failure): {missing}'
        )
    return all_tools
