# Copyright (c) 2026 LazyAGI. All rights reserved.
import ast
import dataclasses
import os
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import lazyllm


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ImportEdge:
    source: str
    target: str
    symbol: str
    line: int


# ---------------------------------------------------------------------------
# Layer 1: Language-specific import extractors
# ---------------------------------------------------------------------------

# --- path resolution helpers ---

def _resolve_python_import(
    source_file: str, module: Optional[str], level: int, clone_dir: str,
) -> Optional[str]:
    '''Resolve a Python import to a relative file path under clone_dir.'''
    if level > 0:
        # relative import
        source_dir = os.path.dirname(os.path.join(clone_dir, source_file))
        for _ in range(level - 1):
            source_dir = os.path.dirname(source_dir)
        if module:
            parts = module.split('.')
            candidate_dir = os.path.join(source_dir, *parts)
        else:
            candidate_dir = source_dir
    else:
        if not module:
            return None
        parts = module.split('.')
        candidate_dir = os.path.join(clone_dir, *parts)

    # try xxx.py then xxx/__init__.py
    as_file = candidate_dir + '.py'
    if os.path.isfile(as_file):
        return os.path.relpath(as_file, clone_dir)
    init_file = os.path.join(candidate_dir, '__init__.py')
    if os.path.isfile(init_file):
        return os.path.relpath(init_file, clone_dir)
    return None


def _resolve_js_import(source_file: str, raw_path: str, clone_dir: str) -> Optional[str]:
    '''Resolve a JS/TS import path to a relative file path under clone_dir.'''
    if not raw_path.startswith('.'):
        return None  # node_modules / bare specifier
    source_dir = os.path.dirname(os.path.join(clone_dir, source_file))
    base = os.path.normpath(os.path.join(source_dir, raw_path))
    extensions = ['.js', '.ts', '.jsx', '.tsx', '.mjs', '.cjs']
    # exact match with extension already present
    if os.path.isfile(base):
        return os.path.relpath(base, clone_dir)
    for ext in extensions:
        candidate = base + ext
        if os.path.isfile(candidate):
            return os.path.relpath(candidate, clone_dir)
    # directory index
    for idx in ['index.js', 'index.ts', 'index.jsx', 'index.tsx']:
        candidate = os.path.join(base, idx)
        if os.path.isfile(candidate):
            return os.path.relpath(candidate, clone_dir)
    return None


def _resolve_go_import(raw_path: str, clone_dir: str) -> Optional[str]:
    '''Resolve a Go import path using go.mod module prefix.'''
    go_mod = os.path.join(clone_dir, 'go.mod')
    if not os.path.isfile(go_mod):
        return None
    try:
        with open(go_mod, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.startswith('module '):
                    mod_prefix = line.split()[1].strip()
                    break
            else:
                return None
    except OSError:
        return None
    if not raw_path.startswith(mod_prefix):
        return None
    local = raw_path[len(mod_prefix):].lstrip('/')
    candidate = os.path.join(clone_dir, local)
    if os.path.isdir(candidate):
        return os.path.relpath(candidate, clone_dir)
    return None


def _resolve_java_import(raw_path: str, clone_dir: str) -> Optional[str]:
    '''Resolve a Java import (com.foo.Bar) to a file path.'''
    parts = raw_path.split('.')
    if not parts:
        return None
    rel = os.path.join(*parts[:-1], parts[-1] + '.java')
    # search common source roots
    for root in ['src/main/java', 'src', '']:
        candidate = os.path.join(clone_dir, root, rel) if root else os.path.join(clone_dir, rel)
        if os.path.isfile(candidate):
            return os.path.relpath(candidate, clone_dir)
    return None


# --- Python extractor (AST + regex fallback) ---

def _extract_from_ast(tree: ast.AST, file_path: str, clone_dir: str) -> List[ImportEdge]:
    edges: List[ImportEdge] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            level = node.level or 0
            if node.module:
                target = _resolve_python_import(file_path, node.module, level, clone_dir)
                if target and target != file_path:
                    names = ', '.join(a.name for a in node.names) if node.names else ''
                    edges.append(ImportEdge(source=file_path, target=target, symbol=names, line=node.lineno))
            elif level > 0 and node.names:
                # `from .. import alpha` — each name could be a submodule
                for alias in node.names:
                    target = _resolve_python_import(file_path, alias.name, level, clone_dir)
                    if target and target != file_path:
                        edges.append(ImportEdge(
                            source=file_path, target=target, symbol=alias.name, line=node.lineno,
                        ))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                target = _resolve_python_import(file_path, alias.name, 0, clone_dir)
                if target and target != file_path:
                    edges.append(ImportEdge(source=file_path, target=target, symbol=alias.name, line=node.lineno))
    return edges


_PYTHON_IMPORT_RE = re.compile(
    r'^\s*(?:from\s+(\.?\.?\.?\S+)\s+import\s+(.+)|import\s+(\S+))', re.MULTILINE,
)


def _extract_from_regex_python(content: str, file_path: str, clone_dir: str) -> List[ImportEdge]:
    edges: List[ImportEdge] = []
    for i, line in enumerate(content.splitlines(), 1):
        m = _PYTHON_IMPORT_RE.match(line)
        if not m:
            continue
        if m.group(1):
            raw_module = m.group(1)
            symbols = m.group(2).strip()
            level = len(raw_module) - len(raw_module.lstrip('.'))
            module = raw_module.lstrip('.') or None
            target = _resolve_python_import(file_path, module, level, clone_dir)
        else:
            module = m.group(3)
            symbols = module
            target = _resolve_python_import(file_path, module, 0, clone_dir)
        if target and target != file_path:
            edges.append(ImportEdge(source=file_path, target=target, symbol=symbols, line=i))
    return edges


def extract_imports_python(file_path: str, content: str, clone_dir: str) -> List[ImportEdge]:
    try:
        tree = ast.parse(content, filename=file_path)
        return _extract_from_ast(tree, file_path, clone_dir)
    except SyntaxError:
        lazyllm.LOG.debug(f'AST parse failed for {file_path}, falling back to regex')
        return _extract_from_regex_python(content, file_path, clone_dir)


# --- JS/TS extractor ---

_JS_IMPORT_RE = re.compile(
    r'''(?:import\s+.*?\s+from\s+['"](.+?)['"]'''
    r'''|require\(\s*['"](.+?)['"]\s*\)'''
    r'''|export\s+.*?\s+from\s+['"](.+?)['"])''',
)


def extract_imports_js(file_path: str, content: str, clone_dir: str) -> List[ImportEdge]:
    edges: List[ImportEdge] = []
    for i, line in enumerate(content.splitlines(), 1):
        for m in _JS_IMPORT_RE.finditer(line):
            raw = m.group(1) or m.group(2) or m.group(3)
            if not raw:
                continue
            target = _resolve_js_import(file_path, raw, clone_dir)
            if target and target != file_path:
                edges.append(ImportEdge(source=file_path, target=target, symbol=raw, line=i))
    return edges


# --- Go extractor ---

_GO_IMPORT_SINGLE_RE = re.compile(r'^\s*import\s+"(.+?)"', re.MULTILINE)
_GO_IMPORT_BLOCK_RE = re.compile(r'import\s*\((.*?)\)', re.DOTALL)
_GO_IMPORT_LINE_RE = re.compile(r'"(.+?)"')


def extract_imports_go(file_path: str, content: str, clone_dir: str) -> List[ImportEdge]:
    edges: List[ImportEdge] = []
    for i, line in enumerate(content.splitlines(), 1):
        m = _GO_IMPORT_SINGLE_RE.match(line)
        if m:
            target = _resolve_go_import(m.group(1), clone_dir)
            if target and target != file_path:
                edges.append(ImportEdge(source=file_path, target=target, symbol=m.group(1), line=i))
    for block_m in _GO_IMPORT_BLOCK_RE.finditer(content):
        block = block_m.group(1)
        block_start = content[:block_m.start()].count('\n') + 1
        for j, bline in enumerate(block.splitlines()):
            lm = _GO_IMPORT_LINE_RE.search(bline)
            if lm:
                target = _resolve_go_import(lm.group(1), clone_dir)
                if target and target != file_path:
                    edges.append(ImportEdge(
                        source=file_path, target=target,
                        symbol=lm.group(1), line=block_start + j + 1,
                    ))
    return edges


# --- Java extractor ---

_JAVA_IMPORT_RE = re.compile(r'^\s*import\s+([\w.]+)\s*;', re.MULTILINE)


def extract_imports_java(file_path: str, content: str, clone_dir: str) -> List[ImportEdge]:
    edges: List[ImportEdge] = []
    for i, line in enumerate(content.splitlines(), 1):
        m = _JAVA_IMPORT_RE.match(line)
        if m:
            raw = m.group(1)
            target = _resolve_java_import(raw, clone_dir)
            if target and target != file_path:
                edges.append(ImportEdge(source=file_path, target=target, symbol=raw, line=i))
    return edges


# --- Extractor registry ---

_IMPORT_EXTRACTORS: Dict[str, Callable[..., List[ImportEdge]]] = {
    'py': extract_imports_python,
    'js': extract_imports_js,
    'ts': extract_imports_js,
    'jsx': extract_imports_js,
    'tsx': extract_imports_js,
    'go': extract_imports_go,
    'java': extract_imports_java,
}


# ---------------------------------------------------------------------------
# Layer 2: Graph construction
# ---------------------------------------------------------------------------

def _parse_changed_files(diff_text: str) -> List[str]:
    '''Extract list of changed file paths from unified diff.'''
    files: List[str] = []
    for line in diff_text.splitlines():
        if line.startswith('+++ b/'):
            files.append(line[6:].strip())
    return files


def _extract_new_import_lines(diff_text: str) -> Dict[str, List[Tuple[int, str]]]:
    '''Extract added import lines from diff, grouped by file.
    Returns {file_path: [(approx_line_no, line_content), ...]}.'''
    result: Dict[str, List[Tuple[int, str]]] = {}
    current_file = ''
    current_line = 0
    for line in diff_text.splitlines():
        if line.startswith('+++ b/'):
            current_file = line[6:].strip()
        elif line.startswith('@@'):
            m = re.search(r'\+(\d+)', line)
            current_line = int(m.group(1)) if m else 0
        elif current_file and current_line > 0:
            if line.startswith('+') and not line.startswith('+++'):
                body = line[1:]
                result.setdefault(current_file, []).append((current_line, body))
                current_line += 1
            elif not line.startswith('-'):
                current_line += 1
    return result


def _read_file_content(clone_dir: str, rel_path: str) -> Optional[str]:
    abs_path = os.path.join(clone_dir, rel_path)
    if not os.path.isfile(abs_path):
        return None
    try:
        with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except OSError:
        return None


def _get_extractor(file_path: str) -> Optional[Callable[..., List[ImportEdge]]]:
    ext = os.path.splitext(file_path)[1].lstrip('.')
    return _IMPORT_EXTRACTORS.get(ext)


def _extract_edges_for_file(
    file_path: str, clone_dir: str,
) -> List[ImportEdge]:
    '''Read a file and extract all its import edges.'''
    extractor = _get_extractor(file_path)
    if not extractor:
        return []
    content = _read_file_content(clone_dir, file_path)
    if not content:
        return []
    try:
        return extractor(file_path, content, clone_dir)
    except Exception as e:
        lazyllm.LOG.debug(f'Import extraction failed for {file_path}: {e}')
        return []


def build_dep_graph(
    diff_text: str,
    clone_dir: str,
) -> Tuple[Dict[str, Set[str]], Set[Tuple[str, str]], Dict[Tuple[str, str], List[ImportEdge]]]:
    '''Build a local dependency graph from diff-changed files + 1-hop neighbors.

    Returns:
        graph: adjacency dict {node: set_of_targets}
        new_edges: edges introduced by this PR (source, target)
        edge_details: maps (source, target) to ImportEdge list for issue reporting
    '''
    changed_files = set(_parse_changed_files(diff_text))
    new_import_lines = _extract_new_import_lines(diff_text)

    graph: Dict[str, Set[str]] = {}
    edge_details: Dict[Tuple[str, str], List[ImportEdge]] = {}
    new_edges: Set[Tuple[str, str]] = set()

    # Step 1: extract edges from changed files, identify new edges
    neighbor_targets: Set[str] = set()
    changed_file_edges: Dict[str, List[ImportEdge]] = {}
    for fpath in changed_files:
        edges = _extract_edges_for_file(fpath, clone_dir)
        changed_file_edges[fpath] = edges
        for e in edges:
            graph.setdefault(e.source, set()).add(e.target)
            graph.setdefault(e.target, set())
            edge_details.setdefault((e.source, e.target), []).append(e)
            neighbor_targets.add(e.target)

    # Identify which edges are new: match full-file edges against diff added lines.
    # An edge is "new" if its import line falls on a line that was added in the diff.
    added_line_sets: Dict[str, Set[int]] = {}
    for fpath, added_lines in new_import_lines.items():
        added_line_sets[fpath] = {lineno for lineno, _ in added_lines}

    for fpath, edges in changed_file_edges.items():
        added = added_line_sets.get(fpath, set())
        if not added:
            continue
        for e in edges:
            if e.line in added:
                new_edges.add((e.source, e.target))

    # Step 2: expand 1-hop neighbors (files imported by changed files)
    for target in neighbor_targets:
        if target in changed_files:
            continue
        edges = _extract_edges_for_file(target, clone_dir)
        for e in edges:
            graph.setdefault(e.source, set()).add(e.target)
            graph.setdefault(e.target, set())
            edge_details.setdefault((e.source, e.target), []).append(e)

    return graph, new_edges, edge_details


# ---------------------------------------------------------------------------
# Layer 3: Analysis engine
# ---------------------------------------------------------------------------

# --- 3a. Module-level aggregation ---

def _infer_module_for_file(file_path: str, clone_dir: str) -> str:
    '''Infer the module (package directory) a file belongs to.
    Walks up from the file's directory looking for __init__.py (Python),
    package.json (JS), or go.mod (Go). Falls back to parent directory.'''
    parts = file_path.replace('\\', '/').split('/')
    if len(parts) <= 1:
        return file_path
    # for Python: find deepest dir with __init__.py
    for depth in range(len(parts) - 1, 0, -1):
        candidate = os.path.join(clone_dir, *parts[:depth])
        if os.path.isfile(os.path.join(candidate, '__init__.py')):
            return '/'.join(parts[:depth])
        if os.path.isfile(os.path.join(candidate, 'package.json')):
            return '/'.join(parts[:depth])
    # fallback: immediate parent directory
    return '/'.join(parts[:-1]) if len(parts) > 1 else file_path


def aggregate_to_module_graph(
    file_graph: Dict[str, Set[str]],
    clone_dir: str,
) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    '''Aggregate file-level graph to module-level graph.
    Returns (module_graph, file_to_module_map).'''
    file_to_mod: Dict[str, str] = {}
    for node in file_graph:
        file_to_mod[node] = _infer_module_for_file(node, clone_dir)

    mod_graph: Dict[str, Set[str]] = {}
    for src, targets in file_graph.items():
        src_mod = file_to_mod.get(src, src)
        mod_graph.setdefault(src_mod, set())
        for tgt in targets:
            tgt_mod = file_to_mod.get(tgt, tgt)
            if src_mod != tgt_mod:
                mod_graph[src_mod].add(tgt_mod)
                mod_graph.setdefault(tgt_mod, set())
    return mod_graph, file_to_mod


# --- 3b. Cycle detection (DFS) ---

_MAX_CYCLE_LENGTH = 5


_MAX_CYCLES_TOTAL = 500  # hard cap to prevent combinatorial explosion on dense graphs


def _dfs_collect_cycles(
    start: str, current: str, path: List[str], seen: Set[str],
    graph: Dict[str, Set[str]], cycles: List[List[str]],
    visited_cycles: Set[Tuple[str, ...]], max_length: int,
) -> None:
    '''Recursive DFS helper for _find_cycles_dfs.'''
    if len(cycles) >= _MAX_CYCLES_TOTAL or len(path) > max_length:
        return
    for neighbor in graph.get(current, set()):
        if neighbor == start and len(path) >= 2:
            ring = path[:]
            min_idx = ring.index(min(ring))
            rotated = tuple(ring[min_idx:] + ring[:min_idx])
            if rotated not in visited_cycles:
                visited_cycles.add(rotated)
                cycles.append(path + [neighbor])
            continue
        if neighbor in seen:
            continue
        seen.add(neighbor)
        path.append(neighbor)
        _dfs_collect_cycles(start, neighbor, path, seen, graph, cycles, visited_cycles, max_length)
        path.pop()
        seen.discard(neighbor)


def _cycle_contains_new_edge(cycle: List[str], new_edges: Set[Tuple[str, str]]) -> bool:
    return any((cycle[i], cycle[i + 1]) in new_edges for i in range(len(cycle) - 1))


def _find_cycles_dfs(
    graph: Dict[str, Set[str]],
    new_edges: Set[Tuple[str, str]],
    max_length: int = _MAX_CYCLE_LENGTH,
) -> List[List[str]]:
    '''Find all simple cycles up to max_length that contain at least one new_edge.'''
    cycles: List[List[str]] = []
    visited_cycles: Set[Tuple[str, ...]] = set()

    for node in graph:
        if len(cycles) >= _MAX_CYCLES_TOTAL:
            lazyllm.LOG.warning(
                f'_find_cycles_dfs: reached {_MAX_CYCLES_TOTAL} cycle limit, stopping early'
            )
            break
        _dfs_collect_cycles(node, node, [node], {node}, graph, cycles, visited_cycles, max_length)

    # filter: only keep cycles containing at least one new_edge
    return [c for c in cycles if _cycle_contains_new_edge(c, new_edges)]


def detect_cycles(
    file_graph: Dict[str, Set[str]],
    new_edges: Set[Tuple[str, str]],
    clone_dir: str,
) -> Tuple[List[List[str]], List[List[str]]]:
    '''Detect cycles at both module and file level.
    Returns (module_cycles, file_cycles).'''
    # module-level
    mod_graph, file_to_mod = aggregate_to_module_graph(file_graph, clone_dir)
    mod_new_edges = {
        (file_to_mod.get(s, s), file_to_mod.get(t, t))
        for s, t in new_edges
        if file_to_mod.get(s, s) != file_to_mod.get(t, t)
    }
    mod_cycles = _find_cycles_dfs(mod_graph, mod_new_edges)

    # file-level
    file_cycles = _find_cycles_dfs(file_graph, new_edges)

    # dedup: remove file cycles fully covered by a module cycle
    if mod_cycles:
        mod_cycle_sets = [set(c[:-1]) for c in mod_cycles]
        deduped_file_cycles: List[List[str]] = []
        for fc in file_cycles:
            fc_mods = {file_to_mod.get(n, n) for n in fc[:-1]}
            covered = any(fc_mods <= mcs for mcs in mod_cycle_sets)
            if not covered:
                deduped_file_cycles.append(fc)
        file_cycles = deduped_file_cycles

    return mod_cycles, file_cycles


# --- 3c. Dependency inversion detection ---

@dataclasses.dataclass
class _LayerDef:
    name: str
    level: int
    paths: List[str]


def _load_layer_config(clone_dir: str) -> Optional[List[_LayerDef]]:
    '''Load .dep-layers.yaml from project root if it exists.'''
    try:
        import yaml as _yaml
    except ImportError:
        lazyllm.LOG.debug('PyYAML not installed, skipping .dep-layers.yaml')
        return None
    for fname in ('.dep-layers.yaml', '.dep-layers.yml'):
        fpath = os.path.join(clone_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = _yaml.safe_load(f)
            layers = []
            for item in data.get('layers', []):
                layers.append(_LayerDef(
                    name=item.get('name', ''),
                    level=int(item.get('level', 0)),
                    paths=[p.rstrip('/') for p in item.get('paths', [])],
                ))
            return layers if layers else None
        except Exception as e:
            lazyllm.LOG.warning(f'Failed to parse {fpath}: {e}')
            return None
    return None


_HEURISTIC_LAYERS: Dict[str, int] = {
    'core': 0, 'base': 0, 'common': 0, 'shared': 0, 'lib': 0,
    'utils': 1, 'util': 1, 'helpers': 1, 'helper': 1,
    'models': 2, 'model': 2, 'domain': 2, 'entities': 2,
    'services': 3, 'service': 3,
    'tools': 3, 'plugins': 3,
    'api': 4, 'routes': 4, 'views': 4, 'handlers': 4,
    'controllers': 4, 'endpoints': 4,
    'engine': 4,  # engine depends on tools in this project (tools ← engine)
    'app': 5, 'application': 5, 'main': 5, 'cmd': 5, 'cli': 5,
}


def _get_level_for_path(
    file_path: str, layer_config: Optional[List[_LayerDef]],
) -> Optional[int]:
    '''Get the layer level for a file path.'''
    if layer_config:
        for layer in layer_config:
            for prefix in layer.paths:
                norm_prefix = prefix.rstrip('/')
                if file_path.startswith(norm_prefix + '/') or file_path == norm_prefix:
                    return layer.level
        return None
    # heuristic: check directory names in path
    parts = file_path.replace('\\', '/').split('/')
    for part in parts:
        lower = part.lower()
        if lower in _HEURISTIC_LAYERS:
            return _HEURISTIC_LAYERS[lower]
    return None


def detect_inversions(
    file_graph: Dict[str, Set[str]],
    new_edges: Set[Tuple[str, str]],
    clone_dir: str,
    arch_doc: str = '',
) -> List[Dict[str, Any]]:
    '''Detect dependency inversions: lower-layer importing higher-layer.
    Only reports inversions on new edges.'''
    layer_config = _load_layer_config(clone_dir)
    issues: List[Dict[str, Any]] = []

    for src, tgt in new_edges:
        src_level = _get_level_for_path(src, layer_config)
        tgt_level = _get_level_for_path(tgt, layer_config)
        if src_level is None or tgt_level is None:
            continue
        if src_level < tgt_level:
            issues.append({
                'source': src,
                'target': tgt,
                'src_level': src_level,
                'tgt_level': tgt_level,
            })
    return issues


# ---------------------------------------------------------------------------
# Entry point: _run_dep_analysis
# ---------------------------------------------------------------------------

def _format_cycle_issue(
    cycle: List[str], edge_details: Dict[Tuple[str, str], List[ImportEdge]],
    new_edges: Set[Tuple[str, str]],
    severity: str, is_module_level: bool,
) -> Dict[str, Any]:
    '''Format a cycle into a standard review issue dict.
    Pins the issue to the new_edge import line(s), merging multiple lines.'''
    cycle_str = ' -> '.join(cycle)
    level_label = 'Module' if is_module_level else 'File'
    n = len(cycle) - 1
    # collect all new_edge import lines in this cycle, grouped by file
    file_lines: Dict[str, List[int]] = {}
    for i in range(len(cycle) - 1):
        pair = (cycle[i], cycle[i + 1])
        if pair in new_edges:
            for e in edge_details.get(pair, []):
                file_lines.setdefault(e.source, []).append(e.line)
    # fallback: any edge in the cycle
    if not file_lines:
        for i in range(len(cycle) - 1):
            for e in edge_details.get((cycle[i], cycle[i + 1]), []):
                file_lines.setdefault(e.source, []).append(e.line)
                break
            if file_lines:
                break
    # pick the file with the most lines as primary
    if file_lines:
        primary_file = max(file_lines, key=lambda f: len(file_lines[f]))
        lines = sorted(set(file_lines[primary_file]))
        primary_line = lines[0]
        if len(lines) > 1:
            also = ', '.join(str(ln) for ln in lines[1:])
            line_note = f' (also at line {also})'
        else:
            line_note = ''
    else:
        primary_file = cycle[0]
        primary_line = 1
        line_note = ''
    return {
        'path': primary_file,
        'line': primary_line,
        'severity': severity,
        'bug_category': 'design',
        'problem': f'{level_label}-level circular dependency ({n} nodes): {cycle_str}{line_note}',
        'suggestion': (
            'Break the cycle by extracting shared interfaces/types into a separate module, '
            'or use dependency injection / lazy imports to decouple the modules.'
        ),
        'source': 'dep_check',
    }


def _format_inversion_issue(
    src: str, tgt: str, src_level: int, tgt_level: int,
    all_edges: List[ImportEdge],
) -> Dict[str, Any]:
    '''Format an inversion into a standard review issue dict.
    Merges multiple import lines of the same (src, tgt) pair into one issue.'''
    lines = sorted({e.line for e in all_edges})
    primary_line = lines[0]
    if len(lines) > 1:
        also_lines = ', '.join(str(ln) for ln in lines[1:])
        line_note = f' (also at line {also_lines})'
    else:
        line_note = ''
    return {
        'path': src,
        'line': primary_line,
        'severity': 'normal',
        'bug_category': 'design',
        'problem': (
            f'Dependency inversion: {src} (layer {src_level}) '
            f'imports {tgt} (layer {tgt_level}). '
            f'Lower-layer modules should not depend on higher-layer modules.{line_note}'
        ),
        'suggestion': (
            'Move the shared logic into a lower layer, or define an interface/protocol '
            'in the lower layer that the higher layer implements.'
        ),
        'source': 'dep_check',
    }


def _collect_cycle_issues(
    graph: Dict[str, Set[str]],
    new_edges: Set[Tuple[str, str]],
    edge_details: Dict[Tuple[str, str], List[ImportEdge]],
    clone_dir: str,
) -> List[Dict[str, Any]]:
    '''Run cycle detection and return formatted issues.'''
    issues = []
    mod_cycles, file_cycles = detect_cycles(graph, new_edges, clone_dir)
    for cycle in mod_cycles:
        issues.append(_format_cycle_issue(cycle, edge_details, new_edges, 'medium', True))
    for cycle in file_cycles:
        issues.append(_format_cycle_issue(cycle, edge_details, new_edges, 'normal', False))
    return issues


def _collect_inversion_issues(
    graph: Dict[str, Set[str]],
    new_edges: Set[Tuple[str, str]],
    edge_details: Dict[Tuple[str, str], List[ImportEdge]],
    clone_dir: str,
    arch_doc: str,
) -> List[Dict[str, Any]]:
    '''Run inversion detection and return formatted issues, merged by (src, tgt) pair.'''
    issues = []
    inversions = detect_inversions(graph, new_edges, clone_dir, arch_doc)
    inv_groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for inv in inversions:
        key = (inv['source'], inv['target'])
        if key not in inv_groups:
            inv_groups[key] = inv
    for (src, tgt), inv in inv_groups.items():
        all_edges = edge_details.get((src, tgt), [])
        if not all_edges:
            all_edges = [ImportEdge(source=src, target=tgt, symbol='', line=1)]
        issues.append(_format_inversion_issue(src, tgt, inv['src_level'], inv['tgt_level'], all_edges))
    return issues


def _dedup_issues(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    '''Remove duplicate issues with the same (path, line) location.'''
    seen: Set[Tuple[str, int]] = set()
    result = []
    for issue in issues:
        key = (issue.get('path', ''), issue.get('line', 0))
        if key not in seen:
            seen.add(key)
            result.append(issue)
    return result


def _run_dep_analysis(
    diff_text: str,
    clone_dir: str,
    arch_doc: str = '',
) -> List[Dict[str, Any]]:
    '''Main entry point: run dependency cycle and inversion analysis.
    Returns a list of standard review issue dicts.'''
    if not diff_text or not clone_dir:
        return []

    try:
        graph, new_edges, edge_details = build_dep_graph(diff_text, clone_dir)
    except Exception as e:
        lazyllm.LOG.warning(f'Dep graph construction failed: {e}')
        return []

    if not new_edges:
        return []

    raw: List[Dict[str, Any]] = []
    try:
        raw.extend(_collect_cycle_issues(graph, new_edges, edge_details, clone_dir))
    except Exception as e:
        lazyllm.LOG.warning(f'Cycle detection failed: {e}')
    try:
        raw.extend(_collect_inversion_issues(graph, new_edges, edge_details, clone_dir, arch_doc))
    except Exception as e:
        lazyllm.LOG.warning(f'Inversion detection failed: {e}')

    issues = _dedup_issues(raw)
    if issues:
        lazyllm.LOG.info(f'Dep analysis: {len(issues)} issue(s) found')
    return issues
