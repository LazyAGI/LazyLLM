# Copyright (c) 2026 LazyAGI. All rights reserved.
import os
import re
from typing import List, Optional, Tuple

_SKIP_DIRS = {'.git', '__pycache__', '.cache', '.tox', 'node_modules', '.mypy_cache', '.pytest_cache', 'dist', 'build'}
_SKIP_EXTS = {'.pyc', '.pyo', '.so', '.egg', '.egg-info'}
_LARGE_FILE_THRESHOLD = 600
_CONTEXT_LINES = 50


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
        lines.append(f'{indent}{"." if depth == 0 else os.path.basename(root)}/')
        lines.extend(f'{indent}  {f}' for f in sorted(files) if not any(f.endswith(e) for e in _SKIP_EXTS))
    return '\n'.join(lines)


def _read_file_head(path: str, max_bytes: int) -> str:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read(max_bytes)
    except OSError:
        return ''


def _find_enclosing_scope(lines: List[str], hunk_start: int) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    for i in range(min(hunk_start - 1, len(lines) - 1), -1, -1):
        m = re.match(r'^(\s*)(class|def)\s+(\w+)', lines[i])
        if m:
            return i, m.group(2), m.group(3)
    return None, None, None


def _find_enclosing_class(lines: List[str], from_idx: int) -> Optional[int]:
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
    return [line.rstrip()[:120] for line in lines if re.match(r'^def\s+\w+', line)]


def _skeleton_collect_imports(lines: List[str], i: int) -> Tuple[str, int]:
    stripped = lines[i].strip()
    if lines[i][0] == ' ' or lines[i][0] == '\t':
        return '', i + 1
    return stripped[:120], i + 1


def _skeleton_collect_constant(lines: List[str], i: int) -> Tuple[str, int]:
    stripped = lines[i].strip()
    if "'''" in stripped or '"""' in stripped:
        return '', i + 1
    return stripped[:80], i + 1


def _skeleton_collect_function(lines: List[str], i: int) -> Tuple[str, int]:
    stripped = lines[i].strip()
    sig = stripped.rstrip()
    if ')' not in sig:
        for j in range(i + 1, min(i + 5, len(lines))):
            sig += ' ' + lines[j].strip()
            if ')' in lines[j]:
                break
    return sig.split('\n')[0][:120], i + 1


def _extract_file_skeleton(clone_dir: str, path: str, max_chars: int = 3000) -> str:
    abs_path = os.path.join(clone_dir, path) if clone_dir else path
    if not os.path.isfile(abs_path):
        return ''
    try:
        with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except OSError:
        return ''

    parts: List[str] = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if raw and raw[0] in (' ', '\t'):
            i += 1
            continue
        if re.match(r'^(?:import|from)\s+', stripped):
            entry, i = _skeleton_collect_imports(lines, i)
        elif re.match(r'^[A-Z_][A-Z0-9_]*\s*=', stripped) or re.match(r'^_[A-Z][A-Z0-9_]*\s*=', stripped):
            entry, i = _skeleton_collect_constant(lines, i)
        elif re.match(r'^class\s+\w+', stripped):
            entry = stripped.rstrip()[:120]
            sigs = _extract_class_method_signatures(lines, i)
            parts.append(entry)
            parts.extend(sigs[:20])
            i += 1
            continue
        elif re.match(r'^def\s+\w+', stripped):
            entry, i = _skeleton_collect_function(lines, i)
        else:
            i += 1
            continue
        if entry:
            parts.append(entry)

    return '\n'.join(parts)[:max_chars]


def _extract_abstract_method_names(diff_content: str) -> List[str]:
    names: List[str] = []
    lines = diff_content.splitlines()
    prev_abstract = False
    for line in lines:
        stripped = line.lstrip('+ ')
        if stripped.strip() == '@abstractmethod':
            prev_abstract = True
            continue
        if prev_abstract:
            m = re.match(r'def\s+(\w+)\s*\(', stripped)
            if m:
                names.append(m.group(1))
            prev_abstract = False
        else:
            prev_abstract = False
    return names


def _find_subclass_implementations(clone_dir: str, method_names: List[str], max_files: int = 8) -> str:
    if not method_names or not clone_dir or not os.path.isdir(clone_dir):
        return ''
    results: List[str] = []
    pattern = re.compile(r'^\s{4,}def\s+(' + '|'.join(re.escape(n) for n in method_names) + r')\s*\(')
    files_checked = 0
    for root, dirs, files in os.walk(clone_dir):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fname in files:
            if not fname.endswith('.py'):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                    file_lines = f.readlines()
            except OSError:
                continue
            for i, line in enumerate(file_lines):
                if pattern.match(line):
                    rel = os.path.relpath(fpath, clone_dir)
                    sig = line.rstrip()[:120]
                    results.append(f'{rel}:{i + 1}: {sig}')
                    files_checked += 1
                    if files_checked >= max_files:
                        break
            if files_checked >= max_files:
                break
        if files_checked >= max_files:
            break
    return '\n'.join(results)


def _read_file_context(clone_dir: str, path: str, hunk_start: int, hunk_end: int) -> str:
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
        start, end = max(0, hunk_start - 1 - _CONTEXT_LINES), min(total, hunk_end + _CONTEXT_LINES)
        numbered = ''.join(f'{start + i + 1:4d} | {ln}' for i, ln in enumerate(lines[start:end]))
        base = f'(excerpt lines {start + 1}–{end} of {total})\n{numbered}'

    scope_idx, scope_kind, scope_name = _find_enclosing_scope(lines, hunk_start)
    if scope_idx is None:
        return base

    extras: List[str] = [f'\n[Enclosing scope: {scope_kind} {scope_name} (line {scope_idx + 1})]']
    if scope_kind == 'def':
        class_idx = _find_enclosing_class(lines, scope_idx)
        if class_idx is not None:
            cm = re.match(r'^\s*class\s+(\w+)', lines[class_idx])
            extras[0] += f' inside class {cm.group(1) if cm else "?"} (line {class_idx + 1})'
            sigs = _extract_class_method_signatures(lines, class_idx)
            if sigs:
                extras += ['[Sibling method signatures of enclosing class]'] + sigs
        else:
            sigs = [s for s in _extract_module_function_signatures(lines)
                    if not re.match(rf'^def\s+{re.escape(scope_name)}\s*\(', s)]
            if sigs:
                extras += ['[Other top-level function signatures in this file]'] + sigs[:20]
    else:
        sigs = _extract_class_method_signatures(lines, scope_idx)
        if sigs:
            extras += ['[Method signatures of enclosing class]'] + sigs
    return base + '\n' + '\n'.join(extras)
