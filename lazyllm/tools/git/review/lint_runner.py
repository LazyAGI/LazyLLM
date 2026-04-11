# Copyright (c) 2026 LazyAGI. All rights reserved.
import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Set, Tuple

import lazyllm

# Ordered list of preferred lint tools per language extension.
# First available tool in the list will be used.
_LINT_TOOLS: Dict[str, List[str]] = {
    'py': ['ruff', 'flake8'],
    'js': ['eslint'],
    'ts': ['eslint'],
    'jsx': ['eslint'],
    'tsx': ['eslint'],
    'go': ['golint'],
    'rb': ['rubocop'],
    'rs': ['clippy'],
}

# Severity mapping from lint tool exit codes / message patterns
_SEVERITY_KEYWORDS = {
    'critical': ['error', 'E9', 'F8', 'syntax'],
    'medium': ['warning', 'W', 'C'],
}


def _find_lint_tool(ext: str) -> str:
    for tool in _LINT_TOOLS.get(ext, []):
        if shutil.which(tool):
            return tool
    return ''


def _parse_changed_lines(diff_text: str) -> Dict[str, Set[int]]:
    # Returns {path: set_of_new_file_line_numbers} for all added/modified lines.
    result: Dict[str, Set[int]] = {}
    current_file = ''
    current_line = 0
    for line in diff_text.splitlines():
        if line.startswith('+++ b/'):
            current_file = line[6:].strip()
            result.setdefault(current_file, set())
        elif line.startswith('@@'):
            m = re.search(r'\+(\d+)', line)
            current_line = int(m.group(1)) if m else 0
        elif current_file:
            if line.startswith('+') and not line.startswith('+++'):
                result[current_file].add(current_line)
                current_line += 1
            elif not line.startswith('-'):
                current_line += 1
    return result


def _infer_severity(message: str) -> str:
    msg_lower = message.lower()
    for kw in _SEVERITY_KEYWORDS['critical']:
        if kw in msg_lower:
            return 'critical'
    for kw in _SEVERITY_KEYWORDS['medium']:
        if kw in msg_lower:
            return 'medium'
    return 'normal'


def _run_ruff(file_path: str, cwd: str) -> List[Tuple[int, str]]:
    # Returns list of (line_number, message)
    try:
        result = subprocess.run(
            ['ruff', 'check', '--output-format=text', file_path],
            capture_output=True, text=True, cwd=cwd, timeout=30,
        )
        issues: List[Tuple[int, str]] = []
        for line in result.stdout.splitlines():
            # format: path:line:col: CODE message
            m = re.match(r'.+?:(\d+):\d+:\s+(.+)', line)
            if m:
                issues.append((int(m.group(1)), m.group(2).strip()))
        return issues
    except Exception as e:
        lazyllm.LOG.warning(f'ruff failed on {file_path}: {e}')
        return []


def _run_flake8(file_path: str, cwd: str) -> List[Tuple[int, str]]:
    try:
        result = subprocess.run(
            ['flake8', '--max-line-length=121', file_path],
            capture_output=True, text=True, cwd=cwd, timeout=30,
        )
        issues: List[Tuple[int, str]] = []
        for line in result.stdout.splitlines():
            m = re.match(r'.+?:(\d+):\d+:\s+(.+)', line)
            if m:
                issues.append((int(m.group(1)), m.group(2).strip()))
        return issues
    except Exception as e:
        lazyllm.LOG.warning(f'flake8 failed on {file_path}: {e}')
        return []


def _run_eslint(file_path: str, cwd: str) -> List[Tuple[int, str]]:
    try:
        result = subprocess.run(
            ['eslint', '--format=compact', file_path],
            capture_output=True, text=True, cwd=cwd, timeout=30,
        )
        issues: List[Tuple[int, str]] = []
        for line in result.stdout.splitlines():
            # format: path: line col, severity - message  rule
            m = re.match(r'.+?: line (\d+),.*?- (.+)', line)
            if m:
                issues.append((int(m.group(1)), m.group(2).strip()))
        return issues
    except Exception as e:
        lazyllm.LOG.warning(f'eslint failed on {file_path}: {e}')
        return []


def _run_golint(file_path: str, cwd: str) -> List[Tuple[int, str]]:
    try:
        result = subprocess.run(
            ['golint', file_path],
            capture_output=True, text=True, cwd=cwd, timeout=30,
        )
        issues: List[Tuple[int, str]] = []
        for line in result.stdout.splitlines():
            m = re.match(r'.+?:(\d+):\d+:\s+(.+)', line)
            if m:
                issues.append((int(m.group(1)), m.group(2).strip()))
        return issues
    except Exception as e:
        lazyllm.LOG.warning(f'golint failed on {file_path}: {e}')
        return []


_TOOL_RUNNERS = {
    'ruff': _run_ruff,
    'flake8': _run_flake8,
    'eslint': _run_eslint,
    'golint': _run_golint,
}


def _run_lint_analysis(diff_text: str, clone_dir: str) -> List[Dict[str, Any]]:
    changed_lines = _parse_changed_lines(diff_text)
    issues: List[Dict[str, Any]] = []
    warned_missing: set = set()

    for rel_path, line_set in changed_lines.items():
        if not line_set:
            continue
        ext = os.path.splitext(rel_path)[1].lstrip('.')
        if ext not in _LINT_TOOLS:
            continue

        tool = _find_lint_tool(ext)
        if not tool:
            if ext not in warned_missing:
                lazyllm.LOG.warning(
                    f'Lint: no tool available for .{ext} files '
                    f'(tried: {", ".join(_LINT_TOOLS[ext])}); skipping'
                )
                warned_missing.add(ext)
            continue

        abs_path = os.path.join(clone_dir, rel_path)
        if not os.path.isfile(abs_path):
            continue

        runner = _TOOL_RUNNERS.get(tool)
        if runner is None:
            continue

        raw_issues = runner(abs_path, clone_dir)
        for line_no, message in raw_issues:
            if line_no not in line_set:
                continue  # only keep issues on changed lines
            issues.append({
                'path': rel_path,
                'line': line_no,
                'severity': _infer_severity(message),
                'bug_category': 'style',
                'problem': message,
                'suggestion': f'Fix the lint issue reported by {tool}: {message}',
                'source': 'lint',
            })

    if issues:
        lazyllm.LOG.info(f'Lint analysis: {len(issues)} issues found across {len(changed_lines)} files')
    return issues
