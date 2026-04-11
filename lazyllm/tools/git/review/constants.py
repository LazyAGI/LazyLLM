# Copyright (c) 2026 LazyAGI. All rights reserved.
import math
import re
from typing import Any, Dict, List

# Single-request context budget (chars). Tune if the backend uses token limits.
SINGLE_CALL_CONTEXT_BUDGET = 80000

# Reserve for system prompt + arch + metadata when packing diff into one call
R1_DIFF_BUDGET = SINGLE_CALL_CONTEXT_BUDGET - 25000

# Issue density: at most M issues per LINE_BLOCK lines of effective diff (+/- lines)
ISSUE_DENSITY_LINE_BLOCK = 100
ISSUE_DENSITY_MAX_PER_BLOCK = 5

ISSUE_DENSITY_RULE_TEXT = (
    f'At most {ISSUE_DENSITY_MAX_PER_BLOCK} issues per {ISSUE_DENSITY_LINE_BLOCK} effective diff lines '
    '(lines starting with + or -, excluding +++/--- file headers); if exceeded, keep highest-severity first.'
)


def estimate_prompt_chars(*parts: str) -> int:
    return sum(len(p or '') for p in parts)


def effective_diff_line_count(diff_text: str) -> int:
    n = 0
    for line in diff_text.splitlines():
        if line.startswith('+++') or line.startswith('---'):
            continue
        if line.startswith('+') or line.startswith('-'):
            n += 1
    return max(n, 1)


def max_issues_for_diff(diff_text: str) -> int:
    return max(
        1,
        math.ceil(effective_diff_line_count(diff_text) / ISSUE_DENSITY_LINE_BLOCK)
        * ISSUE_DENSITY_MAX_PER_BLOCK,
    )


def cap_issues_by_severity(issues: List[Dict[str, Any]], max_n: int) -> List[Dict[str, Any]]:
    if len(issues) <= max_n:
        return issues
    order = {'critical': 0, 'medium': 1, 'normal': 2}
    ranked = sorted(issues, key=lambda c: order.get(c.get('severity', 'normal'), 2))
    return ranked[:max_n]


def clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + '\n...(truncated)'


def clip_diff_by_hunk_budget(diff_text: str, max_chars: int) -> str:
    if len(diff_text) <= max_chars:
        return diff_text
    parts = diff_text.split('@@')
    if len(parts) <= 1:
        return diff_text[:max_chars]
    out: List[str] = []
    cur = 0
    for i, p in enumerate(parts):
        block = ('@@' + p) if i else p
        if cur + len(block) > max_chars:
            break
        out.append(block)
        cur += len(block)
    return ''.join(out) if out else diff_text[:max_chars]


def compress_diff_for_agent_heuristic(diff_text: str, max_chars: int) -> str:
    if len(diff_text) <= max_chars:
        return diff_text
    lines = diff_text.splitlines(keepends=True)
    important: List[str] = []
    rest: List[str] = []
    pat = re.compile(r'def\s+\w+|class\s+\w+|import\s+|from\s+\w+\s+import|@@')
    for line in lines:
        if pat.search(line) or line.startswith('@@'):
            important.append(line)
        else:
            rest.append(line)
    merged = ''.join(important)
    if len(merged) >= max_chars * 0.85:
        return merged[:max_chars]
    for line in rest:
        if len(merged) + len(line) > max_chars:
            break
        merged += line
    return merged[:max_chars]
