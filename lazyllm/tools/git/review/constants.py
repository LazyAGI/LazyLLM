# Copyright (c) 2026 LazyAGI. All rights reserved.
import math
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

# Single-request context budget (chars). Tune if the backend uses token limits.
SINGLE_CALL_CONTEXT_BUDGET = 120000

# Reserve for system prompt + arch + metadata when packing diff into one call
R1_DIFF_BUDGET = SINGLE_CALL_CONTEXT_BUDGET - 25000

# Issue density: at most M issues per LINE_BLOCK lines of effective diff (+/- lines)
ISSUE_DENSITY_LINE_BLOCK = 100
ISSUE_DENSITY_MAX_PER_BLOCK = 5

# Call budget: maximum LLM calls for the entire review session
TOTAL_CALL_BUDGET = 60

# R3 throttle limits (agent verification round)
R3_MAX_FILES = 20
R3_MAX_CHUNKS_PER_FILE = 3
# Hard upper bound on chunks per file regardless of strategy (prevents runaway LLM calls)
R3_MAX_CHUNKS_HARD = 8

# R3 unit diff budget: max combined diff chars per review unit (anchor + absorbed small files)
R3_UNIT_DIFF_BUDGET = 40000

# backward-compatible aliases (deprecated — use R3_* directly)
R2_MAX_FILES = R3_MAX_FILES
R2_MAX_CHUNKS_PER_FILE = R3_MAX_CHUNKS_PER_FILE
R2_MAX_CHUNKS_HARD = R3_MAX_CHUNKS_HARD
R2_UNIT_DIFF_BUDGET = R3_UNIT_DIFF_BUDGET

# R1 window limits: split large hunk lists into windows to avoid truncation
R1_WINDOW_MAX_HUNKS = 30
R1_WINDOW_MAX_DIFF_CHARS = 60000

ISSUE_DENSITY_RULE_TEXT = (
    f'At most {ISSUE_DENSITY_MAX_PER_BLOCK} issues per {ISSUE_DENSITY_LINE_BLOCK} effective diff lines '
    '(lines starting with + or -, excluding +++/--- file headers); if exceeded, keep highest-severity first.'
)


def issue_density_rule(diff_text: str) -> str:
    # Pre-compute the exact issue cap from the diff and inject it directly into the prompt,
    # so the LLM doesn't have to count lines itself.
    n = max_issues_for_diff(diff_text)
    eff_lines = effective_diff_line_count(diff_text)
    return (
        f'Output AT MOST {n} issues total for this diff '
        f'({eff_lines} effective diff lines × {ISSUE_DENSITY_MAX_PER_BLOCK}/{ISSUE_DENSITY_LINE_BLOCK} cap). '
        f'If you find more, keep the highest-severity ones first.'
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
        if pat.search(line):
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


class BudgetManager:
    # Allocates a fixed total context budget across named slots with priority ordering.
    # Higher priority slots are filled first; lower priority slots receive remaining space
    # up to their registered max_chars cap.
    # Also tracks LLM call count to enforce a total_calls budget across the review session.
    #
    # Usage:
    #   bm = BudgetManager(80000, total_calls=60)
    #   bm.register('arch', priority=10, max_chars=6000)
    #   bm.register('diff', priority=9, max_chars=54000)
    #   bm.register('spec', priority=5, max_chars=600)
    #   result = bm.allocate(arch=arch_text, diff=diff_text, spec=spec_text)
    #   # result values are clipped strings ready for prompt assembly

    def __init__(self, total: int = SINGLE_CALL_CONTEXT_BUDGET, total_calls: int = TOTAL_CALL_BUDGET) -> None:
        self._total = total
        self._total_calls = total_calls
        self._used_calls = 0
        self._slots: Dict[str, Tuple[int, int]] = {}  # name -> (priority, max_chars)
        self._lock = threading.Lock()

    def register(self, name: str, priority: int, max_chars: int) -> 'BudgetManager':
        self._slots[name] = (priority, max_chars)
        return self

    def consume_call(self, n: int = 1) -> bool:
        # Returns True if budget allows; False if over limit (does not consume on False).
        with self._lock:
            if self._used_calls + n > self._total_calls:
                return False
            self._used_calls += n
            return True

    def remaining_calls(self) -> int:
        with self._lock:
            return max(0, self._total_calls - self._used_calls)

    def allocate_calls(self, default: int = 1) -> int:
        # Returns how many calls are available (capped at remaining budget).
        return min(default, self.remaining_calls())

    def allocate(self, **contents: Optional[str]) -> Dict[str, str]:
        # sort slots by priority descending; ties broken by registration order (dict preserves insertion)
        ordered = sorted(self._slots.items(), key=lambda kv: -kv[1][0])
        remaining = self._total
        result: Dict[str, str] = {}
        for name, (_, max_chars) in ordered:
            raw = contents.get(name) or ''
            cap = min(max_chars, remaining)
            if cap <= 0:
                result[name] = ''
                continue
            clipped = raw[:cap] + '\n...(truncated)' if len(raw) > cap else raw
            result[name] = clipped
            remaining -= cap
        # pass through any contents not registered as slots (no truncation)
        for name, val in contents.items():
            if name not in result:
                result[name] = val or ''
        return result
