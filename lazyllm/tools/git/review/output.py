# Copyright (c) 2026 LazyAGI. All rights reserved.
import json
import os
from typing import Any, Dict

_FIX_GUIDE_INSTRUCTIONS = (
    'You are reviewing AI-generated code review issues. For each issue:\n\n'
    'RESPONSIBILITIES:\n'
    '- Judge whether the issue is reliable (valid concern) or not reliable'
    ' (misunderstanding, incorrect assumption, already fixed, or invalid constraint)\n'
    '- If reliable, apply a concrete fix in code\n'
    '- Do not skip any issue, even if duplicated or low quality\n\n'
    'IMPORTANT RULES:\n'
    '- Do not accept an issue just because "others said so"\n'
    '- New architectural issues introduced by this change MUST be fixed\n'
    '- Pre-existing issues should be: fixed if trivial, OR tracked via an issue (new or linked)\n'
    '- Always verify whether the issue is already resolved before acting\n'
    '- Avoid over-fixing: only modify code when the issue is valid and actionable\n\n'
    'REQUIRED OUTPUT FORMAT (for each issue):\n'
    '- Issue ID\n'
    '- Judgment: Reliable / Not Reliable\n'
    '- Reasoning\n'
    '- Action: Fixed / Not Fixed / Not Applicable\n'
    '- Fix Description (if applicable)\n\n'
    'Group issues by category: correctness, performance, architecture, style, maintainability.\n'
    'Every issue must be independently addressable. No issue may be omitted.\n\n'
    'GOAL: All valid issues fixed. All invalid issues explicitly rejected with reasoning.'
    ' Report is clear, complete, and auditable.'
)


def write_review_json(result: Dict[str, Any], path: str) -> None:
    issues = []
    for idx, c in enumerate(result.get('comments') or [], start=1):
        if c.get('type') == 'meta':
            continue
        issues.append({
            'id': idx,
            'file': c.get('path', ''),
            'line': c.get('line'),
            'severity': c.get('severity', 'normal'),
            'category': c.get('bug_category', 'maintainability'),
            'problem': c.get('problem', ''),
            'suggestion': c.get('suggestion', ''),
        })

    pr_info_raw = result.get('pr_info') or {}
    output = {
        'instructions': _FIX_GUIDE_INSTRUCTIONS,
        'pr_info': {
            'branch': pr_info_raw.get('source_branch', ''),
            'base': pr_info_raw.get('target_branch', ''),
            'summary': result.get('pr_summary', ''),
        },
        'stats': result.get('comment_stats') or _compute_stats(issues),
        'issues': issues,
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def _compute_stats(issues: list) -> Dict[str, int]:
    stats: Dict[str, int] = {'total': len(issues), 'critical': 0, 'normal': 0, 'suggestion': 0}
    for issue in issues:
        sev = issue.get('severity', 'normal')
        stats[sev] = stats.get(sev, 0) + 1
    return stats
