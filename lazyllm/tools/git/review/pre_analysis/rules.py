# Copyright (c) 2026 LazyAGI. All rights reserved.
import json
import re
from typing import Any, Dict, List

from ..utils import _safe_format
from .prompt import _RULE_CARD_TEMPLATE
def _format_rule_card(rule: Dict[str, Any]) -> str:
    detect = rule.get('detect') or []
    detect_bullets = '\n'.join(f'- {d}' for d in detect) if detect else '- (see bad example)'
    rationale = rule.get('rationale', '')
    rationale_line = f'\n[Rationale] {rationale}' if rationale else ''
    return _safe_format(
        _RULE_CARD_TEMPLATE,
        rule_id=rule.get('rule_id', 'RULE000'),
        category=rule.get('category', 'other'),
        title=rule.get('title', ''),
        severity=rule.get('severity', 'P2'),
        scope=rule.get('scope', 'repo_wide'),
        detect_bullets=detect_bullets,
        bad_example=rule.get('bad_example') or '(n/a)',
        good_example=rule.get('good_example') or '(n/a)',
        fix=rule.get('fix') or '',
        rationale_line=rationale_line,
    )


def _sample_diff_for_rules(diff_content: str, total_lines: int = 300) -> str:
    lines = diff_content.splitlines()
    n = len(lines)
    if n <= total_lines:
        return diff_content
    seg = total_lines // 3
    head = lines[:seg]
    mid_start = max(seg, n // 2 - seg // 2)
    mid = lines[mid_start: mid_start + seg]
    tail = lines[max(n - seg, mid_start + seg):]
    return '\n'.join(head + ['...'] + mid + ['...'] + tail)


def _extract_symbol_keywords_from_diff(diff_content: str) -> List[str]:
    pat = re.compile(r'^\+\s*(?:def|class)\s+(\w+)', re.MULTILINE)
    return pat.findall(diff_content)[:30]


def _lookup_relevant_rules(review_spec: str, diff_content: str, max_detail: int = 10) -> str:  # noqa: C901
    if not review_spec or review_spec.startswith('('):
        return review_spec or ''
    try:
        spec_obj = json.loads(review_spec)
    except (json.JSONDecodeError, ValueError):
        return review_spec[:2000]

    summaries = spec_obj.get('summaries', [])
    details = spec_obj.get('details', {})

    sample = _sample_diff_for_rules(diff_content)
    symbol_kws = {s.lower() for s in _extract_symbol_keywords_from_diff(diff_content)}
    keywords: set = set(symbol_kws)
    for line in sample.splitlines():
        if line.startswith('+++ ') or line.startswith('--- '):
            fname = line.split('/')[-1].replace('.py', '').lower()
            if fname:
                keywords.add(fname)
        for m in re.finditer(r'\b([A-Z][a-zA-Z0-9]+|[a-z_][a-z_0-9]{3,})\b', line):
            keywords.add(m.group(1).lower())

    matched_ids, unmatched_titles = [], []
    for s in summaries:
        rule_id, title = s.get('rule_id', ''), s.get('title', '')
        if any(kw in title.lower() for kw in keywords if len(kw) > 3):
            matched_ids.append(rule_id)
        else:
            cat = s.get('category', '')
            unmatched_titles.append(f'[{rule_id}] ({cat}) {title}' if cat else f'[{rule_id}] {title}')

    matched_by_cat: Dict[str, List[str]] = {}
    for rid in matched_ids[:max_detail]:
        rule = details.get(rid)
        if not rule:
            continue
        cat = rule.get('category', 'other')
        matched_by_cat.setdefault(cat, []).append(_format_rule_card(rule))
    parts = []
    for cat, cards in sorted(matched_by_cat.items()):
        parts.append(f'## {cat}\n' + '\n\n'.join(cards))
    if unmatched_titles:
        parts.append('## Other rules (title only)\n' + '\n'.join(unmatched_titles))
    conventions = spec_obj.get('conventions', '')
    if conventions:
        parts.append('## Conventions & AI False Positives\n' + conventions)
    repo_norms = spec_obj.get('repo_norms', {})
    norm_lines = []
    for entry in repo_norms.get('mandatory', []):
        norm_lines.append(f'- [MUST] {entry.get("norm", "")}')
    for entry in repo_norms.get('forbidden', []):
        norm_lines.append(f'- [NEVER] {entry.get("norm", "")}')
    for entry in repo_norms.get('preferences', []):
        norm_lines.append(f'- [PREFER] {entry.get("norm", "")}')
    if norm_lines:
        parts.append('## Repo-Level Norms\n' + '\n'.join(norm_lines))
    return '\n\n'.join(parts) if parts else '(no matching rules found)'
