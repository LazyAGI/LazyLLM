# Copyright (c) 2026 LazyAGI. All rights reserved.
'''RCov: test coverage checker for new and modified functionality.'''
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from typing import Any, Dict, List, Optional, Tuple

import lazyllm

from .utils import (
    _Progress, _safe_llm_call,
    _language_instruction, _normalize_comment_item,
    JSON_OUTPUT_INSTRUCTION,
)
from .constants import clip_text, clip_diff_by_hunk_budget, SINGLE_CALL_CONTEXT_BUDGET

# Per-group LLM call timeout (seconds). One group = 1 _safe_llm_call with up to 3 retries.
# 90s covers a slow model response + retry delays (3+8+20s).
_RCOV_GROUP_TIMEOUT_SECS = 90


# ---------------------------------------------------------------------------
# Step 1 prompt: identify testable symbols from diff
# ---------------------------------------------------------------------------

_RCOV_IDENTIFY_PROMPT_TMPL = '''\
You are a senior engineer analyzing a code diff to identify which functions and classes \
need test coverage.

{lang_instruction}

## PR Summary
{pr_summary}

## Diff
{diff_text}

## Task
Identify functions and classes that need test coverage:
1. Newly ADDED functions/classes (is_new: true)
2. Significantly MODIFIED existing functions/classes (is_new: false) — \
   only include if the modification changes observable behavior or adds new code paths

For each symbol, provide:
- "symbol": the function or class name (string)
- "file": the source file path (relative, as it appears in the diff)
- "is_new": true if newly added, false if modified
- "is_internal": true if this is an internal helper that should NOT need direct tests \
  (see criteria below), false otherwise
- "test_scenarios": list of 2-5 concrete test scenarios that should be covered

## Criteria for is_internal = true (skip direct testing)
Mark is_internal=true if ALL of the following apply:
- The symbol name starts with `_` (Python private convention), OR
- The class/function is not exported in `__all__` and is only referenced within the same file \
  or same package, AND
- It does NOT contain complex independent logic (e.g. pure data container, simple wrapper, \
  trivial delegation), AND
- Its behavior is fully exercised through the public API that calls it

Mark is_internal=false (needs testing) if ANY of the following apply:
- The symbol is part of the public API (no leading `_`, or explicitly in `__all__`)
- The symbol contains non-trivial logic (state machine, parser, algorithm, error handling)
- The symbol is a new class with multiple methods that form an independent contract

''' + JSON_OUTPUT_INSTRUCTION + '''
Output a JSON array of objects with fields: symbol, file, is_new, is_internal, test_scenarios.
If no symbols need testing: output [].
'''

# ---------------------------------------------------------------------------
# Step 1.5 prompt: dependency analysis — group related symbols
# ---------------------------------------------------------------------------

_RCOV_DEPENDENCY_PROMPT_TMPL = '''\
You are analyzing dependencies between symbols to group them for test coverage evaluation.

{lang_instruction}

## Symbols to Analyze
{symbols_json}

## Task
Group symbols that are closely related and should be evaluated for test coverage together.
Two symbols should be in the same group if:
- One calls or instantiates the other
- They share state or are part of the same workflow
- Testing one naturally exercises the other

For each group, provide:
- "group_id": integer starting from 0
- "symbols": list of symbol names in this group
- "rationale": one sentence explaining why they are grouped

Symbols that have no dependencies with others should each be their own group (singleton group).

''' + JSON_OUTPUT_INSTRUCTION + '''
Output a JSON array of group objects with fields: group_id, symbols, rationale.
'''

# ---------------------------------------------------------------------------
# Step 2 prompt: evaluate coverage for a group of related symbols
# ---------------------------------------------------------------------------

_RCOV_EVALUATE_PROMPT_TMPL = '''\
You are a senior engineer evaluating test coverage for a group of related symbols.

{lang_instruction}

## PR Summary
{pr_summary}

## Symbols in This Group
{symbols_json}

## Grouping Rationale
{rationale}

## Test File Search Results
The following shows grep results for each symbol name across all test files.
A symbol with NO grep hits likely has no test coverage.
A symbol with hits may or may not have adequate coverage — use judgment.
Note: if symbols are grouped, coverage of one may imply coverage of another.

{grep_results}

{usage_scenarios_block}
## Task
Evaluate whether test coverage is ADEQUATE, PARTIAL, or MISSING for this group as a whole.

- ADEQUATE: existing tests clearly exercise the main scenarios for the symbols in this group
- PARTIAL: some tests exist but important scenarios are not covered
- MISSING: no tests found, or only trivial/unrelated mentions

Report issues ONLY for PARTIAL and MISSING coverage. For each issue:
- "path": the source file path (not the test file)
- "line": 1 (use 1 when no specific line is applicable)
- "severity": "normal" for all coverage issues
- "bug_category": "maintainability"
- "problem": one sentence describing what coverage is missing
- "suggestion": list the specific test scenarios that should be added

STRICT RULES:
- Do NOT report ADEQUATE coverage as an issue
- Do NOT report trivial getters/setters or pure data classes
- Do NOT report symbols marked is_internal=true — they are covered via public API tests
- Consider indirect coverage: if symbol A calls symbol B, tests for A may cover B
- Focus on behavioral coverage, not line coverage

''' + JSON_OUTPUT_INSTRUCTION + '''
Output a JSON array of issue objects. If coverage is adequate: output [].
'''


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_test_files(clone_dir: str) -> List[str]:
    '''Find all test files in the repository.'''
    skip_dirs = {'.git', '__pycache__', '.cache', 'node_modules', 'dist', 'build'}
    found: List[str] = []
    base_dir = os.path.realpath(clone_dir)
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
        for fname in files:
            if fname.endswith('.py') and (fname.startswith('test_') or fname.endswith('_test.py')):
                final_path = os.path.realpath(os.path.join(root, fname))
                if final_path.startswith(base_dir):
                    found.append(final_path)
    return found


def _grep_find_matching_files(symbol: str, test_files: List[str]) -> List[str]:
    '''Find test files containing symbol, using grep with fallback to plain text search.'''
    try:
        result = subprocess.run(
            ['grep', '-rn', '--include=*.py', '-l', '-F', symbol] + test_files,
            capture_output=True, text=True, timeout=10,
        )
        return [f for f in result.stdout.strip().splitlines() if f]
    except (subprocess.TimeoutExpired, OSError):
        matching: List[str] = []
        for f in test_files:
            try:
                with open(f, errors='ignore') as fh:
                    if any(symbol in line for line in fh):
                        matching.append(f)
            except OSError:
                pass
        return matching


def _grep_collect_sample_lines(symbol: str, matching_files: List[str], clone_dir: str) -> List[str]:
    '''Collect up to 3 sample matching lines per file (first 3 files).'''
    sample_lines: List[str] = []
    for fpath in matching_files[:3]:
        rel = os.path.relpath(fpath, clone_dir)
        try:
            result = subprocess.run(
                ['grep', '-n', '-F', symbol, fpath],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.strip().splitlines()[:3]:
                sample_lines.append(f'  {rel}: {line}')
        except (subprocess.TimeoutExpired, FileNotFoundError):
            sample_lines.append(f'  {rel}: (found but could not read lines)')
    return sample_lines


def _grep_symbol_in_tests(symbol: str, test_files: List[str], clone_dir: str) -> str:
    '''Grep for a symbol name across test files, return a summary string.'''
    if not test_files or not symbol:
        return '(no test files found)'
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*', symbol):
        lazyllm.LOG.warning(f'  [RCov] Skipping unsafe symbol name: {symbol!r}')
        return '(skipped unsafe symbol name)'

    matching_files = _grep_find_matching_files(symbol, test_files)
    if not matching_files:
        return f'No test files contain "{symbol}"'

    sample_lines = _grep_collect_sample_lines(symbol, matching_files, clone_dir)
    files_str = ', '.join(os.path.relpath(f, clone_dir) for f in matching_files[:5])
    extra = f' (+{len(matching_files) - 5} more)' if len(matching_files) > 5 else ''
    header = f'Found in {len(matching_files)} test file(s): {files_str}{extra}'
    return header + ('\n' + '\n'.join(sample_lines) if sample_lines else '')


def _build_grep_results(
    symbols: List[Dict[str, Any]], test_files: List[str], clone_dir: str,
) -> str:
    '''Build grep results string for a list of symbols (parallel grep).'''
    if not symbols:
        return '(no grep results)'
    parts: List[Tuple[str, str]] = []

    def _grep_one(sym_info: Dict[str, Any]) -> Tuple[str, str]:
        symbol = sym_info.get('symbol', '')
        file_path = sym_info.get('file', '')
        is_new = sym_info.get('is_new', True)
        kind = 'NEW' if is_new else 'MODIFIED'
        result = _grep_symbol_in_tests(symbol, test_files, clone_dir)
        return f'### {symbol} ({kind}) in {file_path}', result

    with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as ex:
        futs = {ex.submit(_grep_one, s): i for i, s in enumerate(symbols)}
        ordered: Dict[int, Tuple[str, str]] = {}
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                ordered[idx] = fut.result()
            except Exception as exc:
                sym_name = symbols[idx].get('symbol', '')
                lazyllm.LOG.warning(f'  [RCov] Grep failed for symbol "{sym_name}": {exc}')
                ordered[idx] = (f'### {sym_name}', '(grep failed)')

    for i in range(len(symbols)):
        header, result = ordered.get(i, ('', ''))
        parts.append(f'{header}\n{result}')

    return '\n\n'.join(parts) if parts else '(no grep results)'


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _rcov_identify_symbols(
    llm: Any, diff_text: str, pr_summary: str, lang_instr: str,
) -> List[Dict[str, Any]]:
    '''Step 1: identify testable symbols from diff, filter internal helpers.'''
    diff_use = clip_diff_by_hunk_budget(diff_text, SINGLE_CALL_CONTEXT_BUDGET - 20000)
    identify_prompt = _RCOV_IDENTIFY_PROMPT_TMPL.format(
        lang_instruction=lang_instr,
        pr_summary=(pr_summary or '')[:600],
        diff_text=diff_use,
    )
    symbols_raw = _safe_llm_call(llm, identify_prompt)
    all_symbols = [
        s for s in (symbols_raw if isinstance(symbols_raw, list) else [])
        if isinstance(s, dict) and s.get('symbol') and s.get('file')
    ]
    symbols = [s for s in all_symbols if not s.get('is_internal', False)]
    n_filtered = len(all_symbols) - len(symbols)
    lazyllm.LOG.info(f'  [RCov] {len(symbols)} testable symbols ({n_filtered} internal helpers filtered out)')
    return symbols


def _rcov_group_symbols(
    llm: Any, symbols: List[Dict[str, Any]], lang_instr: str,
) -> List[Dict[str, Any]]:
    '''Step 1.5: dependency analysis — group related symbols via LLM.'''
    groups: List[Dict[str, Any]] = []
    if len(symbols) > 1:
        dep_prompt = _RCOV_DEPENDENCY_PROMPT_TMPL.format(
            lang_instruction=lang_instr,
            symbols_json=json.dumps(
                [{'symbol': s['symbol'], 'file': s['file'], 'is_new': s.get('is_new', True)}
                 for s in symbols],
                ensure_ascii=False, indent=2,
            ),
        )
        groups_raw = _safe_llm_call(llm, dep_prompt)
        groups = [g for g in (groups_raw if isinstance(groups_raw, list) else [])
                  if isinstance(g, dict) and g.get('symbols')]
    if not groups and len(symbols) > 1:
        lazyllm.LOG.warning(
            '  [RCov] Dependency grouping LLM call failed or returned invalid data; '
            'falling back to standalone symbol processing. '
            f'This may result in {len(symbols)} separate LLM calls instead of fewer grouped calls.'
        )
    if not groups:
        groups = [{'group_id': i, 'symbols': [s['symbol']], 'rationale': 'standalone symbol'}
                  for i, s in enumerate(symbols)]
    lazyllm.LOG.info(f'  [RCov] {len(groups)} symbol group(s) for coverage evaluation')
    return groups


def _rcov_evaluate_groups(
    llm: Any, groups: List[Dict[str, Any]], symbols: List[Dict[str, Any]],
    test_files: List[str], clone_dir: Optional[str], pr_summary: str, lang_instr: str,
    usage_scenarios_str: str = '',
) -> List[Dict[str, Any]]:
    '''Step 2: grep test files and evaluate coverage per group (parallel).'''
    sym_by_name = {s['symbol']: s for s in symbols}

    # Build usage scenarios block for prompt injection
    if usage_scenarios_str:
        usage_scenarios_block = (
            '## Inferred Usage Scenarios (from RScene)\n'
            'The following typical usage scenarios were inferred from documentation, tests, and '
            'existing callers. When evaluating coverage, ALSO check whether these scenarios and '
            'their edge_cases are covered by existing tests. If a scenario\'s edge_case is not '
            'covered, report it as a PARTIAL or MISSING issue.\n\n'
            f'{usage_scenarios_str[:3000]}\n\n'
        )
    else:
        usage_scenarios_block = ''

    def _evaluate_group(group: Dict[str, Any]) -> List[Dict[str, Any]]:
        group_syms = [sym_by_name[name] for name in group.get('symbols', []) if name in sym_by_name]
        if not group_syms:
            return []
        grep_results = _build_grep_results(group_syms, test_files, clone_dir or '')
        evaluate_prompt = _RCOV_EVALUATE_PROMPT_TMPL.format(
            lang_instruction=lang_instr,
            pr_summary=(pr_summary or '')[:600],
            symbols_json=json.dumps(group_syms, ensure_ascii=False, indent=2),
            rationale=group.get('rationale', ''),
            grep_results=clip_text(grep_results, 12000),
            usage_scenarios_block=usage_scenarios_block,
        )
        items_raw = _safe_llm_call(llm, evaluate_prompt)
        issues: List[Dict[str, Any]] = []
        for item in (items_raw if isinstance(items_raw, list) else []):
            normalized = _normalize_comment_item(
                item, default_path=group_syms[0].get('file', ''),
                default_category='maintainability',
            )
            if normalized is not None:
                normalized['source'] = 'rcov'
                issues.append(normalized)
        return issues

    all_issues: List[Dict[str, Any]] = []
    n_workers = max(1, min(4, len(groups)))
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_evaluate_group, g): g for g in groups}
        try:
            for fut in as_completed(futs, timeout=_RCOV_GROUP_TIMEOUT_SECS * n_workers):
                try:
                    all_issues.extend(fut.result())
                except Exception as e:
                    lazyllm.LOG.error(
                        f'  [RCov] Group evaluation failed unexpectedly: {e}', exc_info=True
                    )
        except FuturesTimeoutError:
            lazyllm.LOG.warning(
                f'  [RCov] Overall coverage evaluation timed out after '
                f'{_RCOV_GROUP_TIMEOUT_SECS * n_workers}s, cancelling remaining groups'
            )
            for fut in futs:
                fut.cancel()
    return all_issues


def _run_coverage_check(
    llm: Any,
    diff_text: str,
    pr_summary: str,
    clone_dir: Optional[str],
    language: str = 'cn',
    usage_scenarios_str: str = '',
) -> List[Dict[str, Any]]:
    '''
    RCov: three-step test coverage check.
    Step 1 — identify testable symbols from diff (LLM), filter internal helpers.
    Step 1.5 — dependency analysis: group related symbols (LLM).
    Step 2 — grep test files + evaluate coverage per group (LLM, parallel).
    Returns a list of issues with source='rcov'.
    '''
    prog = _Progress('RCov: test coverage check')
    lang_instr = _language_instruction(language)

    symbols = _rcov_identify_symbols(llm, diff_text, pr_summary, lang_instr)
    if not symbols:
        prog.done('no testable symbols identified')
        return []

    groups = _rcov_group_symbols(llm, symbols, lang_instr)

    test_files: List[str] = []
    if clone_dir and os.path.isdir(clone_dir):
        test_files = _find_test_files(clone_dir)
        lazyllm.LOG.info(f'  [RCov] Found {len(test_files)} test files')

    all_issues = _rcov_evaluate_groups(
        llm, groups, symbols, test_files, clone_dir, pr_summary, lang_instr,
        usage_scenarios_str=usage_scenarios_str,
    )

    prog.done(f'{len(all_issues)} coverage issues found across {len(groups)} group(s)')
    return all_issues
