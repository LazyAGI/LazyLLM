# Copyright (c) 2026 LazyAGI. All rights reserved.
# ReviewReport: collect per-stage issue statistics and generate a local Markdown report.
# The report is written to the PR checkpoint directory and never posted to GitHub.

import dataclasses
import datetime
import os
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class StageStats:
    '''Statistics for a single review stage.'''
    name: str           # human-readable stage name
    produced: int       # issues produced by this stage
    discarded: int      # issues discarded (0 if stage is a source, not a filter)
    kept: int           # issues kept after this stage


@dataclasses.dataclass
class DiscardedIssue:
    '''A single issue that was discarded, with its reason.'''
    stage: str          # which stage discarded it
    path: str
    line: Optional[int]
    severity: str
    category: str
    source: str         # original source tag (r1/r2/rchain/…)
    problem: str        # truncated problem description
    reason: str         # discard reason (e.g. "LLM 去重", "R3 验证未通过")


@dataclasses.dataclass
class ReviewReport:
    '''Full report data collected across all review stages.'''
    # meta
    pr_identifier: str          # PR number or local branch name
    repo: str
    pr_title: str
    source_branch: str
    target_branch: str
    review_time: str            # ISO-8601 timestamp
    model_name: str
    report_path: str            # where this report will be written

    # per-stage stats (ordered)
    stages: List[StageStats]

    # discarded issues per stage
    discarded: List[DiscardedIssue]

    # final issues
    final_issues: List[Dict[str, Any]]
    posted_inline: int
    posted_general: int
    post_to_github: bool

    # special conditions
    diff_truncated: bool
    rcov_timed_out: bool
    rchain_scenarios_ok: int
    rchain_scenarios_timeout: int
    rchain_scenarios_error: int
    ragent_verify_files_skipped: List[str]


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def _issue_key(issue: Dict[str, Any]) -> Tuple[str, int, str]:
    return (
        issue.get('path', ''),
        int(issue.get('line') or 0),
        issue.get('bug_category', ''),
    )


def _short_problem(issue: Dict[str, Any], max_len: int = 120) -> str:
    p = issue.get('problem', '')
    return p[:max_len] + '…' if len(p) > max_len else p


def collect_ragent_verify_discarded(
    rhunk_scan_issues: List[Dict[str, Any]],
    rarch_review_issues: List[Dict[str, Any]],
    discarded_keys: set,
) -> List[DiscardedIssue]:
    '''Compute which RHunkScan/RArchReview issues were discarded by RAgentVerify via set-diff on discarded_keys.'''
    result: List[DiscardedIssue] = []
    for issue in list(rhunk_scan_issues) + list(rarch_review_issues):
        path = issue.get('path', '')
        line = issue.get('line')
        cat = issue.get('bug_category', '')
        key_pl = f'{path}:{line}'
        key_plc = f'{path}:{line}:{cat}'
        if key_pl in discarded_keys or key_plc in discarded_keys:
            result.append(DiscardedIssue(
                stage='RAgentVerify (Agent 验证)',
                path=path,
                line=line,
                severity=issue.get('severity', 'normal'),
                category=cat,
                source=issue.get('source', 'rhunk_scan/rarch_review'),
                problem=_short_problem(issue),
                reason='RAgentVerify Agent 验证未通过（误报或逻辑正确）',
            ))
    return result


def collect_rdedup_merge_discarded(
    rdedup_merge_input: List[Dict[str, Any]],
    rdedup_merge_output: List[Dict[str, Any]],
) -> List[DiscardedIssue]:
    '''Compute RDedupMerge discards by set-diff on (path, line, category) keys.'''
    output_keys = {_issue_key(i) for i in rdedup_merge_output}
    result: List[DiscardedIssue] = []
    for issue in rdedup_merge_input:
        if _issue_key(issue) not in output_keys:
            result.append(DiscardedIssue(
                stage='RDedupMerge (合并去重)',
                path=issue.get('path', ''),
                line=issue.get('line'),
                severity=issue.get('severity', 'normal'),
                category=issue.get('bug_category', ''),
                source=issue.get('source', ''),
                problem=_short_problem(issue),
                reason='RDedupMerge LLM 去重（重复或低质量）',
            ))
    return result


def collect_postmerge_discarded(
    merge_input: List[Dict[str, Any]],
    merge_output: List[Dict[str, Any]],
) -> List[DiscardedIssue]:
    '''Compute post-merge dedup discards by set-diff.'''
    output_keys = {_issue_key(i) for i in merge_output}
    result: List[DiscardedIssue] = []
    for issue in merge_input:
        if _issue_key(issue) not in output_keys:
            result.append(DiscardedIssue(
                stage='Post-merge Dedup (跨源去重)',
                path=issue.get('path', ''),
                line=issue.get('line'),
                severity=issue.get('severity', 'normal'),
                category=issue.get('bug_category', ''),
                source=issue.get('source', ''),
                problem=_short_problem(issue),
                reason='跨源 LLM 去重（与其他来源重复）',
            ))
    return result


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

_SEV_ORDER = {'critical': 0, 'medium': 1, 'normal': 2}


def _sev_emoji(sev: str) -> str:
    return {'critical': '🔴', 'medium': '🟡', 'normal': '🔵'}.get(sev, '⚪')


def _render_issue_table(issues: List[Dict[str, Any]], max_rows: int = 200) -> str:
    if not issues:
        return '_（无）_\n'
    rows = ['| # | 文件 | 行号 | 严重度 | 类别 | 来源 | 问题描述 |',
            '|---|------|------|--------|------|------|----------|']
    for i, iss in enumerate(issues[:max_rows], 1):
        sev = iss.get('severity', 'normal')
        rows.append(
            f'| {i} | `{iss.get("path", "")}` | {iss.get("line", "—")} '
            f'| {_sev_emoji(sev)} {sev} | {iss.get("bug_category", "")} '
            f'| {iss.get("source", "")} | {_short_problem(iss)} |'
        )
    if len(issues) > max_rows:
        rows.append(f'| … | _（共 {len(issues)} 条，仅展示前 {max_rows} 条）_ | | | | | |')
    return '\n'.join(rows) + '\n'


def _render_discard_table(items: List[DiscardedIssue], max_rows: int = 200) -> str:
    if not items:
        return '_（无）_\n'
    rows = ['| # | 文件 | 行号 | 严重度 | 类别 | 来源 | 丢弃原因 | 问题描述 |',
            '|---|------|------|--------|------|------|----------|----------|']
    for i, d in enumerate(items[:max_rows], 1):
        sev = d.severity
        rows.append(
            f'| {i} | `{d.path}` | {d.line if d.line is not None else "—"} '
            f'| {_sev_emoji(sev)} {sev} | {d.category} | {d.source} '
            f'| {d.reason} | {d.problem} |'
        )
    if len(items) > max_rows:
        rows.append(f'| … | _（共 {len(items)} 条，仅展示前 {max_rows} 条）_ | | | | | | |')
    return '\n'.join(rows) + '\n'


def render_markdown(report: ReviewReport) -> str:
    lines: List[str] = []

    # ── Header ──
    lines += [
        '# Code Review 报告',
        '',
        f'**PR / 分支**: {report.pr_identifier}',
        f'**仓库**: {report.repo}',
        f'**标题**: {report.pr_title}',
        f'**分支**: `{report.source_branch}` → `{report.target_branch}`',
        f'**Review 时间**: {report.review_time}',
        f'**模型**: {report.model_name}',
        f'**报告路径**: `{report.report_path}`',
        '',
        '---',
        '',
    ]

    # ── Overview table ──
    lines += [
        '## 概览',
        '',
        '| 阶段 | 产出 issue 数 | 丢弃数 | 保留数 |',
        '|------|-------------|--------|--------|',
    ]
    for s in report.stages:
        lines.append(f'| {s.name} | {s.produced} | {s.discarded} | {s.kept} |')

    total_discarded = sum(s.discarded for s in report.stages)
    total_produced = sum(s.produced for s in report.stages if s.discarded == 0)
    lines += [
        '',
        f'> **最终提交**: {report.posted_inline} 条 inline comment'
        + (f' + {report.posted_general} 条 general comment' if report.posted_general else '')
        + (' （本地模式，未提交 GitHub）' if not report.post_to_github else ''),
        '',
        '---',
        '',
    ]

    # ── Discard details ──
    lines += ['## 各阶段丢弃明细', '']

    by_stage: Dict[str, List[DiscardedIssue]] = {}
    for d in report.discarded:
        by_stage.setdefault(d.stage, []).append(d)

    if not report.discarded:
        lines += ['_本次 review 无 issue 被丢弃。_', '']
    else:
        for stage_name, items in by_stage.items():
            lines += [f'### {stage_name}（丢弃 {len(items)} 条）', '']
            lines.append(_render_discard_table(items))

    lines += ['---', '']

    # ── Final issues ──
    lines += ['## 最终 Issue 列表', '']
    sorted_final = sorted(report.final_issues, key=lambda x: _SEV_ORDER.get(x.get('severity', 'normal'), 2))

    for sev_label, sev_key in [('Critical', 'critical'), ('Medium', 'medium'), ('Normal', 'normal')]:
        group = [i for i in sorted_final if i.get('severity') == sev_key]
        if group:
            lines += [f'### {_sev_emoji(sev_key)} {sev_label}（{len(group)} 条）', '']
            lines.append(_render_issue_table(group))

    lines += ['---', '']

    # ── Special conditions ──
    lines += ['## 特殊情况', '']
    lines.append(f'- **Diff 截断**: {"是（部分文件被跳过）" if report.diff_truncated else "否"}')
    lines.append(f'- **RCov 超时**: {"是" if report.rcov_timed_out else "否"}')
    lines.append(
        f'- **RChain 场景**: ok={report.rchain_scenarios_ok}, '
        f'timeout={report.rchain_scenarios_timeout}, '
        f'error={report.rchain_scenarios_error}'
    )
    if report.ragent_verify_files_skipped:
        lines.append(f'- **RAgentVerify 跳过文件**: {", ".join(f"`{f}`" for f in report.ragent_verify_files_skipped)}')
    else:
        lines.append('- **RAgentVerify 跳过文件**: 无')
    lines += ['', '---', '']

    # ── Summary ──
    lines += [
        '## 统计摘要',
        '',
        '```',
        f'各来源 issue 总数（去重前）: {total_produced}',
        f'总丢弃数:                    {total_discarded}',
    ]
    for stage_name, items in by_stage.items():
        lines.append(f'  - {stage_name}: {len(items)}')
    lines += [
        f'最终 issue 数:               {len(report.final_issues)}',
        f'  - inline comment:          {report.posted_inline}',
        f'  - general comment:         {report.posted_general}',
        '```',
        '',
    ]

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Write helper
# ---------------------------------------------------------------------------

def write_report(report: ReviewReport, path: Optional[str] = None) -> str:
    '''Render and write the Markdown report. Returns the path written.'''
    out_path = path or report.report_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    content = render_markdown(report)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return out_path


# ---------------------------------------------------------------------------
# Factory: build a ReviewReport from runner.py result data
# ---------------------------------------------------------------------------

def build_report(
    *,
    pr_identifier: str,
    repo: str,
    pr_title: str,
    source_branch: str,
    target_branch: str,
    model_name: str,
    report_path: str,
    post_to_github: bool,
    # per-stage raw data
    rhunk_scan_issues: List[Dict[str, Any]],
    rarch_review_issues: List[Dict[str, Any]],
    rmod_issues: List[Dict[str, Any]],
    lint_issues: List[Dict[str, Any]],
    dep_issues: List[Dict[str, Any]],
    ragent_verify_output: List[Dict[str, Any]],
    ragent_verify_discarded_keys: set,
    ragent_verify_files_skipped: List[str],
    rdedup_merge_input: List[Dict[str, Any]],
    rdedup_merge_output: List[Dict[str, Any]],
    rchain_issues: List[Dict[str, Any]],
    rcov_issues: List[Dict[str, Any]],
    postmerge_input: List[Dict[str, Any]],
    postmerge_output: List[Dict[str, Any]],
    final_issues: List[Dict[str, Any]],
    posted_inline: int,
    posted_general: int,
    diff_truncated: bool,
    rcov_timed_out: bool,
    rchain_metrics: Dict[str, int],
) -> ReviewReport:
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # ── Collect discarded issues ──
    discarded: List[DiscardedIssue] = []
    discarded += collect_ragent_verify_discarded(rhunk_scan_issues, rarch_review_issues, ragent_verify_discarded_keys)
    discarded += collect_rdedup_merge_discarded(rdedup_merge_input, rdedup_merge_output)
    discarded += collect_postmerge_discarded(postmerge_input, postmerge_output)

    # ── Build stage stats ──
    ragent_verify_input_count = len(rhunk_scan_issues) + len(rarch_review_issues)
    rdedup_merge_input_count = len(rdedup_merge_input)
    rdedup_merge_discarded_count = len([d for d in discarded if d.stage.startswith('RDedupMerge')])
    pm_input_count = len(postmerge_input)
    pm_discarded_count = len([d for d in discarded if d.stage.startswith('Post')])

    # Compute per-source RAgentVerify discard counts
    _r3_discarded = [d for d in discarded if d.stage.startswith('RAgentVerify')]
    _r3_disc_by_src: Dict[str, int] = {}
    for d in _r3_discarded:
        _r3_disc_by_src[d.source] = _r3_disc_by_src.get(d.source, 0) + 1

    def _r3_disc(src: str) -> int:
        return _r3_disc_by_src.get(src, 0)

    stages: List[StageStats] = [
        StageStats('RHunkScan (Hunk 分析)',
                   len(rhunk_scan_issues),
                   _r3_disc('rhunk_scan') + _r3_disc('r1'),
                   len(rhunk_scan_issues) - _r3_disc('rhunk_scan') - _r3_disc('r1')),
        StageStats('RArchReview (架构审查)',
                   len(rarch_review_issues),
                   _r3_disc('rarch_review') + _r3_disc('r2'),
                   len(rarch_review_issues) - _r3_disc('rarch_review') - _r3_disc('r2')),
        StageStats('RMod (改动必要性)', len(rmod_issues), 0, len(rmod_issues)),
        StageStats('Lint 静态分析', len(lint_issues), 0, len(lint_issues)),
        StageStats('Dep 依赖检查', len(dep_issues), 0, len(dep_issues)),
        StageStats(
            f'RAgentVerify (Agent 验证)  [{ragent_verify_input_count} 输入]',
            len(ragent_verify_output), 0, len(ragent_verify_output),
        ),
        StageStats(
            f'RDedupMerge (合并去重)  [{rdedup_merge_input_count} 输入]',
            rdedup_merge_input_count, rdedup_merge_discarded_count, len(rdedup_merge_output),
        ),
        StageStats('RChain (调用链)', len(rchain_issues), 0, len(rchain_issues)),
        StageStats('RCov (覆盖率)', len(rcov_issues), 0, len(rcov_issues)),
        StageStats(
            f'Post-merge Dedup  [{pm_input_count} 输入]',
            pm_input_count, pm_discarded_count, len(postmerge_output),
        ),
    ]

    return ReviewReport(
        pr_identifier=str(pr_identifier),
        repo=repo,
        pr_title=pr_title,
        source_branch=source_branch,
        target_branch=target_branch,
        review_time=now,
        model_name=model_name,
        report_path=report_path,
        stages=stages,
        discarded=discarded,
        final_issues=final_issues,
        posted_inline=posted_inline,
        posted_general=posted_general,
        post_to_github=post_to_github,
        diff_truncated=diff_truncated,
        rcov_timed_out=rcov_timed_out,
        rchain_scenarios_ok=rchain_metrics.get('ok', 0),
        rchain_scenarios_timeout=rchain_metrics.get('timeout', 0),
        rchain_scenarios_error=rchain_metrics.get('error', 0),
        ragent_verify_files_skipped=ragent_verify_files_skipped,
    )
