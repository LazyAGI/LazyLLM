# Copyright (c) 2026 LazyAGI. All rights reserved.
import os
import subprocess
from typing import Any, Optional, Tuple

import lazyllm

from ...base import LazyLLMGitBase
from ..checkpoint import _load_cache, _save_cache, ReviewStage
from ..utils import _Progress, _safe_llm_call_text
from .agent_instructions import _read_agent_instructions, _AGENT_INSTRUCTIONS_MAX_CHARS
from .arch_pipeline import analyze_repo_architecture
from .conventions import analyze_framework_conventions, _merge_conventions_into_spec
from .git_clone import _resolve_clone_target, _fetch_repo_code
from .prompt import _PRE_ROUND_PROMPT_TMPL
from .review_spec import analyze_historical_reviews, _SPEC_CACHE_VERSION


def _pre_round_pr_summary(
    llm: Any,
    pr_title: str,
    pr_body: str,
    diff_text: str,
    language: str = 'cn',
) -> str:
    from ..utils import _language_instruction
    prog = _Progress('Pre-round: summarizing PR changes')
    prompt = _PRE_ROUND_PROMPT_TMPL.format(
        lang_instruction=_language_instruction(language),
        pr_title=pr_title or '(no title)',
        pr_body=(pr_body or '(no description)')[:800],
        diff_text=diff_text[:5000] if diff_text else '',
    )
    summary = _safe_llm_call_text(llm, prompt) or '(PR summary unavailable)'
    prog.done(f'{len(summary)} chars')
    return summary


def _run_arch_analysis(
    llm: Any, pr: Any, repo: str, arch_cache_path: str, ckpt: Any,
    clone_target_dir: Optional[str] = None, head_sha: Optional[str] = None,
) -> Tuple[str, Optional[str], str]:
    arch_doc = ckpt.get('arch_doc') or ''
    prog = _Progress('Pre-analysis: fetch repo & analyze architecture')
    clone_url, branch = _resolve_clone_target(pr, repo)
    lazyllm.LOG.info(f'Cloning {clone_url} @ {branch}')
    try:
        clone_dir, _ = _fetch_repo_code(
            clone_url, branch, work_dir=clone_target_dir,
            pin_sha=head_sha)
    except Exception as e:
        raise RuntimeError(f'Failed to clone repo {clone_url} @ {branch}: {e}') from e
    ckpt.save('clone_dir', clone_dir)
    ckpt.mark_stage_done(ReviewStage.CLONE)
    prog.update('cloned, analyzing...')
    agent_instructions = _read_agent_instructions(clone_dir)
    if agent_instructions:
        _save_cache(arch_cache_path, 'agent_instructions', agent_instructions)
        lazyllm.LOG.info(f'Found agent instructions ({len(agent_instructions)} chars)')
    try:
        arch_doc = analyze_repo_architecture(
            llm, clone_dir, arch_cache_path, agent_instructions, clone_url, base_repo=repo,
        )
    except Exception:
        import shutil
        shutil.rmtree(clone_dir, ignore_errors=True)
        raise
    ckpt.save('arch_doc', arch_doc)
    ckpt.mark_stage_done(ReviewStage.ARCH)
    if arch_doc and arch_cache_path:
        lazyllm.LOG.success(f'Architecture doc saved to: {arch_cache_path}')
    prog.done('architecture doc ready')
    return arch_doc, clone_dir, agent_instructions


def _run_spec_analysis(
    backend_inst: LazyLLMGitBase, llm: Any,
    review_spec_cache_path: str, max_history_prs: int, ckpt: Any,
) -> str:
    review_spec = ckpt.get('review_spec') or ''
    cached_ver_raw = ckpt.get('spec_cache_version')
    try:
        cached_ver = int(cached_ver_raw) if cached_ver_raw else 0
    except (ValueError, TypeError):
        cached_ver = 0
    need_refresh = (
        not review_spec
        or review_spec.startswith('(')
        or cached_ver < _SPEC_CACHE_VERSION
    )
    if not need_refresh:
        _save_cache(review_spec_cache_path, 'review_spec', review_spec)
        _Progress('Pre-analysis: review spec').done('loaded from checkpoint')
        return review_spec
    try:
        review_spec = analyze_historical_reviews(backend_inst, llm, review_spec_cache_path, max_history_prs)
        ckpt.save('review_spec', review_spec)
        ckpt.save('spec_cache_version', str(_SPEC_CACHE_VERSION))
        ckpt.mark_stage_done(ReviewStage.SPEC)
        if review_spec and review_spec_cache_path:
            if review_spec.startswith('('):
                lazyllm.LOG.warning(f'Review spec not generated: {review_spec}')
            else:
                lazyllm.LOG.success(f'Review spec saved to: {review_spec_cache_path}')
    except Exception as e:
        if 'no review comments' in str(e).lower() or 'not found' in str(e).lower():
            lazyllm.LOG.warning(f'Historical review analysis: {e}')
        else:
            raise
    return review_spec


def _resume_arch_from_checkpoint(
    arch_doc: str, clone_dir: Optional[str], agent_instructions: str,
    arch_cache_path: str, pr: Any, repo: str, clone_target_dir: Optional[str],
    head_sha: Optional[str], ckpt: Any,
) -> Tuple[str, Optional[str], str]:
    _save_cache(arch_cache_path, 'arch_doc', arch_doc)
    _Progress('Pre-analysis: architecture').done('loaded from checkpoint')
    if not clone_dir:
        try:
            clone_url, branch = _resolve_clone_target(pr, repo)
            lazyllm.LOG.info(f'Cloning {clone_url} @ {branch} for agent file access')
            clone_dir, _ = _fetch_repo_code(clone_url, branch, work_dir=clone_target_dir, pin_sha=head_sha)
            ckpt.save('clone_dir', clone_dir)
        except (OSError, subprocess.CalledProcessError, ValueError) as e:
            lazyllm.LOG.error(f'Clone for agent failed: {e}')
            raise RuntimeError(f'Clone for agent failed: {e}') from e
    else:
        lazyllm.LOG.info(f'Reusing cached clone at {clone_dir}')
    if not agent_instructions and clone_dir:
        agent_instructions = _read_agent_instructions(clone_dir)
        if agent_instructions:
            _save_cache(arch_cache_path, 'agent_instructions', agent_instructions)
    return arch_doc, clone_dir, agent_instructions


def _run_local_arch_analysis(
    llm: Any, arch_doc: str, local_repo_path: Optional[str],
    arch_cache_path: str, agent_instructions: str, repo: str, ckpt: Any,
) -> Tuple[str, str]:
    if not arch_doc and local_repo_path and os.path.isdir(local_repo_path):
        agent_instructions = _read_agent_instructions(local_repo_path)
        if agent_instructions:
            _save_cache(arch_cache_path, 'agent_instructions', agent_instructions)
        prog = _Progress('Pre-analysis: architecture')
        try:
            arch_doc = analyze_repo_architecture(
                llm, local_repo_path, arch_cache_path, agent_instructions, base_repo=repo,
            )
            ckpt.save('arch_doc', arch_doc)
            prog.done('architecture doc ready')
        except Exception as e:
            lazyllm.LOG.error(
                f'Local arch analysis failed, downstream LLM calls will use empty arch_doc: {e}',
                exc_info=True,
            )
            prog.done('skipped (local arch analysis failed, proceeding without arch context)')
    else:
        _save_cache(arch_cache_path, 'arch_doc', arch_doc)
        _Progress('Pre-analysis: architecture').done('loaded from checkpoint')
    return arch_doc, agent_instructions


def _run_pre_analysis(
    llm: Any, backend_inst: LazyLLMGitBase, repo: str, pr: Any,
    fetch_repo_code: bool, arch_cache_path: Optional[str], review_spec_cache_path: Optional[str],
    max_history_prs: int, ckpt: Any, pr_dir: Optional[str] = None,
    head_sha: Optional[str] = None, local_repo_path: Optional[str] = None,
) -> Tuple[str, str, Optional[str], str]:
    if fetch_repo_code and local_repo_path:
        raise ValueError(
            '`local_repo_path` must not be set when `fetch_repo_code=True`; '
            'use `head_sha` to pin the remote clone instead.'
        )
    from ..checkpoint import _ReviewCheckpoint
    repo_cache_dir = _ReviewCheckpoint.repo_cache_dir(repo)
    arch_cache_path = arch_cache_path or os.path.join(repo_cache_dir, 'arch.json')
    review_spec_cache_path = review_spec_cache_path or os.path.join(repo_cache_dir, 'spec.json')

    arch_doc = ckpt.get('arch_doc') or ''
    clone_dir: Optional[str] = ckpt.get('clone_dir') or None
    if clone_dir and not os.path.isdir(clone_dir):
        clone_dir = None

    clone_target_dir = os.path.join(pr_dir, 'clone') if pr_dir else None
    agent_instructions = _load_cache(arch_cache_path, 'agent_instructions') or ''

    if fetch_repo_code:
        if not arch_doc:
            arch_doc, clone_dir, agent_instructions = _run_arch_analysis(
                llm, pr, repo, arch_cache_path, ckpt, clone_target_dir,
                head_sha=head_sha,
            )
        else:
            arch_doc, clone_dir, agent_instructions = _resume_arch_from_checkpoint(
                arch_doc, clone_dir, agent_instructions, arch_cache_path,
                pr, repo, clone_target_dir, head_sha, ckpt,
            )
    else:
        arch_doc, agent_instructions = _run_local_arch_analysis(
            llm, arch_doc, local_repo_path, arch_cache_path, agent_instructions, repo, ckpt,
        )

    review_spec = _run_spec_analysis(backend_inst, llm, review_spec_cache_path, max_history_prs, ckpt)

    try:
        conventions = analyze_framework_conventions(backend_inst, llm, review_spec_cache_path, max_prs=max_history_prs)
        if conventions:
            agent_instructions = (agent_instructions + '\n\n## Conventions (from historical reviews)\n'
                                  + conventions)[:_AGENT_INSTRUCTIONS_MAX_CHARS]
            _save_cache(arch_cache_path, 'agent_instructions', agent_instructions)
            review_spec = _merge_conventions_into_spec(review_spec, conventions)
            ckpt.save('review_spec', review_spec)
    except Exception as e:
        lazyllm.LOG.warning(f'Framework convention extraction failed (non-fatal): {e}')

    return arch_doc, review_spec, clone_dir, agent_instructions
