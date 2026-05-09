# Copyright (c) 2026 LazyAGI. All rights reserved.
import collections
import json
from typing import Any, Dict, List, Optional, Tuple

import lazyllm

from ..base import LazyLLMGitBase
from ..checkpoint import _load_cache, _save_cache_multi
from ..utils import _Progress, _safe_llm_call
from .prompt import _VALIDATE_CONVENTION_PROMPT
from .review_spec import (
    _is_merged_pr, _fetch_all_pr_comments,
    _preload_bot_templates_from_cache, _accumulate_and_detect_templates,
    _strip_bot_templates, _BOT_USER_PATTERNS, _BOT_BODY_INDICATORS,
    _MAINTAINER_ASSOCIATIONS, _SPEC_CACHE_VERSION,
)

_ACCEPTED_VERDICTS = {'framework_convention', 'ai_false_positive'}


def _resolve_pr_author(pr: Any) -> str:
    pr_user = (pr.get('user') if isinstance(pr, dict) else getattr(pr, 'user', None))
    if isinstance(pr_user, dict):
        return pr_user.get('login', '')
    return pr_user if isinstance(pr_user, str) else ''


def _validate_conventions_batch(llm: Any, high_conf: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    conventions = []
    for pair in high_conf[:10]:
        ack_section = ''
        if pair.get('bot_acknowledgment'):
            ack_section = f'### Bot Acknowledgment\n{pair["bot_acknowledgment"]}'
        try:
            prompt = _VALIDATE_CONVENTION_PROMPT.format(
                pattern=pair.get('pattern', ''),
                bot_comment=pair.get('bot_comment', ''),
                reply_text=pair.get('reply_text', ''),
                ack_section=ack_section,
            )
        except KeyError as e:
            lazyllm.LOG.warning(f'Convention validation skipped — missing key: {e}')
            continue
        result = _safe_llm_call(llm, prompt)
        if isinstance(result, list):
            result = result[0] if result else {}
        if isinstance(result, dict) and result.get('verdict') in _ACCEPTED_VERDICTS:
            conventions.append(result)
    return conventions


def _build_reply_chains(comments: list) -> List[List[dict]]:
    by_id = {}
    children: Dict[Any, list] = {}
    roots = []
    for c in comments:
        cid = c.get('id') if isinstance(c, dict) else getattr(c, 'id', None)
        if cid is not None:
            by_id[cid] = c
    for c in comments:
        raw = (c.get('raw') if isinstance(c, dict) else getattr(c, 'raw', {})) or {}
        parent_id = raw.get('in_reply_to_id')
        cid = c.get('id') if isinstance(c, dict) else getattr(c, 'id', None)
        if parent_id and parent_id in by_id:
            children.setdefault(parent_id, []).append(c)
        else:
            roots.append(c)
    chains = []
    for root in roots:
        rid = root.get('id') if isinstance(root, dict) else getattr(root, 'id', None)
        chain = [root]
        queue = collections.deque(children.get(rid, []))
        while queue:
            node = queue.popleft()
            chain.append(node)
            nid = node.get('id') if isinstance(node, dict) else getattr(node, 'id', None)
            queue.extend(children.get(nid, []))
        if len(chain) >= 2:
            chains.append(chain)
    return chains


def _get_comment_field(c: Any, field: str) -> str:
    return ((c.get(field) if isinstance(c, dict) else getattr(c, field, '')) or '').strip()


def _is_bot_comment(user: str, body: str) -> bool:
    if _BOT_USER_PATTERNS.search(user):
        return True
    return bool(_BOT_BODY_INDICATORS.search(body))


def _filter_high_confidence_chains(  # noqa: C901
    chains: List[List[Any]], pr_author: str = '',
    bot_templates: Optional[Dict[str, List[Tuple[str, str]]]] = None,
) -> List[Dict[str, str]]:
    results = []
    for chain in chains:
        bot_indices = []
        for i, c in enumerate(chain):
            user = _get_comment_field(c, 'user')
            body = _get_comment_field(c, 'body')
            if body and _is_bot_comment(user, body):
                bot_indices.append(i)
        if not bot_indices:
            continue
        for bi in bot_indices:
            bot_c = chain[bi]
            bot_user = _get_comment_field(bot_c, 'user')
            bot_body = _get_comment_field(bot_c, 'body')
            if bot_templates:
                tlist = bot_templates.get(bot_user)
                if tlist:
                    for pfx, sfx in tlist:
                        if (pfx and bot_body.startswith(pfx)) or (sfx and bot_body.endswith(sfx)):
                            if pfx and bot_body.startswith(pfx):
                                bot_body = bot_body[len(pfx):]
                            if sfx and bot_body.endswith(sfx):
                                bot_body = bot_body[:-len(sfx)]
                            bot_body = bot_body.strip()
                            break
            if not bot_body:
                continue
            after = chain[bi + 1:]
            for idx_r, reply in enumerate(after):
                ruser = _get_comment_field(reply, 'user')
                rbody = _get_comment_field(reply, 'body')
                if not rbody:
                    continue
                if ruser != bot_user and _BOT_USER_PATTERNS.search(ruser):
                    continue
                if ruser == bot_user:
                    humans = [
                        c for c in after[:idx_r]
                        if _get_comment_field(c, 'user') != bot_user
                        and not _BOT_USER_PATTERNS.search(_get_comment_field(c, 'user'))
                        and _get_comment_field(c, 'body')
                    ]
                    if humans:
                        results.append({
                            'bot_comment': bot_body[:600],
                            'reply_text': _get_comment_field(humans[0], 'body')[:600],
                            'bot_acknowledgment': rbody[:400],
                            'pattern': 'three_way',
                        })
                    break
                raw = (reply.get('raw') if isinstance(reply, dict) else getattr(reply, 'raw', {})) or {}
                assoc = (raw.get('author_association') or '').upper()
                is_maint = assoc in _MAINTAINER_ASSOCIATIONS and ruser != pr_author
                if is_maint and not _is_bot_comment(ruser, rbody):
                    results.append({
                        'bot_comment': bot_body[:600],
                        'reply_text': rbody[:600],
                        'bot_acknowledgment': '',
                        'pattern': 'maintainer_direct',
                    })
                    break
                if ruser == pr_author:
                    confirm = next(
                        (_get_comment_field(c, 'body') for c in after[idx_r + 1:]
                         if _get_comment_field(c, 'user') == bot_user and _get_comment_field(c, 'body')),
                        None,
                    )
                    if confirm:
                        results.append({
                            'bot_comment': bot_body[:600],
                            'reply_text': rbody[:600],
                            'bot_acknowledgment': confirm[:400],
                            'pattern': 'three_way',
                        })
                    break
            if len(results) >= 40:
                break
        if len(results) >= 40:
            break
    return results


def _collect_framework_conventions_for_pr(  # noqa: C901
    backend: LazyLLMGitBase, llm: Any, pr: Any,
    idx: int, total: int, cache_path: Optional[str],
    prog: Any, all_conventions: List[Dict[str, Any]],
    bot_templates: Optional[Dict[str, List[Tuple[str, str]]]] = None,
    prefetched_comments: Optional[List[Dict[str, Any]]] = None,
) -> None:
    pr_num = getattr(pr, 'number', None) or (pr.get('number') if isinstance(pr, dict) else None)
    if pr_num is None:
        return
    conv_cache_key = f'conv_pr_{pr_num}_conventions'
    conv_ver_key = f'conv_pr_{pr_num}_ver'
    cached_ver_str = _load_cache(cache_path, conv_ver_key)
    cached_ver = int(cached_ver_str) if cached_ver_str and cached_ver_str.isdigit() else 0
    if cached_ver >= _SPEC_CACHE_VERSION:
        cached_str = _load_cache(cache_path, conv_cache_key)
        if cached_str:
            try:
                convs = json.loads(cached_str)
                all_conventions.extend(convs)
                return
            except (json.JSONDecodeError, TypeError):
                pass
    raw_comments = prefetched_comments if prefetched_comments is not None else _fetch_all_pr_comments(backend, pr_num)
    if not raw_comments:
        return
    chains = _build_reply_chains(raw_comments)
    pr_author = _resolve_pr_author(pr)
    high_conf = _filter_high_confidence_chains(chains, pr_author, bot_templates=bot_templates)
    if not high_conf:
        _save_cache_multi(cache_path, {conv_cache_key: '[]', conv_ver_key: str(_SPEC_CACHE_VERSION)})
        return
    conventions = _validate_conventions_batch(llm, high_conf)
    _save_cache_multi(cache_path, {
        conv_cache_key: json.dumps(conventions, ensure_ascii=False),
        conv_ver_key: str(_SPEC_CACHE_VERSION),
    })
    all_conventions.extend(conventions)
    if conventions:
        prog.update(f'[{idx}/{total}] PR #{pr_num} — {len(high_conf)} chains → {len(conventions)} conventions')


def _format_conventions_result(all_conventions: List[Dict[str, Any]]) -> str:
    conv_lines = []
    fp_lines = []
    for c in all_conventions:
        trigger = c.get('trigger_pattern', '')
        guideline = c.get('do_not_flag', '')
        if not guideline:
            continue
        if c.get('verdict') == 'ai_false_positive':
            why = c.get('why_correct', '')
            cat = c.get('category', 'other')
            fp_lines.append(f'- [{cat}] Pattern: {trigger} → {why}. Rule: {guideline}')
        else:
            behavior = c.get('actual_behavior', '')
            conv_lines.append(f'- Pattern: {trigger} → {behavior}. Rule: {guideline}')
    parts = []
    if conv_lines:
        parts.append('### Framework Conventions\n' + '\n'.join(conv_lines))
    if fp_lines:
        parts.append('### AI Common False Positives in This Repo\n' + '\n'.join(fp_lines))
    return '\n\n'.join(parts) if parts else ''


def analyze_framework_conventions(
    backend: LazyLLMGitBase, llm: Any, cache_path: Optional[str] = None, max_prs: int = 50,
) -> str:
    cached_ver_str = _load_cache(cache_path, 'framework_conventions_ver')
    cached_ver = int(cached_ver_str) if cached_ver_str and cached_ver_str.isdigit() else 0
    if cached_ver >= _SPEC_CACHE_VERSION:
        cached = _load_cache(cache_path, 'framework_conventions')
        if cached:
            return cached
    merged: List[Any] = []
    fetch_size = max_prs
    while len(merged) < max_prs:
        pr_list_res = backend.list_pull_requests(state='closed', max_results=fetch_size)
        if not pr_list_res.get('success'):
            return ''
        prs = pr_list_res.get('list') or []
        if not prs:
            break
        merged = [p for p in prs if _is_merged_pr(p)]
        if len(merged) >= max_prs or len(prs) < fetch_size:
            break
        fetch_size = min(fetch_size * 2, 1000)
    target = merged[:max_prs]
    if not target:
        return ''
    prog = _Progress('Conventions: extracting from bot-reply chains', len(target))
    all_conventions: List[Dict[str, Any]] = []
    bot_templates: Dict[str, List[Tuple[str, str]]] = {}
    _preload_bot_templates_from_cache(cache_path, bot_templates)
    for idx, pr in enumerate(target, 1):
        pr_num = getattr(pr, 'number', None) or (pr.get('number') if isinstance(pr, dict) else None)
        if pr_num is None:
            continue
        comments = _fetch_all_pr_comments(backend, pr_num)
        if comments:
            _accumulate_and_detect_templates(comments, bot_templates, cache_path)
        _collect_framework_conventions_for_pr(backend, llm, pr, idx, len(target), cache_path, prog,
                                              all_conventions, bot_templates=bot_templates,
                                              prefetched_comments=comments)
    prog.done(f'{len(all_conventions)} conventions from {len(target)} PRs')
    if not all_conventions:
        _save_cache_multi(cache_path, {
            'framework_conventions': '', 'framework_conventions_ver': str(_SPEC_CACHE_VERSION),
        })
        return ''
    result = _format_conventions_result(all_conventions)
    _save_cache_multi(cache_path, {
        'framework_conventions': result, 'framework_conventions_ver': str(_SPEC_CACHE_VERSION),
    })
    return result


def _merge_conventions_into_spec(review_spec: str, conventions: str) -> str:
    if not conventions:
        return review_spec
    try:
        spec_obj = json.loads(review_spec) if review_spec and not review_spec.startswith('(') else {}
    except (json.JSONDecodeError, ValueError):
        spec_obj = {}
    spec_obj['conventions'] = conventions
    return json.dumps(spec_obj, ensure_ascii=False)
