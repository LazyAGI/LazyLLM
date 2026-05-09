# Copyright (c) 2026 LazyAGI. All rights reserved.
import json
import os
import re
from typing import Dict, Any, List, Optional, Tuple

import lazyllm

_DEEPWIKI_MCP_URL = 'https://mcp.deepwiki.com/mcp'
_DEEPWIKI_STALE_DAYS = 90
_DEEPWIKI_SUMMARY_BUDGET = 20000
_DEEPWIKI_QA_TTL_DAYS = 30


def _parse_owner_repo(clone_url: str) -> Optional[str]:
    m = re.search(r'(?:github\.com|gitlab\.com|gitee\.com)[:/]([^/]+/[^/]+?)(?:\.git)?$', clone_url)
    return m.group(1) if m else None


def _compress_markdown_by_sections(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    heading_re = re.compile(r'^(#{1,4})\s+(.+)', re.MULTILINE)
    sections: List[Tuple[str, int, int]] = []
    for m in heading_re.finditer(text):
        sections.append((m.group(0), m.start(), len(m.group(1))))
    if not sections:
        return text[:budget]
    parts: List[Tuple[str, str, int]] = []
    for i, (heading, start, level) in enumerate(sections):
        body_start = start + len(heading)
        body_end = sections[i + 1][1] if i + 1 < len(sections) else len(text)
        parts.append((heading, text[body_start:body_end].strip(), level))
    result_parts = []
    total = 0
    for heading, body, _level in parts:
        first_para = body.split('\n\n')[0] if body else ''
        chunk = f'{heading}\n{first_para}\n' if first_para else f'{heading}\n'
        result_parts.append((chunk, body, first_para))
        total += len(chunk)
    if total <= budget:
        for i, (chunk, body, first_para) in enumerate(result_parts):
            remaining = body[len(first_para):].strip()
            if remaining and total + len(remaining) + 2 <= budget:
                result_parts[i] = (f'{chunk}{remaining}\n', body, first_para)
                total += len(remaining) + 2
            elif remaining:
                space = budget - total
                if space > 100:
                    result_parts[i] = (f'{chunk}{remaining[:space]}\n', body, first_para)
                    total += space + 1
                break
    return ''.join(chunk for chunk, _, _ in result_parts)[:budget]


def _fetch_deepwiki_summary(owner_repo: str) -> str:
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError:
        lazyllm.LOG.info('mcp package not installed, skipping DeepWiki integration')
        return ''

    import asyncio

    async def _query() -> str:
        try:
            async with streamablehttp_client(_DEEPWIKI_MCP_URL) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        'read_wiki_contents',
                        {'repoName': owner_repo},
                    )
                    if not result or not result.content:
                        return ''
                    text = '\n'.join(
                        c.text for c in result.content if hasattr(c, 'text') and c.text
                    )
                    return _compress_markdown_by_sections(text, _DEEPWIKI_SUMMARY_BUDGET)
        except Exception as e:
            lazyllm.LOG.info(f'DeepWiki query failed for {owner_repo}: {e}')
            return ''

    try:
        return asyncio.run(_query())
    except Exception as e:
        lazyllm.LOG.info(f'DeepWiki asyncio run failed: {e}')
        return ''


def _deepwiki_qa_cache_path(owner_repo: str) -> str:
    owner, repo = (owner_repo.split('/', 1) + [''])[:2]
    return os.path.join(lazyllm.config['home'], 'review', 'cache', owner, repo, 'deepwiki_qa.json')


def _deepwiki_qa_load(qa_cache_file: str, cache_key: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(qa_cache_file):
        return None
    try:
        with open(qa_cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get(cache_key) if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _deepwiki_qa_find_similar(
    qa_cache_file: str, question: str, ttl_secs: float, now: float,
) -> Optional[str]:
    if not os.path.isfile(qa_cache_file):
        return None
    try:
        with open(qa_cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    q_words = set(re.findall(r'[a-zA-Z_]\w*', question.lower()))
    best_score, best_answer = 0.0, None
    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        if now - entry.get('fetched_at', 0) >= ttl_secs:
            continue
        cached_q = entry.get('question', '')
        c_words = set(re.findall(r'[a-zA-Z_]\w*', cached_q.lower()))
        if not c_words:
            continue
        overlap = len(q_words & c_words) / max(len(q_words | c_words), 1)
        if overlap > best_score:
            best_score, best_answer = overlap, entry.get('answer', '')
    return best_answer if best_score >= 0.6 else None


def _deepwiki_qa_save(qa_cache_file: str, cache_key: str, entry: Dict[str, Any]) -> None:
    try:
        data: Dict[str, Any] = {}
        if os.path.isfile(qa_cache_file):
            try:
                with open(qa_cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}
        data[cache_key] = entry
        os.makedirs(os.path.dirname(os.path.abspath(qa_cache_file)), exist_ok=True)
        with open(qa_cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError as e:
        lazyllm.LOG.info(f'DeepWiki: failed to write QA cache: {e}')


def _deepwiki_ask_cached(owner_repo: str, question: str, max_chars: int = 2000) -> str:
    import hashlib
    import time

    qa_cache_file = _deepwiki_qa_cache_path(owner_repo)
    cache_key = hashlib.md5(question.encode()).hexdigest()[:16]
    now = time.time()
    ttl_secs = _DEEPWIKI_QA_TTL_DAYS * 86400

    entry = _deepwiki_qa_load(qa_cache_file, cache_key)
    if entry and isinstance(entry, dict) and (now - entry.get('fetched_at', 0)) < ttl_secs:
        return entry.get('answer', '')

    similar = _deepwiki_qa_find_similar(qa_cache_file, question, ttl_secs, now)
    if similar:
        return similar

    answer = _deepwiki_fetch_answer(owner_repo, question, max_chars)

    if answer:
        _deepwiki_qa_save(qa_cache_file, cache_key, {'answer': answer, 'fetched_at': now, 'question': question})
    elif entry:
        lazyllm.LOG.info(f'DeepWiki: extending stale cache for {owner_repo!r} (network failure)')
        _deepwiki_qa_save(qa_cache_file, cache_key, dict(entry, fetched_at=now))
        answer = entry.get('answer', '')

    return answer


def _deepwiki_fetch_answer(owner_repo: str, question: str, max_chars: int) -> str:
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError:
        return ''

    import asyncio

    async def _query() -> str:
        try:
            async with streamablehttp_client(_DEEPWIKI_MCP_URL) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        'ask_question',
                        {'repoName': owner_repo, 'question': question},
                    )
                    if not result or not result.content:
                        return ''
                    return '\n'.join(
                        c.text for c in result.content if hasattr(c, 'text') and c.text
                    )[:max_chars]
        except Exception as e:
            lazyllm.LOG.info(f'DeepWiki ask_question failed ({owner_repo!r}): {e}')
            return ''

    try:
        return asyncio.run(_query())
    except Exception as e:
        lazyllm.LOG.info(f'DeepWiki asyncio run failed: {e}')
        return ''
