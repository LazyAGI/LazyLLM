import os
import re
import time
import json
import requests
from typing import Dict, Any, Iterator, Optional


def parse_github_repo(repo_link: str) -> str:
    repo_link = repo_link.strip()
    if repo_link.startswith("http"):
        m = re.search(r"github\.com/([^/]+)/([^/]+)", repo_link)
        if not m:
            raise ValueError(f"Invalid GitHub repo link: {repo_link}")
        owner, repo = m.group(1), m.group(2)
        return f"{owner}/{repo.replace('.git', '')}"
    if "/" in repo_link:
        return repo_link.replace(".git", "")
    raise ValueError(f"Invalid repo identifier: {repo_link}")


def _gh_headers(token: Optional[str]) -> Dict[str, str]:
    h = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "lazyllm-review-collector",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _maybe_wait_rate_limit(resp: requests.Response) -> None:
    # If we hit rate-limit, wait until reset.
    if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
        reset = resp.headers.get("X-RateLimit-Reset")
        wait_s = 30
        if reset and str(reset).isdigit():
            wait_s = max(1, int(reset) - int(time.time()) + 2)
        time.sleep(wait_s)


def _gh_get(session: requests.Session, url: str, headers: Dict[str, str], params: Dict[str, Any] = None) -> requests.Response:
    resp = session.get(url, headers=headers, params=params, timeout=60)
    # print(f'[_gh_get] resp = {resp}, headers = {resp.headers}')
    _maybe_wait_rate_limit(resp)
    if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
        # retry once after waiting
        resp = session.get(url, headers=headers, params=params, timeout=60)
    resp.raise_for_status()
    return resp


def _paginate(session: requests.Session, url: str, headers: Dict[str, str], params: Dict[str, Any] = None, max_pages: Optional[int] = None) -> Iterator[Any]:
    page = 1
    while True:
        # print(f"[_paginate]")
        if max_pages and page > max_pages:
            return
        p = dict(params or {})
        p.update({"per_page": 100, "page": page})
        resp = _gh_get(session, url, headers, p)
        data = resp.json()
        # print(f"data = {data}")
        if not isinstance(data, list):
            yield data
            return
        if not data:
            return
        for item in data:
            yield item
        page += 1


def _actor_login_and_type(obj: Dict[str, Any]) -> (Optional[str], Optional[str]):
    u = obj.get("user") or {}
    return u.get("login"), u.get("type")


def _should_skip_actor(item: Dict[str, Any], pr_author: Optional[str]) -> bool:
    user_name, utype = _actor_login_and_type(item)
    if not user_name:
        return True

    if user_name == "gemini-code-assist[bot]":
        return True

    if pr_author and user_name == pr_author:
        return True

    if utype.lower() == "bot":
        return True

    return False


def collect_merged_pr_all_comments(
    repo_link: str,
    token: Optional[str] = None,
    base_api: str = "https://api.github.com",
    out_jsonl: str = "artifacts/pr_comments.jsonl",
    max_prs: Optional[int] = None,
    progress_every: int = 20,
) -> Dict[str, int]:
    """
    Only merged PRs.
    Write one JSONL record per comment/review-item (streaming).
    Include:
      - inline review comments (/pulls/{n}/comments) with diff_hunk (code snippet)
      - PR conversation comments (/issues/{n}/comments) no code snippet -> diff_hunk=None
      - PR reviews (/pulls/{n}/reviews) no code snippet -> diff_hunk=None
    Any missing attr -> None.
    """
    repo = parse_github_repo(repo_link)
    headers = _gh_headers(token)
    session = requests.Session()

    pulls_url = f"{base_api}/repos/{repo}/pulls"
    pulls_params = {"state": "closed", "sort": "updated", "direction": "desc"}

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)

    counts = {
        "prs_scanned": 0,
        "prs_merged": 0,
        "inline_review_comments": 0,
        "conversation_comments": 0,
        "review_summaries": 0,
        "records_written": 0,
    }

    def _write_record(f_out, record: Dict[str, Any]) -> None:
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        f_out.flush()  # 关键：立刻落盘，你会马上看到输出
        counts["records_written"] += 1

    with open(out_jsonl, "w", encoding="utf-8") as f_out:
        for pr in _paginate(session, pulls_url, headers, pulls_params, max_pages=max_prs):
            counts["prs_scanned"] += 1

            merged_at = pr.get("merged_at")
            if not merged_at:
                continue
            counts["prs_merged"] += 1

            pr_number = pr.get("number")
            if not pr_number:
                continue

            print(f"--- processing pr {pr_number} ---")

            pr_meta = {
                "repo": repo,
                "pr_number": pr_number,
                "pr_title": pr.get("title"),
                "pr_url": pr.get("html_url"),
                "merged_at": merged_at,
                "created_at": pr.get("created_at"),
                "updated_at": pr.get("updated_at"),
                "author": (pr.get("user") or {}).get("login"),
                "base_ref": (pr.get("base") or {}).get("ref"),
                "head_ref": (pr.get("head") or {}).get("ref"),
            }

            # 1) Inline review comments (with diff_hunk)
            inline_url = f"{base_api}/repos/{repo}/pulls/{pr_number}/comments"
            for c in _paginate(session, inline_url, headers):
                print(f"- 1. inline comments {c.get('id')} -")
                if _should_skip_actor(c, pr_meta.get("author")):
                    continue

                record = {
                    "type": "inline_review_comment",
                    "pr": pr_meta,
                    "comment": {
                        "id": c.get("id"),
                        "url": c.get("html_url") or c.get("url"),
                        "user": (c.get("user") or {}).get("login"),
                        "created_at": c.get("created_at"),
                        "updated_at": c.get("updated_at"),
                        "body": c.get("body"),
                    },
                    "code": {
                        "path": c.get("path"),
                        "side": c.get("side"),
                        "start_side": c.get("start_side"),
                        "line": c.get("line"),
                        "start_line": c.get("start_line"),
                        "diff_hunk": c.get("diff_hunk"),  # 对应代码段（可能 None）
                        "commit_id": c.get("commit_id"),
                    },
                }
                _write_record(f_out, record)
                counts["inline_review_comments"] += 1

            # 2) PR conversation comments (issue comments on PR)
            conv_url = f"{base_api}/repos/{repo}/issues/{pr_number}/comments"
            for ic in _paginate(session, conv_url, headers):
                print(f"- 2. conversation comments {ic.get('id')} -")
                if _should_skip_actor(ic, pr_meta.get("author")):
                    continue

                record = {
                    "type": "pr_conversation_comment",
                    "pr": pr_meta,
                    "comment": {
                        "id": ic.get("id"),
                        "url": ic.get("html_url") or ic.get("url"),
                        "user": (ic.get("user") or {}).get("login"),
                        "created_at": ic.get("created_at"),
                        "updated_at": ic.get("updated_at"),
                        "body": ic.get("body"),
                    },
                    "code": {
                        "path": None,
                        "side": None,
                        "start_side": None,
                        "line": None,
                        "start_line": None,
                        "diff_hunk": None,  # conversation comment 通常没有对应代码段
                        "commit_id": None,
                    },
                }
                _write_record(f_out, record)
                counts["conversation_comments"] += 1

            # 3) Review summaries (APPROVED / CHANGES_REQUESTED / COMMENTED)
            reviews_url = f"{base_api}/repos/{repo}/pulls/{pr_number}/reviews"
            for r in _paginate(session, reviews_url, headers):
                print(f"- 3. review summaries {r.get('id')} -")
                if _should_skip_actor(r, pr_meta.get("author")):
                    continue

                record = {
                    "type": "pr_review_summary",
                    "pr": pr_meta,
                    "review": {
                        "id": r.get("id"),
                        "url": r.get("html_url") or r.get("url"),
                        "user": (r.get("user") or {}).get("login"),
                        "state": r.get("state"),
                        "submitted_at": r.get("submitted_at"),
                        "body": r.get("body"),
                        "commit_id": r.get("commit_id"),
                    },
                    "code": {
                        "path": None,
                        "side": None,
                        "start_side": None,
                        "line": None,
                        "start_line": None,
                        "diff_hunk": None,
                        "commit_id": None,
                    },
                }
                _write_record(f_out, record)
                counts["review_summaries"] += 1

            if progress_every and counts["prs_merged"] % progress_every == 0:
                print(
                    f"[progress] merged_prs={counts['prs_merged']} "
                    f"written={counts['records_written']} "
                    f"(inline={counts['inline_review_comments']}, "
                    f"conv={counts['conversation_comments']}, "
                    f"reviews={counts['review_summaries']})"
                )

            if max_prs and counts["prs_merged"] >= max_prs:
                break

    return counts


# Example:
counts = collect_merged_pr_all_comments(
    "https://github.com/LazyAGI/LazyLLM",
    token=os.environ.get("GITHUB_TOKEN"),
    out_jsonl="artifacts/pr_comments_100.jsonl",
    max_prs=100,
)
print(counts)
