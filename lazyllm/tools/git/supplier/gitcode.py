# Copyright (c) 2026 LazyAGI. All rights reserved.
from typing import Any, Dict, List, Optional

import requests

from ..base import LazyLLMGitBase, PrInfo, ReviewCommentInfo, _sanitize_path
from .gitee import _head_base_ref


class GitCode(LazyLLMGitBase):
    def __init__(self, token: str, repo: Optional[str] = None, user: Optional[str] = None,
                 api_base: Optional[str] = None, return_trace: bool = False):
        super().__init__(
            token=token,
            repo=repo,
            api_base=api_base or 'https://api.gitcode.com/api/v5',
            user=user,
            return_trace=return_trace,
        )
        if self._repo:
            self._owner, self._repo_name = self._parse_owner_repo(self._repo)
        else:
            self._owner, self._repo_name = None, None
        self._session.params = {'access_token': self._token}
        self._current_user_login: Optional[str] = None

    def _repo_url(self, path: str) -> str:
        self._require_repo()
        return f'{self._api_base}/repos/{self._owner}/{self._repo_name}{_sanitize_path(path)}'

    def _user_api_url(self, path: str, use_current: bool = False) -> str:
        if use_current or not self._user:
            return f'{self._api_base}/user{_sanitize_path(path)}'
        return f'{self._api_base}/users/{self._user}{_sanitize_path(path)}'

    def _get_current_user(self) -> str:
        if self._current_user_login is not None:
            return self._current_user_login
        r = self._session.get(self._user_api_url('', use_current=True))
        if r.status_code != 200:
            raise RuntimeError(f'Failed to get current user: {r.text or r.reason}')
        data = r.json()
        self._current_user_login = data.get('login', data.get('name', ''))
        return self._current_user_login

    def _req_repo(self, method: str, path: str, **kwargs) -> 'requests.Response':
        return self._session.request(method, self._repo_url(path), **kwargs)

    def _req_user(self, path: str, use_current: bool = False, **kwargs) -> 'requests.Response':
        url = self._user_api_url(path, use_current=use_current)
        return self._session.get(url, **kwargs)

    # ---------- Repo-related (require repo) ----------

    def create_pull_request(self, source_branch: str, target_branch: str,
                            title: str, body: str = '') -> Dict[str, Any]:
        self._require_repo()
        payload = {
            'title': title,
            'head': source_branch,
            'base': target_branch,
            'body': body,
        }
        r = self._req_repo('POST', '/pulls', json=payload)
        if r.status_code not in (200, 201):
            return {'success': False, 'message': r.text or r.reason}
        data = r.json()
        return {
            'success': True,
            'number': data.get('number', data.get('id')),
            'html_url': data.get('html_url', data.get('url', '')),
            'message': 'created',
        }

    def update_pull_request(self, number: int, title: Optional[str] = None,
                            body: Optional[str] = None, state: Optional[str] = None) -> Dict[str, Any]:
        self._require_repo()
        payload = {}
        if title is not None:
            payload['title'] = title
        if body is not None:
            payload['body'] = body
        if state is not None:
            payload['state'] = 'closed' if state == 'closed' else 'open'
        if not payload:
            return {'success': True, 'message': 'nothing to update'}
        r = self._req_repo('PATCH', f'/pulls/{number}', json=payload)
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        return {'success': True, 'message': 'updated'}

    def add_pr_labels(self, number: int, labels: List[str]) -> Dict[str, Any]:
        self._require_repo()
        r = self._req_repo('PATCH', f'/pulls/{number}', json={'labels': labels})
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        return {'success': True, 'message': 'labels updated'}

    def get_pull_request(self, number: int) -> Dict[str, Any]:
        self._require_repo()
        r = self._req_repo('GET', f'/pulls/{number}')
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        data = r.json()
        pr = PrInfo(
            number=data.get('number', data.get('id')),
            title=data.get('title', ''),
            state=data.get('state', 'open'),
            body=data.get('body', data.get('description', '')) or '',
            source_branch=_head_base_ref(data, 'head'),
            target_branch=_head_base_ref(data, 'base'),
            html_url=data.get('html_url', data.get('url', '')),
            raw=data,
        )
        return {'success': True, 'pr': pr}

    def list_pull_requests(self, state: str = 'open', head: Optional[str] = None,
                           base: Optional[str] = None) -> Dict[str, Any]:
        self._require_repo()
        params = {'state': state}
        if head is not None:
            params['head'] = head
        if base is not None:
            params['base'] = base
        r = self._req_repo('GET', '/pulls', params=params)
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        out = []
        for data in r.json():
            out.append(PrInfo(
                number=data.get('number', data.get('id')),
                title=data.get('title', ''),
                state=data.get('state', 'open'),
                body=data.get('body', data.get('description', '')) or '',
                source_branch=_head_base_ref(data, 'head'),
                target_branch=_head_base_ref(data, 'base'),
                html_url=data.get('html_url', data.get('url', '')),
                raw=data,
            ))
        return {'success': True, 'list': out}

    def get_pr_diff(self, number: int) -> Dict[str, Any]:
        self._require_repo()
        r = self._req_repo('GET', f'/pulls/{number}')
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        data = r.json()
        diff_url = data.get('diff_url') or data.get('patch_url')
        if diff_url and diff_url.startswith((self._api_base.rstrip('/'), 'https://gitcode.com')):
            rr = requests.get(diff_url, params={'access_token': self._token}, timeout=60)
            if rr.status_code == 200:
                return {'success': True, 'diff': rr.text}
        r2 = self._req_repo('GET', f'/pulls/{number}/files')
        if r2.status_code != 200:
            return {'success': False, 'message': r2.text or 'no diff available'}
        parts = [f.get('patch', '') for f in r2.json()]
        return {'success': True, 'diff': '\n'.join(parts)}

    def list_review_comments(self, number: int) -> Dict[str, Any]:
        self._require_repo()
        r = self._req_repo('GET', f'/pulls/{number}/comments')
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        out = []
        for c in r.json():
            user = c.get('user', {})
            out.append(ReviewCommentInfo(
                id=c.get('id'),
                body=c.get('body', ''),
                path=c.get('path', ''),
                line=c.get('line'),
                side='RIGHT',
                user=user.get('login', '') if isinstance(user, dict) else '',
                raw=c,
            ))
        return {'success': True, 'comments': out}

    def create_review_comment(self, number: int, body: str, path: str,
                              line: Optional[int] = None, side: str = 'RIGHT',
                              commit_id: Optional[str] = None) -> Dict[str, Any]:
        self._require_repo()
        payload = {'body': body, 'path': path}
        if line is not None:
            payload['line'] = line
        if commit_id:
            payload['commit_id'] = commit_id
        r = self._req_repo('POST', f'/pulls/{number}/comments', json=payload)
        if r.status_code not in (200, 201):
            return {'success': False, 'message': r.text or r.reason}
        data = r.json()
        return {'success': True, 'comment_id': data.get('id'), 'message': 'created'}

    def submit_review(self, number: int, event: str, body: str = '',
                      comments: Optional[List[Dict[str, Any]]] = None,
                      commit_id: Optional[str] = None) -> Dict[str, Any]:
        self._require_repo()
        payload = {'body': body, 'event': event.upper() == 'APPROVE' and 'approve' or event}
        if comments:
            # GitCode may not support GitHub-like batch inline comments; fallback to creating comments one-by-one.
            for c in comments:
                if not isinstance(c, dict) or not c.get('body'):
                    continue
                self.create_review_comment(
                    number=number,
                    body=c['body'],
                    path=c.get('path', ''),
                    line=c.get('line'),
                    commit_id=commit_id,
                )
        r = self._req_repo('POST', f'/pulls/{number}/review', json=payload)
        if r.status_code not in (200, 201):
            return {'success': False, 'message': r.text or r.reason}
        return {'success': True, 'message': 'submitted'}

    def approve_pull_request(self, number: int) -> Dict[str, Any]:
        return self.submit_review(number, 'APPROVE')

    def merge_pull_request(self, number: int, merge_method: Optional[str] = None,
                           commit_title: Optional[str] = None,
                           commit_message: Optional[str] = None) -> Dict[str, Any]:
        self._require_repo()
        payload = {}
        if commit_title is not None:
            payload['merge_commit_message'] = commit_title
        if commit_message is not None:
            payload['merge_commit_message'] = (payload.get('merge_commit_message') or '') + '\n\n' + commit_message
        r = self._req_repo('PUT', f'/pulls/{number}/merge', json=payload)
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        data = r.json() if r.content else {}
        return {'success': True, 'sha': data.get('sha'), 'message': 'merged'}

    # ---------- Repo extra: stargazers, reply, resolve, stash/batch ----------

    def list_repo_stargazers(self, page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        self._require_repo()
        r = self._req_repo('GET', '/stargazers', params={'page': page, 'per_page': per_page})
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        return {'success': True, 'list': r.json()}

    def reply_to_review_comment(self, number: int, comment_id: Any, body: str,
                                path: str, line: Optional[int] = None,
                                commit_id: Optional[str] = None) -> Dict[str, Any]:
        self._require_repo()
        payload = {'body': body, 'path': path, 'parent_id': comment_id}
        if line is not None:
            payload['line'] = line
        if commit_id:
            payload['commit_id'] = commit_id
        r = self._req_repo('POST', f'/pulls/{number}/comments', json=payload)
        if r.status_code not in (200, 201):
            return {'success': False, 'message': r.text or r.reason}
        data = r.json()
        return {'success': True, 'comment_id': data.get('id'), 'message': 'replied'}

    def resolve_review_comment(self, number: int, comment_id: Any) -> Dict[str, Any]:
        self._require_repo()
        r = self._req_repo('PATCH', f'/pulls/{number}/comments/{comment_id}', json={'resolved': True})
        if r.status_code != 200:
            return {
                'success': False,
                'message': r.text or r.reason or 'Resolve may not be supported by this API.',
            }
        return {'success': True, 'message': 'resolved'}

    # ---------- User-related (default to current user when user not set) ----------

    def get_user_info(self, username: Optional[str] = None) -> Dict[str, Any]:
        if username:
            r = self._session.get(f'{self._api_base}/users/{username}')
        else:
            # Use instance user or current user
            if self._user:
                r = self._session.get(f'{self._api_base}/users/{self._user}')
            else:
                r = self._session.get(f'{self._api_base}/user')
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        return {'success': True, 'user': r.json()}

    def list_user_starred_repos(self, username: Optional[str] = None,
                                page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        if username:
            url = f'{self._api_base}/users/{username}/starred'
        else:
            url = f'{self._api_base}/user/starred' if not self._user else f'{self._api_base}/users/{self._user}/starred'
        r = self._session.get(url, params={'page': page, 'per_page': per_page})
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        return {'success': True, 'list': r.json()}
