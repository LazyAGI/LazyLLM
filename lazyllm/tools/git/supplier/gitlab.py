# Copyright (c) 2026 LazyAGI. All rights reserved.
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

from ..base import LazyLLMGitBase, PrInfo, ReviewCommentInfo, _sanitize_path


class GitLab(LazyLLMGitBase):
    def __init__(self, token: str, repo: Optional[str] = None, user: Optional[str] = None,
                 api_base: Optional[str] = None, return_trace: bool = False):
        super().__init__(
            token=token,
            repo=repo,
            api_base=api_base or 'https://gitlab.com/api/v4',
            user=user,
            return_trace=return_trace,
        )
        self._project_path = (self._repo or '').strip().strip('/')
        self._session.headers.update({'PRIVATE-TOKEN': self._token})
        self._current_user_id: Optional[int] = None

    def _url(self, path: str) -> str:
        self._require_repo()
        proj = quote(self._project_path, safe='')
        return f'{self._api_base}/projects/{proj}{_sanitize_path(path)}'

    def _get_current_user_id(self) -> int:
        if self._current_user_id is not None:
            return self._current_user_id
        r = self._session.get(f'{self._api_base}/user')
        if r.status_code != 200:
            raise RuntimeError(f'Failed to get current user: {r.text or r.reason}')
        data = r.json()
        self._current_user_id = data.get('id')
        return self._current_user_id

    def _req(self, method: str, path: str, **kwargs) -> 'requests.Response':
        return self._session.request(method, self._url(path), **kwargs)

    def create_pull_request(self, source_branch: str, target_branch: str,
                            title: str, body: str = '') -> Dict[str, Any]:
        payload = {
            'source_branch': source_branch,
            'target_branch': target_branch,
            'title': title,
            'description': body,
        }
        r = self._req('POST', '/merge_requests', json=payload)
        if r.status_code not in (200, 201):
            return {'success': False, 'message': r.text or r.reason}
        data = r.json()
        return {
            'success': True,
            'number': data['iid'],
            'html_url': data.get('web_url', ''),
            'message': 'created',
        }

    def update_pull_request(self, number: int, title: Optional[str] = None,
                            body: Optional[str] = None, state: Optional[str] = None) -> Dict[str, Any]:
        payload = {}
        if title is not None:
            payload['title'] = title
        if body is not None:
            payload['description'] = body
        if state is not None:
            payload['state_event'] = 'close' if state == 'closed' else 'reopen'
        if not payload:
            return {'success': True, 'message': 'nothing to update'}
        r = self._req('PUT', f'/merge_requests/{number}', json=payload)
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        return {'success': True, 'message': 'updated'}

    def add_pr_labels(self, number: int, labels: List[str]) -> Dict[str, Any]:
        r = self._req('PUT', f'/merge_requests/{number}', json={'labels': ','.join(labels)})
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        return {'success': True, 'message': 'labels updated'}

    def get_pull_request(self, number: int) -> Dict[str, Any]:
        r = self._req('GET', f'/merge_requests/{number}')
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        data = r.json()
        pr = PrInfo(
            number=data['iid'],
            title=data['title'],
            state=data.get('state', 'opened'),
            body=data.get('description') or '',
            source_branch=data.get('source_branch', ''),
            target_branch=data.get('target_branch', ''),
            html_url=data.get('web_url', ''),
            raw=data,
        )
        return {'success': True, 'pr': pr}

    def list_pull_requests(self, state: str = 'open', head: Optional[str] = None,
                           base: Optional[str] = None, max_results: int = 100) -> Dict[str, Any]:
        state_map = {'open': 'opened', 'closed': 'closed', 'all': 'all'}
        params: Dict[str, Any] = {'state': state_map.get(state, state), 'per_page': min(100, max_results), 'page': 1}
        if base is not None:
            params['target_branch'] = base
        if head is not None:
            params['source_branch'] = head
        out = []
        while len(out) < max_results:
            r = self._req('GET', '/merge_requests', params=params)
            if r.status_code != 200:
                return {'success': False, 'message': r.text or r.reason}
            page_data = r.json()
            if not page_data:
                break
            for data in page_data:
                out.append(PrInfo(
                    number=data['iid'],
                    title=data['title'],
                    state=data.get('state', 'opened'),
                    body=data.get('description') or '',
                    source_branch=data.get('source_branch', ''),
                    target_branch=data.get('target_branch', ''),
                    html_url=data.get('web_url', ''),
                    raw=data,
                ))
                if len(out) >= max_results:
                    break
            next_page = r.headers.get('x-next-page', '')
            if not next_page:
                break
            params = {'page': int(next_page), 'per_page': params['per_page']}
        return {'success': True, 'list': out}

    def get_pr_diff(self, number: int) -> Dict[str, Any]:
        r = self._req('GET', f'/merge_requests/{number}/changes')
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        data = r.json()
        diffs = [c.get('diff', '') for c in data.get('changes', []) if c.get('diff')]
        return {'success': True, 'diff': '\n'.join(diffs)}

    def list_review_comments(self, number: int) -> Dict[str, Any]:
        r = self._req('GET', f'/merge_requests/{number}/discussions')
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        out = []
        for d in r.json():
            for note in d.get('notes', []):
                if note.get('system'):
                    continue
                pos = note.get('position', {})
                out.append(ReviewCommentInfo(
                    id=note['id'],
                    body=note.get('body', ''),
                    path=pos.get('new_path') or pos.get('old_path', ''),
                    line=pos.get('new_line') or pos.get('old_line'),
                    side='RIGHT',
                    user=note.get('author', {}).get('username', ''),
                    raw=note,
                ))
        return {'success': True, 'comments': out}

    def create_review_comment(self, number: int, body: str, path: str,
                              line: Optional[int] = None, side: str = 'RIGHT',
                              commit_id: Optional[str] = None,
                              position: Optional[Dict[str, Any]] = None,
                              base_sha: Optional[str] = None,
                              start_sha: Optional[str] = None) -> Dict[str, Any]:
        pos = position
        if not pos and path and line is not None and commit_id:
            pos = {
                'new_path': path,
                'new_line': line,
                'position_type': 'text',
                'base_sha': base_sha,
                'head_sha': commit_id,
                'start_sha': start_sha or commit_id,
            }
        if not pos:
            r = self._req('POST', f'/merge_requests/{number}/notes', json={'body': body})
        else:
            r = self._req('POST', f'/merge_requests/{number}/discussions', json={'body': body, 'position': pos})
        if r.status_code not in (200, 201):
            return {'success': False, 'message': r.text or r.reason}
        data = r.json()
        cid = data.get('id') or (data.get('notes', [{}])[0].get('id') if data.get('notes') else None)
        return {'success': True, 'comment_id': cid, 'message': 'created'}

    def submit_review(self, number: int, event: str, body: str = '',
                      comments: Optional[List[Dict[str, Any]]] = None,
                      commit_id: Optional[str] = None) -> Dict[str, Any]:
        if event.upper() == 'APPROVE':
            return self.approve_pull_request(number)
        if body:
            self._req('POST', f'/merge_requests/{number}/notes', json={'body': body})
        if comments:
            for c in comments:
                if not isinstance(c, dict) or not c.get('body'):
                    continue
                # GitLab needs precise position; fallback to MR note if not provided.
                path = c.get('path', '')
                line = c.get('line')
                if path and line is not None and commit_id:
                    self.create_review_comment(
                        number=number, body=c['body'], path=path, line=int(line), commit_id=commit_id
                    )
                else:
                    self._req('POST', f'/merge_requests/{number}/notes', json={'body': c['body']})
        return {'success': True, 'message': 'submitted'}

    def approve_pull_request(self, number: int, sha: Optional[str] = None) -> Dict[str, Any]:
        payload = {'sha': sha} if sha else {}
        r = self._req('POST', f'/merge_requests/{number}/approve', json=payload)
        if r.status_code not in (200, 201):
            return {'success': False, 'message': r.text or r.reason}
        return {'success': True, 'message': 'approved'}

    def merge_pull_request(self, number: int, merge_method: Optional[str] = None,
                           commit_title: Optional[str] = None,
                           commit_message: Optional[str] = None,
                           merge_when_pipeline_succeeds: bool = False,
                           sha: Optional[str] = None) -> Dict[str, Any]:
        payload = {}
        if merge_method and merge_method.lower() == 'squash':
            payload['squash'] = True
        if merge_when_pipeline_succeeds:
            payload['merge_when_pipeline_succeeds'] = True
        if sha:
            payload['sha'] = sha
        r = self._req('PUT', f'/merge_requests/{number}/merge', json=payload)
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        data = r.json()
        return {'success': True, 'sha': data.get('merge_commit_sha'), 'message': 'merged'}

    def list_repo_stargazers(self, page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        self._require_repo()
        return {
            'success': False,
            'message': 'GitLab API does not provide an endpoint to list users who starred a project.',
        }

    def reply_to_review_comment(self, number: int, comment_id: Any, body: str,
                                path: str, line: Optional[int] = None,
                                commit_id: Optional[str] = None,
                                discussion_id: Optional[Any] = None) -> Dict[str, Any]:
        self._require_repo()
        if discussion_id:
            r = self._req('POST', f'/merge_requests/{number}/discussions/{discussion_id}/notes', json={'body': body})
        else:
            r = self._req('POST', f'/merge_requests/{number}/notes', json={'body': body})
        if r.status_code not in (200, 201):
            return {'success': False, 'message': r.text or r.reason}
        data = r.json()
        return {'success': True, 'comment_id': data.get('id'), 'message': 'replied'}

    def resolve_review_comment(self, number: int, comment_id: Any,
                               discussion_id: Optional[Any] = None) -> Dict[str, Any]:
        self._require_repo()
        if not discussion_id:
            return {
                'success': False,
                'message': 'GitLab requires discussion_id (pass as keyword argument) to resolve a discussion.',
            }
        r = self._req('PUT', f'/merge_requests/{number}/discussions/{discussion_id}',
                      json={'resolved': True})
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        return {'success': True, 'message': 'resolved'}

    def get_user_info(self, username: Optional[str] = None) -> Dict[str, Any]:
        if username:
            r = self._session.get(f'{self._api_base}/users', params={'username': username})
            if r.status_code != 200:
                return {'success': False, 'message': r.text or r.reason}
            data = r.json()
            if not data:
                return {'success': False, 'message': f'User not found: {username}'}
            return {'success': True, 'user': data[0]}
        if self._user:
            r = self._session.get(f'{self._api_base}/users', params={'username': self._user})
            if r.status_code != 200 or not r.json():
                return {'success': False, 'message': r.text or r.reason or 'User not found'}
            return {'success': True, 'user': r.json()[0]}
        r = self._session.get(f'{self._api_base}/user')
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        return {'success': True, 'user': r.json()}

    def list_user_starred_repos(self, username: Optional[str] = None,
                                page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        if username:
            ru = self._session.get(f'{self._api_base}/users', params={'username': username})
            if ru.status_code != 200 or not ru.json():
                return {'success': False, 'message': ru.text or ru.reason or 'User not found'}
            user_id = ru.json()[0].get('id')
            url = f'{self._api_base}/users/{user_id}/starred_projects'
        else:
            if self._user:
                ru = self._session.get(f'{self._api_base}/users', params={'username': self._user})
                if ru.status_code != 200 or not ru.json():
                    return {'success': False, 'message': ru.text or ru.reason or 'User not found'}
                user_id = ru.json()[0].get('id')
            else:
                user_id = self._get_current_user_id()
            url = f'{self._api_base}/users/{user_id}/starred_projects'
        r = self._session.get(url, params={'page': page, 'per_page': per_page})
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        return {'success': True, 'list': r.json()}
