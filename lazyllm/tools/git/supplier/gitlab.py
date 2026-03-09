# Copyright (c) 2026 LazyAGI. All rights reserved.
'''GitLab backend using REST API v4.'''
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

from ..base import LazyLLMGitBase, PrInfo, ReviewCommentInfo, _sanitize_path


def _parse_repo(repo: str) -> str:
    return repo.strip().strip('/')


class GitLab(LazyLLMGitBase):
    '''GitLab backend: REST API (gitlab.com/api/v4), push via local git.'''

    def __init__(self, token: str, repo: str, api_base: Optional[str] = None, **kwargs):
        super().__init__(token=token, repo=repo, api_base=api_base or 'https://gitlab.com/api/v4', **kwargs)
        self._project_path = _parse_repo(self._repo)
        self._session = requests.Session()
        self._session.headers.update({'PRIVATE-TOKEN': self._token})

    def _url(self, path: str) -> str:
        proj = quote(self._project_path, safe='')
        return f'{self._api_base}/projects/{proj}{_sanitize_path(path)}'

    def _req(self, method: str, path: str, **kwargs) -> 'requests.Response':
        return self._session.request(method, self._url(path), **kwargs)

    def create_pull_request(self, source_branch: str, target_branch: str,
                            title: str, body: str = '', **kwargs) -> Dict[str, Any]:
        payload = {
            'source_branch': source_branch,
            'target_branch': target_branch,
            'title': title,
            'description': body,
            **{k: v for k, v in kwargs.items() if k in ('assignee_ids', 'labels', 'remove_source_branch')},
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
                            body: Optional[str] = None, state: Optional[str] = None,
                            **kwargs) -> Dict[str, Any]:
        payload = {}
        if title is not None:
            payload['title'] = title
        if body is not None:
            payload['description'] = body
        if state is not None:
            payload['state_event'] = 'close' if state == 'closed' else 'reopen'
        payload.update(kwargs)
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
                           base: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        state_map = {'open': 'opened', 'closed': 'closed', 'all': 'all'}
        params = {'state': state_map.get(state, state)}
        if base is not None:
            params['target_branch'] = base
        if head is not None:
            params['source_branch'] = head
        params.update({k: v for k, v in kwargs.items() if k in ('page', 'per_page', 'order_by', 'sort')})
        r = self._req('GET', '/merge_requests', params=params)
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        out = []
        for data in r.json():
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
                              commit_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        position = kwargs.get('position')
        if not position and path and line is not None and commit_id:
            position = {
                'new_path': path,
                'new_line': line,
                'position_type': 'text',
                'base_sha': kwargs.get('base_sha'),
                'head_sha': commit_id,
                'start_sha': kwargs.get('start_sha', commit_id),
            }
        if not position:
            r = self._req('POST', f'/merge_requests/{number}/notes', json={'body': body})
        else:
            r = self._req('POST', f'/merge_requests/{number}/discussions', json={'body': body, 'position': position})
        if r.status_code not in (200, 201):
            return {'success': False, 'message': r.text or r.reason}
        data = r.json()
        cid = data.get('id') or (data.get('notes', [{}])[0].get('id') if data.get('notes') else None)
        return {'success': True, 'comment_id': cid, 'message': 'created'}

    def submit_review(self, number: int, event: str, body: str = '',
                      comment_ids: Optional[List[Any]] = None, **kwargs) -> Dict[str, Any]:
        if event.upper() == 'APPROVE':
            return self.approve_pull_request(number, **kwargs)
        if body:
            self._req('POST', f'/merge_requests/{number}/notes', json={'body': body})
        return {'success': True, 'message': 'submitted'}

    def approve_pull_request(self, number: int, **kwargs) -> Dict[str, Any]:
        payload = {'sha': kwargs['sha']} if kwargs.get('sha') else {}
        r = self._req('POST', f'/merge_requests/{number}/approve', json=payload)
        if r.status_code not in (200, 201):
            return {'success': False, 'message': r.text or r.reason}
        return {'success': True, 'message': 'approved'}

    def merge_pull_request(self, number: int, merge_method: Optional[str] = None,
                           commit_title: Optional[str] = None,
                           commit_message: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        payload = {}
        if merge_method and merge_method.lower() == 'squash':
            payload['squash'] = True
        payload.update({k: v for k, v in kwargs.items() if k in ('merge_when_pipeline_succeeds', 'sha')})
        r = self._req('PUT', f'/merge_requests/{number}/merge', json=payload)
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        data = r.json()
        return {'success': True, 'sha': data.get('merge_commit_sha'), 'message': 'merged'}
