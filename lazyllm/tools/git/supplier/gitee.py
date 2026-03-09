# Copyright (c) 2026 LazyAGI. All rights reserved.
'''Gitee backend using OpenAPI v5.'''
from typing import Any, Dict, List, Optional

import requests

from ..base import LazyLLMGitBase, PrInfo, ReviewCommentInfo, _sanitize_path


def _parse_repo(repo: str) -> tuple:
    parts = repo.split('/', 1)
    if len(parts) != 2:
        raise ValueError(f'repo must be \'owner/repo\', got: {repo!r}')
    return parts[0], parts[1]


def _head_base_ref(data: dict, key: str) -> str:
    val = data.get(key, {})
    if isinstance(val, dict):
        return val.get('ref', val.get('label', ''))
    return ''


class Gitee(LazyLLMGitBase):
    '''Gitee backend: OpenAPI v5 (gitee.com/api/v5), push via local git.'''

    def __init__(self, token: str, repo: str, api_base: Optional[str] = None, **kwargs):
        super().__init__(token=token, repo=repo, api_base=api_base or 'https://gitee.com/api/v5', **kwargs)
        self._owner, self._repo_name = _parse_repo(self._repo)
        self._session = requests.Session()
        self._session.params = {'access_token': self._token}

    def _url(self, path: str) -> str:
        return f'{self._api_base}/repos/{self._owner}/{self._repo_name}{_sanitize_path(path)}'

    def _req(self, method: str, path: str, **kwargs) -> 'requests.Response':
        return self._session.request(method, self._url(path), **kwargs)

    def create_pull_request(self, source_branch: str, target_branch: str,
                            title: str, body: str = '', **kwargs) -> Dict[str, Any]:
        payload = {
            'title': title,
            'head': source_branch,
            'base': target_branch,
            'body': body,
            **{k: v for k, v in kwargs.items() if k in ('assignees', 'labels', 'prune_source_branch')},
        }
        r = self._req('POST', '/pulls', json=payload)
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
                            body: Optional[str] = None, state: Optional[str] = None,
                            **kwargs) -> Dict[str, Any]:
        payload = {}
        if title is not None:
            payload['title'] = title
        if body is not None:
            payload['body'] = body
        if state is not None:
            payload['state'] = 'closed' if state == 'closed' else 'open'
        payload.update(kwargs)
        if not payload:
            return {'success': True, 'message': 'nothing to update'}
        r = self._req('PATCH', f'/pulls/{number}', json=payload)
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        return {'success': True, 'message': 'updated'}

    def add_pr_labels(self, number: int, labels: List[str]) -> Dict[str, Any]:
        r = self._req('PATCH', f'/pulls/{number}', json={'labels': labels})
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        return {'success': True, 'message': 'labels updated'}

    def get_pull_request(self, number: int) -> Dict[str, Any]:
        r = self._req('GET', f'/pulls/{number}')
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
                           base: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        params = {'state': state}
        if head is not None:
            params['head'] = head
        if base is not None:
            params['base'] = base
        params.update({k: v for k, v in kwargs.items() if k in ('page', 'per_page', 'sort', 'direction')})
        r = self._req('GET', '/pulls', params=params)
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
        r = self._req('GET', f'/pulls/{number}')
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        data = r.json()
        diff_url = data.get('diff_url') or data.get('patch_url')
        if diff_url and diff_url.startswith((self._api_base.rstrip('/'), 'https://gitee.com')):
            rr = requests.get(diff_url, params={'access_token': self._token}, timeout=60)
            if rr.status_code == 200:
                return {'success': True, 'diff': rr.text}
        r2 = self._req('GET', f'/pulls/{number}/files')
        if r2.status_code != 200:
            return {'success': False, 'message': r2.text or 'no diff available'}
        parts = [f.get('patch', '') for f in r2.json()]
        return {'success': True, 'diff': '\n'.join(parts)}

    def list_review_comments(self, number: int) -> Dict[str, Any]:
        r = self._req('GET', f'/pulls/{number}/comments')
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
                              commit_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        payload = {'body': body, 'path': path}
        if line is not None:
            payload['line'] = line
        if commit_id:
            payload['commit_id'] = commit_id
        payload.update(kwargs)
        r = self._req('POST', f'/pulls/{number}/comments', json=payload)
        if r.status_code not in (200, 201):
            return {'success': False, 'message': r.text or r.reason}
        data = r.json()
        return {'success': True, 'comment_id': data.get('id'), 'message': 'created'}

    def submit_review(self, number: int, event: str, body: str = '',
                      comment_ids: Optional[List[Any]] = None, **kwargs) -> Dict[str, Any]:
        payload = {'body': body, 'event': event.upper() == 'APPROVE' and 'approve' or event}
        if comment_ids is not None:
            payload['comments'] = comment_ids
        payload.update(kwargs)
        r = self._req('POST', f'/pulls/{number}/review', json=payload)
        if r.status_code not in (200, 201):
            return {'success': False, 'message': r.text or r.reason}
        return {'success': True, 'message': 'submitted'}

    def approve_pull_request(self, number: int, **kwargs) -> Dict[str, Any]:
        return self.submit_review(number, 'APPROVE', **kwargs)

    def merge_pull_request(self, number: int, merge_method: Optional[str] = None,
                           commit_title: Optional[str] = None,
                           commit_message: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        payload = {}
        if commit_title is not None:
            payload['merge_commit_message'] = commit_title
        if commit_message is not None:
            payload['merge_commit_message'] = (payload.get('merge_commit_message') or '') + '\n\n' + commit_message
        payload.update({k: v for k, v in kwargs.items() if k in ('prune_source_branch', 'merge_method')})
        r = self._req('PUT', f'/pulls/{number}/merge', json=payload)
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        data = r.json() if r.content else {}
        return {'success': True, 'sha': data.get('sha'), 'message': 'merged'}
