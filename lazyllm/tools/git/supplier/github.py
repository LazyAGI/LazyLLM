# Copyright (c) 2026 LazyAGI. All rights reserved.
'''GitHub backend using REST API v3.'''
import subprocess
from typing import Any, Dict, List, Optional

import requests

from ..base import LazyLLMGitBase, PrInfo, ReviewCommentInfo


def _parse_repo(repo: str) -> tuple:
    parts = repo.split('/', 1)
    if len(parts) != 2:
        raise ValueError(f'repo must be \'owner/repo\', got: {repo!r}')
    return parts[0], parts[1]


class GitHub(LazyLLMGitBase):
    '''GitHub backend: REST API (api.github.com), push via local git.'''

    def __init__(self, token: str, repo: str, api_base: Optional[str] = None, **kwargs):
        super().__init__(token=token, repo=repo, api_base=api_base or 'https://api.github.com', **kwargs)
        self._owner, self._repo_name = _parse_repo(self._repo)
        self._session = requests.Session()
        self._session.headers.update({
            'Accept': 'application/vnd.github.v3+json',
            'Authorization': f'Bearer {self._token}',
        })

    def _url(self, path: str) -> str:
        return f'{self._api_base}/repos/{self._owner}/{self._repo_name}{path}'

    def _req(self, method: str, path: str, **kwargs) -> 'requests.Response':
        return self._session.request(method, self._url(path), **kwargs)

    def push_branch(self, local_branch: str, remote_branch: Optional[str] = None,
                    remote_name: str = 'origin', repo_path: Optional[str] = None) -> Dict[str, Any]:
        remote_branch = remote_branch or local_branch
        cwd = repo_path or '.'
        try:
            out = subprocess.run(
                ['git', 'push', remote_name, f'{local_branch}:{remote_branch}'],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=cwd,
            )
            if out.returncode != 0:
                return {'success': False, 'message': out.stderr or out.stdout or 'git push failed'}
            return {'success': True, 'message': out.stdout or 'pushed'}
        except FileNotFoundError:
            return {'success': False, 'message': 'git not found'}
        except subprocess.TimeoutExpired:
            return {'success': False, 'message': 'git push timeout'}
        except Exception as e:
            return {'success': False, 'message': str(e)}

    def create_pull_request(self, source_branch: str, target_branch: str,
                            title: str, body: str = '', **kwargs) -> Dict[str, Any]:
        payload = {
            'title': title,
            'head': source_branch,
            'base': target_branch,
            'body': body,
            **{k: v for k, v in kwargs.items() if k in ('draft', 'maintainer_can_modify')},
        }
        r = self._req('POST', '/pulls', json=payload)
        if r.status_code not in (200, 201):
            return {'success': False, 'message': r.text or r.reason}
        data = r.json()
        return {
            'success': True,
            'number': data['number'],
            'html_url': data.get('html_url', ''),
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
            payload['state'] = state
        payload.update(kwargs)
        if not payload:
            return {'success': True, 'message': 'nothing to update'}
        r = self._req('PATCH', f'/pulls/{number}', json=payload)
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        return {'success': True, 'message': 'updated'}

    def add_pr_labels(self, number: int, labels: List[str]) -> Dict[str, Any]:
        r = self._req('POST', f'/issues/{number}/labels', json=labels)
        if r.status_code not in (200, 201):
            return {'success': False, 'message': r.text or r.reason}
        return {'success': True, 'message': 'labels added'}

    def get_pull_request(self, number: int) -> Dict[str, Any]:
        r = self._req('GET', f'/pulls/{number}')
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        data = r.json()
        pr = PrInfo(
            number=data['number'],
            title=data['title'],
            state=data.get('state', 'open'),
            body=data.get('body') or '',
            source_branch=data.get('head', {}).get('ref', ''),
            target_branch=data.get('base', {}).get('ref', ''),
            html_url=data.get('html_url', ''),
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
                number=data['number'],
                title=data['title'],
                state=data.get('state', 'open'),
                body=data.get('body') or '',
                source_branch=data.get('head', {}).get('ref', ''),
                target_branch=data.get('base', {}).get('ref', ''),
                html_url=data.get('html_url', ''),
                raw=data,
            ))
        return {'success': True, 'list': out}

    def get_pr_diff(self, number: int) -> Dict[str, Any]:
        r = self._req('GET', f'/pulls/{number}', headers={'Accept': 'application/vnd.github.v3.diff'})
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        return {'success': True, 'diff': r.text}

    def list_review_comments(self, number: int) -> Dict[str, Any]:
        r = self._req('GET', f'/pulls/{number}/comments')
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        out = []
        for c in r.json():
            out.append(ReviewCommentInfo(
                id=c['id'],
                body=c.get('body', ''),
                path=c.get('path', ''),
                line=c.get('line'),
                side=c.get('side', 'RIGHT'),
                user=c.get('user', {}).get('login', ''),
                raw=c,
            ))
        return {'success': True, 'comments': out}

    def create_review_comment(self, number: int, body: str, path: str,
                              line: Optional[int] = None, side: str = 'RIGHT',
                              commit_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        payload = {'body': body, 'path': path}
        if commit_id:
            payload['commit_id'] = commit_id
        if line is not None:
            payload['line'] = line
        payload['side'] = side
        payload.update({k: v for k, v in kwargs.items() if k in ('start_line', 'start_side', 'in_reply_to')})
        r = self._req('POST', f'/pulls/{number}/comments', json=payload)
        if r.status_code not in (200, 201):
            return {'success': False, 'message': r.text or r.reason}
        data = r.json()
        return {'success': True, 'comment_id': data['id'], 'message': 'created'}

    def submit_review(self, number: int, event: str, body: str = '',
                      comment_ids: Optional[List[Any]] = None, **kwargs) -> Dict[str, Any]:
        payload = {'event': event}
        if body:
            payload['body'] = body
        if comment_ids is not None:
            payload['comments'] = []
        payload.update(kwargs)
        r = self._req('POST', f'/pulls/{number}/reviews', json=payload)
        if r.status_code not in (200, 201):
            return {'success': False, 'message': r.text or r.reason}
        return {'success': True, 'message': 'submitted'}

    def approve_pull_request(self, number: int, **kwargs) -> Dict[str, Any]:
        return self.submit_review(number, 'APPROVE', **kwargs)

    def merge_pull_request(self, number: int, merge_method: Optional[str] = None,
                           commit_title: Optional[str] = None,
                           commit_message: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        payload = {}
        if merge_method:
            payload['merge_method'] = merge_method
        if commit_title is not None:
            payload['commit_title'] = commit_title
        if commit_message is not None:
            payload['commit_message'] = commit_message
        payload.update(kwargs)
        r = self._req('PUT', f'/pulls/{number}/merge', json=payload)
        if r.status_code != 200:
            return {'success': False, 'message': r.text or r.reason}
        data = r.json()
        return {'success': True, 'sha': data.get('sha'), 'message': 'merged'}
