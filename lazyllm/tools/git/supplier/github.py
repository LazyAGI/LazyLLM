# Copyright (c) 2026 LazyAGI. All rights reserved.
import os
import time
from typing import Any, Dict, List, Optional

import requests
from lazyllm import config
from lazyllm.thirdparty import jwt

from ..base import LazyLLMGitBase, PrInfo, ReviewCommentInfo, _sanitize_path

config.add('github_app_private_key_path', str,
           os.path.expanduser('~/.config/lazyllm/github-app-private-key.pem'),
           'GITHUB_APP_PRIVATE_KEY_PATH',
           description='Path to GitHub App private key PEM file for App authentication.')


def _get_installation_token(app_id: str, private_key_pem: str, installation_id: int,
                            api_base: str = 'https://api.github.com') -> str:
    now = int(time.time())
    payload = {'iat': now - 60, 'exp': now + 600, 'iss': str(app_id)}
    j = jwt.encode(payload, private_key_pem, algorithm='RS256')
    r = requests.post(
        f'{api_base}/app/installations/{installation_id}/access_tokens',
        headers={'Authorization': f'Bearer {j}', 'Accept': 'application/vnd.github.v3+json'},
        timeout=30,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f'Failed to get installation token: {r.status_code} {r.text}')
    return r.json()['token']


class GitHub(LazyLLMGitBase):
    def __init__(self, token: Optional[str] = None, repo: Optional[str] = None, user: Optional[str] = None,
                 api_base: Optional[str] = None, return_trace: bool = False,
                 app_id: Optional[str] = None, installation_id: Optional[int] = None,
                 private_key_pem: Optional[str] = None, private_key_path: Optional[str] = None):
        if app_id and installation_id:
            pem = private_key_pem
            if not pem:
                path = private_key_path or config['github_app_private_key_path']
                with open(path, 'r') as f:
                    pem = f.read()
            resolved_token = _get_installation_token(app_id, pem, installation_id,
                                                     api_base or 'https://api.github.com')
        else:
            if not token:
                raise ValueError('Provide either token or (app_id + installation_id) for GitHub App auth.')
            resolved_token = token
        super().__init__(
            token=resolved_token,
            repo=repo,
            api_base=api_base or 'https://api.github.com',
            user=user,
            return_trace=return_trace,
        )
        if self._repo:
            self._owner, self._repo_name = self._parse_owner_repo(self._repo)
        else:
            self._owner, self._repo_name = None, None
        self._session.headers.update({
            'Accept': 'application/vnd.github.v3+json',
            'Authorization': f'Bearer {self._token}',
        })
        self._current_user_login: Optional[str] = None

    def _url(self, path: str) -> str:
        self._require_repo()
        return f'{self._api_base}/repos/{self._owner}/{self._repo_name}{_sanitize_path(path)}'

    def _check_401(self, r: 'requests.Response') -> None:
        if r.status_code == 401:
            raise RuntimeError(
                'GitHub API returned 401 Unauthorized. '
                'Please authenticate by running: gh auth login'
            )

    def _fail(self, r: 'requests.Response') -> Dict[str, Any]:
        self._check_401(r)
        return {'success': False, 'message': r.text or r.reason}

    def _get_current_user(self) -> str:
        if self._current_user_login is not None:
            return self._current_user_login
        r = self._session.get(f'{self._api_base}/user')
        if r.status_code != 200:
            self._check_401(r)
            raise RuntimeError(f'Failed to get current user: {r.text or r.reason}')
        data = r.json()
        self._current_user_login = data.get('login', '')
        return self._current_user_login

    def _req(self, method: str, path: str, **kwargs) -> 'requests.Response':
        return self._session.request(method, self._url(path), **kwargs)

    def create_pull_request(self, source_branch: str, target_branch: str,
                            title: str, body: str = '') -> Dict[str, Any]:
        payload = {
            'title': title,
            'head': source_branch,
            'base': target_branch,
            'body': body,
        }
        r = self._req('POST', '/pulls', json=payload)
        if r.status_code not in (200, 201):
            return self._fail(r)
        data = r.json()
        return {
            'success': True,
            'number': data['number'],
            'html_url': data.get('html_url', ''),
            'message': 'created',
        }

    def update_pull_request(self, number: int, title: Optional[str] = None,
                            body: Optional[str] = None, state: Optional[str] = None) -> Dict[str, Any]:
        payload = {}
        if title is not None:
            payload['title'] = title
        if body is not None:
            payload['body'] = body
        if state is not None:
            payload['state'] = state
        if not payload:
            return {'success': True, 'message': 'nothing to update'}
        r = self._req('PATCH', f'/pulls/{number}', json=payload)
        if r.status_code != 200:
            return self._fail(r)
        return {'success': True, 'message': 'updated'}

    def add_pr_labels(self, number: int, labels: List[str]) -> Dict[str, Any]:
        r = self._req('POST', f'/issues/{number}/labels', json=labels)
        if r.status_code not in (200, 201):
            return self._fail(r)
        return {'success': True, 'message': 'labels added'}

    def get_pull_request(self, number: int) -> Dict[str, Any]:
        r = self._req('GET', f'/pulls/{number}')
        if r.status_code != 200:
            return self._fail(r)
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
                           base: Optional[str] = None, max_results: int = 100) -> Dict[str, Any]:
        params: Dict[str, Any] = {'state': state, 'per_page': min(100, max_results)}
        if head is not None:
            params['head'] = head
        if base is not None:
            params['base'] = base
        out = []
        url: Optional[str] = self._url('/pulls')
        while url and len(out) < max_results:
            r = self._session.get(url, params=params)
            params = {}  # only pass params on first request; subsequent URLs are fully formed
            if r.status_code != 200:
                return self._fail(r)
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
                if len(out) >= max_results:
                    break
            link = r.headers.get('Link', '')
            next_url = None
            for part in link.split(','):
                part = part.strip()
                if 'rel="next"' in part:
                    next_url = part.split(';')[0].strip().strip('<>')
                    break
            url = next_url
        return {'success': True, 'list': out}

    def get_pr_diff(self, number: int) -> Dict[str, Any]:
        # GitHub's diff endpoint truncates at ~3000 lines for large PRs.
        # Use /files (paginated) to fetch each file's patch and assemble a proper unified diff.
        self._require_repo()
        files = []
        url: Optional[str] = self._url(f'/pulls/{number}/files')
        params: Dict[str, Any] = {'per_page': 100}
        while url:
            r = self._session.get(url, params=params)
            params = {}
            if r.status_code != 200:
                return self._fail(r)
            files.extend(r.json())
            link = r.headers.get('Link', '')
            url = None
            for part in link.split(','):
                part = part.strip()
                if 'rel="next"' in part:
                    url = part.split(';')[0].strip().strip('<>')
                    break
        # Assemble unified diff from per-file patches
        diff_parts = []
        for f in files:
            filename = f.get('filename', '')
            prev = f.get('previous_filename', filename)
            patch = f.get('patch', '')
            status = f.get('status', '')
            if status == 'removed':
                diff_parts.append(f'diff --git a/{prev} b/{filename}')
                diff_parts.append(f'--- a/{prev}')
                diff_parts.append('+++ /dev/null')
            elif status == 'added':
                diff_parts.append(f'diff --git a/{filename} b/{filename}')
                diff_parts.append('--- /dev/null')
                diff_parts.append(f'+++ b/{filename}')
            elif status == 'renamed':
                diff_parts.append(f'diff --git a/{prev} b/{filename}')
                diff_parts.append(f'--- a/{prev}')
                diff_parts.append(f'+++ b/{filename}')
            else:
                diff_parts.append(f'diff --git a/{filename} b/{filename}')
                diff_parts.append(f'--- a/{filename}')
                diff_parts.append(f'+++ b/{filename}')
            if patch:
                diff_parts.append(patch)
        return {'success': True, 'diff': '\n'.join(diff_parts) + '\n'}

    def list_review_comments(self, number: int) -> Dict[str, Any]:
        self._require_repo()
        out = []
        url: Optional[str] = self._url(f'/pulls/{number}/comments')
        params: Dict[str, Any] = {'per_page': 100}
        while url:
            r = self._session.get(url, params=params)
            params = {}
            if r.status_code != 200:
                return self._fail(r)
            for c in r.json():
                out.append(ReviewCommentInfo(
                    id=c.get('id'),
                    body=c.get('body', ''),
                    path=c.get('path', ''),
                    line=c.get('line'),
                    side=c.get('side', 'RIGHT'),
                    user=c.get('user', {}).get('login', ''),
                    raw=c,
                ))
            link = r.headers.get('Link', '')
            url = None
            for part in link.split(','):
                part = part.strip()
                if 'rel="next"' in part:
                    url = part.split(';')[0].strip().strip('<>')
                    break
        return {'success': True, 'comments': out}

    def create_review_comment(self, number: int, body: str, path: str,
                              line: Optional[int] = None, side: str = 'RIGHT',
                              commit_id: Optional[str] = None,
                              in_reply_to: Optional[Any] = None,
                              start_line: Optional[int] = None,
                              start_side: Optional[str] = None) -> Dict[str, Any]:
        payload = {'body': body, 'path': path}
        if commit_id:
            payload['commit_id'] = commit_id
        if line is not None:
            payload['line'] = line
        payload['side'] = side
        if in_reply_to is not None:
            payload['in_reply_to'] = in_reply_to
        if start_line is not None:
            payload['start_line'] = start_line
        if start_side is not None:
            payload['start_side'] = start_side
        r = self._req('POST', f'/pulls/{number}/comments', json=payload)
        if r.status_code not in (200, 201):
            return self._fail(r)
        data = r.json()
        return {'success': True, 'comment_id': data['id'], 'message': 'created'}

    def add_issue_comment(self, number: int, body: str) -> Dict[str, Any]:
        r = self._req('POST', f'/issues/{number}/comments', json={'body': body})
        if r.status_code not in (200, 201):
            return self._fail(r)
        return {'success': True, 'message': 'created', 'url': r.json().get('html_url', '')}

    def list_issue_comments(self, number: int) -> Dict[str, Any]:
        out = []
        url: Optional[str] = self._url(f'/issues/{number}/comments')
        params: Dict[str, Any] = {'per_page': 100}
        while url:
            r = self._session.get(url, params=params)
            params = {}
            if r.status_code != 200:
                if out:
                    return {'success': True, 'comments': out, 'partial': True}
                return self._fail(r)
            for c in r.json():
                out.append({
                    'id': c.get('id'),
                    'body': c.get('body', ''),
                    'user': c.get('user', {}).get('login', ''),
                    'raw': c,
                })
            link = r.headers.get('Link', '')
            url = None
            for part in link.split(','):
                part = part.strip()
                if 'rel="next"' in part:
                    url = part.split(';')[0].strip().strip('<>')
                    break
        return {'success': True, 'comments': out}

    def submit_review(self, number: int, event: str, body: str = '',
                      comments: Optional[List[Dict[str, Any]]] = None,
                      commit_id: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {'event': event}
        if body:
            payload['body'] = body
        if commit_id:
            payload['commit_id'] = commit_id
        if comments:
            valid_comments = [
                {
                    'path': c['path'],
                    'line': int(c['line']),
                    'body': c['body'],
                    'side': c.get('side', 'RIGHT'),
                }
                for c in comments if c.get('path') and c.get('line') and c.get('body')
            ]
            if valid_comments:
                payload['comments'] = valid_comments
        r = self._req('POST', f'/pulls/{number}/reviews', json=payload)
        if r.status_code not in (200, 201):
            import lazyllm
            lazyllm.LOG.warning(f'submit_review HTTP {r.status_code}: {r.text[:400]}')
            self._check_401(r)
            return {'success': False, 'message': r.text or r.reason, 'status_code': r.status_code}
        data = r.json()
        return {'success': True, 'review_id': data.get('id'), 'message': 'submitted'}

    def approve_pull_request(self, number: int) -> Dict[str, Any]:
        return self.submit_review(number, 'APPROVE')

    def merge_pull_request(self, number: int, merge_method: Optional[str] = None,
                           commit_title: Optional[str] = None,
                           commit_message: Optional[str] = None) -> Dict[str, Any]:
        payload = {}
        if merge_method:
            payload['merge_method'] = merge_method
        if commit_title is not None:
            payload['commit_title'] = commit_title
        if commit_message is not None:
            payload['commit_message'] = commit_message
        r = self._req('PUT', f'/pulls/{number}/merge', json=payload)
        if r.status_code != 200:
            return self._fail(r)
        data = r.json()
        return {'success': True, 'sha': data.get('sha'), 'message': 'merged'}

    def list_repo_stargazers(self, page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        self._require_repo()
        r = self._req('GET', '/stargazers', params={'page': page, 'per_page': per_page})
        if r.status_code != 200:
            return self._fail(r)
        return {'success': True, 'list': r.json()}

    def reply_to_review_comment(self, number: int, comment_id: Any, body: str,
                                path: str, line: Optional[int] = None,
                                commit_id: Optional[str] = None) -> Dict[str, Any]:
        self._require_repo()
        return self.create_review_comment(
            number=number, body=body, path=path, line=line,
            commit_id=commit_id, in_reply_to=comment_id,
        )

    def resolve_review_comment(self, number: int, comment_id: Any) -> Dict[str, Any]:
        self._require_repo()
        return {
            'success': False,
            'message': 'GitHub REST API does not support resolving review comments; use GraphQL or the web UI.',
        }

    def get_user_info(self, username: Optional[str] = None) -> Dict[str, Any]:
        if username:
            r = self._session.get(f'{self._api_base}/users/{username}')
        else:
            r = self._session.get(
                f'{self._api_base}/users/{self._user}' if self._user else f'{self._api_base}/user'
            )
        if r.status_code != 200:
            return self._fail(r)
        return {'success': True, 'user': r.json()}

    def list_user_starred_repos(self, username: Optional[str] = None,
                                page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        if username:
            url = f'{self._api_base}/users/{username}/starred'
        else:
            url = f'{self._api_base}/user/starred' if not self._user else f'{self._api_base}/users/{self._user}/starred'
        r = self._session.get(url, params={'page': page, 'per_page': per_page})
        if r.status_code != 200:
            return self._fail(r)
        return {'success': True, 'list': r.json()}
