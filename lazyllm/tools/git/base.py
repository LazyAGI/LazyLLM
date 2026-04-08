# Copyright (c) 2026 LazyAGI. All rights reserved.
import re
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import requests

from lazyllm.module import ModuleBase
from lazyllm.common.registry import LazyLLMRegisterMetaABCClass

# Safe remote name: alphanumeric, underscore, hyphen only. Reject ext:: and other protocols.
_REMOTE_NAME_RE = re.compile(r'^[a-zA-Z0-9_-]+$')


def _validate_remote_name(remote_name: str) -> None:
    if not remote_name or not isinstance(remote_name, str):
        raise ValueError('remote_name must be a non-empty string')
    if '::' in remote_name or not _REMOTE_NAME_RE.match(remote_name):
        raise ValueError(
            'remote_name must be a safe identifier (alphanumeric, underscore, hyphen). '
            'Dangerous protocols like ext:: are not allowed.'
        )


def _sanitize_path(path: str) -> str:
    if '..' in path: raise ValueError('Path must not contain ".."')
    return path


class PrInfo:
    def __init__(self, number: int, title: str, state: str, body: str = '',
                 source_branch: str = '', target_branch: str = '',
                 html_url: str = '', raw: Optional[Dict[str, Any]] = None):
        self.number = number
        self.title = title
        self.state = state
        self.body = body or ''
        self.source_branch = source_branch
        self.target_branch = target_branch
        self.html_url = html_url
        self.raw = raw or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'number': self.number,
            'title': self.title,
            'state': self.state,
            'body': self.body,
            'source_branch': self.source_branch,
            'target_branch': self.target_branch,
            'html_url': self.html_url,
            'raw': self.raw,
        }


class ReviewCommentInfo:
    def __init__(self, id: Any, body: str, path: str = '', line: Optional[int] = None,
                 side: str = 'RIGHT', user: str = '', raw: Optional[Dict[str, Any]] = None):
        self.id = id
        self.body = body
        self.path = path
        self.line = line
        self.side = side
        self.user = user
        self.raw = raw or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'body': self.body,
            'path': self.path,
            'line': self.line,
            'side': self.side,
            'user': self.user,
            'raw': self.raw,
        }


class LazyLLMGitBase(ModuleBase, ABC, metaclass=LazyLLMRegisterMetaABCClass):
    def __init__(self, token: str, repo: Optional[str] = None, api_base: Optional[str] = None,
                 user: Optional[str] = None, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self._token = token
        self._repo = (repo or '').strip().strip('/')
        self._api_base = (api_base or '').rstrip('/')
        self._user = (user or '').strip() or None
        self._session = requests.Session()

    def _parse_owner_repo(self, repo: str) -> Tuple[str, str]:
        parts = repo.split('/', 1)
        if len(parts) != 2:
            raise ValueError(f'repo must be \'owner/repo\', got: {repo!r}')
        return parts[0], parts[1]

    def _require_repo(self) -> None:
        if not self._repo:
            raise ValueError(
                f'repo is not set; pass repo when constructing {self.__class__.__name__} '
                'to use repo-related APIs.'
            )

    def push_branch(self, local_branch: str, remote_branch: Optional[str] = None,
                    remote_name: str = 'origin', repo_path: Optional[str] = None) -> Dict[str, Any]:
        _validate_remote_name(remote_name)
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

    @abstractmethod
    def create_pull_request(self, source_branch: str, target_branch: str,
                            title: str, body: str = '') -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def update_pull_request(self, number: int, title: Optional[str] = None,
                            body: Optional[str] = None, state: Optional[str] = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def add_pr_labels(self, number: int, labels: List[str]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_pull_request(self, number: int) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_pull_requests(self, state: str = 'open', head: Optional[str] = None,
                           base: Optional[str] = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_pr_diff(self, number: int) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_review_comments(self, number: int) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def create_review_comment(self, number: int, body: str, path: str,
                              line: Optional[int] = None, side: str = 'RIGHT',
                              commit_id: Optional[str] = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def submit_review(self, number: int, event: str, body: str = '',
                      comments: Optional[List[Dict[str, Any]]] = None,
                      commit_id: Optional[str] = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def approve_pull_request(self, number: int) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def merge_pull_request(self, number: int, merge_method: Optional[str] = None,
                           commit_title: Optional[str] = None,
                           commit_message: Optional[str] = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_repo_stargazers(self, page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def reply_to_review_comment(self, number: int, comment_id: Any, body: str,
                                path: str, line: Optional[int] = None,
                                commit_id: Optional[str] = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def resolve_review_comment(self, number: int, comment_id: Any) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_user_info(self, username: Optional[str] = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_user_starred_repos(self, username: Optional[str] = None,
                                page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        raise NotImplementedError

    def check_review_resolution(self, number: int, comment_ids: Optional[List[Any]] = None
                                ) -> Dict[str, Any]:
        out = self.list_review_comments(number)
        if not out.get('success'):
            return out
        comments = out.get('comments') or []
        if comment_ids is not None:
            id_set = set(comment_ids)
            comments = [
                c for c in comments
                if (c.get('id') if isinstance(c, dict) else getattr(c, 'id', None)) in id_set
            ]
        return {
            'success': True,
            'resolved': None,
            'comments': [c.to_dict() if hasattr(c, 'to_dict') else c for c in comments],
            'message': (
                'Use list_review_comments for resolution check; '
                'override check_review_resolution for platform-specific logic.'
            ),
        }

    def _stashed_comments(self) -> List[Dict[str, Any]]:
        if not hasattr(self, '_comment_stash'):
            self._comment_stash = []
        return self._comment_stash

    def stash_review_comment(self, number: int, body: str, path: str,
                             line: Optional[int] = None) -> Dict[str, Any]:
        self._require_repo()
        self._stashed_comments().append({
            'number': number,
            'body': body,
            'path': path,
            'line': line,
        })
        return {'success': True, 'message': 'stashed', 'stash_size': len(self._stashed_comments())}

    def batch_commit_review_comments(self, clear_stash: bool = True) -> Dict[str, Any]:
        self._require_repo()
        stash = self._stashed_comments()
        if not stash:
            return {'success': True, 'message': 'no stashed comments', 'created': 0}
        created = 0
        errors = []
        for item in stash:
            r = self.create_review_comment(
                number=item['number'],
                body=item['body'],
                path=item['path'],
                line=item.get('line'),
            )
            if r.get('success'):
                created += 1
            else:
                errors.append(r.get('message', 'unknown'))
        if clear_stash:
            stash.clear()
        if errors:
            return {'success': False, 'message': '; '.join(errors), 'created': created}
        return {'success': True, 'message': 'committed', 'created': created}

    def submit_review_with_comments(
        self,
        number: int,
        body: str,
        comments: List[Dict[str, Any]],
        commit_id: Optional[str] = None,
        event: str = 'COMMENT',
    ) -> Dict[str, Any]:
        return self.submit_review(number=number, event=event, body=body,
                                  comments=comments, commit_id=commit_id)
