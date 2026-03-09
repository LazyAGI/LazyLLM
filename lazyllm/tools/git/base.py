# Copyright (c) 2026 LazyAGI. All rights reserved.
import re
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

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
    def __init__(self, token: str, repo: str, api_base: Optional[str] = None,
                 return_trace: bool = False, **kwargs):
        super().__init__(return_trace=return_trace)
        self._token = token
        self._repo = repo.strip().strip('/')
        self._api_base = (api_base or '').rstrip('/')
        self._kwargs = kwargs

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
                            title: str, body: str = '', **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def update_pull_request(self, number: int, title: Optional[str] = None,
                            body: Optional[str] = None, state: Optional[str] = None,
                            **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def add_pr_labels(self, number: int, labels: List[str]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_pull_request(self, number: int) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_pull_requests(self, state: str = 'open', head: Optional[str] = None,
                           base: Optional[str] = None, **kwargs) -> Dict[str, Any]:
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
                              commit_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def submit_review(self, number: int, event: str, body: str = '',
                      comment_ids: Optional[List[Any]] = None, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def approve_pull_request(self, number: int, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def merge_pull_request(self, number: int, merge_method: Optional[str] = None,
                           commit_title: Optional[str] = None,
                           commit_message: Optional[str] = None, **kwargs) -> Dict[str, Any]:
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
