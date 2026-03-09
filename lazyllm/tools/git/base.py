# Copyright (c) 2026 LazyAGI. All rights reserved.
'''
Git backend base: cross-platform Git operations (push, PR, review, merge) for agents.
Implementations are registered via registry (GitHub, GitLab, Gitee, GitCode).
'''
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from lazyllm.module import ModuleBase
from lazyllm.common.registry import LazyLLMRegisterMetaABCClass


class PrInfo:
    '''Pull Request / Merge Request summary.'''
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
    '''Single review comment (optionally line-level).'''
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
    '''
    Unified Git platform base; implementations (GitHub, GitLab, Gitee, GitCode) are
    registered via registry. Subclasses implement auth, API base URL, and abstract methods.
    Agents get instances via lazyllm.git.github / lazyllm.git.gitlab etc.
    '''

    def __init__(self, token: str, repo: str, api_base: Optional[str] = None,
                 return_trace: bool = False, **kwargs):
        super().__init__(return_trace=return_trace)
        self._token = token
        self._repo = repo.strip().strip('/')
        self._api_base = (api_base or '').rstrip('/')
        self._kwargs = kwargs

    @abstractmethod
    def push_branch(self, local_branch: str, remote_branch: Optional[str] = None,
                    remote_name: str = 'origin', repo_path: Optional[str] = None) -> Dict[str, Any]:
        '''Push local branch to remote. Returns dict with success, message.'''
        raise NotImplementedError

    @abstractmethod
    def create_pull_request(self, source_branch: str, target_branch: str,
                            title: str, body: str = '', **kwargs) -> Dict[str, Any]:
        '''Create a Pull Request / Merge Request. Returns success, number, html_url, message.'''
        raise NotImplementedError

    @abstractmethod
    def update_pull_request(self, number: int, title: Optional[str] = None,
                            body: Optional[str] = None, state: Optional[str] = None,
                            **kwargs) -> Dict[str, Any]:
        '''Update PR/MR title, body or state. Returns success, message.'''
        raise NotImplementedError

    @abstractmethod
    def add_pr_labels(self, number: int, labels: List[str]) -> Dict[str, Any]:
        '''Add labels to PR/MR. Returns success, message.'''
        raise NotImplementedError

    @abstractmethod
    def get_pull_request(self, number: int) -> Dict[str, Any]:
        '''Get single PR/MR. Returns success, pr (PrInfo or dict), message.'''
        raise NotImplementedError

    @abstractmethod
    def list_pull_requests(self, state: str = 'open', head: Optional[str] = None,
                           base: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        '''List PRs/MRs. Returns success, list, message.'''
        raise NotImplementedError

    @abstractmethod
    def get_pr_diff(self, number: int) -> Dict[str, Any]:
        '''Get PR/MR diff text. Returns success, diff, message.'''
        raise NotImplementedError

    @abstractmethod
    def list_review_comments(self, number: int) -> Dict[str, Any]:
        '''List all review comments on PR/MR. Returns success, comments, message.'''
        raise NotImplementedError

    @abstractmethod
    def create_review_comment(self, number: int, body: str, path: str,
                              line: Optional[int] = None, side: str = 'RIGHT',
                              commit_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        '''Create a single review comment. Returns success, comment_id, message.'''
        raise NotImplementedError

    @abstractmethod
    def submit_review(self, number: int, event: str, body: str = '',
                      comment_ids: Optional[List[Any]] = None, **kwargs) -> Dict[str, Any]:
        '''Submit review (APPROVE / REQUEST_CHANGES / COMMENT). Returns success, message.'''
        raise NotImplementedError

    @abstractmethod
    def approve_pull_request(self, number: int, **kwargs) -> Dict[str, Any]:
        '''Approve PR/MR. Returns success, message.'''
        raise NotImplementedError

    @abstractmethod
    def merge_pull_request(self, number: int, merge_method: Optional[str] = None,
                           commit_title: Optional[str] = None,
                           commit_message: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        '''Merge PR/MR. Returns success, sha, message.'''
        raise NotImplementedError

    def check_review_resolution(self, number: int, comment_ids: Optional[List[Any]] = None
                                ) -> Dict[str, Any]:
        '''Check if review comments are resolved. Default: list comments; override for platform logic.'''
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
