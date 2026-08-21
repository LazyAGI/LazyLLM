# Copyright (c) 2026 LazyAGI. All rights reserved.
import re
import subprocess
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import requests

from lazyllm.module import ModuleBase
from lazyllm.common.registry import LazyLLMRegisterMetaABCClass
from lazyllm.tools.agent.toolError import ToolExecutionError

# Safe remote name: alphanumeric, underscore, hyphen only. Reject ext:: and other protocols.
_REMOTE_NAME_RE = re.compile(r'^[a-zA-Z0-9_-]+$')


def _raise_git_failure(operation: str, result: Any) -> Any:
    if not isinstance(result, dict) or result.get('success') is not False:
        return result

    message = str(result.get('message') or f'Git operation {operation} failed.')
    status_code = result.get('status_code')
    context = f'Git operation {operation} failed'
    if status_code is not None:
        context += f' with HTTP {status_code}'
    message = f'{context}: {message}'
    raise ToolExecutionError(message)


def _git_api(method):
    if getattr(method, '__git_failure_boundary__', False):
        return method

    @wraps(method)
    def wrapped(*args, **kwargs):
        operation = method.__name__
        try:
            return _raise_git_failure(operation, method(*args, **kwargs))
        except ToolExecutionError:
            raise
        except Exception as error:
            response = getattr(error, 'response', None)
            status_code = getattr(response, 'status_code', None)
            context = f'Git operation {operation} failed'
            if status_code is not None:
                context += f' with HTTP {status_code}'
            message = f'{context}: {error}'
            raise ToolExecutionError(message) from error

    wrapped.__git_failure_boundary__ = True
    return wrapped


def _validate_remote_name(remote_name: str) -> None:
    if not remote_name or not isinstance(remote_name, str):
        raise ToolExecutionError(f'remote_name must be a non-empty string, got {remote_name!r}.')
    if '::' in remote_name or not _REMOTE_NAME_RE.match(remote_name):
        raise ToolExecutionError(
            'remote_name must be a safe identifier (alphanumeric, underscore, hyphen). '
            f'Dangerous protocols like ext:: are not allowed; got {remote_name!r}.',
        )


def _sanitize_path(path: str) -> str:
    if '..' in path:
        raise ToolExecutionError(f'Path must not contain ".."; got {path!r}.')
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
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name, member in list(cls.__dict__.items()):
            if not name.startswith('_') and callable(member):
                setattr(cls, name, _git_api(member))

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
            raise ToolExecutionError(f'repo must be \'owner/repo\', got: {repo!r}')
        return parts[0], parts[1]

    def _require_repo(self) -> None:
        if not self._repo:
            raise ToolExecutionError(
                f'repo is not set; pass repo when constructing {self.__class__.__name__} '
                'to use repo-related APIs.'
            )

    @staticmethod
    def _http_failure(response, message: Optional[str] = None, **details) -> Dict[str, Any]:
        failure_message = message or response.text or response.reason
        failure = {
            'success': False,
            'message': failure_message,
            'status_code': response.status_code,
            **details,
        }
        return failure

    @staticmethod
    def _raise_http_error(response, message: Optional[str] = None) -> None:
        raise requests.HTTPError(
            message or response.text or response.reason,
            response=response,
        )

    @_git_api
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
                           base: Optional[str] = None, max_results: int = 100) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_pr_diff(self, number: int) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_review_comments(self, number: int) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_issue_comments(self, number: int) -> Dict[str, Any]:
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

    @_git_api
    def check_review_resolution(self, number: int, comment_ids: Optional[List[Any]] = None
                                ) -> Dict[str, Any]:
        out = self.list_review_comments(number)
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

    @_git_api
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

    @_git_api
    def batch_commit_review_comments(self, clear_stash: bool = True) -> Dict[str, Any]:
        self._require_repo()
        stash = self._stashed_comments()
        if not stash:
            return {'success': True, 'message': 'no stashed comments', 'created': 0}
        created = 0
        errors: List[ToolExecutionError] = []
        for item in stash:
            try:
                self.create_review_comment(
                    number=item['number'],
                    body=item['body'],
                    path=item['path'],
                    line=item.get('line'),
                )
                created += 1
            except ToolExecutionError as error:
                errors.append(error)
        if clear_stash:
            stash.clear()
        if errors:
            first_error = errors[0]
            message = (
                f'Failed to submit review comments: {created} created and {len(errors)} failed. '
                + '; '.join(str(error) for error in errors)
            )
            raise type(first_error)(message) from first_error
        return {'success': True, 'message': 'committed', 'created': created}

    @_git_api
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
