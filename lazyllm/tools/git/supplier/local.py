# Copyright (c) 2026 LazyAGI. All rights reserved.
import os
import re
import subprocess
from typing import Any, Dict, List, Optional

from ..base import LazyLLMGitBase, PrInfo


def _git(args: List[str], cwd: str, timeout: int = 30) -> str:
    try:
        out = subprocess.run(['git'] + args, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired as e:
        if e.process is not None:
            e.process.kill()
            e.process.communicate()
        raise RuntimeError(f'git {" ".join(args)} timed out after {timeout}s')
    if out.returncode != 0:
        raise RuntimeError(f'git {" ".join(args)} failed: {out.stderr.strip() or out.stdout.strip() or "(no output)"}')
    return out.stdout.strip()


class LocalGit(LazyLLMGitBase):
    def __init__(self, repo_path: str = '.', base: str = 'main',
                 include_uncommitted: bool = True, return_trace: bool = False):
        super().__init__(token='', repo='', api_base='', return_trace=return_trace)
        self._repo_path = os.path.abspath(repo_path)
        self._base = base
        self._include_uncommitted = include_uncommitted

    @property
    def local_repo_path(self) -> str:
        return self._repo_path

    def _merge_base(self) -> str:
        return _git(['merge-base', self._base, 'HEAD'], cwd=self._repo_path)

    def _current_branch(self) -> str:
        try:
            return _git(['rev-parse', '--abbrev-ref', 'HEAD'], cwd=self._repo_path)
        except RuntimeError:
            return 'HEAD'

    def get_origin_repo(self) -> str:
        # parse "owner/repo" from git remote -v origin URL
        # supports https://github.com/owner/repo.git and git@github.com:owner/repo.git
        try:
            out = _git(['remote', '-v'], cwd=self._repo_path)
        except RuntimeError:
            return os.path.basename(self._repo_path)
        for line in out.splitlines():
            if not line.startswith('origin'):
                continue
            url = line.split()[1] if len(line.split()) >= 2 else ''
            url = re.sub(r'\.git$', '', url)
            m = re.match(r'https?://[^/]+/(.+)', url)
            if m:
                return m.group(1)
            m = re.match(r'git@[^:]+:(.+)', url)
            if m:
                return m.group(1)
        return os.path.basename(self._repo_path)

    def get_pull_request(self, number: int) -> Dict[str, Any]:
        branch = self._current_branch()
        return {
            'success': True,
            'pr': PrInfo(
                number=0,
                title=f'Local review: {branch} -> {self._base}',
                state='open',
                body='',
                source_branch=branch,
                target_branch=self._base,
                html_url='',
            ),
        }

    def get_pr_diff(self, number: int) -> Dict[str, Any]:
        try:
            merge_base = self._merge_base()
            if self._include_uncommitted:
                diff = _git(['diff', merge_base], cwd=self._repo_path, timeout=60)
            else:
                diff = _git(['diff', merge_base, 'HEAD'], cwd=self._repo_path, timeout=60)
            return {'success': True, 'diff': diff + '\n' if diff and not diff.endswith('\n') else diff}
        except RuntimeError as e:
            return {'success': False, 'message': str(e)}
        except subprocess.TimeoutExpired:
            return {'success': False, 'message': 'git diff timed out'}

    def list_review_comments(self, number: int) -> Dict[str, Any]:
        return {'success': True, 'comments': []}

    def list_issue_comments(self, number: int) -> Dict[str, Any]:
        return {'success': True, 'comments': []}

    def _not_supported(self, op: str) -> Dict[str, Any]:
        return {'success': False, 'message': f'{op} is not supported in local mode'}

    def create_pull_request(self, source_branch: str, target_branch: str,
                            title: str, body: str = '') -> Dict[str, Any]:
        return self._not_supported('create_pull_request')

    def update_pull_request(self, number: int, title: Optional[str] = None,
                            body: Optional[str] = None, state: Optional[str] = None) -> Dict[str, Any]:
        return self._not_supported('update_pull_request')

    def add_pr_labels(self, number: int, labels: List[str]) -> Dict[str, Any]:
        return self._not_supported('add_pr_labels')

    def list_pull_requests(self, state: str = 'open', head: Optional[str] = None,
                           base: Optional[str] = None, max_results: int = 100) -> Dict[str, Any]:
        return self._not_supported('list_pull_requests')

    def create_review_comment(self, number: int, body: str, path: str,
                              line: Optional[int] = None, side: str = 'RIGHT',
                              commit_id: Optional[str] = None,
                              in_reply_to: Optional[Any] = None,
                              start_line: Optional[int] = None,
                              start_side: Optional[str] = None) -> Dict[str, Any]:
        return self._not_supported('create_review_comment')

    def add_issue_comment(self, number: int, body: str) -> Dict[str, Any]:
        return self._not_supported('add_issue_comment')

    def submit_review(self, number: int, event: str, body: str = '',
                      comments: Optional[List[Dict[str, Any]]] = None,
                      commit_id: Optional[str] = None) -> Dict[str, Any]:
        return self._not_supported('submit_review')

    def approve_pull_request(self, number: int) -> Dict[str, Any]:
        return self._not_supported('approve_pull_request')

    def merge_pull_request(self, number: int, merge_method: Optional[str] = None,
                           commit_title: Optional[str] = None,
                           commit_message: Optional[str] = None) -> Dict[str, Any]:
        return self._not_supported('merge_pull_request')

    def list_repo_stargazers(self, page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        return self._not_supported('list_repo_stargazers')

    def reply_to_review_comment(self, number: int, comment_id: Any, body: str,
                                path: str, line: Optional[int] = None,
                                commit_id: Optional[str] = None) -> Dict[str, Any]:
        return self._not_supported('reply_to_review_comment')

    def resolve_review_comment(self, number: int, comment_id: Any) -> Dict[str, Any]:
        return self._not_supported('resolve_review_comment')

    def get_user_info(self, username: Optional[str] = None) -> Dict[str, Any]:
        return self._not_supported('get_user_info')

    def list_user_starred_repos(self, username: Optional[str] = None,
                                page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        return self._not_supported('list_user_starred_repos')
