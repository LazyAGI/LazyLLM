# Copyright (c) 2026 LazyAGI. All rights reserved.
from __future__ import annotations

import base64
import hashlib
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

import requests

from lazyllm import globals
from lazyllm.common import KeyAuthError

from ..base import CloudFSBufferedFile, LazyLLMFSBase, clean_document_ref

_GITHUB_HOSTS = {'github.com', 'www.github.com'}
_MARKDOWN_SUFFIXES = {'.md', '.markdown'}
_SAFE_OPERATION_RE = re.compile(r'[^A-Za-z0-9._-]+')


class GitHubFSError(RuntimeError):
    '''Stable, provider-safe GitHub filesystem failure.'''

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 0,
        retryable: bool = False,
    ) -> None:
        super().__init__(code, message, status_code, retryable)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable

    def __str__(self) -> str:
        return self.args[1]


def _validate_repo_part(value: str, label: str) -> str:
    value = str(value or '').strip()
    if not value or value in {'.', '..'} or '/' in value or '\\' in value:
        raise ValueError(f'Invalid GitHub {label}: {value!r}.')
    return value


def _validate_repo_path(value: str, *, markdown_only: bool = False) -> str:
    value = unquote(str(value or '')).strip()
    if value.startswith(('/', '\\')) or '//' in value:
        raise ValueError(f'Invalid GitHub repository path: {value!r}.')
    path = PurePosixPath(value)
    if not value or value.endswith('/') or any(part in {'', '.', '..'} for part in path.parts):
        raise ValueError(f'Invalid GitHub repository path: {value!r}.')
    if '\\' in value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f'Invalid GitHub repository path: {value!r}.')
    if markdown_only and path.suffix.lower() not in _MARKDOWN_SUFFIXES:
        raise ValueError('GitHub Writer targets must be .md or .markdown files.')
    return path.as_posix()


def _validate_wiki_path(value: str) -> str:
    value = unquote(str(value or '')).strip().strip('/')
    if not value:
        raise ValueError('GitHub Wiki page is required.')
    if not value.lower().endswith(tuple(_MARKDOWN_SUFFIXES)):
        value += '.md'
    return _validate_repo_path(value, markdown_only=True)


def _repo_uri(owner: str, repo: str, path: str, ref: str) -> str:
    return f'githubrepo:/{owner}/{repo}/{quote(path, safe="/")}?ref={quote(ref, safe="")}'


def _wiki_uri(owner: str, repo: str, path: str) -> str:
    return f'githubwiki:/{owner}/{repo}/{quote(path, safe="/")}'


def _repo_browser_url(owner: str, repo: str, ref: str, path: str) -> str:
    return (
        f'https://github.com/{quote(owner, safe="")}/{quote(repo, safe="")}/blob/'
        f'{quote(ref, safe="/")}/{quote(path, safe="/")}'
    )


def _repo_parent_browser_url(owner: str, repo: str, ref: str = '', directory: str = '') -> str:
    base = f'https://github.com/{quote(owner, safe="")}/{quote(repo, safe="")}'
    if not ref:
        return base
    suffix = f'/{quote(directory, safe="/")}' if directory else ''
    return f'{base}/tree/{quote(ref, safe="/")}{suffix}'


def _wiki_parent_browser_url(owner: str, repo: str) -> str:
    return f'https://github.com/{quote(owner, safe="")}/{quote(repo, safe="")}/wiki'


def _markdown_filename(title: str) -> str:
    name = re.sub(r'[\x00-\x1f\x7f/\\:*?"<>|]+', '-', str(title or '').strip())
    name = re.sub(r'\s+', '-', name)
    name = re.sub(r'-{2,}', '-', name).strip(' .-_')
    if not name:
        name = 'untitled'
    suffix = PurePosixPath(name).suffix.lower()
    if suffix not in _MARKDOWN_SUFFIXES:
        name = f'{name[:120].rstrip(" .-_")}.md'
    return _validate_repo_path(name, markdown_only=True)


def _wiki_browser_url(owner: str, repo: str, path: str) -> str:
    page = path.rsplit('.', 1)[0]
    return (
        f'https://github.com/{quote(owner, safe="")}/{quote(repo, safe="")}/wiki/'
        f'{quote(page, safe="/")}'
    )


def _operation_id_for_files(
    owner: str,
    repo: str,
    target_path: str,
    target_ref: str,
    expected_revision: str,
    publish_mode: str,
    files: Mapping[str, bytes],
) -> str:
    digest = hashlib.sha256()
    for value in (
        owner,
        repo,
        target_path,
        target_ref,
        expected_revision,
        publish_mode,
    ):
        digest.update(value.encode('utf-8'))
        digest.update(b'\0')
    for file_path in sorted(files):
        digest.update(file_path.encode('utf-8'))
        digest.update(b'\0')
        digest.update(hashlib.sha256(files[file_path]).digest())
    return digest.hexdigest()[:20]


class _MemoryGitHubFile(CloudFSBufferedFile):
    def __init__(self, fs: LazyLLMFSBase, path: str, data: bytes, **kwargs: Any) -> None:
        self._github_data = data
        super().__init__(fs, path, size=len(data), **kwargs)

    def _fetch_range(self, start: int, end: int) -> bytes:
        return self._github_data[start:end]


class _GitHubFSBase(LazyLLMFSBase):
    __lazyllm_registry_disable__ = True
    credential_key = 'github'

    def __init__(
        self,
        token: str | None = None,
        base_url: str | None = None,
        dynamic_auth: bool = False,
        **storage_options: Any,
    ) -> None:
        if dynamic_auth:
            if token:
                raise ValueError('token must be None when dynamic_auth=True')
            token = ''
        super().__init__(
            token=token or '',
            base_url=base_url or 'https://api.github.com',
            dynamic_auth=dynamic_auth,
            **storage_options,
        )

    def _setup_auth(self) -> None:
        self._session.headers.update({
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28',
            'User-Agent': 'LazyLLM-GitHubFS',
        })

    def _resolve_dynamic_token(self):
        mapping = globals.config['dynamic_fs_auth'] or {}
        return mapping.get(self.credential_key, '')

    def _missing_dynamic_token_error(self) -> str:
        return (
            'dynamic_fs_auth["github"] is not set in globals.config; '
            'configure a GitHub connection before accessing GitHub documents'
        )

    def _is_key_auth_error(self, resp: Any) -> bool:
        # A GitHub 403 commonly means repository permission or rate limiting,
        # not an invalid token. Only rotate/fail credentials on 401.
        return getattr(resp, 'status_code', 0) == 401

    @staticmethod
    def _response_detail(response: requests.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            body = {}
        if isinstance(body, dict):
            detail = str(body.get('message') or '').strip()
            if detail:
                return detail[:300]
        return str(response.reason or 'GitHub request failed')[:300]

    def _api(
        self,
        method: str,
        path: str,
        *,
        expected: tuple[int, ...] = (200,),
        not_found_code: str = 'GITHUB_TARGET_NOT_FOUND',
        **kwargs: Any,
    ) -> Any:
        url = path if path.startswith(('http://', 'https://')) else f'{self._base_url}{path}'
        try:
            response = self._request(method, url, timeout=30, **kwargs)
        except KeyAuthError as exc:
            raise GitHubFSError(
                'GITHUB_AUTH_REQUIRED',
                'GitHub authorization is missing, expired, or revoked.',
                status_code=401,
            ) from exc
        except requests.HTTPError as exc:
            response = exc.response
            status = int(getattr(response, 'status_code', 0) or 0)
            detail = self._response_detail(response) if response is not None else 'GitHub request failed'
            if status == 403 and response is not None \
                    and response.headers.get('X-RateLimit-Remaining') == '0':
                raise GitHubFSError(
                    'GITHUB_RATE_LIMITED',
                    'GitHub API rate limit exceeded.',
                    status_code=status,
                    retryable=True,
                ) from exc
            if status == 403:
                lowered_detail = detail.lower()
                if (
                    'resource not accessible' in lowered_detail
                    or 'insufficient scope' in lowered_detail
                    or 'oauth app access' in lowered_detail
                ):
                    raise GitHubFSError(
                        'GITHUB_SCOPE_INSUFFICIENT',
                        'The GitHub authorization does not grant the required repository scope.',
                        status_code=status,
                    ) from exc
                raise GitHubFSError(
                    'GITHUB_PERMISSION_DENIED',
                    'The current GitHub account does not have permission for this operation.',
                    status_code=status,
                ) from exc
            if status == 404:
                if not_found_code == 'GITHUB_PERMISSION_DENIED':
                    detail = (
                        'The current GitHub account does not have permission to access '
                        'this repository.'
                    )
                    status = 403
                raise GitHubFSError(not_found_code, detail, status_code=status) from exc
            if status == 409:
                raise GitHubFSError(
                    'GITHUB_REVISION_CONFLICT', detail, status_code=status,
                ) from exc
            raise GitHubFSError(
                'GITHUB_WRITE_FAILED' if method.upper() != 'GET' else 'GITHUB_READ_FAILED',
                detail,
                status_code=status,
                retryable=status == 429 or status >= 500,
            ) from exc
        if response.status_code not in expected:
            raise GitHubFSError(
                'GITHUB_WRITE_FAILED' if method.upper() != 'GET' else 'GITHUB_READ_FAILED',
                self._response_detail(response),
                status_code=response.status_code,
                retryable=response.status_code == 429 or response.status_code >= 500,
            )
        if response.status_code == 204 or not response.content:
            return {}
        return response.json()

    def _open(
        self,
        path: str,
        mode: str = 'rb',
        block_size: int | None = None,
        autocommit: bool = True,
        cache_options: dict | None = None,
        **kwargs: Any,
    ) -> CloudFSBufferedFile:
        if 'b' not in mode:
            raise ValueError(f'{type(self).__name__} only supports binary mode')
        if 'r' in mode:
            data = self.read_bytes(path)
            return _MemoryGitHubFile(
                self, path, data,
                mode=mode,
                block_size=block_size or self.blocksize,
                autocommit=autocommit,
                cache_options=cache_options,
            )
        return CloudFSBufferedFile(
            self, path,
            mode=mode,
            block_size=block_size or self.blocksize,
            autocommit=autocommit,
            cache_options=cache_options,
        )

    def _download_range(self, path: str, start: int, end: int) -> bytes:
        return self.read_bytes(path)[start:end]


class GitHubRepoFS(_GitHubFSBase):
    '''Read and atomically publish files in GitHub repositories.'''

    __public_apis__ = LazyLLMFSBase.__public_apis__ + [
        'resolve_target', 'resolve_create_parent', 'resolve_create_target',
        'apply_document_patch', 'create_document',
    ]

    @staticmethod
    def _repo_api(owner: str, repo: str, suffix: str = '') -> str:
        base = f'/repos/{quote(owner, safe="")}/{quote(repo, safe="")}'
        return base + suffix

    def _repository(self, owner: str, repo: str) -> dict[str, Any]:
        return self._api(
            'GET', self._repo_api(owner, repo),
            # GitHub deliberately returns 404 for private repositories that the
            # current account cannot see. Treat that boundary as permission so
            # callers do not misreport an authorization failure as a bad link.
            not_found_code='GITHUB_PERMISSION_DENIED',
        )

    def _branch_head(
        self,
        owner: str,
        repo: str,
        ref: str,
        *,
        missing_ok: bool = False,
    ) -> str | None:
        try:
            data = self._api(
                'GET', self._repo_api(owner, repo, f'/git/ref/heads/{quote(ref, safe="")}'),
                not_found_code='GITHUB_BRANCH_NOT_FOUND',
            )
        except GitHubFSError as exc:
            if missing_ok and exc.status_code == 404:
                return None
            raise
        sha = str(((data.get('object') or {}).get('sha')) or '').strip()
        if not sha:
            raise GitHubFSError('GITHUB_READ_FAILED', 'GitHub returned an empty branch HEAD.')
        return sha

    def _resolve_browser_ref(
        self,
        owner: str,
        repo: str,
        tail: list[str],
        *,
        markdown_only: bool,
    ) -> tuple[str, str]:
        self._repository(owner, repo)
        for split_at in range(len(tail) - 1, 0, -1):
            ref = unquote('/'.join(tail[:split_at]))
            path = unquote('/'.join(tail[split_at:]))
            if not path or (
                markdown_only and PurePosixPath(path).suffix.lower() not in _MARKDOWN_SUFFIXES
            ):
                continue
            if self._branch_head(owner, repo, ref, missing_ok=True):
                return ref, _validate_repo_path(path, markdown_only=markdown_only)
        raise GitHubFSError(
            'GITHUB_TARGET_INVALID',
            'The GitHub URL does not identify an existing Markdown file and branch.',
            status_code=400,
        )

    @classmethod
    def matches_create_parent(cls, value: str) -> bool:
        parsed = urlparse(clean_document_ref(str(value or '').strip()))
        if parsed.scheme not in {'http', 'https'} or (parsed.hostname or '').lower() not in _GITHUB_HOSTS:
            return False
        parts = [part for part in parsed.path.split('/') if part]
        return len(parts) == 2 or (
            len(parts) >= 4 and parts[2].lower() == 'tree'
        )

    def _parse_create_parent(self, value: str) -> tuple[str, str, str, str, dict[str, Any]]:
        parsed = urlparse(clean_document_ref(str(value or '').strip()))
        if parsed.scheme not in {'http', 'https'} or (parsed.hostname or '').lower() not in _GITHUB_HOSTS:
            raise ValueError('GitHub create parent must be a github.com repository or tree URL.')
        parts = [unquote(part) for part in parsed.path.split('/') if part]
        if len(parts) < 2:
            raise ValueError('GitHub create parent must include owner and repository.')
        owner = _validate_repo_part(parts[0], 'owner')
        repo_value = parts[1][:-4] if parts[1].lower().endswith('.git') else parts[1]
        repo = _validate_repo_part(repo_value, 'repository')
        repository = self._repository(owner, repo)
        if len(parts) == 2:
            ref = str(repository.get('default_branch') or '').strip()
            if not ref:
                raise GitHubFSError(
                    'GITHUB_BRANCH_NOT_FOUND', 'GitHub repository has no default branch.',
                )
            return owner, repo, ref, '', repository
        if len(parts) < 4 or parts[2].lower() != 'tree':
            raise ValueError('GitHub create parent must be a repository root or /tree/{ref}/{directory} URL.')
        tail = parts[3:]
        for split_at in range(len(tail), 0, -1):
            ref = '/'.join(tail[:split_at])
            directory = '/'.join(tail[split_at:])
            if self._branch_head(owner, repo, ref, missing_ok=True):
                if directory:
                    directory = _validate_repo_path(directory)
                return owner, repo, ref, directory, repository
        raise GitHubFSError(
            'GITHUB_BRANCH_NOT_FOUND',
            'The GitHub tree URL does not identify an existing branch.',
            status_code=404,
        )

    def resolve_create_parent(self, parent: str) -> dict[str, Any]:
        owner, repo, ref, directory, repository = self._parse_create_parent(parent)
        permissions = repository.get('permissions')
        if isinstance(permissions, Mapping) and permissions.get('push') is False:
            raise GitHubFSError(
                'GITHUB_PERMISSION_DENIED',
                'The current GitHub account cannot create files in this repository.',
                status_code=403,
            )
        revision = self._branch_head(owner, repo, ref)
        if directory:
            content = self._content(owner, repo, directory, ref)
            if not isinstance(content, list) and content.get('type') != 'dir':
                raise GitHubFSError(
                    'GITHUB_TARGET_INVALID',
                    'The GitHub tree URL does not identify a repository directory.',
                    status_code=400,
                )
        return {
            'uri': _repo_parent_browser_url(owner, repo, ref, directory),
            'browser_url': _repo_parent_browser_url(owner, repo, ref, directory),
            'owner': owner,
            'repo': repo,
            'ref': ref,
            'base_ref': ref,
            'directory': directory,
            'revision': revision,
            'target_type': 'repository',
            'fs_scheme': 'githubrepo',
            'publish_mode': 'pull_request',
            'create_pending': True,
            'parent_uri': str(parent or '').strip(),
        }

    def resolve_create_target(
        self,
        parent: Mapping[str, Any],
        title: str,
    ) -> dict[str, Any]:
        owner = _validate_repo_part(str(parent.get('owner') or ''), 'owner')
        repo = _validate_repo_part(str(parent.get('repo') or ''), 'repository')
        ref = str(parent.get('base_ref') or parent.get('ref') or '').strip()
        revision = str(parent.get('revision') or '').strip()
        if not ref or not revision:
            raise GitHubFSError(
                'GITHUB_TARGET_INVALID',
                'Pending GitHub target is missing its base branch or revision.',
                status_code=400,
            )
        directory = str(parent.get('directory') or '').strip('/')
        if directory:
            directory = _validate_repo_path(directory)
        filename = _markdown_filename(title)
        repo_path = f'{directory}/{filename}' if directory else filename
        try:
            self._content(owner, repo, repo_path, ref)
        except GitHubFSError as exc:
            if exc.code != 'GITHUB_TARGET_NOT_FOUND':
                raise
        else:
            raise GitHubFSError(
                'GITHUB_TARGET_EXISTS',
                f'GitHub target file already exists: {repo_path}.',
                status_code=409,
            )
        return {
            'doc_id': f'{owner}/{repo}:{ref}:{repo_path}',
            'uri': _repo_uri(owner, repo, repo_path, ref),
            'browser_url': _repo_browser_url(owner, repo, ref, repo_path),
            'title': PurePosixPath(repo_path).stem,
            'owner': owner,
            'repo': repo,
            'ref': ref,
            'base_ref': ref,
            'path': repo_path,
            'directory': directory,
            'revision': revision,
            'target_type': 'repository',
            'fs_scheme': 'githubrepo',
            'publish_mode': str(parent.get('publish_mode') or 'pull_request'),
            'create_pending': False,
            'parent_uri': str(parent.get('parent_uri') or parent.get('uri') or ''),
        }

    def _parse_target(
        self,
        path: str,
        *,
        markdown_only: bool = False,
    ) -> tuple[str, str, str, str]:
        value = clean_document_ref(path)
        parsed = urlparse(value)
        if parsed.scheme in ('http', 'https'):
            if (parsed.hostname or '').lower() not in _GITHUB_HOSTS:
                raise ValueError(f'Unsupported GitHub host: {parsed.hostname!r}.')
            parts = [part for part in parsed.path.split('/') if part]
            if len(parts) < 5 or parts[2].lower() != 'blob':
                raise ValueError('GitHub repository document URL must use /blob/{ref}/{path}.')
            owner = _validate_repo_part(unquote(parts[0]), 'owner')
            repo = _validate_repo_part(unquote(parts[1]), 'repository')
            ref, repo_path = self._resolve_browser_ref(
                owner, repo, parts[3:], markdown_only=markdown_only,
            )
            return owner, repo, ref, repo_path

        if value.startswith('githubrepo:'):
            parsed = urlparse(value)
            raw_path = parsed.path
            query = parse_qs(parsed.query)
        else:
            parsed = urlparse(f'githubrepo:{value if value.startswith("/") else "/" + value}')
            raw_path = parsed.path
            query = parse_qs(parsed.query)
        parts = [unquote(part) for part in raw_path.split('/') if part]
        if len(parts) < 3:
            raise ValueError('githubrepo URI must contain owner, repository, and file path.')
        owner = _validate_repo_part(parts[0], 'owner')
        repo = _validate_repo_part(parts[1], 'repository')
        repo_path = _validate_repo_path('/'.join(parts[2:]), markdown_only=markdown_only)
        ref = str((query.get('ref') or [''])[0]).strip()
        if not ref:
            ref = str(self._repository(owner, repo).get('default_branch') or '').strip()
        if not ref:
            raise GitHubFSError('GITHUB_BRANCH_NOT_FOUND', 'GitHub repository has no default branch.')
        return owner, repo, ref, repo_path

    def resolve_target(self, path: str) -> dict[str, Any]:
        owner, repo, ref, repo_path = self._parse_target(path, markdown_only=True)
        self._repository(owner, repo)
        revision = self._branch_head(owner, repo, ref)
        content = self._content(owner, repo, repo_path, ref)
        return {
            'doc_id': f'{owner}/{repo}:{ref}:{repo_path}',
            'uri': _repo_uri(owner, repo, repo_path, ref),
            'browser_url': _repo_browser_url(owner, repo, ref, repo_path),
            'title': PurePosixPath(repo_path).stem,
            'owner': owner,
            'repo': repo,
            'ref': ref,
            'path': repo_path,
            'revision': revision,
            'blob_sha': str(content.get('sha') or ''),
            'size': int(content.get('size') or 0),
            'target_type': 'repository',
            'fs_scheme': 'githubrepo',
        }

    def _content(self, owner: str, repo: str, path: str, ref: str) -> dict[str, Any]:
        return self._api(
            'GET', self._repo_api(owner, repo, f'/contents/{quote(path, safe="/")}'),
            params={'ref': ref},
            not_found_code='GITHUB_TARGET_NOT_FOUND',
        )

    def _content_sha(self, owner: str, repo: str, path: str, ref: str) -> str | None:
        try:
            content = self._content(owner, repo, path, ref)
        except GitHubFSError as exc:
            if exc.code == 'GITHUB_TARGET_NOT_FOUND':
                return None
            raise
        return str(content.get('sha') or '')

    def _paths_unchanged(
        self,
        owner: str,
        repo: str,
        paths: list[str],
        old_revision: str,
        new_revision: str,
    ) -> bool:
        return all(
            self._content_sha(owner, repo, path, old_revision)
            == self._content_sha(owner, repo, path, new_revision)
            for path in paths
        )

    def read_bytes(self, path: str) -> bytes:
        owner, repo, ref, repo_path = self._parse_target(path)
        data = self._content(owner, repo, repo_path, ref)
        if data.get('type') != 'file':
            raise GitHubFSError('GITHUB_TARGET_INVALID', 'GitHub target is not a file.')
        encoded = str(data.get('content') or '').replace('\n', '')
        if encoded:
            try:
                return base64.b64decode(encoded, validate=True)
            except ValueError as exc:
                raise GitHubFSError('GITHUB_READ_FAILED', 'GitHub returned invalid file content.') from exc
        download_url = str(data.get('download_url') or '').strip()
        if not download_url:
            return b''
        try:
            response = self._request('GET', download_url, timeout=30)
        except Exception as exc:
            raise GitHubFSError('GITHUB_READ_FAILED', 'Failed to download GitHub file.') from exc
        return response.content

    def info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        target = self.resolve_target(path)
        return self._entry(
            target['uri'],
            size=target['size'],
            ftype='file',
            sha=target['blob_sha'],
            revision=target['revision'],
            browser_url=target['browser_url'],
        )

    def ls(self, path: str, detail: bool = True, **kwargs: Any) -> list:
        value = clean_document_ref(path)
        parsed = urlparse(value if value.startswith('githubrepo:') else f'githubrepo:{value}')
        parts = [unquote(part) for part in parsed.path.split('/') if part]
        if len(parts) < 2:
            raise ValueError('githubrepo directory URI must contain owner and repository.')
        owner = _validate_repo_part(parts[0], 'owner')
        repo = _validate_repo_part(parts[1], 'repository')
        repo_path = '/'.join(parts[2:])
        ref = str((parse_qs(parsed.query).get('ref') or [''])[0]).strip() \
            or str(self._repository(owner, repo).get('default_branch') or '').strip()
        data = self._api(
            'GET', self._repo_api(owner, repo, f'/contents/{quote(repo_path, safe="/")}'),
            params={'ref': ref},
            not_found_code='GITHUB_TARGET_NOT_FOUND',
        )
        rows = data if isinstance(data, list) else [data]
        entries = [
            self._entry(
                _repo_uri(owner, repo, str(row.get('path') or ''), ref),
                size=int(row.get('size') or 0),
                ftype='directory' if row.get('type') == 'dir' else 'file',
                sha=str(row.get('sha') or ''),
            )
            for row in rows
        ]
        return entries if detail else [entry['name'] for entry in entries]

    def get_document_id(self, path: str) -> str:
        return str(self.resolve_target(path)['doc_id'])

    def _create_blob(self, owner: str, repo: str, content: bytes) -> str:
        data = self._api(
            'POST', self._repo_api(owner, repo, '/git/blobs'),
            expected=(201,),
            json={'content': base64.b64encode(content).decode('ascii'), 'encoding': 'base64'},
        )
        return str(data.get('sha') or '')

    def _commit_files(
        self,
        owner: str,
        repo: str,
        branch: str,
        parent_sha: str,
        files: Mapping[str, bytes],
        message: str,
    ) -> str:
        parent = self._api(
            'GET', self._repo_api(owner, repo, f'/git/commits/{quote(parent_sha, safe="")}'),
            not_found_code='GITHUB_REVISION_CONFLICT',
        )
        base_tree = str(((parent.get('tree') or {}).get('sha')) or '').strip()
        entries = []
        for file_path, content in files.items():
            clean_path = _validate_repo_path(file_path)
            blob_sha = self._create_blob(owner, repo, bytes(content))
            entries.append({'path': clean_path, 'mode': '100644', 'type': 'blob', 'sha': blob_sha})
        tree = self._api(
            'POST', self._repo_api(owner, repo, '/git/trees'),
            expected=(201,),
            json={'base_tree': base_tree, 'tree': entries},
        )
        tree_sha = str(tree.get('sha') or '').strip()
        if not tree_sha:
            raise GitHubFSError('GITHUB_WRITE_FAILED', 'GitHub returned an empty tree SHA.')
        if tree_sha == base_tree:
            return parent_sha
        commit = self._api(
            'POST', self._repo_api(owner, repo, '/git/commits'),
            expected=(201,),
            json={
                'message': message,
                'tree': tree_sha,
                'parents': [parent_sha],
            },
        )
        sha = str(commit.get('sha') or '').strip()
        if not sha:
            raise GitHubFSError('GITHUB_WRITE_FAILED', 'GitHub returned an empty commit SHA.')
        return sha

    def _create_ref(self, owner: str, repo: str, branch: str, sha: str) -> None:
        self._api(
            'POST', self._repo_api(owner, repo, '/git/refs'),
            expected=(201,),
            json={'ref': f'refs/heads/{branch}', 'sha': sha},
        )

    def _update_ref(self, owner: str, repo: str, branch: str, sha: str) -> None:
        self._api(
            'PATCH', self._repo_api(owner, repo, f'/git/refs/heads/{quote(branch, safe="")}'),
            json={'sha': sha, 'force': False},
        )

    def _find_open_pr(self, owner: str, repo: str, branch: str, base_ref: str) -> dict[str, Any]:
        rows = self._api(
            'GET', self._repo_api(owner, repo, '/pulls'),
            params={'state': 'open', 'head': f'{owner}:{branch}', 'base': base_ref, 'per_page': 10},
        )
        return rows[0] if isinstance(rows, list) and rows else {}

    def _ensure_pr(
        self,
        owner: str,
        repo: str,
        branch: str,
        base_ref: str,
        title: str,
    ) -> dict[str, Any]:
        existing = self._find_open_pr(owner, repo, branch, base_ref)
        if existing:
            return existing
        return self._api(
            'POST', self._repo_api(owner, repo, '/pulls'),
            expected=(201,),
            json={'title': title, 'head': branch, 'base': base_ref, 'body': 'Created by LazyMind Writer.'},
        )

    def _commit_has_operation(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
        operation_id: str,
    ) -> bool:
        commit = self._api(
            'GET',
            self._repo_api(owner, repo, f'/git/commits/{quote(commit_sha, safe="")}'),
            not_found_code='GITHUB_REVISION_CONFLICT',
        )
        message = str(commit.get('message') or '')
        return f'({operation_id})' in message

    @staticmethod
    def _publication_result(
        *,
        owner: str,
        repo: str,
        repo_path: str,
        target_branch: str,
        pull_request_base: str,
        mode: str,
        commit_sha: str,
        operation_id: str,
        pr: Mapping[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        pr = pr or {}
        return {
            'success': True,
            'provider': 'github',
            'target_type': 'repository',
            'doc_id': f'{owner}/{repo}:{target_branch}:{repo_path}',
            'uri': _repo_uri(owner, repo, repo_path, target_branch),
            'browser_url': str(
                pr.get('html_url')
                or _repo_browser_url(owner, repo, target_branch, repo_path)
            ),
            'publish_mode': mode,
            'commit_sha': commit_sha,
            'revision': commit_sha,
            'base_ref': pull_request_base,
            'work_branch': target_branch if mode == 'pull_request' else '',
            'pull_request_url': str(pr.get('html_url') or ''),
            'pull_request_number': pr.get('number'),
            'operation_id': operation_id,
            'warnings': warnings or [],
        }

    @staticmethod
    def _operation_branch(operation_id: str) -> str:
        normalized = _SAFE_OPERATION_RE.sub('-', operation_id).strip('-._')[:80]
        if not normalized:
            normalized = hashlib.sha256(operation_id.encode('utf-8')).hexdigest()[:16]
        return f'lazymind/{normalized}'

    def apply_document_patch(
        self,
        uri: str,
        markdown: str,
        *,
        files: Mapping[str, bytes] | None = None,
        expected_revision: str = '',
        operation_id: str = '',
        publish_mode: str = 'pull_request',
        commit_message: str = '',
        pull_request_title: str = '',
        work_branch: str = '',
        base_ref: str = '',
    ) -> dict[str, Any]:
        owner, repo, document_ref, repo_path = self._parse_target(uri, markdown_only=True)
        pull_request_base = str(base_ref or '').strip() or document_ref
        mode = str(publish_mode or 'pull_request').strip().lower()
        if mode not in {'pull_request', 'direct'}:
            raise ValueError('publish_mode must be pull_request or direct')
        all_files = self._patch_files(repo_path, markdown, files)
        operation_id = str(operation_id or '').strip() or _operation_id_for_files(
            owner,
            repo,
            repo_path,
            pull_request_base,
            expected_revision,
            mode,
            all_files,
        )
        message = commit_message.strip() or f'Update {repo_path} via LazyMind'
        if f'({operation_id})' not in message:
            message = f'{message} ({operation_id})'

        warnings: list[str] = []
        target_branch, current, branch_existed, base_update_allowed = self._prepare_patch_branch(
            owner, repo, document_ref, mode, work_branch, operation_id,
            pull_request_base, expected_revision, all_files)

        possible_retry = branch_existed or (
            mode == 'direct' and bool(expected_revision and current != expected_revision)
        )
        if possible_retry and self._commit_has_operation(
            owner, repo, str(current), operation_id,
        ):
            pr: dict[str, Any] = {}
            if mode == 'pull_request':
                pr = self._ensure_pr(
                    owner,
                    repo,
                    target_branch,
                    pull_request_base,
                    pull_request_title.strip() or f'Update {PurePosixPath(repo_path).name}',
                )
            return self._publication_result(
                owner=owner,
                repo=repo,
                repo_path=repo_path,
                target_branch=target_branch,
                pull_request_base=pull_request_base,
                mode=mode,
                commit_sha=str(current),
                operation_id=operation_id,
                pr=pr,
            )

        if expected_revision and current != expected_revision and not base_update_allowed:
            raise GitHubFSError(
                'GITHUB_REVISION_CONFLICT',
                'GitHub branch changed after the document was loaded.',
                status_code=409,
            )
        commit_sha = self._commit_files(
            owner, repo, target_branch, str(current), all_files, message,
        )
        if commit_sha != current:
            try:
                self._update_ref(owner, repo, target_branch, commit_sha)
            except GitHubFSError as exc:
                if mode != 'direct' or exc.code not in {
                    'GITHUB_PERMISSION_DENIED', 'GITHUB_WRITE_FAILED',
                }:
                    raise
                target_branch = self._operation_branch(operation_id)
                if self._branch_head(owner, repo, target_branch, missing_ok=True) is None:
                    self._create_ref(owner, repo, target_branch, commit_sha)
                else:
                    self._update_ref(owner, repo, target_branch, commit_sha)
                mode = 'pull_request'
                warnings.append('DIRECT_PUSH_FALLBACK_TO_PR')

        pr: dict[str, Any] = {}
        if mode == 'pull_request':
            pr = self._ensure_pr(
                owner,
                repo,
                target_branch,
                pull_request_base,
                pull_request_title.strip() or f'Update {PurePosixPath(repo_path).name}',
            )
        return self._publication_result(
            owner=owner,
            repo=repo,
            repo_path=repo_path,
            target_branch=target_branch,
            pull_request_base=pull_request_base,
            mode=mode,
            commit_sha=commit_sha,
            operation_id=operation_id,
            pr=pr,
            warnings=warnings,
        )

    def create_document(
        self,
        title: str,
        parent: str,
        *,
        operation_id: str = '',
        publish_mode: str = 'pull_request',
    ) -> dict[str, Any]:
        owner, repo, ref, parent_path = self._parse_target(parent, markdown_only=True)
        directory = str(PurePosixPath(parent_path).parent)
        directory = '' if directory == '.' else directory
        filename = _SAFE_OPERATION_RE.sub('-', title.strip()).strip('-._') or 'untitled'
        repo_path = f'{directory}/{filename}.md' if directory else f'{filename}.md'
        uri = _repo_uri(owner, repo, repo_path, ref)
        result = self.apply_document_patch(
            uri,
            f'# {title.strip() or "Untitled"}\n',
            operation_id=operation_id,
            publish_mode=publish_mode,
            pull_request_title=f'Create {repo_path}',
        )
        return {**result, 'title': title.strip() or 'Untitled', 'path': repo_path}

    def _upload_data(self, path: str, data: bytes) -> None:
        target = self.resolve_target(path)
        self.apply_document_patch(
            target['uri'],
            data.decode('utf-8'),
            expected_revision=target['revision'],
            publish_mode='direct',
        )

    def _prepare_patch_branch(self, owner, repo, document_ref, mode, work_branch, operation_id,
                              pull_request_base, expected_revision, all_files):
        target_branch = document_ref
        branch_existed = False
        base_update_allowed = False
        if mode == 'pull_request':
            target_branch = work_branch.strip() or self._operation_branch(operation_id)
            current = self._branch_head(owner, repo, target_branch, missing_ok=True)
            branch_existed = current is not None
            if current is None:
                current = self._branch_head(owner, repo, pull_request_base)
                if expected_revision and current != expected_revision and not self._paths_unchanged(
                    owner,
                    repo,
                    list(all_files),
                    expected_revision,
                    current,
                ):
                    raise GitHubFSError(
                        'GITHUB_REVISION_CONFLICT',
                        'GitHub branch changed after the document was loaded.',
                        status_code=409,
                    )
                base_update_allowed = bool(expected_revision and current != expected_revision)
                self._create_ref(owner, repo, target_branch, current)
        else:
            current = self._branch_head(owner, repo, document_ref)
        return target_branch, current, branch_existed, base_update_allowed

    @staticmethod
    def _patch_files(repo_path, markdown, files):
        all_files = {repo_path: markdown.encode('utf-8')}
        for file_path, data in (files or {}).items():
            clean_path = _validate_repo_path(file_path)
            if clean_path == repo_path:
                raise ValueError('Resource path cannot replace the target Markdown path.')
            all_files[clean_path] = bytes(data)
        return all_files


class GitHubWikiFS(_GitHubFSBase):
    '''Read and safely publish pages in a repository's separate Wiki Git repository.'''

    __public_apis__ = LazyLLMFSBase.__public_apis__ + [
        'resolve_target', 'resolve_create_parent', 'resolve_create_target',
        'apply_document_patch', 'create_document',
    ]

    def _repository(self, owner: str, repo: str) -> dict[str, Any]:
        return self._api(
            'GET',
            f'/repos/{quote(owner, safe="")}/{quote(repo, safe="")}',
            not_found_code='GITHUB_PERMISSION_DENIED',
        )

    @classmethod
    def matches_create_parent(cls, value: str) -> bool:
        parsed = urlparse(clean_document_ref(str(value or '').strip()))
        if parsed.scheme not in {'http', 'https'} or (parsed.hostname or '').lower() not in _GITHUB_HOSTS:
            return False
        parts = [part for part in parsed.path.split('/') if part]
        return len(parts) == 3 and parts[2].lower() == 'wiki'

    def _parse_create_parent(self, value: str) -> tuple[str, str, dict[str, Any]]:
        parsed = urlparse(clean_document_ref(str(value or '').strip()))
        if parsed.scheme not in {'http', 'https'} or (parsed.hostname or '').lower() not in _GITHUB_HOSTS:
            raise ValueError('GitHub Wiki create parent must be a github.com Wiki root URL.')
        parts = [unquote(part) for part in parsed.path.split('/') if part]
        if len(parts) != 3 or parts[2].lower() != 'wiki':
            raise ValueError('GitHub Wiki create parent must use /{owner}/{repo}/wiki.')
        owner = _validate_repo_part(parts[0], 'owner')
        repo_value = parts[1][:-4] if parts[1].lower().endswith('.git') else parts[1]
        repo = _validate_repo_part(repo_value, 'repository')
        return owner, repo, self._repository(owner, repo)

    def resolve_create_parent(self, parent: str) -> dict[str, Any]:
        owner, repo, repository = self._parse_create_parent(parent)
        permissions = repository.get('permissions')
        if isinstance(permissions, Mapping) and permissions.get('push') is False:
            raise GitHubFSError(
                'GITHUB_PERMISSION_DENIED',
                'The current GitHub account cannot create pages in this Wiki.',
                status_code=403,
            )
        if repository.get('has_wiki') is False:
            raise GitHubFSError(
                'GITHUB_WIKI_NOT_FOUND',
                'GitHub Wiki is not enabled for this repository.',
                status_code=404,
            )
        with tempfile.TemporaryDirectory(prefix='lazyllm-github-wiki-') as root:
            checkout = self._clone(owner, repo, root, materialize=False)
            revision = self._git(['rev-parse', 'HEAD'], cwd=str(checkout))
            branch = self._git(['rev-parse', '--abbrev-ref', 'HEAD'], cwd=str(checkout))
        browser_url = _wiki_parent_browser_url(owner, repo)
        return {
            'uri': browser_url,
            'browser_url': browser_url,
            'owner': owner,
            'repo': repo,
            'ref': branch,
            'base_ref': branch,
            'directory': '',
            'revision': revision,
            'target_type': 'wiki',
            'fs_scheme': 'githubwiki',
            'publish_mode': 'direct',
            'create_pending': True,
            'parent_uri': str(parent or '').strip(),
        }

    def resolve_create_target(
        self,
        parent: Mapping[str, Any],
        title: str,
    ) -> dict[str, Any]:
        owner = _validate_repo_part(str(parent.get('owner') or ''), 'owner')
        repo = _validate_repo_part(str(parent.get('repo') or ''), 'repository')
        revision = str(parent.get('revision') or '').strip()
        if not revision:
            raise GitHubFSError(
                'GITHUB_TARGET_INVALID',
                'Pending GitHub Wiki target is missing its revision.',
                status_code=400,
            )
        page_path = _markdown_filename(title)
        with tempfile.TemporaryDirectory(prefix='lazyllm-github-wiki-') as root:
            checkout = self._clone(owner, repo, root, materialize=False)
            current = self._git(['rev-parse', 'HEAD'], cwd=str(checkout))
            if current == revision and self._git(
                ['ls-tree', '--name-only', 'HEAD', '--', page_path], cwd=str(checkout),
            ):
                raise GitHubFSError(
                    'GITHUB_TARGET_EXISTS',
                    f'GitHub Wiki page already exists: {page_path}.',
                    status_code=409,
                )
        branch = str(parent.get('base_ref') or parent.get('ref') or '').strip()
        return {
            'doc_id': f'{owner}/{repo}.wiki:{page_path}',
            'uri': _wiki_uri(owner, repo, page_path),
            'browser_url': _wiki_browser_url(owner, repo, page_path),
            'title': PurePosixPath(page_path).stem,
            'owner': owner,
            'repo': repo,
            'ref': branch,
            'base_ref': branch,
            'path': page_path,
            'directory': '',
            'revision': revision,
            'target_type': 'wiki',
            'fs_scheme': 'githubwiki',
            'publish_mode': 'direct',
            'create_pending': False,
            'parent_uri': str(parent.get('parent_uri') or parent.get('uri') or ''),
        }

    def _parse_target(
        self,
        path: str,
        *,
        document_only: bool = False,
    ) -> tuple[str, str, str]:
        value = clean_document_ref(path)
        parsed = urlparse(value)
        if parsed.scheme in ('http', 'https'):
            if (parsed.hostname or '').lower() not in _GITHUB_HOSTS:
                raise ValueError(f'Unsupported GitHub host: {parsed.hostname!r}.')
            parts = [unquote(part) for part in parsed.path.split('/') if part]
            if len(parts) < 4 or parts[2].lower() != 'wiki':
                raise ValueError('GitHub Wiki URL must use /{owner}/{repo}/wiki/{page}.')
            owner = _validate_repo_part(parts[0], 'owner')
            repo = _validate_repo_part(parts[1], 'repository')
            page_path = _validate_wiki_path('/'.join(parts[3:]))
            return owner, repo, page_path
        if value.startswith('githubwiki:'):
            parsed = urlparse(value)
            raw_path = parsed.path
        else:
            parsed = urlparse(f'githubwiki:{value if value.startswith("/") else "/" + value}')
            raw_path = parsed.path
        parts = [unquote(part) for part in raw_path.split('/') if part]
        if len(parts) < 3:
            raise ValueError('githubwiki URI must contain owner, repository, and page.')
        resource_path = '/'.join(parts[2:])
        return (
            _validate_repo_part(parts[0], 'owner'),
            _validate_repo_part(parts[1], 'repository'),
            _validate_wiki_path(resource_path)
            if document_only else _validate_repo_path(resource_path),
        )

    def _git_env(self) -> dict[str, str]:
        self.ensure_token()
        token = self.get_current_token()
        encoded = base64.b64encode(f'x-access-token:{token}'.encode()).decode('ascii')
        env = os.environ.copy()
        env.update({
            'GIT_TERMINAL_PROMPT': '0',
            'GIT_CONFIG_COUNT': '1',
            'GIT_CONFIG_KEY_0': 'http.extraHeader',
            'GIT_CONFIG_VALUE_0': f'Authorization: Basic {encoded}',
            'GIT_AUTHOR_NAME': 'LazyMind Writer',
            'GIT_AUTHOR_EMAIL': 'writer@lazymind.local',
            'GIT_COMMITTER_NAME': 'LazyMind Writer',
            'GIT_COMMITTER_EMAIL': 'writer@lazymind.local',
        })
        return env

    def _git(self, args: list[str], *, cwd: str | None = None) -> str:
        try:
            completed = subprocess.run(
                ['git', *args],
                cwd=cwd,
                env=self._git_env(),
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or 'GitHub Wiki Git operation failed').strip()
            lowered = detail.lower()
            if 'authentication failed' in lowered or 'could not read username' in lowered:
                code = 'GITHUB_AUTH_REQUIRED'
            elif 'permission' in lowered or '403' in lowered:
                code = 'GITHUB_PERMISSION_DENIED'
            elif 'not found' in lowered or 'repository not found' in lowered:
                code = 'GITHUB_WIKI_NOT_FOUND'
            elif (
                'non-fast-forward' in lowered
                or 'fetch first' in lowered
                or 'rejected' in lowered
            ):
                code = 'GITHUB_REVISION_CONFLICT'
            else:
                code = 'GITHUB_WRITE_FAILED'
            raise GitHubFSError(code, detail[:300]) from exc
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GitHubFSError(
                'GITHUB_WRITE_FAILED', 'GitHub Wiki Git operation failed.', retryable=True,
            ) from exc
        return completed.stdout.strip()

    def _clone(
        self,
        owner: str,
        repo: str,
        root: str,
        *,
        materialize: bool = True,
    ) -> Path:
        self._repository(owner, repo)
        checkout = Path(root) / 'wiki'
        remote = f'https://github.com/{owner}/{repo}.wiki.git'
        args = ['clone', '--depth', '1']
        if not materialize:
            args.extend(['--filter=blob:none', '--no-checkout'])
        self._git([*args, remote, str(checkout)])
        return checkout

    def _checkout_file(self, checkout: Path, path: str, revision: str = 'HEAD') -> Path:
        file_path = checkout / path
        if file_path.is_file() and not file_path.is_symlink():
            return file_path
        entry = self._git(
            ['--literal-pathspecs', 'ls-tree', '-z', revision, '--', path], cwd=str(checkout),
        )
        if not entry.startswith(('100644 blob ', '100755 blob ')):
            raise GitHubFSError(
                'GITHUB_WIKI_NOT_FOUND', 'GitHub Wiki file does not exist.', status_code=404,
            )
        self._git(
            ['--literal-pathspecs', 'checkout', revision, '--', path], cwd=str(checkout),
        )
        return file_path

    def _resolved_target(
        self,
        owner: str,
        repo: str,
        page_path: str,
        checkout: Path,
    ) -> dict[str, Any]:
        revision = self._git(['rev-parse', 'HEAD'], cwd=str(checkout))
        file_path = checkout / page_path
        if not file_path.is_file():
            raise GitHubFSError(
                'GITHUB_WIKI_NOT_FOUND', 'GitHub Wiki page does not exist.', status_code=404,
            )
        return {
            'doc_id': f'{owner}/{repo}.wiki:{page_path}',
            'uri': _wiki_uri(owner, repo, page_path),
            'browser_url': _wiki_browser_url(owner, repo, page_path),
            'title': PurePosixPath(page_path).stem,
            'owner': owner,
            'repo': repo,
            'path': page_path,
            'revision': revision,
            'size': file_path.stat().st_size,
            'target_type': 'wiki',
            'fs_scheme': 'githubwiki',
        }

    @contextmanager
    def read_session(
        self,
        path: str,
    ) -> Iterator[tuple[dict[str, Any], Callable[[str], bytes]]]:
        '''Resolve and read files from one immutable Wiki checkout.'''
        owner, repo, page_path = self._parse_target(path, document_only=True)
        with tempfile.TemporaryDirectory(prefix='lazyllm-github-wiki-') as root:
            checkout = self._clone(owner, repo, root, materialize=False)
            self._checkout_file(checkout, page_path)
            resolved = self._resolved_target(owner, repo, page_path, checkout)

            def read_bytes(resource: str) -> bytes:
                resource_owner, resource_repo, resource_path = self._parse_target(resource)
                if (resource_owner, resource_repo) != (owner, repo):
                    raise ValueError('GitHub Wiki read session cannot cross repositories.')
                file_path = self._checkout_file(checkout, resource_path, resolved['revision'])
                return file_path.read_bytes()

            yield resolved, read_bytes

    def resolve_target(self, path: str) -> dict[str, Any]:
        with self.read_session(path) as (resolved, _):
            return resolved

    def read_bytes(self, path: str) -> bytes:
        owner, repo, page_path = self._parse_target(path)
        with tempfile.TemporaryDirectory(prefix='lazyllm-github-wiki-') as root:
            checkout = self._clone(owner, repo, root, materialize=False)
            file_path = self._checkout_file(checkout, page_path)
            return file_path.read_bytes()

    def info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        target = self.resolve_target(path)
        return self._entry(
            target['uri'],
            size=target['size'],
            ftype='file',
            revision=target['revision'],
            browser_url=target['browser_url'],
        )

    def ls(self, path: str, detail: bool = True, **kwargs: Any) -> list:
        owner, repo, page_path = self._parse_target(path)
        directory = str(PurePosixPath(page_path).parent)
        directory = '' if directory == '.' else directory
        with tempfile.TemporaryDirectory(prefix='lazyllm-github-wiki-') as root:
            checkout = self._clone(owner, repo, root)
            base = checkout / directory
            if not base.is_dir():
                return []
            entries = [
                self._entry(
                    _wiki_uri(owner, repo, str(item.relative_to(checkout).as_posix())),
                    size=item.stat().st_size if item.is_file() else 0,
                    ftype='file' if item.is_file() else 'directory',
                )
                for item in base.iterdir()
                if item.name != '.git'
            ]
        return entries if detail else [entry['name'] for entry in entries]

    def get_document_id(self, path: str) -> str:
        owner, repo, page_path = self._parse_target(path, document_only=True)
        return f'{owner}/{repo}.wiki:{page_path}'

    def apply_document_patch(
        self,
        uri: str,
        markdown: str,
        *,
        files: Mapping[str, bytes] | None = None,
        expected_revision: str = '',
        operation_id: str = '',
        commit_message: str = '',
        **_: Any,
    ) -> dict[str, Any]:
        owner, repo, page_path = self._parse_target(uri, document_only=True)
        all_files = {page_path: markdown.encode('utf-8')}
        for relative, data in (files or {}).items():
            clean_path = _validate_repo_path(relative)
            if clean_path == page_path:
                raise ValueError('Resource path cannot replace the target Wiki Markdown path.')
            all_files[clean_path] = bytes(data)
        operation_id = operation_id.strip() or _operation_id_for_files(
            owner,
            repo,
            page_path,
            'wiki',
            expected_revision,
            'direct',
            all_files,
        )
        message = commit_message.strip() or f'Update {page_path} via LazyMind'
        if f'({operation_id})' not in message:
            message = f'{message} ({operation_id})'
        with tempfile.TemporaryDirectory(prefix='lazyllm-github-wiki-') as root:
            checkout = self._clone(owner, repo, root, materialize=False)
            current = self._git(['rev-parse', 'HEAD'], cwd=str(checkout))
            current_message = self._git(['log', '-1', '--format=%B'], cwd=str(checkout))
            if f'({operation_id})' in current_message:
                commit_sha = current
            else:
                if expected_revision and current != expected_revision:
                    raise GitHubFSError(
                        'GITHUB_REVISION_CONFLICT',
                        'GitHub Wiki changed after the page was loaded.',
                        status_code=409,
                    )
                self._git(['read-tree', 'HEAD'], cwd=str(checkout))
                for relative, data in all_files.items():
                    destination = checkout / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
                self._git(['add', '--', *all_files], cwd=str(checkout))
                staged = self._git(['diff', '--cached', '--name-only'], cwd=str(checkout))
                if staged:
                    tree_sha = self._git(
                        ['write-tree', '--missing-ok'], cwd=str(checkout),
                    )
                    commit_sha = self._git(
                        ['commit-tree', tree_sha, '-p', current, '-m', message],
                        cwd=str(checkout),
                    )
                    self._git(
                        ['update-ref', 'HEAD', commit_sha, current], cwd=str(checkout),
                    )
                    branch = self._git(
                        ['rev-parse', '--abbrev-ref', 'HEAD'], cwd=str(checkout),
                    )
                    self._git(['fetch', 'origin', branch], cwd=str(checkout))
                    remote_head = self._git(
                        ['rev-parse', f'refs/remotes/origin/{branch}'], cwd=str(checkout),
                    )
                    if remote_head != current:
                        raise GitHubFSError(
                            'GITHUB_REVISION_CONFLICT',
                            'GitHub Wiki changed while the page was being written.',
                            status_code=409,
                        )
                    self._git(['push', 'origin', f'HEAD:{branch}'], cwd=str(checkout))
                else:
                    commit_sha = current
        return {
            'success': True,
            'provider': 'github',
            'target_type': 'wiki',
            'doc_id': f'{owner}/{repo}.wiki:{page_path}',
            'uri': _wiki_uri(owner, repo, page_path),
            'browser_url': _wiki_browser_url(owner, repo, page_path),
            'publish_mode': 'direct',
            'commit_sha': commit_sha,
            'revision': commit_sha,
            'work_branch': '',
            'pull_request_url': '',
            'operation_id': operation_id,
            'warnings': [],
        }

    def create_document(
        self,
        title: str,
        parent: str,
        *,
        operation_id: str = '',
        **kwargs: Any,
    ) -> dict[str, Any]:
        owner, repo, _ = self._parse_target(parent, document_only=True)
        filename = _SAFE_OPERATION_RE.sub('-', title.strip()).strip('-._') or 'Untitled'
        page_path = _validate_wiki_path(filename)
        result = self.apply_document_patch(
            _wiki_uri(owner, repo, page_path),
            f'# {title.strip() or "Untitled"}\n',
            operation_id=operation_id,
            **kwargs,
        )
        return {**result, 'title': title.strip() or 'Untitled', 'path': page_path}

    def _upload_data(self, path: str, data: bytes) -> None:
        target = self.resolve_target(path)
        self.apply_document_patch(
            target['uri'],
            data.decode('utf-8'),
            expected_revision=target['revision'],
        )


__all__ = ['GitHubFSError', 'GitHubRepoFS', 'GitHubWikiFS']
