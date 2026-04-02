# Copyright (c) 2026 LazyAGI. All rights reserved.
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import lazyllm
from lazyllm import config

from ..base import LazyLLMFSBase, CloudFSBufferedFile

config.add('yuque_token', str, None, 'YUQUE_TOKEN', description='Yuque API token (yuque-sdk official env).')

_API_BASE = 'https://www.yuque.com/api/v2'


class YuqueFS(LazyLLMFSBase):

    def __init__(self, token: Optional[str] = None, base_url: Optional[str] = None, **storage_options):
        token = token or config['yuque_token'] or os.environ.get('YUQUE_TOKEN') or ''
        super().__init__(token=token, base_url=base_url or _API_BASE, **storage_options)

    def _setup_auth(self) -> None:
        self._session.headers.update({
            'X-Auth-Token': self._secret_key,
            'Content-Type': 'application/json',
            'User-Agent': 'lazyllm-fs (https://github.com/LazyAGI/lazyllm)',
        })

    def ls(self, path: str, detail: bool = True, **kwargs) -> List:
        parts = self._parse_path(path)
        if not parts:
            return self._list_user_repos(detail)
        if len(parts) == 1:
            return self._list_user_repos(detail, login=parts[0])
        login, repo_slug = parts[0], parts[1]
        return self._list_docs(login, repo_slug, detail)

    def info(self, path: str, **kwargs) -> Dict[str, Any]:
        parts = self._parse_path(path)
        if not parts:
            data = self._get(f'{self._base_url}/user')
            user = data.get('data', {})
            return self._entry(user.get('login', '/'), ftype='directory',
                               title=user.get('name', ''))
        if len(parts) == 1:
            login = parts[0]
            data = self._get(f'{self._base_url}/users/{login}')
            user = data.get('data', {})
            return self._entry(login, ftype='directory', title=user.get('name', login))
        if len(parts) == 2:
            login, repo_slug = parts
            data = self._get(f'{self._base_url}/repos/{login}/{repo_slug}')
            return self._repo_to_entry(data.get('data', {}))
        login, repo_slug, doc_slug = parts[0], parts[1], parts[2]
        data = self._get(f'{self._base_url}/repos/{login}/{repo_slug}/docs/{doc_slug}')
        return self._doc_to_entry(data.get('data', {}))

    def _open(self, path: str, mode: str = 'rb',
              block_size: Optional[int] = None,
              autocommit: bool = True,
              cache_options: Optional[Dict] = None,
              **kwargs) -> CloudFSBufferedFile:
        return CloudFSBufferedFile(
            self, path, mode=mode,
            block_size=block_size or self.blocksize,
            autocommit=autocommit, cache_options=cache_options,
        )

    def mkdir(self, path: str, create_parents: bool = True, **kwargs) -> None:
        parts = self._parse_path(path)
        if len(parts) < 2:
            raise ValueError('path must be /<login>/<repo_name>')
        login, name = parts[0], parts[1]
        payload = {
            'name': name,
            'slug': name.lower().replace(' ', '-'),
            'type': 'Book',
        }
        self._post(f'{self._base_url}/groups/{login}/repos', json=payload)

    def rm_file(self, path: str) -> None:
        parts = self._parse_path(path)
        if len(parts) < 3:
            raise ValueError('path must be /<login>/<repo_slug>/<doc_id>')
        login, repo_slug, doc_id = parts[0], parts[1], parts[2]
        self._delete(f'{self._base_url}/repos/{login}/{repo_slug}/docs/{doc_id}')

    def rmdir(self, path: str) -> None:
        parts = self._parse_path(path)
        if len(parts) < 2:
            return
        login, repo_slug = parts[0], parts[1]
        self._delete(f'{self._base_url}/repos/{login}/{repo_slug}')

    def copy(self, path1: str, path2: str, recursive: bool = False, **kwargs) -> None:
        raise NotImplementedError('YuqueFS: Yuque official API does not support copy')

    def move(self, path1: str, path2: str, recursive: bool = False, **kwargs) -> None:
        raise NotImplementedError('YuqueFS: Yuque official API does not support move')

    def _download_range(self, path: str, start: int, end: int) -> bytes:
        parts = self._parse_path(path)
        if len(parts) < 3:
            raise FileNotFoundError(path)
        login, repo_slug, doc_slug = parts[0], parts[1], parts[2]
        data = self._get(f'{self._base_url}/repos/{login}/{repo_slug}/docs/{doc_slug}')
        body = data.get('data', {}).get('body', '')
        encoded = body.encode('utf-8')
        return encoded[start:end]

    def _upload_data(self, path: str, data: bytes) -> None:
        parts = self._parse_path(path)
        if len(parts) < 3:
            raise ValueError('path must be /<login>/<repo_slug>/<doc_slug>')
        login, repo_slug, doc_slug = parts[0], parts[1], parts[2]
        body = data.decode('utf-8', errors='replace')
        try:
            existing = self._get(f'{self._base_url}/repos/{login}/{repo_slug}/docs/{doc_slug}')
            doc_id = existing.get('data', {}).get('id')
            if doc_id:
                self._put(f'{self._base_url}/repos/{login}/{repo_slug}/docs/{doc_id}',
                          json={'body': body})
                return
        except Exception:
            pass
        self._post(
            f'{self._base_url}/repos/{login}/{repo_slug}/docs',
            json={'title': doc_slug, 'slug': doc_slug, 'body': body, 'format': 'markdown'},
        )

    def _platform_supports_webhook(self) -> bool:
        return True

    def _register_webhook(self, webhook_url: str, events: List[str], path: str) -> Dict[str, Any]:
        parts = self._parse_path(path)
        if len(parts) < 2:
            raise ValueError('path must include <login>/<repo_slug> for webhook')
        login, repo_slug = parts[0], parts[1]
        payload = {
            'url': webhook_url,
            'events': events or ['doc_publish', 'doc_update', 'doc_delete'],
        }
        return self._post(f'{self._base_url}/repos/{login}/{repo_slug}/webhooks', json=payload)

    def _list_user_repos(self, detail: bool, login: Optional[str] = None) -> List:
        if login:
            url = f'{self._base_url}/users/{login}/repos'
        else:
            url = f'{self._base_url}/mine/repos'
        data = self._get(url, params={'limit': 100})
        repos = data.get('data', [])
        if detail:
            return [self._repo_to_entry(r) for r in repos]
        return [f'{r.get("user", {}).get("login", "")}/{r.get("slug", "")}' for r in repos]

    def _list_docs(self, login: str, repo_slug: str, detail: bool) -> List:
        url = f'{self._base_url}/repos/{login}/{repo_slug}/docs'
        data = self._get(url)
        docs = data.get('data', [])
        if detail:
            entries = [self._doc_to_entry(d) for d in docs]
            for e in entries:
                e['name'] = f'{login}/{repo_slug}/{e["name"]}'
            return entries
        return [f'{login}/{repo_slug}/{d.get("slug", d.get("id", ""))}' for d in docs]

    @staticmethod
    def _repo_to_entry(repo: Dict[str, Any]) -> Dict[str, Any]:
        login = repo.get('user', {}).get('login', '')
        slug = repo.get('slug', '')
        name = f'{login}/{slug}' if login and slug else slug
        mtime = None
        ts = repo.get('updated_at')
        if ts:
            try:
                mtime = datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
            except (ValueError, TypeError) as e:
                lazyllm.LOG.debug(f"Failed to parse timestamp '{ts}': {e}")
        return LazyLLMFSBase._entry(
            name=name, ftype='directory', mtime=mtime,
            title=repo.get('name', ''), namespace=repo.get('namespace', ''),
        )

    @staticmethod
    def _doc_to_entry(doc: Dict[str, Any]) -> Dict[str, Any]:
        mtime = None
        ts = doc.get('updated_at')
        if ts:
            try:
                mtime = datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
            except (ValueError, TypeError) as e:
                lazyllm.LOG.debug(f"Failed to parse timestamp '{ts}': {e}")
        return LazyLLMFSBase._entry(
            name=str(doc.get('id', doc.get('slug', ''))),
            ftype='file', mtime=mtime, size=doc.get('word_count', 0),
            title=doc.get('title', ''), slug=doc.get('slug', ''),
        )
