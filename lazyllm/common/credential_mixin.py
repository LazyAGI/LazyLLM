# Copyright (c) 2026 LazyAGI. All rights reserved.
import os
import requests
import threading
import time
import uuid
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qs, urlparse

from .auth import (AllKeysExhaustedError, AuthStrategy, BearerTokenStrategy,
                   Credential, KeyAuthError, KeyPool, KeySelectPolicy)
from .globals import globals as lazyllm_globals, locals as lazyllm_locals


class CredentialMixin:

    _TOKEN_REFRESH_BUFFER: float = 60.0

    def __init_credential__(
        self,
        credential: Credential,
        strategy: Optional[AuthStrategy] = None,
        skip_auth: bool = False,
        dynamic_key_policy: KeySelectPolicy = KeySelectPolicy.RANDOM,
    ) -> None:
        self._credential: Credential = credential
        self._auth_strategy: AuthStrategy = strategy or BearerTokenStrategy()
        self._skip_auth: bool = skip_auth
        self._dynamic_key_policy: KeySelectPolicy = dynamic_key_policy
        self._token_lock: threading.Lock = threading.Lock()
        self._credential_id: str = uuid.uuid4().hex

    def __key_source__(self) -> Any:
        if self._skip_auth:
            return True
        return bool(self._get_token())

    @property
    def _dynamic_auth(self) -> bool:
        return self._credential.kind == 'dynamic'

    @property
    def _secret_key(self) -> Any:
        return self._credential.secret_key

    def _get_token(self) -> str:
        curr = lazyllm_locals['curr_key'].get(self._credential_id)
        if curr is not None:
            return curr
        pool = self._get_active_pool()
        if pool is not None:
            return pool.peek()
        cred = self._credential
        if cred.kind == 'dynamic':
            raw = self._resolve_dynamic_token()
            return (raw[0] if raw else '') if isinstance(raw, (list, tuple)) else (raw or '')
        if cred.kind == 'static':
            return cred.secret_key if isinstance(cred.secret_key, str) else ''
        return cred.access_token or ''

    def get_current_token(self) -> str:
        return self._get_token()

    def _resolve_dynamic_token(self) -> Union[str, List[str]]:
        return ''

    def _missing_dynamic_token_error(self) -> str:
        return f'{type(self).__name__}: dynamic credential is not configured.'

    def _default_credential(
        self, token: Any, dynamic_auth: bool,
        policy: KeySelectPolicy = KeySelectPolicy.RANDOM,
    ) -> Credential:
        if dynamic_auth:
            return Credential(kind='dynamic')
        if isinstance(token, (list, tuple)) and len(token) > 1:
            return Credential(kind='static', key_pool=KeyPool(list(token), policy))
        sk = token if isinstance(token, str) else (token[0] if token else None)
        return Credential(kind='static', secret_key=sk)

    def _make_credential(self, token: Any, dynamic_auth: bool) -> Credential:
        return self._default_credential(token, dynamic_auth)

    def ensure_token(self) -> None:
        cred = self._credential
        if cred.kind == 'dynamic':
            if not self._resolve_dynamic_token():
                raise ValueError(self._missing_dynamic_token_error())
            return
        if cred.kind in ('oauth2', 'app_credentials'):
            if not cred.access_token or self._is_token_expired(cred):
                self._refresh_credential()

    @classmethod
    def _is_token_expired(cls, cred: Credential) -> bool:
        if cred.token_expire_at is None:
            return False
        return time.time() >= (cred.token_expire_at - cls._TOKEN_REFRESH_BUFFER)

    def inject_auth_header(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        self.ensure_token()
        out = dict(headers or {})
        out.update(self._auth_strategy.build_header(self._get_token()))
        return out

    def inject_auth_params(self, params: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        self.ensure_token()
        out = dict(params or {})
        out.update(self._auth_strategy.build_params(self._get_token()))
        return out

    def _get_active_pool(self) -> Optional[KeyPool]:
        cred = self._credential
        if cred.kind == 'static' and cred.key_pool:
            return cred.key_pool
        if cred.kind == 'dynamic':
            raw = self._resolve_dynamic_token()
            if isinstance(raw, (list, tuple)) and len(raw) > 1:
                pool_state = lazyllm_globals['key_pool_state']
                pool = pool_state.get(self._credential_id)
                if pool is None:
                    pool = KeyPool(list(raw), self._dynamic_key_policy)
                    pool_state[self._credential_id] = pool
                return pool
        return None

    def _is_key_auth_error(self, resp: Any) -> bool:
        return getattr(resp, 'status_code', 0) in (401, 403)

    def _http_execute(self, method: str, url: str, **kwargs) -> Any:
        resp = requests.request(method, url, **kwargs)
        if self._is_key_auth_error(resp):
            raise KeyAuthError(f'{resp.status_code} for {url}')
        if not resp.ok:
            raise requests.HTTPError(response=resp)
        return resp

    def _request(self, method: str, url: str, **kwargs) -> Any:
        incoming_headers = kwargs.pop('headers', None)
        pool = self._get_active_pool()
        if pool is None:
            headers = self.inject_auth_header(incoming_headers)
            return self._http_execute(method, url, headers=headers, **kwargs)
        last_err: Optional[Exception] = None
        curr_key = lazyllm_locals['curr_key']
        for key in pool.ordered_keys():
            curr_key[self._credential_id] = key
            try:
                headers = self.inject_auth_header(incoming_headers)
                result = self._http_execute(method, url, headers=headers, **kwargs)
                pool.report_success(key)
                return result
            except KeyAuthError as e:
                pool.report_failure(key)
                last_err = e
            finally:
                curr_key.pop(self._credential_id, None)
        raise AllKeysExhaustedError(f'{type(self).__name__}: all keys exhausted') from last_err

    def _refresh_credential(self) -> None:
        with self._token_lock:
            cred = self._credential
            if cred.access_token and not self._is_token_expired(cred):
                return
            new_token, new_expires, new_refresh = self._acquire_credential(cred)
            self._credential = replace(
                cred, access_token=new_token, token_expire_at=new_expires,
                refresh_token=new_refresh or cred.refresh_token,
            )
            if new_refresh:
                self._save_persisted_refresh_token(new_refresh)

    def _acquire_credential(self, cred: Credential) -> Tuple[str, Optional[float], str]:
        '''Acquire a fresh access token. Subclasses implement the kind-specific paths.'''
        if cred.kind == 'app_credentials':
            return self._do_acquire_without_refresh()
        rt = cred.refresh_token
        if rt == 'auto':
            rt = self._load_persisted_refresh_token()
        if rt:
            try:
                return self._do_refresh_token(rt)
            except Exception as err:
                if not cred.oauth_auto:
                    raise
                from lazyllm import LOG
                LOG.warning(f'refresh_token invalid for {self._get_persist_key()}: {err}; '
                            'falling back to OAuth flow.')
                self._save_persisted_refresh_token('')
        return self._do_oauth_flow()

    def _do_refresh_token(self, refresh_token: str) -> Tuple[str, Optional[float], str]:
        raise NotImplementedError(
            f'{type(self).__name__} must implement _do_refresh_token() to support refresh-based OAuth.'
        )

    def _do_acquire_without_refresh(self) -> Tuple[str, Optional[float], str]:
        raise NotImplementedError(
            f'{type(self).__name__} must implement _do_acquire_without_refresh() for kind="app_credentials".'
        )

    def _do_oauth_flow(self) -> Tuple[str, Optional[float], str]:
        raise NotImplementedError(
            f'{type(self).__name__} must implement _do_oauth_flow() for interactive OAuth.'
        )

    @staticmethod
    def _run_local_oauth_server(port: int, timeout: float) -> str:
        captured: Dict[str, str] = {}
        done = threading.Event()

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                qs = parse_qs(urlparse(self.path).query)
                if 'code' in qs:
                    captured['code'] = qs['code'][0]
                    body = b'<html><body>Authorization successful. You can close this tab.</body></html>'
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    captured['error'] = qs.get('error', ['unknown'])[0]
                    self.send_response(400)
                    self.end_headers()
                done.set()

            def log_message(self, fmt: str, *args: Any) -> None:
                return None

        server = HTTPServer(('localhost', port), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            if not done.wait(timeout=timeout):
                raise TimeoutError(f'OAuth callback timed out after {timeout}s')
        finally:
            server.shutdown()
            server.server_close()
        if 'code' not in captured:
            raise RuntimeError(f'OAuth flow failed: {captured.get("error", "unknown")}')
        return captured['code']

    def _get_persist_key(self) -> str:
        return f'{type(self).__name__}:{id(self)}'

    @staticmethod
    def _persist_path() -> str:
        from lazyllm import config
        return os.path.join(config['home'], '.lazyllm/tokens.txt')

    def _load_persisted_refresh_token(self) -> str:
        path, key = self._persist_path(), self._get_persist_key()
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    k, sep, v = line.strip().partition(': ')
                    if sep and k.strip() == key:
                        return v.strip()
        except FileNotFoundError:
            pass
        return ''

    def _save_persisted_refresh_token(self, refresh_token: str) -> None:
        path, key = self._persist_path(), self._get_persist_key()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        lines: List[str] = []
        replaced = False
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    k, sep, _ = line.strip().partition(': ')
                    if sep and k.strip() == key:
                        if refresh_token:
                            lines.append(f'{key}: {refresh_token}\n')
                            replaced = True
                    else:
                        lines.append(line if line.endswith('\n') else line + '\n')
        except FileNotFoundError:
            pass
        if not replaced and refresh_token:
            lines.append(f'{key}: {refresh_token}\n')
        with open(path, 'w', encoding='utf-8') as fh:
            fh.writelines(lines)

    def _bootstrap_token(self) -> None:
        if self._credential.kind in ('oauth2', 'app_credentials'):
            self._refresh_credential()


__all__ = ['CredentialMixin']
