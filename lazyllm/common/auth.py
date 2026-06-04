# Copyright (c) 2026 LazyAGI. All rights reserved.
import random
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from .globals import globals as lazyllm_globals

_AUTH_KINDS = ('static', 'dynamic', 'oauth2', 'app_credentials')


class KeySelectPolicy(str, Enum):
    RANDOM = 'random'
    ROUND_ROBIN = 'round_robin'
    PREFER_LAST_SUCCESS = 'prefer_last_success'


class KeyAuthError(Exception):
    pass


class AllKeysExhaustedError(Exception):
    pass


class KeyPool:
    def __init__(self, keys: List[str], policy: KeySelectPolicy) -> None:
        self._keys = list(keys)
        self._policy = policy
        self._id = uuid.uuid4().hex

    def _get_state(self) -> dict:
        return lazyllm_globals['key_pool_state'].setdefault(self._id, {})

    def ordered_keys(self) -> List[str]:
        state = self._get_state()
        failed = state.get('failed', set())
        if self._policy == KeySelectPolicy.RANDOM:
            candidates = [k for k in self._keys if k not in failed]
            random.shuffle(candidates)
            return candidates
        if self._policy == KeySelectPolicy.ROUND_ROBIN:
            candidates = [k for k in self._keys if k not in failed]
            if not candidates:
                return candidates
            idx = state.get('rr_index', 0) % len(candidates)
            state['rr_index'] = (idx + 1) % len(candidates)
            return candidates[idx:] + candidates[:idx]
        last = state.get('last_success')
        return sorted(self._keys, key=lambda k: (k in failed, k != last))

    def report_success(self, key: str) -> None:
        self._get_state()['last_success'] = key

    def report_failure(self, key: str) -> None:
        self._get_state().setdefault('failed', set()).add(key)

    def peek(self) -> str:
        state = self._get_state()
        failed = state.get('failed', set())
        candidates = [k for k in self._keys if k not in failed]
        if not candidates:
            return ''
        if self._policy == KeySelectPolicy.PREFER_LAST_SUCCESS:
            last = state.get('last_success')
            return last if (last and last not in failed) else candidates[0]
        return candidates[0]


@dataclass(frozen=True)
class Credential:
    kind: str = 'static'
    secret_key: Any = None
    access_token: str = ''
    token_expire_at: Optional[float] = None
    refresh_token: str = ''
    oauth_auto: bool = False
    key_pool: Optional['KeyPool'] = None

    def __post_init__(self) -> None:
        if self.kind not in _AUTH_KINDS:
            raise ValueError(f'Invalid Credential.kind: {self.kind!r}, must be one of {_AUTH_KINDS}')
        if self.key_pool is not None and self.kind not in ('static', 'dynamic'):
            raise ValueError(f'key_pool only allowed for static/dynamic, got {self.kind!r}')


class AuthStrategy(ABC):
    @abstractmethod
    def build_header(self, token: str) -> Dict[str, str]: ...

    @abstractmethod
    def build_params(self, token: str) -> Dict[str, str]: ...


class BearerTokenStrategy(AuthStrategy):

    def build_header(self, token: str) -> Dict[str, str]:
        return {'Authorization': f'Bearer {token}'} if token else {}

    def build_params(self, token: str) -> Dict[str, str]:
        return {}


class ApiKeyHeaderStrategy(AuthStrategy):

    def __init__(self, header_name: str, value_template: Optional[str] = None) -> None:
        self._header_name = header_name
        self._value_template = value_template

    def build_header(self, token: str) -> Dict[str, str]:
        if not token:
            return {}
        value = self._value_template.format(token=token) if self._value_template else token
        return {self._header_name: value}

    def build_params(self, token: str) -> Dict[str, str]:
        return {}


class QueryParamStrategy(AuthStrategy):

    def __init__(self, param_name: str) -> None:
        self._param_name = param_name

    def build_header(self, token: str) -> Dict[str, str]:
        return {}

    def build_params(self, token: str) -> Dict[str, str]:
        return {self._param_name: token} if token else {}
