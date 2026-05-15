# Copyright (c) 2026 LazyAGI. All rights reserved.
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable

_AUTH_KINDS = ('static', 'dynamic', 'oauth2', 'app_credentials')


@dataclass(frozen=True)
class Credential:
    kind: str = 'static'
    secret_key: Any = None
    access_token: str = ''
    token_expire_at: Optional[float] = None
    refresh_token: str = ''
    oauth_auto: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _AUTH_KINDS:
            raise ValueError(f'Invalid Credential.kind: {self.kind!r}, must be one of {_AUTH_KINDS}')


@runtime_checkable
class AuthStrategy(Protocol):
    def build_header(self, token: str) -> Dict[str, str]: ...
    def build_params(self, token: str) -> Dict[str, str]: ...


class BearerTokenStrategy:

    def build_header(self, token: str) -> Dict[str, str]:
        return {'Authorization': f'Bearer {token}'} if token else {}

    def build_params(self, token: str) -> Dict[str, str]:
        return {}


class ApiKeyHeaderStrategy:

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


class QueryParamStrategy:

    def __init__(self, param_name: str) -> None:
        self._param_name = param_name

    def build_header(self, token: str) -> Dict[str, str]:
        return {}

    def build_params(self, token: str) -> Dict[str, str]:
        return {self._param_name: token} if token else {}
