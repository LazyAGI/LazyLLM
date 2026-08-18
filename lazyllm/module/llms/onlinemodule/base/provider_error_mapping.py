from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping, Optional

from .model_outcome import ModelFailureCode


@dataclass(frozen=True)
class ProviderErrorMapping:
    code_map: Mapping[str, ModelFailureCode]
    type_map: Mapping[str, ModelFailureCode]
    http_map: Mapping[int, ModelFailureCode]


_PROVIDER_ERROR_MAPPINGS: Dict[str, ProviderErrorMapping] = {}


def _normalized_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError('Provider error mapping name cannot be empty.')
    return normalized


def _normalized_string_map(mapping: Optional[Mapping[str, ModelFailureCode]]) -> Dict[str, ModelFailureCode]:
    return {str(key).lower(): value for key, value in (mapping or {}).items()}


def register_provider_error_mapping(
    name: str,
    *,
    extends: Optional[str] = None,
    code_map: Optional[Mapping[str, ModelFailureCode]] = None,
    type_map: Optional[Mapping[str, ModelFailureCode]] = None,
    http_map: Optional[Mapping[int, ModelFailureCode]] = None,
) -> ProviderErrorMapping:
    name = _normalized_name(name)
    if name in _PROVIDER_ERROR_MAPPINGS:
        raise ValueError(f'Provider error mapping {name!r} is already registered.')

    if extends is None:
        merged_code_map: Dict[str, ModelFailureCode] = {}
        merged_type_map: Dict[str, ModelFailureCode] = {}
        merged_http_map: Dict[int, ModelFailureCode] = {}
    else:
        parent = get_provider_error_mapping(extends)
        merged_code_map = dict(parent.code_map)
        merged_type_map = dict(parent.type_map)
        merged_http_map = dict(parent.http_map)

    merged_code_map.update(_normalized_string_map(code_map))
    merged_type_map.update(_normalized_string_map(type_map))
    merged_http_map.update(http_map or {})
    mapping = ProviderErrorMapping(
        code_map=MappingProxyType(merged_code_map),
        type_map=MappingProxyType(merged_type_map),
        http_map=MappingProxyType(merged_http_map),
    )
    _PROVIDER_ERROR_MAPPINGS[name] = mapping
    return mapping


def get_provider_error_mapping(name: str) -> ProviderErrorMapping:
    normalized = _normalized_name(name)
    try:
        return _PROVIDER_ERROR_MAPPINGS[normalized]
    except KeyError as exc:
        raise KeyError(f'Provider error mapping {normalized!r} is not registered.') from exc


register_provider_error_mapping(
    'openai_compatible',
    http_map={
        400: ModelFailureCode.INVALID_REQUEST,
        401: ModelFailureCode.AUTHENTICATION_FAILED,
        403: ModelFailureCode.PERMISSION_DENIED,
        404: ModelFailureCode.NOT_FOUND,
        408: ModelFailureCode.REQUEST_TIMEOUT,
        409: ModelFailureCode.CONFLICT,
        422: ModelFailureCode.UNPROCESSABLE_ENTITY,
        429: ModelFailureCode.TOO_MANY_REQUESTS,
        500: ModelFailureCode.PROVIDER_INTERNAL_ERROR,
        502: ModelFailureCode.PROVIDER_INTERNAL_ERROR,
        503: ModelFailureCode.SERVICE_UNAVAILABLE,
        504: ModelFailureCode.REQUEST_TIMEOUT,
    },
)
