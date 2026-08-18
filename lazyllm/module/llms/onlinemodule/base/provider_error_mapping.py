from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Mapping, Optional

from .model_outcome import ModelFailureCode


@dataclass(frozen=True)
class ProviderErrorProfile:
    code_map: Mapping[str, ModelFailureCode] = field(default_factory=dict)
    type_map: Mapping[str, ModelFailureCode] = field(default_factory=dict)
    http_map: Mapping[int, ModelFailureCode] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, 'code_map', MappingProxyType(self._normalized_string_map(self.code_map)))
        object.__setattr__(self, 'type_map', MappingProxyType(self._normalized_string_map(self.type_map)))
        object.__setattr__(self, 'http_map', MappingProxyType(dict(self.http_map)))

    @staticmethod
    def _normalized_string_map(mapping: Optional[Mapping[str, ModelFailureCode]]) -> Dict[str, ModelFailureCode]:
        return {str(key).lower(): value for key, value in (mapping or {}).items()}

    def extend(
        self,
        *,
        code_map: Optional[Mapping[str, ModelFailureCode]] = None,
        type_map: Optional[Mapping[str, ModelFailureCode]] = None,
        http_map: Optional[Mapping[int, ModelFailureCode]] = None,
    ) -> 'ProviderErrorProfile':
        merged_code_map = dict(self.code_map)
        merged_type_map = dict(self.type_map)
        merged_http_map = dict(self.http_map)
        merged_code_map.update(self._normalized_string_map(code_map))
        merged_type_map.update(self._normalized_string_map(type_map))
        merged_http_map.update(http_map or {})
        return ProviderErrorProfile(
            code_map=merged_code_map,
            type_map=merged_type_map,
            http_map=merged_http_map,
        )


OPENAI_COMPATIBLE_PROFILE = ProviderErrorProfile(
    http_map={
        400: ModelFailureCode.INVALID_REQUEST,
        401: ModelFailureCode.AUTHENTICATION_FAILED,
        403: ModelFailureCode.PERMISSION_DENIED,
        404: ModelFailureCode.NOT_FOUND,
        408: ModelFailureCode.REQUEST_TIMEOUT,
        409: ModelFailureCode.CONFLICT,
        422: ModelFailureCode.UNPROCESSABLE_ENTITY,
        429: ModelFailureCode.RATE_LIMITED,
        500: ModelFailureCode.PROVIDER_INTERNAL_ERROR,
        502: ModelFailureCode.PROVIDER_INTERNAL_ERROR,
        503: ModelFailureCode.SERVICE_UNAVAILABLE,
        504: ModelFailureCode.REQUEST_TIMEOUT,
    },
)
