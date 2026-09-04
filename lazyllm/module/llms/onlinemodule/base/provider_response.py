from dataclasses import dataclass, field
import json
import re
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import uuid

import lazyllm
import requests

from .model_call_runner import ModelAttemptState, is_retryable_transport_error
from .model_outcome import (
    ModelFailure,
    ModelFailureCode,
    ModelFailureOrigin,
    ModelFinish,
    _ModelResponseError,
)


ProviderErrorFields = Optional[Tuple[Optional[str], Optional[str]]]


@dataclass(frozen=True)
class ProviderResponseProfile:
    code_map: Mapping[str, ModelFailureCode] = field(default_factory=dict)
    type_map: Mapping[str, ModelFailureCode] = field(default_factory=dict)
    http_map: Mapping[int, ModelFailureCode] = field(default_factory=dict)
    finish_map: Mapping[str, ModelFinish] = field(default_factory=dict)
    error_at_top_level: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'code_map', MappingProxyType(self._normalized_string_map(self.code_map)))
        object.__setattr__(self, 'type_map', MappingProxyType(self._normalized_string_map(self.type_map)))
        object.__setattr__(self, 'http_map', MappingProxyType(dict(self.http_map)))
        object.__setattr__(self, 'finish_map', MappingProxyType(dict(self.finish_map)))

    @staticmethod
    def _normalized_string_map(mapping: Optional[Mapping[str, ModelFailureCode]]) -> Dict[str, ModelFailureCode]:
        return {str(key).lower(): value for key, value in (mapping or {}).items()}

    def extend(
        self,
        *,
        code_map: Optional[Mapping[str, ModelFailureCode]] = None,
        type_map: Optional[Mapping[str, ModelFailureCode]] = None,
        http_map: Optional[Mapping[int, ModelFailureCode]] = None,
        finish_map: Optional[Mapping[str, ModelFinish]] = None,
        error_at_top_level: Optional[bool] = None,
    ) -> 'ProviderResponseProfile':
        return ProviderResponseProfile(
            code_map={**self.code_map, **self._normalized_string_map(code_map)},
            type_map={**self.type_map, **self._normalized_string_map(type_map)},
            http_map={**self.http_map, **(http_map or {})},
            finish_map={**self.finish_map, **(finish_map or {})},
            error_at_top_level=(
                self.error_at_top_level if error_at_top_level is None else error_at_top_level
            ),
        )

    def classify(
        self,
        *,
        origin: ModelFailureOrigin,
        provider_error_code: Optional[str] = None,
        provider_error_type: Optional[str] = None,
        provider_http_status: Optional[int] = None,
    ) -> ModelFailureCode:
        if origin == ModelFailureOrigin.PROTOCOL:
            return ModelFailureCode.PROTOCOL_ERROR
        if origin == ModelFailureOrigin.TRANSPORT:
            return ModelFailureCode.TRANSPORT_ERROR
        if provider_error_code:
            mapped = self.code_map.get(provider_error_code.lower())
            if mapped is not None: return mapped
        if provider_error_type:
            mapped = self.type_map.get(provider_error_type.lower())
            if mapped is not None: return mapped
        if provider_http_status is not None:
            mapped = self.http_map.get(provider_http_status)
            if mapped is not None: return mapped
        return ModelFailureCode.PROVIDER_REJECTED

    def extract_error(self, message: Dict[str, Any]) -> ProviderErrorFields:
        error = message.get('error')
        if isinstance(error, dict): return self._error_fields(error)
        if error is not None and not self.error_at_top_level: return None, None
        if self.error_at_top_level:
            fields = self._error_fields(message)
            if fields != (None, None): return fields
        return None

    def map_finish(self, raw_finish_reason: Any) -> ModelFinish:
        return self.finish_map.get(raw_finish_reason, ModelFinish.UNKNOWN)

    def error(
        self,
        message: str,
        origin: ModelFailureOrigin,
        *,
        provider_error_code: Optional[str] = None,
        provider_error_type: Optional[str] = None,
        provider_http_status: Optional[int] = None,
    ) -> _ModelResponseError:
        return _ModelResponseError(message, ModelFailure(
            origin=origin,
            code=self.classify(
                origin=origin,
                provider_error_code=provider_error_code,
                provider_error_type=provider_error_type,
                provider_http_status=provider_http_status,
            ),
            provider_error_code=provider_error_code,
            provider_error_type=provider_error_type,
            provider_http_status=provider_http_status,
            diagnostic_id=uuid.uuid4().hex,
        ))

    @staticmethod
    def _error_fields(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        code = payload.get('code')
        error_type = payload.get('type')
        return (
            str(code) if code is not None else None,
            str(error_type) if error_type is not None else None,
        )


OPENAI_COMPATIBLE_PROFILE = ProviderResponseProfile(
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
    finish_map={
        'stop': ModelFinish.STOP,
        'tool_calls': ModelFinish.TOOL_CALLS,
        'function_call': ModelFinish.TOOL_CALLS,
        'length': ModelFinish.LENGTH,
        'content_filter': ModelFinish.CONTENT_FILTER,
    },
)


def raise_for_http_error(response: Any, profile: ProviderResponseProfile) -> None:
    if response.status_code == 200: return
    try:
        error_message = response.json()
    except (TypeError, ValueError, requests.RequestException):
        error_message = {}
    fields = profile.extract_error(error_message) if isinstance(error_message, dict) else None
    code, error_type = fields or (None, None)
    raise profile.error(
        f'Provider returned HTTP status {response.status_code}.',
        ModelFailureOrigin.HTTP,
        provider_error_code=code,
        provider_error_type=error_type,
        provider_http_status=response.status_code,
    )


def select_primary_choice(choices: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not choices: return None
    if any(choice.get('index') is not None for choice in choices):
        return next((choice for choice in choices if choice.get('index') == 0), None)
    return choices[0]


class _OpenAICompatibleResponseParser:
    def __init__(
        self,
        *,
        profile: ProviderResponseProfile,
        convert_message: Callable[[Dict[str, Any]], Any],
        emit_message: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self._profile = profile
        self._convert_message = convert_message
        self._emit_message = emit_message

    def extract_sse_payload(self, raw_frame: Union[str, bytes]) -> Optional[str]:
        if isinstance(raw_frame, bytes):
            try:
                raw_frame = raw_frame.decode('utf-8')
            except UnicodeDecodeError as exc:
                raise self._profile.error(
                    'Provider returned a non-UTF-8 frame.',
                    ModelFailureOrigin.PROTOCOL,
                ) from exc
        content = raw_frame.strip()
        if not content or content.startswith(':') or re.match(r'^(event|id|retry):', content): return None
        return re.sub(r'^data:\s?', '', content)

    def parse_json_payload(self, payload: str) -> Union[Dict[str, Any], str]:
        try:
            raw_message = json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise self._profile.error(
                'Provider returned an invalid JSON frame.', ModelFailureOrigin.PROTOCOL,
            ) from exc
        if not isinstance(raw_message, dict):
            raise self._profile.error(
                'Provider returned a non-object JSON frame.', ModelFailureOrigin.PROTOCOL,
            )
        error_fields = self._extract_provider_error(raw_message)
        if error_fields is not None:
            code, error_type = error_fields
            raise self._profile.error(
                'Provider returned an error frame.',
                ModelFailureOrigin.PROVIDER,
                provider_error_code=code,
                provider_error_type=error_type,
            )
        try:
            message = self._convert_message(raw_message)
        except Exception as exc:
            raise self._profile.error(
                'Provider response conversion failed.', ModelFailureOrigin.PROTOCOL,
            ) from exc
        if not isinstance(message, dict):
            if message in ('', None): return ''
            raise self._profile.error(
                'Provider response conversion returned an invalid frame.',
                ModelFailureOrigin.PROTOCOL,
            )
        if self._emit_message is not None: self._emit_message(message)
        lazyllm.LOG.debug(f'message: {message}')
        return message

    def parse_response_frame(self, raw_frame: Union[str, bytes]) -> Union[Dict[str, Any], str]:
        payload = self.extract_sse_payload(raw_frame)
        if payload is None or payload == '[DONE]': return ''
        return self.parse_json_payload(payload)

    def _consume_frame(
        self,
        raw_frame: Union[str, bytes],
        messages: List[Dict[str, Any]],
        state: Optional[ModelAttemptState],
    ) -> bool:
        if raw_frame in (b'', ''): return False
        payload = self.extract_sse_payload(raw_frame)
        if payload is None: return False
        if payload == '[DONE]':
            if state is None or state.finish is None:
                raise self._profile.error(
                    'Provider stream ended without a finish_reason.',
                    ModelFailureOrigin.PROTOCOL,
                )
            return True
        message = self.parse_json_payload(payload)
        if not message: return False
        messages.append(message)
        if state is not None: self.update_attempt_state(message, state)
        return False

    def collect(
        self,
        frames: Any,
        state: Optional[ModelAttemptState],
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if state is not None: state.frames = messages
        frame_iter = iter(frames)
        while True:
            try:
                raw_frame = next(frame_iter)
            except StopIteration:
                break
            except Exception as exc:
                if state is not None and state.finish is not None and is_retryable_transport_error(exc):
                    break
                raise
            if self._consume_frame(raw_frame, messages, state):
                break
        if state is None or state.finish is None:
            raise self._profile.error(
                'Provider response ended without a finish_reason.',
                ModelFailureOrigin.PROTOCOL,
            )
        return messages

    def update_attempt_state(self, message: Dict[str, Any], state: ModelAttemptState) -> None:
        choices = message.get('choices')
        if choices is None: return
        if not isinstance(choices, list):
            raise self._profile.error(
                'Provider response has an invalid choices field.', ModelFailureOrigin.PROTOCOL,
            )
        for choice in choices:
            if not isinstance(choice, dict):
                raise self._profile.error(
                    'Provider response contains an invalid choice.', ModelFailureOrigin.PROTOCOL,
                )
            output = choice.get('message') or choice.get('delta') or {}
            if not isinstance(output, dict):
                raise self._profile.error(
                    'Provider response contains an invalid choice payload.',
                    ModelFailureOrigin.PROTOCOL,
                )
        choice = select_primary_choice(choices)
        if choice is None: return
        output = choice.get('message') or choice.get('delta') or {}
        if (output.get('content') or output.get('reasoning_content')
                or output.get('tool_calls') or output.get('function_call')):
            state.semantic_output = True
        raw_finish_reason = choice.get('finish_reason')
        if raw_finish_reason not in (None, ''):
            state.finish = self._profile.map_finish(raw_finish_reason)

    def _extract_provider_error(self, message: Dict[str, Any]) -> ProviderErrorFields:
        return self._profile.extract_error(message)


def usage_from_frames(msg_json: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    if not isinstance(msg_json, list):
        return None
    for frame in reversed(msg_json):
        if isinstance(frame, dict) and isinstance(frame.get('usage'), dict):
            return frame['usage']
    return None
