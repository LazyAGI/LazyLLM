from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from lazyllm.common import HandledException


class ModelFinish(str, Enum):
    STOP = 'stop'
    TOOL_CALLS = 'tool_calls'
    LENGTH = 'length'
    CONTENT_FILTER = 'content_filter'
    INSUFFICIENT_SYSTEM_RESOURCE = 'insufficient_system_resource'
    UNKNOWN = 'unknown'


class ModelFailureOrigin(str, Enum):
    TRANSPORT = 'transport'
    HTTP = 'http'
    PROVIDER = 'provider'
    PROTOCOL = 'protocol'
    CANCELLED = 'cancelled'


class ModelFailureCode(str, Enum):
    INVALID_REQUEST = 'invalid_request'
    AUTHENTICATION_FAILED = 'authentication_failed'
    PERMISSION_DENIED = 'permission_denied'
    NOT_FOUND = 'not_found'
    RATE_LIMITED = 'rate_limited'
    TOO_MANY_REQUESTS = 'too_many_requests'
    QUOTA_EXHAUSTED = 'quota_exhausted'
    BALANCE_EXHAUSTED = 'balance_exhausted'
    ORGANIZATION_SPEND_LIMIT_EXCEEDED = 'organization_spend_limit_exceeded'
    PROJECT_SPEND_LIMIT_EXCEEDED = 'project_spend_limit_exceeded'
    ORGANIZATION_USAGE_LIMIT_EXCEEDED = 'organization_usage_limit_exceeded'
    INPUT_FILTERED = 'input_filtered'
    OUTPUT_FILTERED = 'output_filtered'
    TOKEN_LIMIT = 'token_limit'
    REQUEST_TIMEOUT = 'request_timeout'
    PROVIDER_OVERLOADED = 'provider_overloaded'
    SERVICE_UNAVAILABLE = 'service_unavailable'
    PROVIDER_INTERNAL_ERROR = 'provider_internal_error'
    PROVIDER_REJECTED = 'provider_rejected'
    CONFLICT = 'conflict'
    UNPROCESSABLE_ENTITY = 'unprocessable_entity'
    PROTOCOL_ERROR = 'protocol_error'
    TRANSPORT_ERROR = 'transport_error'


@dataclass
class ModelFailure:
    origin: ModelFailureOrigin
    code: ModelFailureCode
    provider_error_code: Optional[str] = None
    provider_error_type: Optional[str] = None
    provider_http_status: Optional[int] = None
    retry_after_ms: Optional[int] = None
    diagnostic_id: Optional[str] = None
    has_semantic_output: bool = False
    response_started: bool = False

    def public_dict(self) -> Dict[str, Any]:
        result = {
            'origin': self.origin.value,
            'code': self.code.value,
            'has_semantic_output': self.has_semantic_output,
        }
        if self.diagnostic_id:
            result['diagnostic_id'] = self.diagnostic_id
        return result


@dataclass
class ModelCallTerminal:
    model_call_id: str
    attempt_count: int
    kind: str
    has_semantic_output: bool
    finish: Optional[ModelFinish] = None
    raw_finish_reason: Optional[str] = None
    failure: Optional[ModelFailure] = None

    def public_dict(self) -> Dict[str, Any]:
        result = {
            'model_call_id': self.model_call_id,
            'attempt_count': self.attempt_count,
            'kind': self.kind,
            'has_semantic_output': self.has_semantic_output,
        }
        if self.finish is not None:
            result['finish'] = self.finish.value
        if self.failure is not None:
            result['failure'] = self.failure.public_dict()
        return result


class ModelResponseError(Exception):
    def __init__(self, message: str, failure: ModelFailure):
        super().__init__(message)
        self.failure = failure


class ModelCallError(HandledException):
    def __init__(self, message: str, terminal: ModelCallTerminal,
                 partial_response: Optional[List[dict]] = None):
        super().__init__(message)
        self.terminal = terminal
        self.partial_response = list(partial_response or [])
        self.usage: Optional[Dict[str, int]] = None


class ModelCallInterrupted(ModelCallError): pass
class ModelCallFailed(ModelCallError): pass
