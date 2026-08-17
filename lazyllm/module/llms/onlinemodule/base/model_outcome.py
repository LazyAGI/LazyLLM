from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class ModelFinish(str, Enum):
    STOP = 'stop'
    TOOL_CALLS = 'tool_calls'
    LENGTH = 'length'
    CONTENT_FILTER = 'content_filter'
    UNKNOWN = 'unknown'


class ModelFailureOrigin(str, Enum):
    TRANSPORT = 'transport'
    HTTP = 'http'
    PROTOCOL = 'protocol'
    CANCELLED = 'cancelled'


@dataclass
class ModelFailure:
    origin: ModelFailureOrigin
    provider_error_code: Optional[str] = None
    diagnostic_id: Optional[str] = None
    has_semantic_output: bool = False

    def public_dict(self) -> Dict[str, Any]:
        result = {
            'origin': self.origin.value,
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


class ModelCallError(Exception):
    def __init__(self, message: str, terminal: ModelCallTerminal):
        super().__init__(message)
        self.terminal = terminal


class ModelCallInterrupted(ModelCallError): pass
class ModelCallFailed(ModelCallError): pass
