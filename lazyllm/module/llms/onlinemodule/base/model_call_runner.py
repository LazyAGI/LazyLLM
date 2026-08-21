import random
import socket
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import requests

from .model_outcome import (
    ModelCallError,
    ModelCallTerminal,
    ModelFailure,
    ModelFailureCode,
    ModelFailureOrigin,
    ModelFinish,
    _ModelResponseError,
)


def exception_chain(exc: Exception):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def is_retryable_transport_error(exc: Exception) -> bool:
    chain = tuple(exception_chain(exc))
    if any(isinstance(item, (requests.exceptions.SSLError, requests.exceptions.ProxyError)) for item in chain):
        return False
    dns_errors = tuple(item for item in chain if isinstance(item, socket.gaierror))
    if dns_errors:
        return any(item.errno == socket.EAI_AGAIN for item in dns_errors)
    retryable_types = (
        requests.exceptions.ConnectTimeout,
        requests.exceptions.ReadTimeout,
        requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ConnectionError,
        ConnectionResetError,
        ConnectionAbortedError,
        BrokenPipeError,
    )
    retryable_names = {'RemoteDisconnected', 'IncompleteRead', 'ProtocolError'}
    return any(isinstance(item, retryable_types) or type(item).__name__ in retryable_names for item in chain)


@dataclass
class ModelAttemptState:
    semantic_output: bool = False
    response_started: bool = False
    finish: Optional[ModelFinish] = None
    frames: List[dict] = field(default_factory=list)


class _ModelCallRunner:
    def __init__(self, *, emit_event: Callable[[str, dict], None],
                 is_retryable_transport_error: Callable[[Exception], bool] = is_retryable_transport_error,
                 report_failure: Optional[Callable[[ModelFailure], None]] = None,
                 sleep: Callable[[float], None] = time.sleep):
        self._emit_event = emit_event
        self._is_retryable_transport_error = is_retryable_transport_error
        self._report_failure = report_failure
        self._sleep = sleep

    def run(self, execute_attempt: Callable[[ModelAttemptState], List[dict]], *, max_attempts: int) -> List[dict]:
        model_call_id = uuid.uuid4().hex
        max_attempts = max(1, int(max_attempts))
        semantic_output = False
        for attempt_index in range(1, max_attempts + 1):
            state = ModelAttemptState()
            try:
                result = execute_attempt(state)
                state.frames = result
                semantic_output = semantic_output or state.semantic_output
                terminal = ModelCallTerminal(
                    model_call_id=model_call_id,
                    attempt_count=attempt_index,
                    kind='finish',
                    finish=state.finish,
                    has_semantic_output=semantic_output,
                )
                self._emit_event('model_call_finished', terminal.public_dict())
                if state.finish in (ModelFinish.STOP, ModelFinish.TOOL_CALLS):
                    return result
                raise ModelCallError(
                    'Model call ended without a complete answer.', terminal, state.frames,
                )
            except ModelCallError:
                raise
            except _ModelResponseError as exc:
                semantic_output = semantic_output or state.semantic_output
                exc.failure.response_started = state.response_started
                self._report(exc.failure)
                terminal = ModelCallTerminal(
                    model_call_id=model_call_id,
                    attempt_count=attempt_index,
                    kind='failure',
                    failure=exc.failure,
                    has_semantic_output=semantic_output,
                )
                self._emit_event('model_call_finished', terminal.public_dict())
                raise ModelCallError(str(exc), terminal, state.frames) from exc
            except Exception as exc:
                semantic_output = semantic_output or state.semantic_output
                retryable = self._is_retryable_transport_error(exc)
                if retryable and not state.response_started and not semantic_output and attempt_index < max_attempts:
                    delay = self._retry_delay(attempt_index)
                    self._emit_event('model_retry_scheduled', {
                        'model_call_id': model_call_id,
                        'retry_index': attempt_index,
                        'max_attempts': max_attempts,
                        'delay_ms': int(delay * 1000),
                    })
                    self._sleep(delay)
                    continue
                failure = ModelFailure(
                    origin=ModelFailureOrigin.TRANSPORT,
                    code=ModelFailureCode.TRANSPORT_ERROR,
                    diagnostic_id=uuid.uuid4().hex,
                    response_started=state.response_started,
                )
                self._report(failure)
                terminal = ModelCallTerminal(
                    model_call_id=model_call_id,
                    attempt_count=attempt_index,
                    kind='failure',
                    failure=failure,
                    has_semantic_output=semantic_output,
                )
                self._emit_event('model_call_finished', terminal.public_dict())
                raise ModelCallError('Model transport failed.', terminal, state.frames) from exc
        raise AssertionError('model call runner exhausted without a terminal')

    def _report(self, failure: ModelFailure) -> None:
        if self._report_failure is not None:
            self._report_failure(failure)

    @staticmethod
    def _retry_delay(retry_index: int) -> float:
        base = min(2 ** (retry_index - 1), 16)
        return random.uniform(base * 0.8, base * 1.2)
