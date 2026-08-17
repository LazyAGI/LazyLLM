import random
import time
import uuid
from dataclasses import dataclass
from typing import Callable, List, Optional

from .model_outcome import (
    ModelCallError,
    ModelCallFailed,
    ModelCallInterrupted,
    ModelCallTerminal,
    ModelFailure,
    ModelFailureOrigin,
    ModelFinish,
    ModelResponseError,
)


@dataclass
class ModelAttemptState:
    semantic_output: bool = False
    finish: Optional[ModelFinish] = None
    raw_finish_reason: Optional[str] = None


class ModelCallRunner:
    def __init__(self, *, emit_event: Callable[[str, dict], None],
                 is_retryable_transport_error: Callable[[Exception], bool],
                 sleep: Callable[[float], None] = time.sleep):
        self._emit_event = emit_event
        self._is_retryable_transport_error = is_retryable_transport_error
        self._sleep = sleep

    def run(self, execute_attempt: Callable[[ModelAttemptState], List[dict]], *, max_attempts: int) -> List[dict]:
        model_call_id = uuid.uuid4().hex
        max_attempts = max(1, int(max_attempts))
        semantic_output = False
        for attempt_index in range(1, max_attempts + 1):
            state = ModelAttemptState()
            try:
                result = execute_attempt(state)
                semantic_output = semantic_output or state.semantic_output
                terminal = ModelCallTerminal(
                    model_call_id=model_call_id,
                    attempt_count=attempt_index,
                    kind='finish',
                    finish=state.finish,
                    raw_finish_reason=state.raw_finish_reason,
                    has_semantic_output=semantic_output,
                )
                self._emit_event('model_call_finished', terminal.public_dict())
                if state.finish in (ModelFinish.STOP, ModelFinish.TOOL_CALLS):
                    return result
                raise ModelCallInterrupted('Model call ended without a complete answer.', terminal)
            except ModelCallError:
                raise
            except ModelResponseError as exc:
                semantic_output = semantic_output or state.semantic_output
                exc.failure.has_semantic_output = semantic_output
                terminal = ModelCallTerminal(
                    model_call_id=model_call_id,
                    attempt_count=attempt_index,
                    kind='failure',
                    failure=exc.failure,
                    has_semantic_output=semantic_output,
                )
                self._emit_event('model_call_finished', terminal.public_dict())
                error_cls = ModelCallInterrupted if semantic_output else ModelCallFailed
                raise error_cls(str(exc), terminal) from exc
            except Exception as exc:
                semantic_output = semantic_output or state.semantic_output
                retryable = self._is_retryable_transport_error(exc)
                if retryable and not semantic_output and attempt_index < max_attempts:
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
                    diagnostic_id=uuid.uuid4().hex,
                    has_semantic_output=semantic_output,
                )
                terminal = ModelCallTerminal(
                    model_call_id=model_call_id,
                    attempt_count=attempt_index,
                    kind='failure',
                    failure=failure,
                    has_semantic_output=semantic_output,
                )
                self._emit_event('model_call_finished', terminal.public_dict())
                error_cls = ModelCallInterrupted if semantic_output else ModelCallFailed
                raise error_cls('Model transport failed.', terminal) from exc
        raise AssertionError('model call runner exhausted without a terminal')

    @staticmethod
    def _retry_delay(retry_index: int) -> float:
        base = min(2 ** (retry_index - 1), 16)
        return random.uniform(base * 0.8, base * 1.2)
