from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
import time


class FSReadLimitError(ValueError):
    pass


@dataclass
class _ReadBudget:
    deadline: float
    remaining_bytes: int
    remaining_requests: int

    def remaining_seconds(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Cloud read deadline exceeded')
        return remaining


_read_budget = ContextVar('fs_read_budget', default=None)


@contextmanager
def fs_read_limits(*, seconds: float = 20, max_bytes: int = 16 * 1024 * 1024, max_requests: int = 256):
    if not math.isfinite(seconds) or seconds <= 0 or max_bytes <= 0 or max_requests <= 0:
        raise ValueError('Read limits must be finite and positive')
    token = _read_budget.set(_ReadBudget(time.monotonic() + seconds, max_bytes, max_requests))
    try:
        yield
    finally:
        _read_budget.reset(token)


def _bounded_request(session, method, url, **kwargs):
    budget = _read_budget.get()
    if budget is None:
        return session.request(method, url, **kwargs)
    remaining = budget.remaining_seconds()
    budget.remaining_requests -= 1
    if budget.remaining_requests < 0:
        raise FSReadLimitError('Cloud read request limit exceeded')
    timeout = kwargs.get('timeout')
    values = timeout if isinstance(timeout, tuple) else (timeout, timeout)
    kwargs['timeout'] = tuple(min(value or 5, 5, remaining) for value in values)
    # Redirect bodies must not be eagerly downloaded outside the byte budget.
    kwargs.update(stream=True, allow_redirects=False)
    response = session.request(method, url, **kwargs)
    try:
        if 300 <= response.status_code < 400:
            raise FSReadLimitError('Cloud read redirects are not supported')
        length = response.headers.get('Content-Length', '')
        if length.isdigit() and int(length) > budget.remaining_bytes:
            raise FSReadLimitError('Cloud read byte limit exceeded')
        content = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            budget.remaining_seconds()
            budget.remaining_bytes -= len(chunk)
            if budget.remaining_bytes < 0:
                raise FSReadLimitError('Cloud read byte limit exceeded')
            content.extend(chunk)
        budget.remaining_seconds()
        response._content = bytes(content)
        response._content_consumed = True
        return response
    finally:
        response.close()
