import asyncio
import json
import mimetypes
import re
import socket
import time
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests

from lazyllm.common import HandledException
from ....module import ModuleExecutionError
from .protocol_ir import RequestIR


class _ModelCallDeadlineExceeded(ModuleExecutionError):
    pass


@dataclass(frozen=True)
class ModelCallControl:
    cancel_check: Optional[Callable[[], bool]] = field(default=None, repr=False)
    request_check: Optional[Callable] = field(default=None, repr=False)
    deadline: Optional[float] = None

    def check(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise _ModelCallDeadlineExceeded('Model call deadline exceeded.')
        if self.cancel_check is not None:
            try:
                cancelled = self.cancel_check()
            except Exception as exc:
                raise ModuleExecutionError('Model call cancellation check failed.') from exc
            if cancelled:
                raise asyncio.CancelledError('Model call cancelled.')

    def validate_request(self, request):
        self.check()
        if self.request_check is not None:
            try:
                self.request_check(request)
            except HandledException:
                raise
            except Exception as exc:
                raise ModuleExecutionError('Model request validation failed.') from exc
        self.check()

    def wait(self, duration):
        until = time.monotonic() + duration
        while True:
            self.check()
            remaining = until - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.05, remaining))


class _ControlledModelCall:
    def _call_impl(self, *args, **kwargs):
        control = kwargs.get('call_control')
        if control is not None: control.check()
        result = super()._call_impl(*args, **kwargs)
        if control is not None: control.check()
        return result


def _requires_vision(value, files):
    def contains_image(item):
        if isinstance(item, dict):
            return item.get('type') in {'image_url', 'input_image', 'image'} or any(
                contains_image(child) for child in item.values())
        return isinstance(item, (list, tuple)) and any(contains_image(child) for child in item)
    return contains_image(value) or any(
        (mimetypes.guess_type(str(file).split('?', 1)[0])[0] or '').startswith('image/') for file in files or [])


@dataclass(frozen=True)
class ModelRequest:
    _json: str = field(repr=False)
    model_type: str
    requires_vision: bool = False
    _semantic: Optional[RequestIR] = field(default=None, repr=False)
    is_estimate: bool = field(default=True, init=False)

    @property
    def payload(self):
        return json.loads(self._json)

    def estimate_tokens(self, text_estimator, message_estimator=None):
        return self.ir.estimate_tokens(text_estimator, message_estimator)

    @property
    def ir(self):
        return deepcopy(self._semantic) if self._semantic is not None else RequestIR.from_payload(self.payload)


def response_chunks(response, control):
    raw = getattr(response, 'raw', None)
    read = getattr(raw, 'read', None)
    fp_read = getattr(raw, '_fp_read', None)
    # Keep Requests' decoding/error path, but do not wait to fill a buffer for
    # non-chunked SSE. These changes belong only to this response iterator.
    if callable(read1 := getattr(raw, 'read1', None)):
        raw.read = read1
    elif fp_read is not None and callable(read1 := getattr(getattr(raw, '_fp', None), 'read1', None)):
        raw._fp_read = read1
        if hasattr(raw, '_raw_read'):
            # urllib3 2.0 also fills a decoded buffer; 1.26's read already
            # returns after one _fp_read. Later 2.x provides read1 directly.
            def read_available(amt, decode_content=True):
                raw._init_decoder()
                data = raw._raw_read(amt)
                return raw._decode(data, decode_content, not data)
            raw.read = read_available
    try:
        for chunk in response.iter_content(chunk_size=8192):
            control.check()
            yield chunk
            control.check()
    finally:
        if read is not None: raw.read = read
        if fp_read is not None: raw._fp_read = fp_read
    control.check()


def response_frames(response, stream, control):
    if control is None:
        yield from response.iter_lines() if stream else [response.text]
        return
    if not stream:
        content = b''.join(response_chunks(response, control))
        yield content.decode(response.encoding or 'utf-8', errors='replace')
        return
    pending = b''
    for chunk in response_chunks(response, control):
        lines = re.split(b'[\r\n]', pending + chunk)
        pending = lines.pop()
        for line in lines:
            control.check()
            yield line
    if pending:
        yield pending


class _ControlledReadSocket:
    def __init__(self, sock, control):
        self._sock, self._control = sock, control

    def __getattr__(self, name):
        return getattr(self._sock, name)

    def recv_into(self, *args, **kwargs):
        timeout = self._sock.gettimeout()
        expires = None if timeout is None else time.monotonic() + timeout
        try:
            while True:
                self._control.check()
                remaining = None if expires is None else expires - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise socket.timeout('timed out')
                interval = 0.05 if remaining is None else min(0.05, remaining)
                if self._control.deadline is not None:
                    interval = min(interval, max(0, self._control.deadline - time.monotonic()))
                self._sock.settimeout(interval)
                try:
                    return self._sock.recv_into(*args, **kwargs)
                except socket.timeout:
                    # Poll below SocketIO: allowing the poll timeout to escape
                    # would poison its buffered reader and lose partial headers.
                    self._control.check()
                    if expires is not None and time.monotonic() >= expires:
                        raise
        finally:
            self._sock.settimeout(timeout)


class _ControlledHTTPAdapter(requests.adapters.HTTPAdapter):
    def __init__(self, control):
        self._control = control
        self._connection_classes = {}
        super().__init__()

    def _controlled_pool(self, pool):
        base = pool.ConnectionCls
        if base not in self._connection_classes.values():
            if base not in self._connection_classes:
                control = self._control

                class ControlledResponse(base.response_class):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        # HTTP and TLS makefile readers both use SocketIO. Check
                        # cancellation in the reading thread; Windows timed reads
                        # cannot rely on another thread's shutdown waking select.
                        raw = self.fp.raw
                        raw._sock = _ControlledReadSocket(raw._sock, control)

                class ControlledConnection(base):
                    response_class = ControlledResponse

                self._connection_classes[base] = ControlledConnection
            # Pools belong to this request's Session, including proxy/SOCKS pools.
            pool.ConnectionCls = self._connection_classes[base]
        return pool

    def get_connection(self, *args, **kwargs):
        return self._controlled_pool(super().get_connection(*args, **kwargs))

    def get_connection_with_tls_context(self, *args, **kwargs):
        return self._controlled_pool(super().get_connection_with_tls_context(*args, **kwargs))


@contextmanager
def model_response(url, *, control=None, **kwargs):
    if control is not None: control.check()
    if control is not None and control.deadline is not None:
        remaining = control.deadline - time.monotonic()
        timeout = kwargs.get('timeout')
        if isinstance(timeout, requests.adapters.TimeoutSauce):
            timeout = timeout.clone()
        elif isinstance(timeout, tuple):
            timeout = requests.adapters.TimeoutSauce(connect=timeout[0], read=timeout[1])
        else:
            timeout = requests.adapters.TimeoutSauce.from_float(timeout)
        timeout.total = min(timeout.total, remaining) if timeout.total is not None else remaining
        kwargs['timeout'] = timeout
    try:
        with (requests.Session() if control is not None else nullcontext(requests)) as client:
            if control is not None and (control.deadline is not None or control.cancel_check is not None):
                adapter = _ControlledHTTPAdapter(control)
                client.mount('http://', adapter)
                client.mount('https://', adapter)
            with client.post(url, **kwargs) as response:
                # urllib3 1.x otherwise accepts a truncated body as a clean EOF.
                response.raw.enforce_content_length = True
                if control is not None: control.check()
                yield response
                if control is not None: control.check()
    except BaseException:
        if control is not None: control.check()
        raise
