import asyncio
import json
import mimetypes
import re
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests

from lazyllm.common import HandledException
from ....module import ModuleExecutionError


@dataclass(frozen=True)
class ModelCallControl:
    cancel_check: Optional[Callable[[], bool]] = field(default=None, repr=False)
    request_check: Optional[Callable] = field(default=None, repr=False)

    def check(self):
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
    is_estimate: bool = field(default=True, init=False)

    @property
    def payload(self):
        return json.loads(self._json)

    def estimate_tokens(self, text_estimator, message_estimator=None):
        data = self.payload
        messages = data.pop('messages', [])
        estimate_message = message_estimator or (
            lambda message: text_estimator(json.dumps(message, ensure_ascii=False)))
        return text_estimator(json.dumps({**data, 'messages': []}, ensure_ascii=False)) + sum(
            estimate_message(message) + 4 for message in messages)


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


@contextmanager
def model_response(url, *, control=None, **kwargs):
    if control is not None: control.check()
    try:
        with (requests.Session() if control is not None else nullcontext(requests)) as client, \
                client.post(url, **kwargs) as response:
            # urllib3 1.x otherwise accepts a truncated body as a clean EOF.
            response.raw.enforce_content_length = True
            if control is not None: control.check()
            yield response
            if control is not None: control.check()
    except BaseException:
        if control is not None: control.check()
        raise
