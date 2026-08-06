from __future__ import annotations

from contextvars import copy_context
from queue import Empty, Queue
import re
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

from lazyllm.common import ThreadPoolExecutor
from lazyllm.configs import config

_DRAFT_STREAM_EXECUTOR = ThreadPoolExecutor(max_workers=config['thread_pool_worker_num'])


class MarkdownStreamNormalizer:
    _think_open_pattern = re.compile(r'^<think\b[^>]*>', re.IGNORECASE)
    _think_close_pattern = re.compile(r'</think\s*>', re.IGNORECASE)

    def __init__(self):
        self._leading = ''
        self._trailing = ''
        self._started = False
        self._parts: List[str] = []

    @property
    def body(self) -> str:
        return ''.join(self._parts)

    def feed(self, text: str) -> List[str]:
        if not text:
            return []
        if self._started:
            return self._feed_body(text)
        self._leading += text
        return self._drain_leading()

    def finish(self) -> List[str]:
        deltas = self._drain_leading(final=True) if not self._started else []
        self._trailing = ''
        return deltas

    def _drain_leading(self, final: bool = False) -> List[str]:
        while True:
            stripped = self._leading.lstrip()
            if not stripped:
                if final:
                    self._leading = ''
                return []

            open_match = self._think_open_pattern.match(stripped)
            if open_match:
                close_match = self._think_close_pattern.search(stripped, open_match.end())
                if close_match is None:
                    if not final:
                        return []
                    self._started = True
                    self._leading = ''
                    return self._feed_body(stripped)
                self._leading = stripped[close_match.end():]
                continue

            if not final and self._could_be_think_prefix(stripped):
                return []

            self._started = True
            self._leading = ''
            return self._feed_body(stripped)

    @staticmethod
    def _could_be_think_prefix(text: str) -> bool:
        lowered = text.lower()
        if '<think'.startswith(lowered):
            return True
        return bool(re.match(r'^<think\b', lowered)) and '>' not in lowered

    def _feed_body(self, text: str) -> List[str]:
        combined = self._trailing + text
        trailing_match = re.search(r'\s*$', combined)
        content = combined[:trailing_match.start()]
        self._trailing = combined[trailing_match.start():]
        if not content:
            return []
        self._parts.append(content)
        return [content]


class DraftMarkdownStream(Iterator[str]):
    def __init__(
        self,
        call: Callable[[Callable[[Dict[str, Any]], None]], str],
        finalize: Callable[[str], dict],
        prefix: str,
        idle_timeout: float,
    ):
        self._queue: Queue[Dict[str, Any]] = Queue()
        self._cancelled = False
        self._last_activity = time.monotonic()
        self._idle_timeout = idle_timeout
        self._prefix = prefix
        self._finalize = finalize
        self._result: Optional[dict] = None
        self._error: Optional[BaseException] = None
        context = copy_context()
        self._future = _DRAFT_STREAM_EXECUTOR.submit(context.run, call, self._sink)
        self._iterator = self._iterate()

    def __iter__(self) -> 'DraftMarkdownStream':
        return self

    def __next__(self) -> str:
        return next(self._iterator)

    def __enter__(self) -> 'DraftMarkdownStream':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def result(self) -> dict:
        if self._error is not None:
            raise self._error
        if self._result is None:
            raise RuntimeError('Draft Markdown stream must be fully consumed before result().')
        return self._result

    def close(self) -> None:
        self._cancelled = True
        self._future.cancel()
        self._iterator.close()
        self._drain_queue()

    def _sink(self, payload: Dict[str, Any]) -> None:
        if self._cancelled:
            return
        self._last_activity = time.monotonic()
        self._queue.put(dict(payload))

    def _iterate(self) -> Iterator[str]:
        normalizer = MarkdownStreamNormalizer()
        try:
            yield self._prefix
            while not self._future.done():
                try:
                    payload = self._queue.get(timeout=self._poll_timeout())
                except Empty:
                    self._raise_if_idle()
                    continue
                yield from self._visible_deltas(payload, normalizer)

            for payload in self._drain_queue():
                yield from self._visible_deltas(payload, normalizer)

            body = self._future.result().strip()
            final_deltas = normalizer.finish()
            if normalizer.body != body:
                raise ValueError('Streamed Markdown body does not match the normalized LLM response.')
            self._result = self._finalize(body)
            yield from final_deltas
            yield '\n'
        except BaseException as exc:
            self._error = exc
            raise
        finally:
            if self._result is None:
                self._cancelled = True
                self._future.cancel()

    @staticmethod
    def _visible_deltas(
        payload: Dict[str, Any],
        normalizer: MarkdownStreamNormalizer,
    ) -> List[str]:
        if payload.get('tag') != 'text':
            return []
        return normalizer.feed(str(payload.get('delta') or ''))

    def _poll_timeout(self) -> float:
        remaining = self._idle_timeout - (time.monotonic() - self._last_activity)
        return max(min(remaining, 0.1), 0.001)

    def _raise_if_idle(self) -> None:
        if not self._future.done() and time.monotonic() - self._last_activity >= self._idle_timeout:
            raise TimeoutError(f'Draft Markdown stream was idle for {self._idle_timeout:g} seconds.')

    def _drain_queue(self) -> List[Dict[str, Any]]:
        payloads = []
        while True:
            try:
                payloads.append(self._queue.get_nowait())
            except Empty:
                return payloads
