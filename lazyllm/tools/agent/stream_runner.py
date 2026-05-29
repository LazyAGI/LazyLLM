import re
import time
from queue import Empty, Queue
from typing import Any, Callable

import lazyllm
from lazyllm import FileSystemQueue, locals

from .events import (
    AgentEvent,
    REASONING_DELTA,
    REASONING_FINISHED,
    TEXT_DELTA,
    TEXT_FINISHED,
)

_POLL_INTERVAL = 0.01
_ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*m')
_POOL = lazyllm.ThreadPoolExecutor(max_workers=lazyllm.config['thread_pool_worker_num'])


class StreamRunner:
    def __init__(self, agent_name: str):
        self._agent_name = agent_name
        self._result = None
        self._error = None
        self._future = None
        self._queue: Queue = Queue()
        self._saw_text = False
        self._saw_reasoning = False
        self._buffer: list = []
        self._emitted_reasoning_finished = False
        self._emitted_text_finished = False

    def start(self, target: Callable[[Callable[[AgentEvent], None]], Any]):
        sid = lazyllm.globals._sid
        lazyllm.globals._init_sid(sid)
        locals._init_sid(sid)
        FileSystemQueue().clear()
        FileSystemQueue.get_instance('think').clear()

        def _worker():
            lazyllm.globals._init_sid(sid)
            locals._init_sid(sid)
            try:
                self._result = target(self._emit)
            except Exception as exc:
                self._error = exc

        self._future = _POOL.submit(_worker)

    def _emit(self, event: AgentEvent):
        self._queue.put(event)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self._buffer:
                return self._buffer.pop(0)

            self._buffer = (
                list(self._drain_queue())
                + list(self._drain_fsq_text())
                + list(self._drain_fsq_think())
            )

            if self._buffer:
                return self._buffer.pop(0)

            if self._future.done():
                if self._saw_reasoning and not self._emitted_reasoning_finished:
                    self._emitted_reasoning_finished = True
                    return self._mk(REASONING_FINISHED)
                if self._saw_text and not self._emitted_text_finished:
                    self._emitted_text_finished = True
                    return self._mk(TEXT_FINISHED)
                if self._error:
                    raise self._error
                raise StopIteration

            time.sleep(_POLL_INTERVAL)

    @property
    def result(self):
        if self._error:
            raise self._error
        return self._result

    def _mk(self, event_type: str, **kwargs) -> AgentEvent:
        return AgentEvent(type=event_type, agent=self._agent_name, **kwargs)

    def _drain_queue(self):
        while True:
            try:
                yield self._queue.get_nowait()
            except Empty:
                break

    def _drain_fsq_text(self):
        queue = FileSystemQueue()
        values = queue.dequeue()
        if not values:
            return
        for value in values:
            chunk = _strip_ansi(value)
            if not chunk:
                continue
            self._saw_text = True
            yield self._mk(TEXT_DELTA, delta=chunk)

    def _drain_fsq_think(self):
        queue = FileSystemQueue.get_instance('think')
        values = queue.dequeue()
        if not values:
            return
        for value in values:
            chunk = _strip_ansi(value)
            if not chunk:
                continue
            self._saw_reasoning = True
            yield self._mk(REASONING_DELTA, delta=chunk)


def _strip_ansi(text: str) -> str:
    if not text:
        return ''
    return _ANSI_ESCAPE_RE.sub('', text)
