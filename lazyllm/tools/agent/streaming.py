import asyncio
import json
import re

import lazyllm
from .events import (
    AGENT_FAILED,
    AGENT_FINISHED,
    AgentEvent,
    PLAN_FINISHED,
    PLAN_STARTED,
    REASONING_DELTA,
    TEXT_DELTA,
    TOOL_CALLS,
    TOOL_RESULTS,
    TOOLS_EVENT_QUEUE,
)


_ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*m')

_TAG_TO_EVENT_TYPE = {
    'tool_calls': TOOL_CALLS,
    'tool_results': TOOL_RESULTS,
    'plan_started': PLAN_STARTED,
    'plan_finished': PLAN_FINISHED,
}


def _clean_chunk(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub('', text)


class AgentStreamAdapter:
    def __init__(self, agent, interval: float):
        self._agent = agent
        self._sleep_interval = interval
        self._agent_name = agent.__class__.__name__

    def prepare(self):
        lazyllm.FileSystemQueue().clear()
        lazyllm.FileSystemQueue.get_instance('think').clear()
        lazyllm.FileSystemQueue.get_instance(TOOLS_EVENT_QUEUE).clear()

    def drain(self, future, sleep):
        text_q = lazyllm.FileSystemQueue()
        think_q = lazyllm.FileSystemQueue.get_instance('think')
        tools_q = lazyllm.FileSystemQueue.get_instance(TOOLS_EVENT_QUEUE)

        while not future.done():
            drained = False
            for evt in self._drain_text(text_q):
                drained = True
                yield evt
            for evt in self._drain_think(think_q):
                drained = True
                yield evt
            for evt in self._drain_tools(tools_q):
                drained = True
                yield evt
            if not drained:
                sleep(self._sleep_interval)
        for evt in self._drain_text(text_q):
            yield evt
        for evt in self._drain_think(think_q):
            yield evt
        for evt in self._drain_tools(tools_q):
            yield evt

        for evt in self._agent_finish(future):
            yield evt

    async def adrain(self, future):
        text_q = lazyllm.FileSystemQueue()
        think_q = lazyllm.FileSystemQueue.get_instance('think')
        tools_q = lazyllm.FileSystemQueue.get_instance(TOOLS_EVENT_QUEUE)

        while not future.done():
            drained = False
            for evt in self._drain_text(text_q):
                drained = True
                yield evt
            for evt in self._drain_think(think_q):
                drained = True
                yield evt
            for evt in self._drain_tools(tools_q):
                drained = True
                yield evt
            if not drained:
                await asyncio.sleep(self._sleep_interval)
        for evt in self._drain_text(text_q):
            yield evt
        for evt in self._drain_think(think_q):
            yield evt
        for evt in self._drain_tools(tools_q):
            yield evt

        for evt in self._agent_finish(future):
            yield evt

    def _agent_finish(self, future):
        try:
            result = future.result()
        except Exception as exc:
            yield AgentEvent(type=AGENT_FAILED, agent=self._agent_name, error=str(exc))
            return
        if isinstance(result, tuple) and len(result) == 2:
            yield AgentEvent(type=AGENT_FINISHED, agent=self._agent_name,
                             text=result[0] if isinstance(result[0], str) else None,
                             metadata={'result': result[0], 'trace_id': result[1]})
        elif isinstance(result, str):
            yield AgentEvent(type=AGENT_FINISHED, agent=self._agent_name, text=result)
        else:
            yield AgentEvent(type=AGENT_FINISHED, agent=self._agent_name,
                             metadata={'result': result})

    @staticmethod
    def _drain_text(q):
        if values := q.dequeue():
            for v in values:
                chunk = _clean_chunk(str(v))
                if chunk:
                    yield AgentEvent(type=TEXT_DELTA, delta=chunk)
            return True
        return False

    @staticmethod
    def _drain_think(q):
        if values := q.dequeue():
            for v in values:
                chunk = _clean_chunk(str(v))
                if chunk:
                    yield AgentEvent(type=REASONING_DELTA, delta=chunk)
            return True
        return False

    def _drain_tools(self, q):
        if values := q.dequeue():
            for raw in values:
                payload = json.loads(raw)
                tag = payload.pop('tag', None)
                event_type = _TAG_TO_EVENT_TYPE.get(tag)
                if event_type:
                    yield AgentEvent(type=event_type, agent=self._agent_name, **payload)
            return True
        return False
