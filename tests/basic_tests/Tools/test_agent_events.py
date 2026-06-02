import copy
import json
import threading
import time
from types import SimpleNamespace

import lazyllm
from lazyllm.tools import PlanAndSolveAgent, ReactAgent


def add_one(value: int) -> int:
    '''
    Add one to the input integer.

    Args:
        value (int): Input integer.

    Returns:
        int: Incremented integer.
    '''
    return value + 1


class _FakeLLM(object):
    def __init__(self, outputs, *, stream=False):
        self._outputs = outputs
        self._cursor = 0
        self._stream = stream
        self._module_id = f'fake-llm-{id(self)}'

    def share(self, prompt=None, format=None, stream=None, history=None, copy_static_params=False):
        cloned = copy.copy(self)
        if stream is not None:
            cloned._stream = stream
        return cloned

    def used_by(self, module_id):
        return self

    def __call__(self, input, **kwargs):
        output = self._outputs[self._cursor]
        self._cursor += 1
        if self._stream:
            if isinstance(output, dict):
                reasoning = output.get('reasoning_content', '')
                content = output.get('content', '')
                if reasoning:
                    lazyllm.FileSystemQueue().enqueue(json.dumps({'tag': 'think', 'delta': reasoning}))
                if content:
                    lazyllm.FileSystemQueue().enqueue(json.dumps({'tag': 'text', 'delta': content}))
            elif output:
                lazyllm.FileSystemQueue().enqueue(json.dumps({'tag': 'text', 'delta': str(output)}))
        return output


class _SlowStreamingLLM(_FakeLLM):
    def __init__(self, outputs, *, stream=False):
        super().__init__(outputs, stream=stream)
        self.release = threading.Event()

    def share(self, prompt=None, format=None, stream=None, history=None, copy_static_params=False):
        cloned = copy.copy(self)
        if stream is not None:
            cloned._stream = stream
        return cloned

    def __call__(self, input, **kwargs):
        output = self._outputs[self._cursor]
        self._cursor += 1
        if self._stream:
            lazyllm.FileSystemQueue().enqueue(
                json.dumps({'tag': 'think', 'delta': output.get('reasoning_content', '')}))
            time.sleep(0.05)
            lazyllm.FileSystemQueue().enqueue(
                json.dumps({'tag': 'text', 'delta': output.get('content', '')}))
            self.release.wait(timeout=1)
        return output


def _read_agent_events():
    events = []
    for raw in lazyllm.FileSystemQueue().dequeue():
        if raw:
            payload = json.loads(raw)
            events.append(SimpleNamespace(**payload))
    return events


class TestReactAgentEvents(object):
    def test_react_agent_stream_emits_text_reasoning_and_tool_events(self):
        llm = _FakeLLM([
            {
                'role': 'assistant',
                'content': 'Let me use a tool.',
                'reasoning_content': 'Need one calculation.',
                'tool_calls': [{
                    'id': 'call-1',
                    'type': 'function',
                    'function': {'name': 'add_one', 'arguments': '{"value": 1}'},
                }],
            },
            {
                'role': 'assistant',
                'content': 'The answer is 2.',
                'reasoning_content': 'Now I can answer.',
            },
        ])
        agent = ReactAgent(llm=llm, tools=[add_one], max_retries=3, stream=True,
                           enable_builtin_tools=False)

        result = agent('add one to 1')
        events = _read_agent_events()

        event_types = [event.type for event in events]
        assert result == 'The answer is 2.'
        assert 'think' in event_types
        assert 'text' in event_types
        assert 'tool_calls' in event_types
        assert 'tool_results' in event_types

    def test_react_agent_stream_writes_events_before_forward_returns(self):
        llm = _SlowStreamingLLM([{
            'role': 'assistant',
            'content': 'The answer is already streaming.',
            'reasoning_content': 'Thinking in real time.',
        }])
        agent = ReactAgent(llm=llm, tools=[add_one], max_retries=1, stream=True,
                           enable_builtin_tools=False)
        result_holder = {}
        sid = lazyllm.globals._sid

        def _run_agent():
            lazyllm.globals._init_sid(sid)
            lazyllm.locals._init_sid(sid)
            result_holder['result'] = agent('stream now')

        thread = threading.Thread(
            target=_run_agent,
        )
        thread.start()

        seen_types = []
        deadline = time.time() + 1
        while thread.is_alive() and time.time() < deadline:
            events = _read_agent_events()
            seen_types.extend(event.type for event in events)
            if 'think' in seen_types or 'text' in seen_types:
                break
            time.sleep(0.01)

        assert thread.is_alive()
        assert 'think' in seen_types or 'text' in seen_types
        llm.release.set()
        thread.join(timeout=1)
        assert result_holder['result'] == 'The answer is already streaming.'


class TestPlanAndSolveAgentEvents(object):
    def test_plan_and_solve_agent_reuses_shared_execution_for_forward_and_stream(self):
        plan_outputs = ['Plan:\n1. Use add_one to compute the final answer.\n<END_OF_PLAN>']
        solve_outputs = [
            {
                'role': 'assistant',
                'content': 'Let me use a tool.',
                'reasoning_content': 'Need one calculation.',
                'tool_calls': [{
                    'id': 'call-1',
                    'type': 'function',
                    'function': {'name': 'add_one', 'arguments': '{"value": 1}'},
                }],
            },
            {
                'role': 'assistant',
                'content': 'The answer is 2.',
                'reasoning_content': 'Now I can answer.',
            },
        ]
        forward_agent = PlanAndSolveAgent(plan_llm=_FakeLLM(plan_outputs), solve_llm=_FakeLLM(solve_outputs),
                                          tools=[add_one], max_retries=3, stream=False,
                                          enable_builtin_tools=False)
        assert forward_agent('add one to 1') == 'The answer is 2.'

        stream_agent = PlanAndSolveAgent(plan_llm=_FakeLLM(plan_outputs), solve_llm=_FakeLLM(solve_outputs),
                                         tools=[add_one], max_retries=3, stream=True,
                                         enable_builtin_tools=False)
        result = stream_agent('add one to 1')
        events = _read_agent_events()

        event_types = [event.type for event in events]
        assert result == 'The answer is 2.'
        assert event_types.index('plan_started') < event_types.index('plan_finished')
        assert 'think' in event_types
        assert 'text' in event_types
        assert 'tool_calls' in event_types
        assert 'tool_results' in event_types
