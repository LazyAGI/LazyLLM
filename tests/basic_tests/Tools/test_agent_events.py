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


def get_status() -> dict:
    '''Return a structured status result.

    Returns:
        dict: Structured status information.
    '''
    return {'status': 'ok', 'content': 'Error handling reference'}


class _FakeLLM(object):
    def __init__(self, outputs, *, stream=False):
        self._outputs = outputs
        self._cursor = 0
        self._stream = stream
        self.inputs = []
        self._module_id = f'fake-llm-{id(self)}'

    def share(self, prompt=None, format=None, stream=None, history=None, copy_static_params=False):
        cloned = copy.copy(self)
        if stream is not None:
            cloned._stream = stream
        return cloned

    def used_by(self, module_id):
        return self

    def __call__(self, input, **kwargs):
        self.inputs.append(input)
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


class _SharedCursorLLM(_FakeLLM):
    def __init__(self, outputs, *, stream=False):
        super().__init__(outputs, stream=stream)
        self._cursor_state = {'value': 0}

    def __call__(self, input, **kwargs):
        self.inputs.append(input)
        cursor = self._cursor_state['value']
        output = self._outputs[cursor]
        self._cursor_state['value'] = cursor + 1
        if self._stream:
            if isinstance(output, dict):
                content = output.get('content', '')
                if content:
                    lazyllm.FileSystemQueue().enqueue(json.dumps({'tag': 'text', 'delta': content}))
            elif output:
                lazyllm.FileSystemQueue().enqueue(json.dumps({'tag': 'text', 'delta': str(output)}))
        return output


def _read_agent_events():
    events = []
    for raw in lazyllm.FileSystemQueue().dequeue():
        if raw:
            payload = json.loads(raw)
            events.append(SimpleNamespace(**payload))
    return events


class TestReactAgentEvents(object):
    def test_force_summary_is_emitted_after_streamed_tool_progress(self):
        llm = _SharedCursorLLM([
            {
                'role': 'assistant',
                'content': 'First step.',
                'tool_calls': [{
                    'id': 'call-1',
                    'type': 'function',
                    'function': {'name': 'add_one', 'arguments': '{"value": 1}'},
                }],
            },
            {
                'role': 'assistant',
                'content': 'Second step.',
                'tool_calls': [{
                    'id': 'call-2',
                    'type': 'function',
                    'function': {'name': 'add_one', 'arguments': '{"value": 2}'},
                }],
            },
            'Final summary after the limit.',
        ])
        agent = ReactAgent(
            llm=llm,
            tools=[add_one],
            max_retries=1,
            stream=True,
            enable_builtin_tools=False,
            force_summarize=True,
        )

        assert agent('complete all steps') == 'Final summary after the limit.'
        events = _read_agent_events()
        assert any(
            event.tag == 'text' and event.delta == 'Final summary after the limit.'
            for event in events
        )

    def test_react_agent_summarizes_when_round_limit_callback_declines_expansion(self):
        llm = _SharedCursorLLM([
            {
                'role': 'assistant',
                'content': 'First step.',
                'tool_calls': [{
                    'id': 'call-1',
                    'type': 'function',
                    'function': {'name': 'add_one', 'arguments': '{"value": 1}'},
                }],
            },
            {
                'role': 'assistant',
                'content': 'Second step.',
                'tool_calls': [{
                    'id': 'call-2',
                    'type': 'function',
                    'function': {'name': 'add_one', 'arguments': '{"value": 2}'},
                }],
            },
            'Summary after decision timeout.',
        ])
        agent = ReactAgent(
            llm=llm,
            tools=[add_one],
            max_retries=1,
            stream=True,
            enable_builtin_tools=False,
            on_max_retries=lambda output, used, current: None,
            force_summarize=True,
        )

        assert agent('complete all steps') == 'Summary after decision timeout.'

    def test_react_agent_uses_generic_round_limit_callback(self):
        llm = _FakeLLM([
            {
                'role': 'assistant',
                'content': 'First step.',
                'tool_calls': [{
                    'id': 'call-1',
                    'type': 'function',
                    'function': {'name': 'add_one', 'arguments': '{"value": 1}'},
                }],
            },
            {
                'role': 'assistant',
                'content': 'Second step.',
                'tool_calls': [{
                    'id': 'call-2',
                    'type': 'function',
                    'function': {'name': 'add_one', 'arguments': '{"value": 2}'},
                }],
            },
            {'role': 'assistant', 'content': 'Finished after explicit continuation.'},
        ])
        limit_calls = []

        def expand(output, used, current):
            limit_calls.append((used, current))
            return 10

        agent = ReactAgent(
            llm=llm,
            tools=[add_one],
            max_retries=1,
            stream=True,
            enable_builtin_tools=False,
            on_max_retries=expand,
        )

        assert agent('complete all steps') == 'Finished after explicit continuation.'
        assert limit_calls == [(2, 2)]

    def test_react_agent_stream_preserves_structured_tool_results(self):
        llm = _FakeLLM([
            {
                'role': 'assistant',
                'content': 'Let me read the status.',
                'tool_calls': [{
                    'id': 'call-status',
                    'type': 'function',
                    'function': {'name': 'get_status', 'arguments': '{}'},
                }],
            },
            {'role': 'assistant', 'content': 'Done.'},
        ])
        agent = ReactAgent(llm=llm, tools=[get_status], max_retries=1, stream=True,
                           enable_builtin_tools=False)

        assert agent('read status') == 'Done.'
        events = _read_agent_events()
        result_event = next(event for event in events if event.tag == 'tool_results')

        assert result_event.tool_results[0]['result'] == {
            'status': 'ok',
            'content': 'Error handling reference',
        }
        tool_message = llm.inputs[1]['input'][0]

        assert tool_message['role'] == 'tool'
        assert tool_message['content'] == str({
            'status': 'ok',
            'content': 'Error handling reference',
        })

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
        agent = ReactAgent(llm=llm, tools=[add_one], max_retries=1, stream=True,
                           enable_builtin_tools=False)

        result = agent('add one to 1')
        events = _read_agent_events()

        event_types = [event.tag for event in events]
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
            seen_types.extend(event.tag for event in events)
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

        event_types = [event.tag for event in events]
        assert result == 'The answer is 2.'
        assert event_types.index('plan_started') < event_types.index('plan_finished')
        assert 'think' in event_types
        assert 'text' in event_types
        assert 'tool_calls' in event_types
        assert 'tool_results' in event_types
