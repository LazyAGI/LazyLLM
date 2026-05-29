import copy

import lazyllm
from lazyllm.tools import FunctionCall, FunctionCallAgent, PlanAndSolveAgent


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
                    lazyllm.FileSystemQueue.get_instance('think').enqueue(reasoning)
                if content:
                    lazyllm.FileSystemQueue().enqueue(content)
            elif output:
                lazyllm.FileSystemQueue().enqueue(str(output))
        return output


class TestFunctionCallEvents(object):
    def test_function_call_emits_tool_events_outside_tool_manager(self):
        llm = _FakeLLM([{
            'role': 'assistant',
            'content': 'Let me use a tool.',
            'tool_calls': [{
                'id': 'call-1',
                'type': 'function',
                'function': {'name': 'add_one', 'arguments': '{"value": 1}'},
            }],
        }])
        fc = FunctionCall(llm=llm, tools=[add_one])
        captured = []

        result = fc._run_once('add one to 1', event_callback=captured.append)

        assert result['tool_calls'][0]['function']['name'] == 'add_one'
        assert [event.type for event in captured] == ['agent.tool.calls', 'agent.tool.results']
        assert captured[0].tool_calls[0]['function']['name'] == 'add_one'
        assert captured[1].tool_results[0]['name'] == 'add_one'
        assert captured[1].tool_results[0]['result'] == '2'

    def test_function_call_emits_tool_call_event_before_tool_execution(self):
        llm = _FakeLLM([{
            'role': 'assistant',
            'content': 'Let me use a tool.',
            'tool_calls': [{
                'id': 'call-1',
                'type': 'function',
                'function': {'name': 'add_one', 'arguments': '{"value": 1}'},
            }],
        }])
        fc = FunctionCall(llm=llm, tools=[add_one])
        captured = []

        def _fake_tools_manager(tool_calls):
            assert [event.type for event in captured] == ['agent.tool.calls']
            return ['2']

        fc._tools_manager = _fake_tools_manager
        result = fc._run_once('add one to 1', event_callback=captured.append)

        assert result['tool_calls'][0]['function']['name'] == 'add_one'
        assert [event.type for event in captured] == ['agent.tool.calls', 'agent.tool.results']


class TestFunctionCallAgentEvents(object):
    def test_function_call_agent_stream_emits_text_reasoning_and_tool_events(self):
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
        agent = FunctionCallAgent(llm=llm, tools=[add_one], max_retries=3, stream=True)

        stream = agent.stream('add one to 1')
        events = []
        try:
            while True:
                events.append(next(stream))
        except StopIteration as stop:
            result = stop.value

        event_types = [event.type for event in events]
        assert result == 'The answer is 2.'
        assert 'agent.reasoning.delta' in event_types
        assert 'agent.text.delta' in event_types
        assert 'agent.tool.calls' in event_types
        assert 'agent.tool.results' in event_types
        assert event_types[-1] == 'agent.finished'


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
        stream = stream_agent.stream('add one to 1')
        events = []
        try:
            while True:
                events.append(next(stream))
        except StopIteration as stop:
            result = stop.value

        event_types = [event.type for event in events]
        assert result == 'The answer is 2.'
        assert event_types[:2] == ['agent.plan.started', 'agent.plan.finished']
        assert 'agent.reasoning.delta' in event_types
        assert 'agent.text.delta' in event_types
        assert 'agent.tool.calls' in event_types
        assert 'agent.tool.results' in event_types
        assert event_types[-1] == 'agent.finished'
