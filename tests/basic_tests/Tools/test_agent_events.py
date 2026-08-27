import copy
import json
import threading
import time
from types import SimpleNamespace

import lazyllm
from lazyllm.tools import PlanAndSolveAgent, ReactAgent
from lazyllm.tools.agent import ToolExecutionError
from lazyllm.tools.agent.base import (
    TOOL_OBSERVATION_KEY,
    is_tool_result_envelope,
    normalize_tool_observation,
)
from lazyllm.tools.agent.functionCall import FunctionCall
from lazyllm.tools.agent.toolsManager import ToolManager


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


def private_status() -> str:
    '''Return a private toolkit status.

    Returns:
        str: Private status.
    '''
    return 'private status'


def exposed_failure(query: str) -> str:
    '''Raise from an exposed prefixed tool.

    Args:
        query (str): Search query.
    '''
    raise ToolExecutionError(f'Search failed for query {query!r}.')


def approval_failure(path: str) -> str:
    '''Require approval before changing a path.

    Args:
        path (str): Path that would be changed.
    '''
    raise ToolExecutionError.approval_required(f'Changing {path} requires approval.')


def _private_tool_group():
    return {
        'name': 'private',
        'desc': 'Private status tools.',
        'lazy': True,
        'prefix': False,
        'tools': [private_status],
    }


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


class _BudgetRecordingLLM(_FakeLLM):
    def __init__(self, outputs, *, stream=False):
        super().__init__(outputs, stream=stream)
        self.budget_histories = []

    def __call__(self, input, **kwargs):
        histories = lazyllm.locals.get('chat_history', {})
        self.budget_histories.append(copy.deepcopy(histories.get(self._module_id, [])))
        return super().__call__(input, **kwargs)


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
        preserved_workspace = lazyllm.locals['_lazyllm_agent'].get('workspace', {})
        assert preserved_workspace.get('history')
        assert preserved_workspace.get('_react_round_number') == 2
        assert '_react_round_limit' not in preserved_workspace
        continuation_llm = _BudgetRecordingLLM([
            {'role': 'assistant', 'content': 'Finished after the follow-up continuation.'},
        ])
        continuation_agent = ReactAgent(
            llm=continuation_llm,
            tools=[add_one],
            max_retries=1,
            stream=False,
            enable_builtin_tools=False,
        )
        assert continuation_agent('continue') == 'Finished after the follow-up continuation.'
        continuation_history = continuation_llm.budget_histories[0]
        assert any(
            message.get('content', '').startswith('complete all steps')
            for message in continuation_history
        )
        assert continuation_llm.inputs[0] == 'continue'
        assert any(
            'Internal ReAct rounds left: 0.' in str(message.get('content', ''))
            for message in continuation_history
        )
        assert not any(message.get('role') == 'system' for message in continuation_history)

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

    def test_react_agent_persists_dynamic_round_budget_in_tool_result(self):
        llm = _BudgetRecordingLLM([
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
            {'role': 'assistant', 'content': 'Finished within the expanded budget.'},
        ])
        agent = ReactAgent(
            llm=llm,
            tools=[add_one],
            max_retries=1,
            stream=True,
            enable_builtin_tools=False,
            on_max_retries=lambda output, used, current: 4,
        )

        result = agent('complete all steps')

        assert result == 'Finished within the expanded budget.'
        input_snapshots = [json.dumps(value) for value in llm.inputs]
        assert 'Internal ReAct rounds left:' not in input_snapshots[0]
        assert 'Internal ReAct rounds left: 0.' in input_snapshots[1]
        assert 'Internal ReAct rounds left: 1.' in input_snapshots[2]
        assert all(
            '"role": "system"' not in json.dumps(history)
            for history in llm.budget_histories
        )
        persisted_history = lazyllm.locals['_lazyllm_agent']['history']
        persisted_snapshot = json.dumps(persisted_history)
        assert [
            f'Internal ReAct rounds left: {remaining}.' in persisted_snapshot
            for remaining in [0, 1]
        ] == [True, True]
        assert sum(
            'Internal ReAct rounds left:' in str(message.get('content', ''))
            for message in persisted_history
        ) == 2
        assert all(
            notice in persisted_snapshot
            for notice in [
                '[Internal runtime notice] Internal ReAct rounds left: 1.',
                '[Internal runtime notice] Internal ReAct rounds left: 0.',
            ]
        )
        assert 'Internal ReAct rounds left:' not in result
        events = _read_agent_events()
        assert 'Internal ReAct rounds left:' not in json.dumps([vars(event) for event in events])

        for input_value in llm.inputs:
            if isinstance(input_value, dict):
                invocation_messages = input_value.get('input', [])
                assert invocation_messages
                assert all(message['role'] == 'tool' for message in invocation_messages)
        assert [
            message['content']
            for message in persisted_history
            if 'Internal ReAct rounds left:' in str(message.get('content', ''))
        ] == [
            llm.inputs[1]['input'][-1]['content'],
            llm.inputs[2]['input'][-1]['content'],
        ]

    def test_react_agent_appends_round_budget_to_last_tool_result(self):
        llm = _BudgetRecordingLLM([
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
            {'role': 'assistant', 'content': 'Finished before the limit.'},
        ])
        agent = ReactAgent(
            llm=llm,
            tools=[add_one],
            max_retries=3,
            stream=False,
            enable_builtin_tools=False,
        )

        assert agent('complete all steps') == 'Finished before the limit.'
        assert len(llm.budget_histories) == 3
        invocation_snapshots = [
            json.dumps({'history': history, 'input': input_value})
            for history, input_value in zip(llm.budget_histories, llm.inputs)
        ]
        assert 'Internal ReAct rounds left:' not in invocation_snapshots[0]
        assert 'Internal ReAct rounds left: 2.' in invocation_snapshots[1]
        assert 'Internal ReAct rounds left: 1.' in invocation_snapshots[2]
        assert [
            input_value['input'][-1]['content']
            for input_value in llm.inputs[1:]
        ] == [
            '2\n\n'
            '[Internal runtime notice] Internal ReAct rounds left: 2.',
            '3\n\n'
            '[Internal runtime notice] Internal ReAct rounds left: 1.',
        ]
        persisted_history = lazyllm.locals['_lazyllm_agent']['history']
        persisted_budget_queries = [
            message['content']
            for message in persisted_history
            if 'Internal ReAct rounds left:' in str(message.get('content', ''))
        ]
        assert persisted_budget_queries == [
            llm.inputs[1]['input'][-1]['content'],
            llm.inputs[2]['input'][-1]['content'],
        ]

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
        agent = ReactAgent(llm=llm, tools=[get_status], max_retries=3, stream=True,
                           enable_builtin_tools=False)

        assert agent('read status') == 'Done.'
        events = _read_agent_events()
        result_event = next(event for event in events if event.tag == 'tool_results')

        assert result_event.tool_results[0]['result'] == {
            'ok': True,
            'value': {
                'status': 'ok',
                'content': 'Error handling reference',
            },
        }
        tool_message = llm.inputs[1]['input'][0]

        assert tool_message['role'] == 'tool'
        assert tool_message['content'] == (
            f'{str({"status": "ok", "content": "Error handling reference"})}\n\n'
            '[Internal runtime notice] Internal ReAct rounds left: 2.'
        )

    def test_history_compactor_receives_structured_observation_without_model_leak(self):
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
        captured = []

        def capture(prior_history, _keep, current_round_messages=None, **_kwargs):
            current = list(current_round_messages or [])
            captured.append(copy.deepcopy(list(prior_history) + current))
            return list(prior_history) + current

        agent = ReactAgent(
            llm=llm,
            tools=[get_status],
            max_retries=3,
            history_compactor=capture,
            enable_builtin_tools=False,
        )

        assert agent('read status') == 'Done.'
        captured_tool = next(
            message
            for history in captured
            for message in history
            if message.get('role') == 'tool'
        )
        assert captured_tool[TOOL_OBSERVATION_KEY] == {
            'version': 1,
            'ok': True,
            'value': {'status': 'ok', 'content': 'Error handling reference'},
            'error': '',
        }
        assert all(
            TOOL_OBSERVATION_KEY not in message
            for invocation in llm.inputs
            if isinstance(invocation, dict)
            for message in invocation.get('input', [])
        )
        persisted_tool = next(
            message for message in lazyllm.locals['_lazyllm_agent']['history']
            if message.get('role') == 'tool'
        )
        assert persisted_tool[TOOL_OBSERVATION_KEY] == captured_tool[TOOL_OBSERVATION_KEY]
        assert normalize_tool_observation({
            'ok': False,
            'value': None,
            'msg': '[Tool Error] failed',
        }) == {
            'version': 1,
            'ok': False,
            'value': None,
            'error': '[Tool Error] failed',
        }
        incidental = {'ok': 'yes', 'path': '/tmp/file', 'content': 'not an envelope'}
        assert is_tool_result_envelope(incidental) is False
        assert normalize_tool_observation(incidental) == {
            'version': 1,
            'ok': None,
            'value': incidental,
            'error': '',
        }

    def test_history_compactor_tuple_return_sends_current_tools_once(self):
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

        def compact(prior_history, _keep, current_round_messages=None, **_kwargs):
            current = list(current_round_messages or [])
            return [{'role': 'user', 'content': 'earlier turns summarized'}], current

        agent = ReactAgent(
            llm=llm,
            tools=[get_status],
            max_retries=3,
            history_compactor=compact,
            enable_builtin_tools=False,
        )

        assert agent('read status') == 'Done.'
        tool_round = next(
            invocation for invocation in llm.inputs
            if isinstance(invocation, dict) and isinstance(invocation.get('input'), list)
        )
        tool_ids = [
            message.get('tool_call_id')
            for message in tool_round['input']
            if isinstance(message, dict) and message.get('role') == 'tool'
        ]
        assert tool_ids == ['call-status']

    def test_react_agent_exposes_only_tool_failure_value_to_next_round(self):
        llm = _FakeLLM([
            {
                'role': 'assistant',
                'content': 'Let me calculate.',
                'tool_calls': [{
                    'id': 'call-invalid',
                    'type': 'function',
                    'function': {'name': 'add_one'},
                }],
            },
            {'role': 'assistant', 'content': 'I corrected the plan.'},
        ])
        agent = ReactAgent(
            llm=llm,
            tools=[add_one],
            max_retries=3,
            stream=True,
            enable_builtin_tools=False,
        )

        assert agent('calculate') == 'I corrected the plan.'
        events = _read_agent_events()
        result_event = next(event for event in events if event.tag == 'tool_results')
        failure = result_event.tool_results[0]['result']
        assert failure['ok'] is False
        assert 'value:' in failure['value']
        tool_message = llm.inputs[1]['input'][0]
        visible_error = tool_message['content'].split('\n\n', 1)[0]
        assert visible_error == failure['value']
        assert not visible_error.startswith('{"ok"')

    def test_prefixed_failure_keeps_exposed_name_in_error_event_and_history(self):
        exposed_name = 'github_exposed_failure'
        llm = _FakeLLM([
            {
                'role': 'assistant',
                'content': 'Search GitHub.',
                'tool_calls': [{
                    'id': 'call-prefixed',
                    'type': 'function',
                    'function': {
                        'name': exposed_name,
                        'arguments': '{"query": "LazyLLM"}',
                    },
                }],
            },
            {'role': 'assistant', 'content': 'Done.'},
        ])
        agent = ReactAgent(
            llm=llm,
            tools=[{
                'name': 'github',
                'desc': 'GitHub tools.',
                'lazy': False,
                'prefix': True,
                'tools': [exposed_failure],
            }],
            max_retries=3,
            stream=True,
            enable_builtin_tools=False,
        )

        assert agent('search') == 'Done.'
        event = next(item for item in _read_agent_events() if item.tag == 'tool_results')
        tool_result = event.tool_results[0]
        tool_message = llm.inputs[1]['input'][0]

        assert tool_result['name'] == exposed_name
        assert tool_result['result']['ok'] is False
        assert tool_result['result'] == {
            'ok': False,
            'value': "Search failed for query 'LazyLLM'.",
        }
        assert tool_message['name'] == exposed_name
        visible_result = tool_message['content'].split('\n\n', 1)[0]
        assert visible_result == tool_result['result']['value']

    def test_react_agent_hides_approval_metadata_from_tool_observation(self):
        llm = _FakeLLM([
            {
                'role': 'assistant',
                'content': 'I need to change the file.',
                'tool_calls': [{
                    'id': 'call-approval',
                    'type': 'function',
                    'function': {
                        'name': 'approval_failure',
                        'arguments': '{"path": "/workspace/a.txt"}',
                    },
                }],
            },
            {'role': 'assistant', 'content': 'Approval is required.'},
        ])
        agent = ReactAgent(
            llm=llm,
            tools=[approval_failure],
            max_retries=3,
            stream=True,
            enable_builtin_tools=False,
        )

        assert agent('change the file') == 'Approval is required.'
        event = next(item for item in _read_agent_events() if item.tag == 'tool_results')
        result = event.tool_results[0]['result']
        tool_message = llm.inputs[1]['input'][0]

        assert result == {
            'ok': False,
            'value': 'Changing /workspace/a.txt requires approval.',
            'needs_approval': True,
        }
        visible_result = tool_message['content'].split('\n\n', 1)[0]
        assert visible_result == result['value']
        assert 'needs_approval' not in visible_result

    def test_react_agent_reuses_one_tool_snapshot_per_round(self):
        llm = _FakeLLM([
            {
                'role': 'assistant',
                'content': 'Activate and inspect.',
                'tool_calls': [
                    {
                        'id': 'call-gateway',
                        'type': 'function',
                        'function': {'name': 'get_private_methods', 'arguments': '{}'},
                    },
                    {
                        'id': 'call-hidden-same-round',
                        'type': 'function',
                        'function': {'name': 'private_status', 'arguments': '{}'},
                    },
                ],
            },
            {
                'role': 'assistant',
                'content': 'Use the activated tool.',
                'tool_calls': [{
                    'id': 'call-hidden-next-round',
                    'type': 'function',
                    'function': {'name': 'private_status', 'arguments': '{}'},
                }],
            },
            {'role': 'assistant', 'content': 'Done.'},
        ])
        agent = ReactAgent(
            llm=llm,
            tools=[_private_tool_group()],
            max_retries=3,
            stream=True,
            enable_builtin_tools=False,
        )

        assert agent('inspect private status') == 'Done.'
        result_events = [event for event in _read_agent_events() if event.tag == 'tool_results']

        first_round = result_events[0].tool_results
        assert first_round[0]['result']['ok'] is True
        assert first_round[0]['result']['value'].startswith('Activated Toolkit "private"')
        assert 'private_status' in first_round[1]['result']['value']
        assert result_events[1].tool_results[0]['result'] == {
            'ok': True,
            'value': 'private status',
        }

    def test_function_call_round_snapshot_is_session_local(self):
        manager = ToolManager([_private_tool_group()])
        function_call = FunctionCall(_FakeLLM([]), _tool_manager=manager)
        barrier = threading.Barrier(2)
        snapshots = {}
        errors = []

        def capture(label, active):
            try:
                with lazyllm.new_session(f'tool-snapshot-{label}'):
                    lazyllm.locals['_lazyllm_agent'] = {
                        'workspace': {'_active_groups': ['private'] if active else []},
                    }
                    function_call._get_current_tools(refresh=True)
                    barrier.wait()
                    snapshots[label] = function_call._get_visible_tool_names()
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=capture, args=('inactive', False)),
            threading.Thread(target=capture, args=('active', True)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors
        assert snapshots == {
            'inactive': {'get_private_methods'},
            'active': {'private_status'},
        }

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
