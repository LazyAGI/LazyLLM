import asyncio
import copy
import uuid

import pytest

import lazyllm
from lazyllm.tools import ReactAgent


def status() -> str:
    '''Return a test observation.'''
    return 'observation A'


def cancel_tool() -> str:
    '''Cancel the current test execution.'''
    raise asyncio.CancelledError('tool cancelled')


def tool_output(name='status'):
    return {'content': '', 'tool_calls': [{
        'id': 'call-1', 'type': 'function', 'function': {'name': name, 'arguments': '{}'}}]}


class ScriptedModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.histories = []
        self._module_id = uuid.uuid4().hex

    def share(self, **kwargs):
        return copy.copy(self)

    def used_by(self, _module_id):
        return self

    def __call__(self, *args, **kwargs):
        self.histories.append(copy.deepcopy(lazyllm.locals['_lazyllm_agent'].get('workspace', {})))
        output = next(self.outputs)
        if isinstance(output, BaseException):
            raise output
        return copy.deepcopy(output)


@pytest.fixture(autouse=True)
def session():
    sid = f'agent-cleanup-{uuid.uuid4().hex}'
    with lazyllm.globals._bind_sid(sid), lazyllm.locals._scope():
        yield
    lazyllm.globals._clear_sid(sid)


@pytest.mark.parametrize('phase', ['build', 'pre', 'stop', 'tool', 'summary', 'model', 'exhausted'])
def test_react_cleans_unfinished_state_across_entire_execution(phase, monkeypatch, tmp_path):
    error = asyncio.CancelledError(f'{phase} cancelled')
    outputs = [tool_output('cancel_tool' if phase == 'tool' else 'status'), tool_output(), error]
    if phase == 'model':
        outputs = [error]
    model = ScriptedModel(outputs)
    state = lazyllm.locals['_lazyllm_agent']
    state.update(completed=['completed before A'], history=['saved before A'])
    histories = lazyllm.locals['chat_history']
    histories['other-model'] = ['unrelated']
    agent = ReactAgent(model, [status, cancel_tool], max_retries=1, force_summarize=phase == 'summary',
                       enable_builtin_tools=False, sandbox=None, workspace=str(tmp_path))

    def fail(*args, **kwargs):
        state['workspace'] = {'history': ['unfinished']}
        raise error

    if phase in ('build', 'pre'):
        monkeypatch.setattr(agent, 'build_agent' if phase == 'build' else '_pre_process', fail)
    if phase == 'stop':
        agent._extra_stop_condition = fail
    with pytest.raises(Exception if phase == 'exhausted' else asyncio.CancelledError):
        agent('task A', llm_chat_history=[{'role': 'user', 'content': 'prior A'}])
    assert 'workspace' not in state
    assert state['completed'] == ['completed before A']
    assert state['history'] == ['saved before A']
    assert histories == {'other-model': ['unrelated']}


def test_react_cleanup_uses_captured_containers_after_sid_switch(tmp_path):
    state = lazyllm.locals['_lazyllm_agent']
    histories = lazyllm.locals['chat_history']
    histories['other'] = ['keep']
    alternate = f'alternate-{uuid.uuid4().hex}'
    alternate_state = {'workspace': {'history': ['other request']}}

    def switch_and_cancel(_output):
        lazyllm.locals._init_sid(alternate)
        lazyllm.locals['_lazyllm_agent'] = alternate_state
        lazyllm.locals['chat_history'] = {'other': ['other request']}
        raise asyncio.CancelledError('switched')

    agent = ReactAgent(ScriptedModel([tool_output()]), [status], extra_stop_condition=switch_and_cancel,
                       enable_builtin_tools=False, sandbox=None, workspace=str(tmp_path))
    try:
        with pytest.raises(asyncio.CancelledError, match='switched'):
            agent('A', llm_chat_history=[{'role': 'user', 'content': 'old'}])
        assert 'workspace' not in state
        assert histories == {'other': ['keep']}
        assert alternate_state['workspace']['history'] == ['other request']
    finally:
        lazyllm.locals._clear_sid(alternate)


@pytest.mark.parametrize('continue_a', [False, True])
def test_summary_retained_only_within_scope_and_completed_history_preserved(tmp_path, continue_a):
    model = ScriptedModel([tool_output(), tool_output(), 'summary A', {'content': 'completed', 'tool_calls': []}])

    def summarize_and_continue():
        agent = ReactAgent(model, [status], max_retries=1, force_summarize=True,
                           enable_builtin_tools=False, sandbox=None, workspace=str(tmp_path))
        assert agent('marker A') == 'summary A'
        state = lazyllm.locals['_lazyllm_agent']
        assert state['workspace']['history'][0]['content'] == 'marker A'
        assert model._module_id not in lazyllm.locals['chat_history']
        if not continue_a:
            return
        assert agent('continue') == 'completed'
        assert 'workspace' not in state
        assert state['completed']
        assert any(item.get('content') == 'marker A' for item in state['history'])

    with lazyllm.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(summarize_and_continue).result()
        assert pool.submit(lambda: dict(lazyllm.locals['_lazyllm_agent'])).result() == {}
