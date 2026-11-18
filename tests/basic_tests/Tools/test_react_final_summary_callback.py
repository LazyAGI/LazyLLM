import asyncio
import copy
import uuid

import pytest
from lazyllm.tools import ReactAgent


pytestmark = pytest.mark.usefixtures('model_call_session')


def final_summary_observation():
    '''Return a completed observation for final-summary tests.'''
    return 'completed observation'


class ScriptedModel:
    def __init__(self):
        self._module_id = uuid.uuid4().hex
        self.calls = []

    def share(self, **kwargs):
        return copy.copy(self)

    def used_by(self, module_id):
        return self

    def __call__(self, *args, **kwargs):
        self.calls.append(args)
        return {'content': '', 'tool_calls': [{
            'id': 'final-summary-tool', 'type': 'function',
            'function': {'name': 'final_summary_observation', 'arguments': '{}'},
        }]}


@pytest.mark.parametrize('cancelled', [False, True])
def test_final_summary_callback_uses_the_public_constructor_and_preserves_llm(tmp_path, cancelled):
    llm = ScriptedModel()
    prompts = []
    agent = None

    def summarize(prompt):
        assert agent._llm is llm
        prompts.append(prompt)
        if cancelled:
            raise asyncio.CancelledError('summary cancelled')
        return 'final answer from the confirmed model'

    agent = ReactAgent(llm, [final_summary_observation], max_retries=1, force_summarize=True,
                       workspace=str(tmp_path), enable_builtin_tools=False, sandbox=None,
                       final_summary_call=summarize)
    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            agent('task')
    else:
        assert agent('task') == 'final answer from the confirmed model'
    assert len(llm.calls) == 2
    assert len(prompts) == 1 and 'completed observation' in prompts[0]
    assert agent._llm is llm


def indexed_summary_observation(index: int) -> str:
    '''Return a distinct observation for each tool round.

    Args:
        index (int): The current tool round.
    '''
    return f'observation-marker-{index}'


@pytest.mark.parametrize('max_retries', [1, 5])
@pytest.mark.parametrize('use_callback', [False, True])
def test_final_summary_includes_the_last_executed_observation(tmp_path, max_retries, use_callback):
    import json
    import lazyllm

    class IndexedModel(ScriptedModel):
        def __call__(self, *args, **kwargs):
            self.calls.append(args)
            index = len(self.calls) - 1
            if index > max_retries:
                return 'final answer'
            return {'content': '', 'tool_calls': [{
                'id': f'observation-{index}', 'type': 'function',
                'function': {'name': 'indexed_summary_observation',
                             'arguments': json.dumps({'index': index})},
            }]}

    llm = IndexedModel()
    prompts = []

    def summarize(prompt):
        prompts.append(prompt)
        return 'final answer'

    agent = ReactAgent(llm, [indexed_summary_observation], max_retries=max_retries, force_summarize=True,
                       workspace=str(tmp_path), enable_builtin_tools=False, sandbox=None,
                       final_summary_call=summarize if use_callback else None)
    assert agent('task', llm_chat_history=[{'role': 'user', 'content': 'old-evidence-marker'}]) == 'final answer'
    prompt = prompts[0] if use_callback else llm.calls[-1][0]
    assert f'observation-marker-{max_retries}' in prompt
    assert prompt.count(f'observation-marker-{max_retries}') == 1
    assert len(llm.calls) == max_retries + 1 + (not use_callback)
    if max_retries == 5:
        assert 'old-evidence-marker' not in prompt
        assert 'observation-marker-0' not in prompt
    # Summary rendering must not consume pending results from the retained workspace.
    workspace = lazyllm.locals['_lazyllm_agent']['workspace']
    assert workspace['tool_call_trace'][0]['tool_call_result']['value'] == f'observation-marker-{max_retries}'
    assert all(f'observation-marker-{max_retries}' not in str(m) for m in workspace['history'])
