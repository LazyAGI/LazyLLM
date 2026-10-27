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
