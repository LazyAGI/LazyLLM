import json
from typing import Any, Dict

import pytest

import lazyllm
from lazyllm.tools.agent.rewooAgent import ReWOOAgent


def echo_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    '''Return the input payload.

    Args:
        payload (Dict[str, Any]): Structured data to return.
    '''
    return payload


@pytest.mark.parametrize('payload', [
    {'values': [1, 2]},
    {'values': [[1], [2, 3]]},
    {'text': 'a]b'},
    {'nested': {'values': [1, 2]}},
    {'text': 'plain'},
    {'value': 3},
])
def test_actual_agent_tool_dispatch_preserves_json_arguments(tmp_path, payload):
    llm = lazyllm.TrainableModule('qwen2-1.5b', trust_remote_code=False)
    agent = ReWOOAgent(llm=llm, tools=[echo_payload], enable_builtin_tools=False, workspace=str(tmp_path))
    agent._pre_process('parse-only')
    call = 'echo_payload[' + json.dumps({'payload': payload}) + ']'
    assert agent._parse_and_call_tool(call, {}) == payload


def test_nested_evidence_references_reach_real_tool(tmp_path):
    llm = lazyllm.TrainableModule('qwen2-1.5b', trust_remote_code=False)
    agent = ReWOOAgent(llm=llm, tools=[echo_payload], enable_builtin_tools=False, workspace=str(tmp_path))
    agent._pre_process('parse-only')
    call = 'echo_payload[' + json.dumps({'payload': {'values': ['#E1']}}) + ']'
    assert agent._parse_and_call_tool(call, {'#E1': [1, 2]}) == {'values': [[1, 2]]}
