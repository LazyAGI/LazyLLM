from lazyllm import locals
from lazyllm.tools.agent.rewooAgent import ReWOOAgent


class _ToolManagerStub:
    _safe_parse_json = staticmethod(lambda raw: __import__('json').loads(raw))

    def __init__(self):
        self.calls = []

    def __call__(self, tool_calls):
        self.calls.append(tool_calls)
        name = tool_calls[0]['function']['name']
        value = 68 if name == 'add_tool' else 1360
        return [{'ok': True, 'value': value}]


def _agent_with_stub():
    agent = object.__new__(ReWOOAgent)
    agent._tools_manager = _ToolManagerStub()
    agent._stream = False
    locals['_lazyllm_agent']['workspace'] = {'tool_call_trace': []}
    return agent


def test_rewoo_unwraps_tool_result_before_resolving_dependent_arguments():
    agent = _agent_with_stub()
    evidence = {}

    evidence['#E1'] = agent._parse_and_call_tool('add_tool[{"a": 45, "b": 23}]', evidence)
    evidence['#E2'] = agent._parse_and_call_tool('multiply_tool[{"a": 20, "b": "#E1"}]', evidence)

    assert evidence == {'#E1': 68, '#E2': 1360}
    assert agent._tools_manager.calls[1][0]['function']['arguments'] == {'a': 20, 'b': 68}


def test_rewoo_resolves_evidence_embedded_in_text_without_corrupting_json():
    resolved = ReWOOAgent._resolve_evidence_refs(
        {'input': 'Use #E1 to continue', 'payload': '#E2'},
        {'#E1': {'ok': True}, '#E2': [1, 2]},
    )

    assert resolved == {'input': 'Use {"ok": true} to continue', 'payload': [1, 2]}
