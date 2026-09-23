import copy

import pytest

from lazyllm.tools.agent import describe_tool_turns


def _turn():
    return [
        {'role': 'assistant', 'tool_calls': [{'id': 'a'}, {'id': 'b'}]},
        {'role': 'tool', 'tool_call_id': 'b', 'content': 'B'},
        {'role': 'tool', 'tool_call_id': 'a', 'content': 'A'},
    ]


@pytest.mark.parametrize('split', [0, 1, 2, 3])
def test_complete_turn_contract_is_independent_of_call_result_split(split):
    messages = _turn()
    original = copy.deepcopy(messages)
    turns = describe_tool_turns(messages[:split], messages[split:])
    assert len(turns) == 1
    turn = turns[0]
    assert (turn.start, turn.stop, turn.result_indexes) == (0, 3, (1, 2))
    assert turn.current is (split < 3)
    assert messages == original


@pytest.mark.parametrize('malformation', [
    'duplicate_call', 'missing_id', 'missing_result', 'foreign_result',
    'duplicate_result', 'interleaved_user', 'interleaved_assistant',
])
def test_malformed_turn_is_not_exposed_as_removable(malformation):
    messages = _turn()
    if malformation == 'duplicate_call':
        messages[0]['tool_calls'][1]['id'] = 'a'
    elif malformation == 'missing_id':
        messages[0]['tool_calls'][1].pop('id')
    elif malformation == 'missing_result':
        messages.pop()
    elif malformation == 'foreign_result':
        messages[1]['tool_call_id'] = 'other'
    elif malformation == 'duplicate_result':
        messages[2]['tool_call_id'] = 'b'
    else:
        messages.insert(2, {'role': malformation.removeprefix('interleaved_'), 'content': 'interruption'})
    assert describe_tool_turns(messages, []) == ()


def test_malformed_round_does_not_prevent_later_complete_round():
    messages = _turn()[:1] + _turn()
    turns = describe_tool_turns(messages, [])
    assert len(turns) == 1
    assert (turns[0].start, turns[0].stop) == (1, 4)
