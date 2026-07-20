import requests

from lazyllm.module.llms.onlinemodule.base.onlineChatModuleBase import (
    _is_input_inspection_failure,
    _remove_prior_tool_traces,
)


def test_input_inspection_failure_detection_is_specific():
    assert _is_input_inspection_failure(
        requests.RequestException('400: {"code":"data_inspection_failed"}')
    )
    assert not _is_input_inspection_failure(requests.RequestException('500: unavailable'))


def test_prior_tool_traces_are_removed_without_losing_conversation_or_current_observation():
    messages = [
        {'role': 'system', 'content': 'system'},
        {'role': 'user', 'content': 'original goal'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'old'}]},
        {'role': 'tool', 'tool_call_id': 'old', 'content': 'untrusted web payload'},
        {'role': 'assistant', 'content': 'prior final answer'},
        {'role': 'user', 'content': 'continue with that goal'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'current'}]},
        {'role': 'tool', 'tool_call_id': 'current', 'content': 'current observation'},
    ]

    sanitized = _remove_prior_tool_traces(messages)

    assert [message.get('content') for message in sanitized] == [
        'system', 'original goal', 'prior final answer', 'continue with that goal', '', 'current observation',
    ]
    assert sanitized[-2]['tool_calls'] == [{'id': 'current'}]
