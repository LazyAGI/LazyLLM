import requests

from lazyllm.module.llms.onlinemodule.base.onlineChatModuleBase import (
    LazyLLMOnlineChatModuleBase,
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


def test_streamed_tool_calls_merge_by_index_when_chunk_lengths_differ():
    module = LazyLLMOnlineChatModuleBase.__new__(LazyLLMOnlineChatModuleBase)
    chunks = [
        [{
            'index': 0,
            'id': 'call-0',
            'type': 'function',
            'function': {'name': 'list_data_sources', 'arguments': ''},
        }],
        [
            {
                'index': 0,
                'id': '',
                'type': 'function',
                'function': {'arguments': '{}'},
            },
            {
                'index': 1,
                'id': 'call-1',
                'type': 'function',
                'function': {'name': 'get_TavilySearch_methods', 'arguments': '{'},
            },
        ],
        [{
            'index': 1,
            'id': '',
            'type': 'function',
            'function': {'arguments': '}'},
        }],
    ]

    assert module._merge_stream_result(chunks) == [
        {
            'index': 0,
            'id': 'call-0',
            'type': 'function',
            'function': {'name': 'list_data_sources', 'arguments': '{}'},
        },
        {
            'index': 1,
            'id': 'call-1',
            'type': 'function',
            'function': {'name': 'get_TavilySearch_methods', 'arguments': '{}'},
        },
    ]


def test_streamed_single_choice_remains_a_list_after_merge():
    module = LazyLLMOnlineChatModuleBase.__new__(LazyLLMOnlineChatModuleBase)
    chunks = [
        {
            'choices': [{
                'index': 0,
                'delta': {'role': 'assistant', 'content': '{"content": "'},
                'finish_reason': None,
            }],
        },
        {
            'choices': [{
                'index': 0,
                'delta': {'content': 'polished"}'},
                'finish_reason': 'stop',
            }],
        },
    ]

    merged = module._merge_stream_result(chunks)

    assert isinstance(merged['choices'], list)
    assert merged['choices'] == [{
        'index': 0,
        'delta': {'role': 'assistant', 'content': '{"content": "polished"}'},
        'finish_reason': 'stop',
    }]
    assert module._extract_specified_key_fields(merged) == {
        'role': 'assistant',
        'content': '{"content": "polished"}',
    }


def test_streamed_single_choice_and_single_tool_call_both_remain_lists():
    module = LazyLLMOnlineChatModuleBase.__new__(LazyLLMOnlineChatModuleBase)
    chunks = [
        {
            'choices': [{
                'index': 0,
                'delta': {
                    'role': 'assistant',
                    'tool_calls': [{
                        'index': 0,
                        'id': 'call-0',
                        'type': 'function',
                        'function': {'name': 'create_subagent', 'arguments': '{'},
                    }],
                },
            }],
        },
        {
            'choices': [{
                'index': 0,
                'delta': {
                    'tool_calls': [{
                        'index': 0,
                        'function': {'arguments': '}'},
                    }],
                },
                'finish_reason': 'tool_calls',
            }],
        },
    ]

    merged = module._merge_stream_result(chunks)

    assert isinstance(merged['choices'], list)
    tool_calls = merged['choices'][0]['delta']['tool_calls']
    assert isinstance(tool_calls, list)
    assert tool_calls == [{
        'index': 0,
        'id': 'call-0',
        'type': 'function',
        'function': {'name': 'create_subagent', 'arguments': '{}'},
    }]
