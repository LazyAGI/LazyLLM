from lazyllm.module.llms.onlinemodule.base.onlineChatModuleBase import (
    LazyLLMOnlineChatModuleBase,
)


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
