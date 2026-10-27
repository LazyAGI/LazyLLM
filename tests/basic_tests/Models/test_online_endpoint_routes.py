from unittest.mock import MagicMock, patch

import pytest

from lazyllm.module.llms.onlinemodule.supplier.openai import OpenAIChat
from lazyllm.module.llms.onlinemodule.supplier.siliconflow import SiliconFlowEmbed, SiliconFlowRerank


@pytest.mark.parametrize(('base', 'expected'), [
    ('https://provider.test/v1', 'https://provider.test/v1/chat/completions'),
    ('https://provider.test/v1/', 'https://provider.test/v1/chat/completions'),
    ('https://provider.test/gateway/v1', 'https://provider.test/gateway/v1/chat/completions'),
    ('https://provider.test', 'https://provider.test/chat/completions'),
    ('https://provider.test/v1?version=1', 'https://provider.test/v1/chat/completions?version=1'),
    ('https://provider.test/v1/chat/completions?version=1',
     'https://provider.test/v1/chat/completions?version=1'),
])
@pytest.mark.parametrize('dynamic', [False, True])
def test_chat_request_preserves_base_path(base, expected, dynamic):
    module = OpenAIChat(api_key='test-key', base_url='https://unused.test/v1/' if dynamic else base)
    response = MagicMock()
    response.status_code = 200
    response.text = '{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}'
    response.__enter__.return_value = response
    with patch('requests.post', return_value=response) as post:
        module.forward('hello', stream=False, **({'url': base} if dynamic else {}))
    assert post.call_args.args[0] == expected
    assert post.call_args.kwargs['json']['messages'][-1]['content'] == 'hello'


@pytest.mark.parametrize('url', [
    'https://provider.test/v1', 'https://provider.test/v1/', 'https://provider.test/v1/rerank',
])
@pytest.mark.parametrize('alias', ['url', 'base_url', 'embed_url'])
def test_siliconflow_rerank_request_uses_rerank_route(url, alias):
    module = SiliconFlowRerank(api_key='test-key')
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {'results': [{'index': 0, 'relevance_score': 0.9}]}
    response.__enter__.return_value = response
    with patch('requests.post', return_value=response) as post:
        result = module.forward('query', documents=['doc'], top_n=1, **{alias: url})
    assert module.type == 'RERANK'
    assert post.call_args.args[0] == 'https://provider.test/v1/rerank'
    assert post.call_args.kwargs['json']['query'] == 'query'
    assert post.call_args.kwargs['json']['documents'] == ['doc']
    assert result == [(0, 0.9)]


def test_embedding_retains_embedding_route():
    module = SiliconFlowEmbed(api_key='test-key')
    assert module.type == 'EMBED'
    assert module._get_embed_url('https://provider.test/v1') == 'https://provider.test/v1/embeddings'
