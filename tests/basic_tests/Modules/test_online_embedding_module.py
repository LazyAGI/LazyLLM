import pytest

import lazyllm
from lazyllm.module.llms.onlinemodule.supplier.qwen import QwenEmbed


class TestOnlineEmbeddingModule:

    def test_accepts_model_keyword_alias(self):
        module = lazyllm.OnlineEmbeddingModule(
            source='qwen',
            api_key='test-key',
            model='text-embedding-v1',
        )

        assert isinstance(module, QwenEmbed)
        assert module._embed_model_name == 'text-embedding-v1'

    def test_accepts_model_as_first_positional_argument(self):
        module = lazyllm.OnlineEmbeddingModule(
            'text-embedding-v1',
            source='qwen',
            api_key='test-key',
        )

        assert isinstance(module, QwenEmbed)
        assert module._embed_model_name == 'text-embedding-v1'

    def test_keeps_source_as_first_positional_argument_for_backwards_compatibility(self):
        module = lazyllm.OnlineEmbeddingModule('qwen', api_key='test-key')

        assert isinstance(module, QwenEmbed)

    def test_rejects_conflicting_model_arguments(self):
        with pytest.raises(ValueError, match='Conflicting values were provided'):
            lazyllm.OnlineEmbeddingModule(
                'text-embedding-v1',
                source='qwen',
                api_key='test-key',
                model='text-embedding-v3',
            )
