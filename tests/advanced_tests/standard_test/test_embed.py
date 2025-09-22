import lazyllm
from lazyllm.tools.rag.embed_wrapper import _EmbedWrapper

class TestEmbed(object):
    def test_embed_batch(self):
        embed_model = lazyllm.TrainableModule('Qwen3-Embedding-8B').deploy_method(lazyllm.deploy.vllm).start()
        embed_model = _EmbedWrapper(embed_model)
        vec1 = embed_model('床前明月光')
        vec2 = embed_model(['床前明月光', '疑是地上霜'])
        assert len(vec2) == 2
        assert len(vec2[0]) == len(vec1)
        assert len(vec2[1]) == len(vec1)
