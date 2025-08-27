import lazyllm
from lazyllm.tools.rag.doc_impl import embed_wrapper

class TestEmbed(object):
    def test_embed_batch(self):
        embed_model = lazyllm.TrainableModule("bge-large-zh-v1.5").start()
        embed_model = embed_wrapper(embed_model)
        vec1 = embed_model("床前明月光")
        vec2 = embed_model(["床前明月光", "疑是地上霜"])
        assert len(vec2) == 2
        assert len(vec2[0]) == len(vec1)
        assert len(vec2[1]) == len(vec1)

    def test_online_embed_batch(self):
        embed_model = lazyllm.OnlineEmbeddingModule()
        vec1 = embed_model("床前明月光")
        vec2 = embed_model(["床前明月光", "疑是地上霜"])
        assert len(vec2) == 2
        assert len(vec2[0]) == len(vec1)
        assert len(vec2[1]) == len(vec1)
