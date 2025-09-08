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
        embed_model2 = lazyllm.OnlineEmbeddingModule(num_worker=4)
        vec1 = embed_model("床前明月光")
        vec2 = embed_model(["床前明月光", "疑是地上霜"])
        assert len(vec2) == 2
        assert len(vec2[0]) == len(vec1)
        assert len(vec2[1]) == len(vec1)
        vec3 = embed_model2(["床前明月光", "疑是地上霜", "床前明月光", "疑是地上霜", "床前明月光", "疑是地上霜", "床前明月光", "疑是地上霜"])
        assert len(vec3) == 8
        assert len(vec3[0]) == len(vec1)
        assert len(vec3[1]) == len(vec1)

    def test__embed_adjust_batch(self):
        embed_model = lazyllm.OnlineEmbeddingModule(source="qwen", embed_model_name="text-embedding-v3",
                                                    num_worker=4, batch_size=20)
        vec2 = embed_model(["床前明月光" for i in range(0, 20)])
        assert len(vec2) == 20
