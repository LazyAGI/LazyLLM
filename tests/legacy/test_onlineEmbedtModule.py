import unittest
import lazyllm

class TestOnlineEmbedModule(unittest.TestCase):

    def test_openai_embedding(self):
        m = lazyllm.OnlineEmbeddingModule(source="openai", embed_url="https://gf.nekoapi.com/v1/embeddings")
        emb = m("hello world")
        print(f"emb: {emb}")

    def test_openai_isinstance(self):
        m = lazyllm.OnlineEmbeddingModule(source="openai")
        self.assertTrue(isinstance(m, lazyllm.OnlineEmbeddingModule))

    def test_glm_embedding(self):
        m = lazyllm.OnlineEmbeddingModule(source="glm")
        emb = m("hello world")
        print(f"emb: {emb}")

    def test_qwen_embedding(self):
        m = lazyllm.OnlineEmbeddingModule(source="qwen")
        emb = m("hello world")
        print(f"emb: {emb}")

    def test_sensenova_embedding(self):
        m = lazyllm.OnlineEmbeddingModule(source="sensenova")
        emb = m("hello world")
        print(f"emb: {emb}")

if __name__ == '__main__':
    unittest.main()
