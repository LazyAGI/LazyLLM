import lazyllm
from lazyllm.tools.rag.transform import SentenceSplitter
from lazyllm.tools.rag import Document, Retriever

class TestDocument:
    def test_multi_embedding_with_document(self):
        self.embed_model1 = lazyllm.TrainableModule("bge-large-zh-v1.5").start()
        self.embed_model2 = lazyllm.TrainableModule("bge-m3").start()

        Document(dataset_path="rag_master")._impl._dlm.release()
        document1 = Document(dataset_path="rag_master", embed=self.embed_model1)
        document1.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
        retriever1 = Retriever(document1, group_name="sentences", similarity="cosine", topk=10)
        nodes1 = retriever1("何为天道?")
        assert len(nodes1) == 10

        document2 = Document(dataset_path="rag_master", embed={"m1": self.embed_model1, "m2": self.embed_model2})
        document2.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
        retriever2 = Retriever(document2, group_name="sentences", similarity="cosine", topk=3)
        nodes2 = retriever2("何为天道?")
        assert len(nodes2) >= 3

        document3 = Document(dataset_path="rag_master", embed={"m1": self.embed_model1, "m2": self.embed_model2})
        document3.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
        retriever3 = Retriever(document3, group_name="sentences", similarity="cosine",
                               similarity_cut_off={"m1": 0.5, "m2": 0.55}, topk=3, output_format='content', join=True)
        nodes3_text = retriever3("何为天道?")
        assert '观天之道' in nodes3_text or '天命之谓性' in nodes3_text
