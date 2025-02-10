import time
import unittest
from unittest.mock import MagicMock
from lazyllm.tools.rag.map_store import MapStore
from lazyllm.tools.rag import DocNode, IndexBase, StoreBase, Document
from lazyllm.tools.rag.default_index import DefaultIndex
from lazyllm.tools.rag.similarity import register_similarity, registered_similarities
from lazyllm.tools.rag.utils import parallel_do_embedding, generic_process_filters
from typing import List, Optional, Dict
from lazyllm.common import override
from lazyllm import SentenceSplitter, Retriever

class TestDefaultIndex(unittest.TestCase):
    def setUp(self):
        self.mock_embed = {
            'default': MagicMock(side_effect=self.delayed_embed),
            'test1': MagicMock(return_value=[0, 1, 0]),
            'test2': MagicMock(return_value=[0, 0, 1]),
        }
        self.mock_store = MapStore(node_groups=['group1'], embed=self.mock_embed)

        # Create instance of DefaultIndex
        self.index = DefaultIndex(embed=self.mock_embed, store=self.mock_store)

        # Create mock DocNodes
        self.doc_node_1 = DocNode(uid="text1", group="group1")
        self.doc_node_1.embedding = {"default": [1, 0, 0], "test1": [1, 0, 0], "test2": [1, 0, 0]}
        self.doc_node_2 = DocNode(uid="text2", group="group1")
        self.doc_node_2.embedding = {"default": [0, 1, 0], "test1": [0, 1, 0], "test2": [0, 1, 0]}
        self.doc_node_3 = DocNode(uid="text3", group="group1")
        self.doc_node_3.embedding = {"default": [0, 0, 1], "test1": [0, 0, 1], "test2": [0, 0, 1]}
        self.nodes = [self.doc_node_1, self.doc_node_2, self.doc_node_3]
        self.mock_store.update_nodes(self.nodes)  # used by index

    def delayed_embed(self, text):
        time.sleep(3)
        return [1, 1, 0]

    def test_register_similarity(self):
        # Register a custom similarity function
        @register_similarity(mode="embedding", batch=True)
        def custom_similarity(query, nodes, **kwargs):
            return [(node, 1.0) for node in nodes]

        self.assertIn("custom_similarity", registered_similarities)
        self.assertEqual(
            registered_similarities["custom_similarity"][1], "embedding"
        )

    def test_query_cosine_similarity(self):
        results = self.index.query(
            query="test",
            group_name="group1",
            similarity_name="cosine",
            similarity_cut_off=0.0,
            topk=2,
            embed_keys=["default"]
        )
        self.assertEqual(len(results), 2)
        self.assertIn(self.doc_node_1, results)
        self.assertIn(self.doc_node_2, results)

    def test_invalid_similarity_name(self):
        with self.assertRaises(ValueError):
            self.index.query(
                query="test",
                group_name="group1",
                similarity_name="invalid_similarity",
                similarity_cut_off=0.0,
                topk=2,
                embed_keys=["default"]
            )

    def test_parallel_do_embedding(self):
        for node in self.nodes:
            node.has_embedding = MagicMock(return_value=False)
        start_time = time.time()
        parallel_do_embedding(self.index.embed, self.index.embed.keys(), self.nodes)
        assert time.time() - start_time < 4, "Parallel not used!"

    def test_query_multi_embed_similarity(self):
        results = self.index.query(
            query="test",
            group_name="group1",
            similarity_name="cosine",
            similarity_cut_off={"default": 0.8, "test1": 0.8, "test2": 0.8},
            topk=2,
        )
        self.assertEqual(len(results), 2)
        self.assertIn(self.doc_node_2, results)
        self.assertIn(self.doc_node_3, results)

    def test_query_multi_embed_one_thresholds(self):
        results = self.index.query(
            query="test",
            group_name="group1",
            similarity_name="cosine",
            similarity_cut_off=0.8,
            embed_keys=["default", "test1"],
            topk=2,
        )
        print(f"results: {results}")
        self.assertEqual(len(results), 1)
        self.assertIn(self.doc_node_2, results)

class KeywordIndex(IndexBase):
    def __init__(self, cstore: StoreBase):
        self.store = cstore

    @override
    def update(self, nodes: List[DocNode]) -> None:
        pass

    @override
    def remove(self, group_name: str, uids: List[str]) -> None:
        pass

    @override
    def query(self, query: str, group_name: Optional[str] = None,
              filters: Optional[Dict[str, List]] = None, topk: int = 5, **kwargs) -> List[DocNode]:
        nodes = self.store.get_nodes(group_name)
        if filters:
            nodes = generic_process_filters(nodes, filters)

        ranked_nodes = self._synthesize_answer(nodes, query)
        return ranked_nodes[:topk]

    def _synthesize_answer(self, nodes: List[DocNode], query: str) -> List[DocNode]:
        relevant_nodes = [(node, self._is_relevant(node, query)) for node in nodes]
        sorted_nodes = [node for node, count in sorted(relevant_nodes, key=lambda item: item[1], reverse=True)
                        if count > 0]
        return sorted_nodes

    def _is_relevant(self, node: DocNode, query: str) -> int:
        return node.text.encode("utf-8", "ignore").decode("utf-8").casefold().count(
            query.encode("utf-8", "ignore").decode("utf-8").casefold())

class TestIndex(unittest.TestCase):
    def test_index_registration(self):
        doc1 = Document(dataset_path="rag_master", manager=False)
        doc1.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
        ret1 = Retriever(doc1, "CoarseChunk", "bm25_chinese", 0.003, topk=3)
        query = "道"
        nodes = ret1(query)
        nums1 = []
        for node in nodes:
            nums1.append(node.text.lower().count(query.lower()))
        assert len(nums1) == 0
        doc2 = Document(dataset_path="rag_master", manager=False)
        doc2.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
        doc2.register_index("keyword_index", KeywordIndex, doc2.get_store())
        ret2 = Retriever(doc2, "CoarseChunk", "bm25_chinese", 0.003, index="keyword_index", topk=3)
        nodes = ret2(query)
        nums2 = []
        for node in nodes:
            nums2.append(node.text.casefold().count(query.casefold()))
        assert all(query.casefold() in node.text.casefold() for node in nodes) and nums2 == sorted(nums2, reverse=True)

if __name__ == "__main__":
    unittest.main()
