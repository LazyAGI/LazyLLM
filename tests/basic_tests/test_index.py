import time
import unittest
from unittest.mock import MagicMock
from lazyllm.tools.rag.store import DocNode, MapStore
from lazyllm.tools.rag.index import DefaultIndex, register_similarity


class TestDefaultIndex(unittest.TestCase):
    def setUp(self):
        self.mock_embed = MagicMock(side_effect=self.delayed_embed)
        self.mock_embed1 = MagicMock(return_value=[0, 1, 0])
        self.mock_embed2 = MagicMock(return_value=[0, 0, 1])
        self.mock_store = MagicMock(spec=MapStore)

        # Create instance of DefaultIndex
        self.index = DefaultIndex(embed={"default": self.mock_embed,
                                         "test1": self.mock_embed1,
                                         "test2": self.mock_embed2},
                                  store=self.mock_store)

        # Create mock DocNodes
        self.doc_node_1 = DocNode("text1")
        self.doc_node_1.embedding = {"default": [1, 0, 0], "test1": [1, 0, 0], "test2": [1, 0, 0]}
        self.doc_node_2 = DocNode("text2")
        self.doc_node_2.embedding = {"default": [0, 1, 0], "test1": [0, 1, 0], "test2": [0, 1, 0]}
        self.doc_node_3 = DocNode("text3")
        self.doc_node_3.embedding = {"default": [0, 0, 1], "test1": [0, 0, 1], "test2": [0, 0, 1]}
        self.nodes = [self.doc_node_1, self.doc_node_2, self.doc_node_3]

    def delayed_embed(self, text):
        time.sleep(3)
        return [1, 1, 0]

    def test_register_similarity(self):
        # Register a custom similarity function
        @register_similarity(mode="embedding", batch=True)
        def custom_similarity(query, nodes, **kwargs):
            return [(node, 1.0) for node in nodes]

        self.assertIn("custom_similarity", DefaultIndex.registered_similarity)
        self.assertEqual(
            DefaultIndex.registered_similarity["custom_similarity"][1], "embedding"
        )

    def test_query_cosine_similarity(self):
        results = self.index.query(
            query="test",
            nodes=self.nodes,
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
                nodes=self.nodes,
                similarity_name="invalid_similarity",
                similarity_cut_off=0.0,
                topk=2,
                embed_keys=["default"]
            )

    def test_parallel_do_embedding(self):
        for node in self.nodes:
            node.has_embedding = MagicMock(return_value=False)
        start_time = time.time()
        self.index._parallel_do_embedding(self.nodes)
        assert time.time() - start_time < 4, "Parallel not used!"

    def test_query_multi_embed_similarity(self):
        results = self.index.query(
            query="test",
            nodes=self.nodes,
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
            nodes=self.nodes,
            similarity_name="cosine",
            similarity_cut_off=0.8,
            embed_keys=["default", "test1"],
            topk=2,
        )
        print(f"results: {results}")
        self.assertEqual(len(results), 1)
        self.assertIn(self.doc_node_2, results)

if __name__ == "__main__":
    unittest.main()
