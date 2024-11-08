import time
import unittest
from unittest.mock import MagicMock
from lazyllm.tools.rag.map_store import MapStore
from lazyllm.tools.rag.doc_node import DocNode
from lazyllm.tools.rag.default_index import DefaultIndex
from lazyllm.tools.rag.similarity import register_similarity, registered_similarities
from lazyllm.tools.rag.utils import parallel_do_embedding

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

if __name__ == "__main__":
    unittest.main()
