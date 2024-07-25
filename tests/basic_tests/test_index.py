import unittest
from unittest.mock import MagicMock
from lazyllm.tools.rag.store import DocNode, MapStore
from lazyllm.tools.rag.index import DefaultIndex, register_similarity


class TestDefaultIndex(unittest.TestCase):
    def setUp(self):
        self.mock_embed = MagicMock(return_value=[1, 1, 0])
        self.mock_store = MagicMock(spec=MapStore)

        # Create instance of DefaultIndex
        self.index = DefaultIndex(embed=self.mock_embed, store=self.mock_store)

        # Create mock DocNodes
        self.doc_node_1 = DocNode("text1")
        self.doc_node_1.embedding = [1, 0, 0]
        self.doc_node_2 = DocNode("text2")
        self.doc_node_2.embedding = [0, 1, 0]
        self.doc_node_3 = DocNode("text3")
        self.doc_node_3.embedding = [0, 0, 1]
        self.nodes = [self.doc_node_1, self.doc_node_2, self.doc_node_3]

    def test_register_similarity(self):
        # Register a custom similarity function
        @DefaultIndex.register_similarity(mode="embedding", batch=True)
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
        )
        self.assertEqual(len(results), 2)
        print(results)
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
            )


if __name__ == "__main__":
    unittest.main()
