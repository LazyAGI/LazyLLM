import unittest
from lazyllm.tools.rag.component.bm25 import BM25
from lazyllm.tools.rag.store import DocNode
import numpy as np


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.nodes = [
            DocNode(text="This is a test document."),
            DocNode(text="This document is for testing BM25."),
            DocNode(text="BM25 is a ranking function used in information retrieval."),
        ]

        self.bm25_en = BM25(self.nodes, language="en", topk=2)

    def test_initialization(self):
        self.assertIsInstance(self.bm25_en, BM25)
        self.assertEqual(self.bm25_en.topk, 2)
        self.assertEqual(len(self.bm25_en.nodes), 3)

    def test_retrieve(self):
        query = "test document"
        results = self.bm25_en.retrieve(query)

        self.assertEqual(len(results), 2)

        for node, score in results:
            self.assertIsInstance(node, DocNode)
            self.assertIsInstance(score, np.float32)

        self.assertIn(self.nodes[0], [result[0] for result in results])
        self.assertIn(self.nodes[1], [result[0] for result in results])


if __name__ == "__main__":
    unittest.main()
