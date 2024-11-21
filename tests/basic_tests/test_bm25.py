import unittest
from lazyllm.tools.rag.component.bm25 import BM25
from lazyllm.tools.rag.doc_node import DocNode
from lazyllm.thirdparty import numpy as np


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


class TestBM25Chinese(unittest.TestCase):
    def setUp(self):
        self.nodes = [
            DocNode(text="这是一个测试文档。这个文档用于测试BM25。"),
            DocNode(
                text="BM25是一种在信息检索中使用的排序函数。信息检索系统通过BM25算法来排序文档和分数。"
            ),
            DocNode(text="中文文档的测试内容。测试文档中包含多个句子。"),
            DocNode(
                text="这个测试是为了验证BM25在中文文档中的表现。我们需要对多个文档进行排序测试。"
            ),
            DocNode(
                text="文档的内容可以影响BM25的评分。排序函数的性能对于信息检索非常重要。"
            ),
        ]

        self.bm25_cn = BM25(self.nodes, language="zh", topk=3)

    def test_retrieve(self):
        query = "测试文档"
        results = self.bm25_cn.retrieve(query)

        self.assertEqual(len(results), 3)

        self.assertIn(self.nodes[0], [result[0] for result in results])
        self.assertIn(self.nodes[2], [result[0] for result in results])
        self.assertIn(self.nodes[3], [result[0] for result in results])


if __name__ == "__main__":
    unittest.main()
