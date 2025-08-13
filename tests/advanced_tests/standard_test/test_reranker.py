import unittest
import os
from lazyllm.tools.rag.doc_node import DocNode
from lazyllm.tools.rag.rerank import Reranker, register_reranker


class TestReranker(unittest.TestCase):

    def setUp(self):
        self.doc1 = DocNode(text="This is a test document with the keyword apple.")
        self.doc2 = DocNode(
            text="This is another test document with the keyword banana."
        )
        self.doc3 = DocNode(text="This document contains the keyword cherry.")
        self.nodes = [self.doc1, self.doc2, self.doc3]
        self.query = "test query"

    def test_keyword_filter_with_required_keys(self):
        required_keys = ["apple"]
        exclude_keys = []
        reranker = Reranker(
            name="KeywordFilter", required_keys=required_keys, exclude_keys=exclude_keys
        )
        results = reranker.forward(self.nodes, query=self.query)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get_text(), self.doc1.get_text())

    def test_keyword_filter_with_exclude_keys(self):
        required_keys = []
        exclude_keys = ["banana"]
        reranker = Reranker(
            name="KeywordFilter", required_keys=required_keys, exclude_keys=exclude_keys
        )
        results = reranker.forward(self.nodes, query=self.query)
        self.assertEqual(len(results), 2)
        self.assertNotIn(self.doc2, results)

    def test_module_reranker(self):
        env_key = 'LAZYLLM_DEFAULT_EMBEDDING_ENGINE'
        test_cases = ['', 'transformers']
        original_value = os.getenv(env_key, None)
        for value in test_cases:
            with self.subTest(value=value):
                os.environ[env_key] = value
                reranker = Reranker(name="ModuleReranker", model="bge-reranker-large", topk=2)
                reranker.start()
                results = reranker.forward(self.nodes, query='cherry')

                self.assertEqual(len(results), 2)
                self.assertEqual(
                    results[0].get_text(), self.doc3.get_text()
                )  # highest score
                assert results[0].relevance_score > results[1].relevance_score
        if original_value:
            os.environ[env_key] = original_value

    def test_register_reranker_decorator(self):
        @register_reranker
        def CustomReranker(node, **kwargs):
            if "custom" in node.get_text():
                return node
            return None

        custom_doc = DocNode(text="This document contains custom keyword.")
        nodes = [self.doc1, self.doc2, self.doc3, custom_doc]

        reranker = Reranker(name="CustomReranker")
        results = reranker.forward(nodes)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get_text(), custom_doc.get_text())
