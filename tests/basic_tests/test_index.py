import os
import time
import unittest
import tempfile
from unittest.mock import MagicMock
from lazyllm.tools.rag.store import MapStore, LAZY_ROOT_NAME, MilvusStore, MilvusField
from lazyllm.tools.rag.doc_node import DocNode
from lazyllm.tools.rag.index import DefaultIndex, register_similarity, parallel_do_embedding
import pymilvus

class TestDefaultIndex(unittest.TestCase):
    def setUp(self):
        self.mock_embed = MagicMock(side_effect=self.delayed_embed)
        self.mock_embed1 = MagicMock(return_value=[0, 1, 0])
        self.mock_embed2 = MagicMock(return_value=[0, 0, 1])
        self.mock_store = MapStore(node_groups=['group1'])

        # Create instance of DefaultIndex
        self.index = DefaultIndex(embed={"default": self.mock_embed,
                                         "test1": self.mock_embed1,
                                         "test2": self.mock_embed2},
                                  store=self.mock_store)

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

        self.assertIn("custom_similarity", DefaultIndex.registered_similarity)
        self.assertEqual(
            DefaultIndex.registered_similarity["custom_similarity"][1], "embedding"
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
        parallel_do_embedding(self.index.embed, self.nodes)
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

class TestMilvusIndex(unittest.TestCase):
    def setUp(self):
        field_list = [
            MilvusField(name="comment", data_type=MilvusField.DTYPE_VARCHAR, max_length=128),
            MilvusField(name="vec1", data_type=MilvusField.DTYPE_FLOAT_VECTOR,
                        index_type='HNSW', metric_type='COSINE'),
            MilvusField(name="vec2", data_type=MilvusField.DTYPE_FLOAT_VECTOR,
                        index_type='HNSW', metric_type='COSINE'),
        ]
        group_fields = {
            "group1": field_list,
            "group2": field_list,
        }

        self.mock_embed = {
            'vec1': MagicMock(return_value=[1.0, 2.0, 3.0]),
            'vec2': MagicMock(return_value=[400.0, 500.0, 600.0, 700.0, 800.0]),
        }

        self.node_groups = [LAZY_ROOT_NAME, "group1", "group2"]
        _, self.store_file = tempfile.mkstemp(suffix=".db")

        self.store = MilvusStore(uri=self.store_file, embed=self.mock_embed,
                                 group_fields=group_fields)
        self.index = self.store.get_index()

        self.node1 = DocNode(uid="1", text="text1", group="group1", parent=None,
                             embedding={"vec1": [8.0, 9.0, 10.0], "vec2": [11.0, 12.0, 13.0, 14.0, 15.0]},
                             metadata={'comment': 'comment1'})
        self.node2 = DocNode(uid="2", text="text2", group="group1", parent=self.node1,
                             embedding={"vec1": [100.0, 200.0, 300.0], "vec2": [400.0, 500.0, 600.0, 700.0, 800.0]},
                             metadata={'comment': 'comment2'})

    def tearDown(self):
        os.remove(self.store_file)

    def test_update_and_query(self):
        self.store.update_nodes([self.node1])
        ret = self.index.query(query='text1', group_name='group1', embed_keys=['vec2'], topk=1)
        self.assertEqual(len(ret), 1)
        self.assertEqual(ret[0].uid, self.node1.uid)

        self.store.update_nodes([self.node2])
        ret = self.index.query(query='text2', group_name='group1', embed_keys=['vec2'], topk=1)
        self.assertEqual(len(ret), 1)
        self.assertEqual(ret[0].uid, self.node2.uid)

    def test_remove_and_query(self):
        self.store.update_nodes([self.node1, self.node2])
        ret = self.index.query(query='test', group_name='group1', embed_keys=['vec2'], topk=1)
        self.assertEqual(len(ret), 1)
        self.assertEqual(ret[0].uid, self.node2.uid)

        self.store.remove_nodes("group1", [self.node2.uid])
        ret = self.index.query(query='test', group_name='group1', embed_keys=['vec2'], topk=1)
        self.assertEqual(len(ret), 1)
        self.assertEqual(ret[0].uid, self.node1.uid)

if __name__ == "__main__":
    unittest.main()
