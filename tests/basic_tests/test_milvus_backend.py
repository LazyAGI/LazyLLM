import os
import unittest
import tempfile
from unittest.mock import MagicMock
from lazyllm.tools.rag.doc_node import DocNode
from lazyllm.tools.rag.store import LAZY_ROOT_NAME
from lazyllm.tools.rag.milvus_backend import MilvusStore, MilvusField

class TestMilvusBackend(unittest.TestCase):
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
