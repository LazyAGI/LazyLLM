import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock
from lazyllm.tools.rag.store_base import LAZY_ROOT_NAME
from lazyllm.tools.rag.map_store import MapStore
from lazyllm.tools.rag.chroma_store import ChromadbStore
from lazyllm.tools.rag.milvus_store import MilvusStore
from lazyllm.tools.rag.doc_node import DocNode
from lazyllm.tools.rag.global_metadata import GlobalMetadataDesc


def clear_directory(directory_path):
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"The directory {directory_path} does not exist.")

# Test class for ChromadbStore
class TestChromadbStore(unittest.TestCase):
    def setUp(self):
        self.node_groups = [LAZY_ROOT_NAME, "group1", "group2"]
        self.store_dir = tempfile.mkdtemp()
        self.mock_embed = {
            'default': MagicMock(return_value=[1.0, 2.0, 3.0]),
        }
        self.embed_dims = {"default": 3}

        embed_keys = set(['default'])
        group_embed_keys = {
            LAZY_ROOT_NAME: embed_keys,
            'group1': embed_keys,
            'group2': embed_keys,
        }
        self.store = ChromadbStore(group_embed_keys=group_embed_keys, embed=self.mock_embed,
                                   embed_dims=self.embed_dims, dir=self.store_dir)

        self.store.update_nodes(
            [DocNode(uid="1", text="text1", group=LAZY_ROOT_NAME, parent=None)],
        )

    def tearDown(self):
        clear_directory(self.store_dir)

    def test_initialization(self):
        self.assertEqual(set(self.store._collections.keys()), set(self.node_groups))

    def test_update_nodes(self):
        node1 = DocNode(uid="1", text="text1", group="group1")
        node2 = DocNode(uid="2", text="text2", group="group2")
        self.store.update_nodes([node1, node2])
        collection = self.store._collections["group1"]
        self.assertEqual(set(collection.peek(collection.count())["ids"]), set(["1", "2"]))
        nodes = self.store.get_nodes("group1")
        self.assertEqual(nodes, [node1])

    def test_remove_group_nodes(self):
        node1 = DocNode(uid="1", text="text1", group="group1")
        node2 = DocNode(uid="2", text="text2", group="group2")
        self.store.update_nodes([node1, node2])
        collection = self.store._collections["group1"]
        self.assertEqual(collection.peek(collection.count())["ids"], ["1", "2"])
        self.store.remove_nodes("group1", "1")
        self.assertEqual(collection.peek(collection.count())["ids"], ["2"])

    def test_load_store(self):
        # Set up initial data to be loaded
        node1 = DocNode(uid="1", text="text1", group="group1", parent=None)
        node2 = DocNode(uid="2", text="text2", group="group1", parent=node1)
        self.store.update_nodes([node1, node2])

        # Reset store and load from "persistent" storage
        self.store._map_store._group2docs = {group: {} for group in self.node_groups}
        self.store._load_store(self.embed_dims)

        nodes = self.store.get_nodes("group1")
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].uid, "1")
        self.assertEqual(nodes[1].uid, "2")
        self.assertEqual(nodes[1].parent.uid, "1")

    def test_insert_dict_as_sparse_embedding(self):
        node1 = DocNode(uid="1", text="text1", group="group1", embedding={'default': {1: 10, 2: 20}})
        node2 = DocNode(uid="2", text="text2", group="group1", embedding={'default': {0: 30, 2: 50}})
        orig_embedding_dict = {
            node1.uid: [0, 10, 20],
            node2.uid: [30, 0, 50],
        }
        self.store.update_nodes([node1, node2])

        results = self.store._peek_all_documents('group1')
        nodes = self.store._build_nodes_from_chroma(results, self.embed_dims)
        nodes_dict = {
            node.uid: node for node in nodes
        }

        assert nodes_dict.keys() == orig_embedding_dict.keys()
        for uid, node in nodes_dict.items():
            assert node.embedding['default'] == orig_embedding_dict.get(uid)

    def test_all_groups(self):
        self.assertEqual(set(self.store.all_groups()), set(self.node_groups))

    def test_query(self):
        node1 = DocNode(uid="1", text="text1", group="group1", parent=None)
        node2 = DocNode(uid="2", text="text2", group="group1", parent=node1)
        self.store.update_nodes([node1, node2])
        res = self.store.query(query='text1', group_name='group1', embed_keys=['default'], topk=2,
                               similarity_name='cosine', similarity_cut_off=0.000001)
        self.assertEqual(set([node1, node2]), set(res))

    def test_group_others(self):
        node1 = DocNode(uid="1", text="text1", group="group1", parent=None)
        node2 = DocNode(uid="2", text="text2", group="group1", parent=node1)
        self.store.update_nodes([node1, node2])
        self.assertEqual(self.store.is_group_active("group1"), True)
        self.assertEqual(self.store.is_group_active("group2"), False)

class TestMapStore(unittest.TestCase):
    def setUp(self):
        self.mock_embed = {
            'default': MagicMock(return_value=[1.0, 2.0, 3.0]),
        }
        self.node_groups = [LAZY_ROOT_NAME, "group1", "group2"]
        self.store = MapStore(node_groups=self.node_groups, embed=self.mock_embed)
        self.node1 = DocNode(uid="1", text="text1", group="group1", parent=None)
        self.node2 = DocNode(uid="2", text="text2", group="group1", parent=self.node1)

    def test_update_nodes(self):
        self.store.update_nodes([self.node1, self.node2])
        nodes = self.store.get_nodes("group1")
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].uid, "1")
        self.assertEqual(nodes[1].uid, "2")
        self.assertEqual(nodes[1].parent.uid, "1")

    def test_get_group_nodes(self):
        self.store.update_nodes([self.node1, self.node2])
        n1 = self.store.get_nodes("group1", ["1"])[0]
        self.assertEqual(n1.text, self.node1.text)
        n2 = self.store.get_nodes("group1", ["2"])[0]
        self.assertEqual(n2.text, self.node2.text)
        ids = set([self.node1.uid, self.node2.uid])
        docs = self.store.get_nodes("group1")
        self.assertEqual(ids, set([doc.uid for doc in docs]))

    def test_remove_group_nodes(self):
        self.store.update_nodes([self.node1, self.node2])

        n1 = self.store.get_nodes("group1", ["1"])[0]
        assert n1.text == self.node1.text
        self.store.remove_nodes("group1", ["1"])
        n1 = self.store.get_nodes("group1", ["1"])
        assert not n1

        n2 = self.store.get_nodes("group1", ["2"])[0]
        assert n2.text == self.node2.text
        self.store.remove_nodes("group1", ["2"])
        n2 = self.store.get_nodes("group1", ["2"])
        assert not n2

    def test_all_groups(self):
        self.assertEqual(set(self.store.all_groups()), set(self.node_groups))

    def test_query(self):
        self.store.update_nodes([self.node1, self.node2])
        res = self.store.query(query='text1', group_name='group1', embed_keys=['default'], topk=2,
                               similarity_name='cosine', similarity_cut_off=0.000001)
        self.assertEqual(set([self.node1, self.node2]), set(res))

    def test_group_others(self):
        self.store.update_nodes([self.node1, self.node2])
        self.assertEqual(self.store.is_group_active("group1"), True)
        self.assertEqual(self.store.is_group_active("group2"), False)

class TestMilvusStore(unittest.TestCase):
    def setUp(self):
        self.mock_embed = {
            'vec1': MagicMock(return_value=[1.0, 2.0, 3.0]),
            'vec2': MagicMock(return_value=[400.0, 500.0, 600.0, 700.0, 800.0]),
        }
        self.global_metadata_desc = {
            'comment': GlobalMetadataDesc(data_type=GlobalMetadataDesc.DTYPE_VARCHAR, max_size=65535, default_value=' '),
        }

        self.node_groups = [LAZY_ROOT_NAME, "group1", "group2"]
        _, self.store_file = tempfile.mkstemp(suffix=".db")

        embed_keys = set(['vec1', 'vec2'])
        self.group_embed_keys = {
            LAZY_ROOT_NAME: embed_keys,
            'group1': embed_keys,
            'group2': embed_keys,
        }
        self.embed_dims = {
            "vec1": 3,
            "vec2": 5,
        }
        self.store = MilvusStore(group_embed_keys=self.group_embed_keys, embed=self.mock_embed,
                                 embed_dims=self.embed_dims, global_metadata_desc=self.global_metadata_desc,
                                 uri=self.store_file)

        self.node1 = DocNode(uid="1", text="text1", group="group1", parent=None,
                             embedding={"vec1": [8.0, 9.0, 10.0], "vec2": [11.0, 12.0, 13.0, 14.0, 15.0]},
                             metadata={'comment': 'comment1'},
                             global_metadata={'comment': 'comment3', 'signature': 'node1', 'tags': [1, 3, 5]})
        self.node2 = DocNode(uid="2", text="text2", group="group1", parent=self.node1,
                             embedding={"vec1": [100.0, 200.0, 300.0], "vec2": [400.0, 500.0, 600.0, 700.0, 800.0]},
                             metadata={'comment': 'comment2', 'signature': 'node2'})
        self.node3 = DocNode(uid="3", text="text3", group="group1", parent=None,
                             embedding={"vec1": [4.0, 5.0, 6.0], "vec2": [16.0, 17.0, 18.0, 19.0, 20.0]},
                             metadata={'comment': 'comment3', 'signature': 'node3'},
                             global_metadata={'tags': [1, 2, 3]})

    def tearDown(self):
        os.remove(self.store_file)

    def test_update_and_query(self):
        self.store.update_nodes([self.node1])
        ret = self.store.query(query='text1', group_name='group1', embed_keys=['vec2'], topk=1)
        self.assertEqual(len(ret), 1)
        self.assertEqual(ret[0].uid, self.node1.uid)

        self.store.update_nodes([self.node2])
        ret = self.store.query(query='text2', group_name='group1', embed_keys=['vec2'], topk=1)
        self.assertEqual(len(ret), 1)
        self.assertEqual(ret[0].uid, self.node2.uid)

    def test_remove_and_query(self):
        self.store.update_nodes([self.node1, self.node2])
        ret = self.store.query(query='test', group_name='group1', embed_keys=['vec2'], topk=1)
        self.assertEqual(len(ret), 1)
        self.assertEqual(ret[0].uid, self.node2.uid)

        self.store.remove_nodes("group1", [self.node2.uid])
        ret = self.store.query(query='test', group_name='group1', embed_keys=['vec2'], topk=1)
        self.assertEqual(len(ret), 1)
        self.assertEqual(ret[0].uid, self.node1.uid)

    def test_all_groups(self):
        self.assertEqual(set(self.store.all_groups()), set(self.node_groups))

    def test_group_others(self):
        self.store.update_nodes([self.node1, self.node2])
        self.assertEqual(self.store.is_group_active("group1"), True)
        self.assertEqual(self.store.is_group_active("group2"), False)

    def test_query_with_filter_exist_1(self):
        self.store.update_nodes([self.node1, self.node3])
        ret = self.store.query(query='test', group_name='group1', embed_keys=['vec2'], topk=10,
                               filters={'comment': ['comment3']})
        self.assertEqual(len(ret), 1)
        self.assertEqual(ret[0].uid, self.node1.uid)

    def test_query_with_filter_exist_2(self):
        self.store.update_nodes([self.node1, self.node2, self.node3])
        ret = self.store.query(query='test', group_name='group1', embed_keys=['vec2'], topk=10,
                               filters={'comment': ['comment3']})
        self.assertEqual(len(ret), 2)
        self.assertEqual(set([ret[0].uid, ret[1].uid]), set([self.node1.uid, self.node2.uid]))

    def test_query_with_filter_non_exist(self):
        self.store.update_nodes([self.node1, self.node3])
        ret = self.store.query(query='test', group_name='group1', embed_keys=['vec1'], topk=10,
                               filters={'comment': ['non-exist']})
        self.assertEqual(len(ret), 0)

    def test_reload(self):
        self.store.update_nodes([self.node1, self.node2, self.node3])
        del self.store
        self.store = MilvusStore(group_embed_keys=self.group_embed_keys, embed=self.mock_embed,
                                 embed_dims=self.embed_dims, global_metadata_desc=self.global_metadata_desc,
                                 uri=self.store_file)
        self.assertEqual(set([node.uid for node in self.store.get_nodes('group1')]),
                         set([self.node1.uid, self.node2.uid, self.node3.uid]))

    # XXX `array_contains_any` is not supported in local(aka lite) mode. skip this ut
    def _test_query_with_array_filter(self):
        self.store.update_nodes([self.node1, self.node3])
        ret = self.store.query(query='test', group_name='group1', embed_keys=['vec1'], topk=10,
                               filters={'tags': [2]})
        self.assertEqual(len(ret), 2)
        self.assertEqual(set([ret[0].uid, ret[1].uid]), set([self.node1.uid, self.node2.uid]))
