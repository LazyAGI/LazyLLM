import os
import shutil
import pytest
import tempfile
import unittest
from unittest.mock import MagicMock
from lazyllm.tools.rag.store_base import LAZY_ROOT_NAME
from lazyllm.tools.rag.map_store import MapStore
from lazyllm.tools.rag.chroma_store import ChromadbStore
from lazyllm.tools.rag.milvus_store import MilvusStore
from lazyllm.tools.rag.doc_node import DocNode
from lazyllm.tools.rag.data_type import DataType
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
@pytest.mark.skip_on_win
@pytest.mark.skip_on_mac
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
        self.assertEqual(nodes[0]._uid, "1")
        self.assertEqual(nodes[1]._uid, "2")
        self.assertEqual(nodes[1].parent._uid, "1")

    def test_insert_dict_as_sparse_embedding(self):
        node1 = DocNode(uid="1", text="text1", group="group1", embedding={'default': {1: 10, 2: 20}})
        node2 = DocNode(uid="2", text="text2", group="group1", embedding={'default': {0: 30, 2: 50}})
        orig_embedding_dict = {
            node1._uid: [0, 10, 20],
            node2._uid: [30, 0, 50],
        }
        self.store.update_nodes([node1, node2])

        results = self.store._peek_all_documents('group1')
        nodes = self.store._build_nodes_from_chroma(results, self.embed_dims)
        nodes_dict = {
            node._uid: node for node in nodes
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
        self.assertEqual(nodes[0]._uid, "1")
        self.assertEqual(nodes[1]._uid, "2")
        self.assertEqual(nodes[1].parent._uid, "1")

    def test_get_group_nodes(self):
        self.store.update_nodes([self.node1, self.node2])
        n1 = self.store.get_nodes("group1", ["1"])[0]
        self.assertEqual(n1.text, self.node1.text)
        n2 = self.store.get_nodes("group1", ["2"])[0]
        self.assertEqual(n2.text, self.node2.text)
        ids = set([self.node1._uid, self.node2._uid])
        docs = self.store.get_nodes("group1")
        self.assertEqual(ids, set([doc._uid for doc in docs]))

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

@pytest.mark.skip_on_win
@pytest.mark.skip_on_mac
class TestMilvusStoreWithNormalEmbedding(unittest.TestCase):
    def setUp(self):
        self.mock_embed = {
            'vec1': MagicMock(return_value=[1.0, 2.0, 3.0]),
            'vec2': MagicMock(return_value=[400.0, 500.0, 600.0, 700.0, 800.0]),
        }
        self.global_metadata_desc = {
            'comment': GlobalMetadataDesc(data_type=DataType.VARCHAR, max_size=65535, default_value=' '),
            'signature': GlobalMetadataDesc(data_type=DataType.VARCHAR, max_size=256, default_value=' '),
            'tags': GlobalMetadataDesc(data_type=DataType.ARRAY, element_type=DataType.INT32, max_size=128,
                                       default_value=[]),
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
        self.embed_datatypes = {
            'vec1': DataType.FLOAT_VECTOR,
            'vec2': DataType.FLOAT_VECTOR,
        }

        self.kwargs = {
            'uri': self.store_file,
            'index_kwargs': {
                'index_type': 'HNSW',
                'metric_type': 'COSINE',
            }
        }

        self.store = MilvusStore(group_embed_keys=self.group_embed_keys, embed=self.mock_embed,
                                 embed_dims=self.embed_dims, embed_datatypes=self.embed_datatypes,
                                 global_metadata_desc=self.global_metadata_desc, **self.kwargs)

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
        self.assertEqual(ret[0]._uid, self.node1._uid)

        self.store.update_nodes([self.node2])
        ret = self.store.query(query='text2', group_name='group1', embed_keys=['vec2'], topk=1)
        self.assertEqual(len(ret), 1)
        self.assertEqual(ret[0]._uid, self.node2._uid)

    def test_remove_and_query(self):
        self.store.update_nodes([self.node1, self.node2])
        ret = self.store.query(query='test', group_name='group1', embed_keys=['vec2'], topk=1)
        self.assertEqual(len(ret), 1)
        self.assertEqual(ret[0]._uid, self.node2._uid)

        self.store.remove_nodes("group1", [self.node2._uid])
        ret = self.store.query(query='test', group_name='group1', embed_keys=['vec2'], topk=1)
        self.assertEqual(len(ret), 1)
        self.assertEqual(ret[0]._uid, self.node1._uid)

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
        self.assertEqual(ret[0]._uid, self.node1._uid)

    def test_query_with_filter_exist_2(self):
        self.store.update_nodes([self.node1, self.node2, self.node3])
        ret = self.store.query(query='test', group_name='group1', embed_keys=['vec2'], topk=10,
                               filters={'comment': ['comment3']})
        self.assertEqual(len(ret), 2)
        self.assertEqual(set([ret[0]._uid, ret[1]._uid]), set([self.node1._uid, self.node2._uid]))

    def test_query_with_filter_non_exist(self):
        self.store.update_nodes([self.node1, self.node3])
        ret = self.store.query(query='test', group_name='group1', embed_keys=['vec1'], topk=10,
                               filters={'comment': ['non-exist']})
        self.assertEqual(len(ret), 0)

    def test_reload(self):
        self.store.update_nodes([self.node1, self.node2, self.node3])
        # reload from storage
        del self.store
        self.store = MilvusStore(group_embed_keys=self.group_embed_keys, embed=self.mock_embed,
                                 embed_dims=self.embed_dims, global_metadata_desc=self.global_metadata_desc,
                                 embed_datatypes=self.embed_datatypes, **self.kwargs)

        nodes = self.store.get_nodes('group1')
        orig_nodes = [self.node1, self.node2, self.node3]
        self.assertEqual(set([node._uid for node in nodes]), set([node._uid for node in orig_nodes]))

        for node in nodes:
            for orig_node in orig_nodes:
                if node._uid == orig_node._uid:
                    self.assertEqual(node.text, orig_node.text)
                    # builtin fields are not in orig node, so we can not use
                    # node.global_metadata == orig_node.global_metadata
                    for k, v in orig_node.global_metadata.items():
                        self.assertEqual(node.global_metadata[k], v)
                    break

    # XXX `array_contains_any` is not supported in local(aka lite) mode. skip this ut
    def _test_query_with_array_filter(self):
        self.store.update_nodes([self.node1, self.node3])
        ret = self.store.query(query='test', group_name='group1', embed_keys=['vec1'], topk=10,
                               filters={'tags': [2]})
        self.assertEqual(len(ret), 2)
        self.assertEqual(set([ret[0]._uid, ret[1]._uid]), set([self.node1._uid, self.node2._uid]))


@pytest.mark.skip_on_win
@pytest.mark.skip_on_mac
class TestMilvusStoreWithSparseEmbedding(unittest.TestCase):
    def setUp(self):
        self.mock_embed = {
            'vec1': MagicMock(return_value={0: 1.0, 1: 2.0, 2: 3.0}),
            'vec2': MagicMock(return_value={0: 400.0, 1: 500.0, 2: 600.0, 3: 700.0, 4: 800.0}),
        }
        self.global_metadata_desc = {
            'comment': GlobalMetadataDesc(data_type=DataType.VARCHAR, max_size=65535, default_value=' '),
        }

        self.node_groups = [LAZY_ROOT_NAME, "group1", "group2"]
        _, self.store_file = tempfile.mkstemp(suffix=".db")

        embed_keys = set(['vec1', 'vec2'])
        self.group_embed_keys = {
            LAZY_ROOT_NAME: embed_keys,
            'group1': embed_keys,
            'group2': embed_keys,
        }
        self.embed_datatypes = {
            'vec1': DataType.SPARSE_FLOAT_VECTOR,
            'vec2': DataType.SPARSE_FLOAT_VECTOR,
        }

        self.kwargs = {
            'uri': self.store_file,
            'index_kwargs': [
                {
                    'embed_key': 'vec1',
                    'index_type': 'SPARSE_INVERTED_INDEX',
                    'metric_type': 'IP',
                },
                {
                    'embed_key': 'vec2',
                    'index_type': 'SPARSE_WAND',
                    'metric_type': 'IP',
                }
            ]
        }

        self.store = MilvusStore(group_embed_keys=self.group_embed_keys, embed=self.mock_embed,
                                 embed_dims=None, embed_datatypes=self.embed_datatypes,
                                 global_metadata_desc=self.global_metadata_desc, **self.kwargs)

        self.node1 = DocNode(uid="1", text="text1", group="group1", parent=None,
                             embedding={"vec1": {0: 1.0, 1: 2.0, 2: 3.0},
                                        "vec2": {0: 400.0, 1: 500.0, 2: 600.0, 3: 700.0, 4: 800.0}})
        self.node2 = DocNode(uid="2", text="text2", group="group1", parent=None,
                             embedding={"vec1": {0: 8.0, 1: 9.0, 2: 10.0},
                                        "vec2": {0: 11.0, 1: 12.0, 2: 13.0, 3: 14.0, 4: 15.0}})

    def tearDown(self):
        os.remove(self.store_file)

    def test_sparse_embedding(self):
        self.store.update_nodes([self.node1, self.node2])

        ret = self.store.query(query='test', group_name='group1', embed_keys=['vec1'], topk=1)
        self.assertEqual(len(ret), 1)
        self.assertEqual(ret[0]._uid, self.node2._uid)

        ret = self.store.query(query='test', group_name='group1', embed_keys=['vec2'], topk=1)
        self.assertEqual(len(ret), 1)
        self.assertEqual(ret[0]._uid, self.node1._uid)
