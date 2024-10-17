import os
import shutil
import unittest
import lazyllm
import tempfile
import pymilvus
from unittest.mock import MagicMock
from lazyllm.tools.rag.store import (
    DocNode,
    MapStore,
    ChromadbStore,
    MilvusStore, MilvusEmbeddingIndexField,
    LAZY_ROOT_NAME
)


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
        self.embed = {"default": MagicMock(side_effect=lambda text: [0.1, 0.2, 0.3])}
        self.store = ChromadbStore(self.node_groups, self.embed)
        self.store.update_nodes(
            [DocNode(uid="1", text="text1", group=LAZY_ROOT_NAME, parent=None)],
        )

    @classmethod
    def tearDownClass(cls):
        clear_directory(lazyllm.config['rag_persistent_path'])

    def test_initialization(self):
        self.assertEqual(set(self.store._collections.keys()), set(self.node_groups))

    def test_add_and_get_nodes(self):
        node1 = DocNode(uid="1", text="text1", group="group1")
        node2 = DocNode(uid="2", text="text2", group="group2")
        self.store.update_nodes([node1, node2])
        nodes = self.store.get_nodes("group1")
        self.assertEqual(nodes, [node1])

    def test_save_nodes(self):
        node1 = DocNode(uid="1", text="text1", group="group1")
        node2 = DocNode(uid="2", text="text2", group="group2")
        self.store.update_nodes([node1, node2])
        collection = self.store._collections["group1"]
        self.assertEqual(collection.peek(collection.count())["ids"], ["1", "2"])

    def test_load_store(self):
        # Set up initial data to be loaded
        node1 = DocNode(uid="1", text="text1", group="group1", parent=None)
        node2 = DocNode(uid="2", text="text2", group="group1", parent=node1)
        self.store.update_nodes([node1, node2])

        # Reset store and load from "persistent" storage
        self.store._map_store._group2docs = {group: {} for group in self.node_groups}
        self.store._load_store()

        nodes = self.store.get_nodes("group1")
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].uid, "1")
        self.assertEqual(nodes[1].uid, "2")
        self.assertEqual(nodes[1].parent.uid, "1")

    def test_all_groups(self):
        self.assertEqual(set(self.store.all_groups()), set(self.node_groups))

    def test_group_others(self):
        node1 = DocNode(uid="1", text="text1", group="group1", parent=None)
        node2 = DocNode(uid="2", text="text2", group="group1", parent=node1)
        self.store.update_nodes([node1, node2])
        self.assertEqual(self.store.has_nodes("group1"), True)
        self.assertEqual(self.store.has_nodes("group2"), True)

class TestMapStore(unittest.TestCase):
    def setUp(self):
        self.node_groups = [LAZY_ROOT_NAME, "group1", "group2"]
        self.store = MapStore(self.node_groups)
        self.node1 = DocNode(uid="1", text="text1", group="group1", parent=None)
        self.node2 = DocNode(uid="2", text="text2", group="group1", parent=self.node1)

    def test_update_nodes(self):
        self.store.update_nodes([self.node1, self.node2])
        nodes = self.store.get_nodes("group1")
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].uid, "1")
        self.assertEqual(nodes[1].uid, "2")
        self.assertEqual(nodes[1].parent.uid, "1")

    def test_get_node(self):
        self.store.update_nodes([self.node1, self.node2])
        n1 = self.store.get_node("group1", "1")
        assert n1.text == self.node1.text
        n2 = self.store.get_node("group1", "2")
        assert n2.text == self.node2.text

    def test_remove_nodes(self):
        self.store.update_nodes([self.node1, self.node2])

        n1 = self.store.get_node("group1", "1")
        assert n1.text == self.node1.text
        self.store.remove_nodes([n1])
        n1 = self.store.get_node("group1", "1")
        assert not n1

        n2 = self.store.get_node("group1", "2")
        assert n2.text == self.node2.text
        self.store.remove_nodes([n2])
        n2 = self.store.get_node("group1", "2")
        assert not n2

    def test_get_nodes(self):
        self.store.update_nodes([self.node1, self.node2])
        ids = set([self.node1.uid, self.node2.uid])

        docs = self.store.get_nodes("group1")
        self.assertEqual(ids, set([doc.uid for doc in docs]))

    def test_all_groups(self):
        self.assertEqual(set(self.store.all_groups()), set(self.node_groups))

    def test_group_others(self):
        self.store.update_nodes([self.node1, self.node2])
        self.assertEqual(self.store.has_nodes("group1"), True)
        self.assertEqual(self.store.has_nodes("group2"), False)

class TestMilvusStore(unittest.TestCase):
    def setUp(self):
        self.node_groups = [LAZY_ROOT_NAME, "group1", "group2"]

        self.map_store = MapStore(self.node_groups)

        index_field_list = [
            MilvusEmbeddingIndexField("vec1", 3, pymilvus.DataType.FLOAT_VECTOR),
            MilvusEmbeddingIndexField("vec2", 5, pymilvus.DataType.FLOAT_VECTOR),
        ]
        _, self.store_file = tempfile.mkstemp(suffix=".db")
        self.store = MilvusStore(node_groups=self.node_groups, uri=self.store_file,
                                 embedding_index_info=index_field_list,
                                 full_data_store=self.map_store)

        self.node1 = DocNode(uid="1", text="text1", group="group1", parent=None,
                             embedding={"vec1": [1, 2, 3], "vec2": [4, 5, 6, 7, 8]})
        self.node2 = DocNode(uid="2", text="text2", group="group1", parent=self.node1,
                             embedding={"vec1": [100, 200, 300], "vec2": [400, 500, 600, 700, 800]})
        self.nodes = [self.node1, self.node2]

    def tearDown(self):
        os.remove(self.store_file)

    def test_update_nodes(self):
        self.store.update_nodes(self.nodes)

        nodes = self.store.get_nodes("group1")
        self.assertEqual(len(nodes), len(self.nodes))

        counter = 0
        for node in nodes:
            for expected_node in self.nodes:
                if node.uid == expected_node.uid:
                    self.assertEqual(node.text, expected_node.text)
                    counter += 1
                    break
        self.assertEqual(counter, len(self.nodes))

    def test_get_node(self):
        self.store.update_nodes(self.nodes)
        n1 = self.store.get_node("group1", "1")
        self.assertEqual(n1.text, self.node1.text)
        n2 = self.store.get_node("group1", "2")
        self.assertEqual(n2.text, self.node2.text)

    def test_remove_nodes(self):
        self.store.update_nodes(self.nodes)

        n1 = self.store.get_node("group1", "1")
        self.assertEqual(n1.text, self.node1.text)
        self.store.remove_nodes([n1])
        n1 = self.store.get_node("group1", "1")
        self.assertEqual(n1, None)

        n2 = self.store.get_node("group1", "2")
        self.assertEqual(n2.text, self.node2.text)
        self.store.remove_nodes([n2])
        n2 = self.store.get_node("group1", "2")
        self.assertEqual(n2, None)

    def test_get_nodes(self):
        self.store.update_nodes(self.nodes)
        ids = set([self.node1.uid, self.node2.uid])

        docs = self.store.get_nodes("group1")
        self.assertEqual(ids, set([doc.uid for doc in docs]))

    def test_all_groups(self):
        self.assertEqual(set(self.store.all_groups()), set(self.node_groups))

    def test_group_others(self):
        self.store.update_nodes([self.node1, self.node2])
        self.assertEqual(self.store.has_nodes("group1"), True)
        self.assertEqual(self.store.has_nodes("group2"), False)

if __name__ == "__main__":
    unittest.main()
