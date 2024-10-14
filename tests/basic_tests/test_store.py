import os
import shutil
import unittest
import lazyllm
import tempfile
from unittest.mock import MagicMock
from lazyllm.tools.rag.store import DocNode, MapStore, ChromadbStore, MilvusStore, LAZY_ROOT_NAME


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

    def test_add_and_traverse_group(self):
        node1 = DocNode(uid="1", text="text1", group="group1")
        node2 = DocNode(uid="2", text="text2", group="group2")
        self.store.update_nodes([node1, node2])
        nodes = self.store.traverse_group("group1")
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
        self.store._mapstore._group2docs = {group: {} for group in self.node_groups}
        self.store._load_store()

        nodes = self.store.traverse_group("group1")
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].uid, "1")
        self.assertEqual(nodes[1].uid, "2")
        self.assertEqual(nodes[1].parent.uid, "1")

class TestMapStore(unittest.TestCase):
    def test_update_nodes(self):
        node_groups = [LAZY_ROOT_NAME, "group1", "group2"]
        store = MapStore(node_groups)

        node1 = DocNode(uid="1", text="text1", group="group1", parent=None)
        node2 = DocNode(uid="2", text="text2", group="group1", parent=node1)
        store.update_nodes([node1, node2])

        nodes = store.traverse_group("group1")
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].uid, "1")
        self.assertEqual(nodes[1].uid, "2")
        self.assertEqual(nodes[1].parent.uid, "1")

    def test_get_node(self):
        node_groups = [LAZY_ROOT_NAME, "group1", "group2"]
        store = MapStore(node_groups)

        node1 = DocNode(uid="1", text="text1", group="group1", parent=None)
        node2 = DocNode(uid="2", text="text2", group="group1", parent=node1)
        store.update_nodes([node1, node2])

        n1 = store.get_node("group1", "1")
        assert n1.text == node1.text

        n2 = store.get_node("group1", "2")
        assert n2.text == node2.text

    def test_remove_nodes(self):
        node_groups = [LAZY_ROOT_NAME, "group1", "group2"]
        store = MapStore(node_groups)

        node1 = DocNode(uid="1", text="text1", group="group1", parent=None)
        node2 = DocNode(uid="2", text="text2", group="group1", parent=node1)
        store.update_nodes([node1, node2])

        n1 = store.get_node("group1", "1")
        assert n1.text == node1.text
        store.remove_nodes([n1])
        n1 = store.get_node("group1", "1")
        assert not n1

        n2 = store.get_node("group1", "2")
        assert n2.text == node2.text
        store.remove_nodes([n2])
        n2 = store.get_node("group1", "2")
        assert not n2

    def test_traverse_group(self):
        node_groups = [LAZY_ROOT_NAME, "group1", "group2"]
        store = MapStore(node_groups)

        node1 = DocNode(uid="1", text="text1", group="group1", parent=None)
        node2 = DocNode(uid="2", text="text2", group="group1", parent=node1)
        store.update_nodes([node1, node2])
        ids = set([node1.uid, node2.uid])

        docs = store.traverse_group("group1")
        self.assertEqual(ids, set([doc.uid for doc in docs]))

    def test_group_others(self):
        node_groups = [LAZY_ROOT_NAME, "group1", "group2"]
        store = MapStore(node_groups)
        self.assertEqual(store.has_group("group1"), True)
        self.assertEqual(store.has_group("group2"), True)

class TestMilvusStore(unittest.TestCase):
    def test_update_nodes(self):
        node_groups = [LAZY_ROOT_NAME, "group1", "group2"]
        _, store_file = tempfile.mkstemp(suffix=".db")
        store = MilvusStore(node_groups, store_file)

        node1 = DocNode(uid="1", text="text1", group="group1", parent=None)
        node2 = DocNode(uid="2", text="text2", group="group1", parent=node1)
        store.update_nodes([node1, node2])

        # Reset store and load from "persistent" storage
        store._map_store._group2docs = {group: {} for group in node_groups}
        store._load_store()

        nodes = store.traverse_group("group1")
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].uid, "1")
        self.assertEqual(nodes[1].uid, "2")
        self.assertEqual(nodes[1].parent.uid, "1")

        os.remove(store_file)

    def test_get_node(self):
        node_groups = [LAZY_ROOT_NAME, "group1", "group2"]
        _, store_file = tempfile.mkstemp(suffix=".db")
        store = MilvusStore(node_groups, store_file)

        node1 = DocNode(uid="1", text="text1", group="group1", parent=None)
        node2 = DocNode(uid="2", text="text2", group="group1", parent=node1)
        store.update_nodes([node1, node2])

        n1 = store.get_node("group1", "1")
        assert n1.text == node1.text

        n2 = store.get_node("group1", "2")
        assert n2.text == node2.text

        os.remove(store_file)

    def test_remove_nodes(self):
        node_groups = [LAZY_ROOT_NAME, "group1", "group2"]
        _, store_file = tempfile.mkstemp(suffix=".db")
        store = MilvusStore(node_groups, store_file)

        node1 = DocNode(uid="1", text="text1", group="group1", parent=None)
        node2 = DocNode(uid="2", text="text2", group="group1", parent=node1)
        store.update_nodes([node1, node2])

        n1 = store.get_node("group1", "1")
        assert n1.text == node1.text
        store.remove_nodes([n1])
        n1 = store.get_node("group1", "1")
        assert not n1

        n2 = store.get_node("group1", "2")
        assert n2.text == node2.text
        store.remove_nodes([n2])
        n2 = store.get_node("group1", "2")
        assert not n2

        os.remove(store_file)

    def test_traverse_group(self):
        node_groups = [LAZY_ROOT_NAME, "group1", "group2"]
        _, store_file = tempfile.mkstemp(suffix=".db")
        store = MilvusStore(node_groups, store_file)

        node1 = DocNode(uid="1", text="text1", group="group1", parent=None)
        node2 = DocNode(uid="2", text="text2", group="group1", parent=node1)
        store.update_nodes([node1, node2])
        ids = set([node1.uid, node2.uid])

        docs = store.traverse_group("group1")
        self.assertEqual(ids, set([doc.uid for doc in docs]))

        os.remove(store_file)

    def test_group_others(self):
        node_groups = [LAZY_ROOT_NAME, "group1", "group2"]
        _, store_file = tempfile.mkstemp(suffix=".db")
        store = MilvusStore(node_groups, store_file)

        self.assertEqual(store.has_group("group1"), True)
        self.assertEqual(store.has_group("group2"), True)

        os.remove(store_file)

if __name__ == "__main__":
    unittest.main()
