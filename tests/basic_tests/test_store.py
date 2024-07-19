import unittest
from unittest.mock import MagicMock
from lazyllm.tools.rag.store import DocNode, ChromadbStore, LAZY_ROOT_NAME


# Test class for ChromadbStore
class TestChromadbStore(unittest.TestCase):
    def setUp(self):
        self.node_groups = [LAZY_ROOT_NAME, "group1", "group2"]
        self.embed = MagicMock(side_effect=lambda text: [0.1, 0.2, 0.3])
        self.store = ChromadbStore(self.node_groups, self.embed)
        self.store.add_nodes(
            LAZY_ROOT_NAME,
            [DocNode(uid="1", text="text1", group="group1", parent=None)],
        )

    def test_initialization(self):
        self.assertEqual(set(self.store._collections.keys()), set(self.node_groups))

    def test_add_and_traverse_nodes(self):
        node1 = DocNode(uid="1", text="text1", group="type1")
        node2 = DocNode(uid="2", text="text2", group="type2")
        self.store.add_nodes("group1", [node1, node2])
        nodes = self.store.traverse_nodes("group1")
        self.assertEqual(nodes, [node1, node2])

    def test_save_nodes(self):
        node1 = DocNode(uid="1", text="text1", group="type1")
        node2 = DocNode(uid="2", text="text2", group="type2")
        self.store.add_nodes("group1", [node1, node2])
        self.store.save_nodes("group1")
        collection = self.store._collections["group1"]
        self.assertEqual(collection.peek(collection.count())["ids"], ["1", "2"])
        self.assertTrue(node1.has_embedding())
        self.assertTrue(node2.has_embedding())

    def test_try_load_store(self):
        # Set up initial data to be loaded
        node1 = DocNode(uid="1", text="text1", group="group1", parent=None)
        node2 = DocNode(uid="2", text="text2", group="group1", parent=node1)
        self.store.add_nodes("group1", [node1, node2])

        # Reset store and load from "persistent" storage
        self.store._store = {group: {} for group in self.node_groups}
        self.store.try_load_store()

        nodes = self.store.traverse_nodes("group1")
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].uid, "1")
        self.assertEqual(nodes[1].uid, "2")
        self.assertEqual(nodes[1].parent.uid, "1")


if __name__ == "__main__":
    unittest.main()
