from lazyllm.tools.rag.utils import generic_process_filters
from lazyllm.tools.rag.doc_node import DocNode
from lazyllm.tools.rag.utils import _FileNodeIndex, sparse2normal, is_sparse
from lazyllm.tools.rag.store_base import LAZY_ROOT_NAME
from lazyllm.tools.rag.global_metadata import RAG_DOC_PATH
import unittest

class TestRagUtils:
    def test_generic_process_filters(self):
        nodes = [
            DocNode(uid='1', global_metadata={'k1': 'v1', 'k2': 'v2', 'k4': 'v4'}),
            DocNode(uid='2', global_metadata={'k1': 'v1', 'k3': 'v3', 'k5': 'v5'}),
            DocNode(uid='3', global_metadata={'k2': 'v2', 'k3': 'v3', 'k6': 'v6'}),
        ]

        res = generic_process_filters(nodes, {'k1': 'v1'})
        assert len(res) == 2
        assert set([res[0]._uid, res[1]._uid]) == set(["1", "2"])

        res = generic_process_filters(nodes, {'k6': 'v6'})
        assert len(res) == 1
        assert res[0]._uid == '3'

        res = generic_process_filters(nodes, {'k2': 'v6'})
        assert len(res) == 0

    def test_sparse2normal(self):
        embedding = {1: 3, 5: 12}
        dim = 6
        res = sparse2normal(embedding, dim)
        assert len(res) == dim
        assert res == [0, 3, 0, 0, 0, 12]

        embedding = [(0, 9), (2, 14), (4, 28)]
        dim = 8
        res = sparse2normal(embedding, dim)
        assert len(res) == dim
        assert res == [9, 0, 14, 0, 28, 0, 0, 0]

    def test_is_sparse(self):
        embedding = {1: 3, 5: 12}
        assert is_sparse(embedding)

        embedding = [(0, 9), (2, 14), (4, 28)]
        assert is_sparse(embedding)

        embedding = [9, 0, 14, 0, 28, 0, 0, 0]
        assert not is_sparse(embedding)

class TestFileNodeIndex(unittest.TestCase):
    def setUp(self):
        self.index = _FileNodeIndex()
        self.node1 = DocNode(uid='1', group=LAZY_ROOT_NAME, global_metadata={RAG_DOC_PATH: "d1"})
        self.node2 = DocNode(uid='2', group=LAZY_ROOT_NAME, global_metadata={RAG_DOC_PATH: "d2"})
        self.files = [self.node1.global_metadata[RAG_DOC_PATH], self.node2.global_metadata[RAG_DOC_PATH]]

    def test_update(self):
        self.index.update([self.node1, self.node2])

        nodes = self.index.query(self.files)
        assert len(nodes) == len(self.files)

        ret = [node.global_metadata[RAG_DOC_PATH] for node in nodes]
        assert set(ret) == set(self.files)

    def test_remove(self):
        self.index.update([self.node1, self.node2])
        self.index.remove([self.node2._uid])
        ret = self.index.query([self.node2.global_metadata[RAG_DOC_PATH]])
        assert len(ret) == 0

    def test_query(self):
        self.index.update([self.node1, self.node2])
        ret = self.index.query([self.node2.global_metadata[RAG_DOC_PATH]])
        assert len(ret) == 1
        assert ret[0] is self.node2
        ret = self.index.query([self.node1.global_metadata[RAG_DOC_PATH]])
        assert len(ret) == 1
        assert ret[0] is self.node1
