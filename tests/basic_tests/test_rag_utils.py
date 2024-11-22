from lazyllm.tools.rag.utils import generic_process_filters
from lazyllm.tools.rag.doc_node import DocNode

class TestRagUtils:
    def test_generic_process_filters(self):
        nodes = [
            DocNode(uid='1', global_metadata={'k1': 'v1', 'k2': 'v2', 'k4': 'v4'}),
            DocNode(uid='2', global_metadata={'k1': 'v1', 'k3': 'v3', 'k5': 'v5'}),
            DocNode(uid='3', global_metadata={'k2': 'v2', 'k3': 'v3', 'k6': 'v6'}),
        ]

        res = generic_process_filters(nodes, {'k1': 'v1'})
        assert len(res) == 2
        assert set([res[0].uid, res[1].uid]) == set(["1", "2"])

        res = generic_process_filters(nodes, {'k6': 'v6'})
        assert len(res) == 1
        assert res[0].uid == '3'

        res = generic_process_filters(nodes, {'k2': 'v6'})
        assert len(res) == 0
