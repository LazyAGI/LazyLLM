from lazyllm import pipeline, parallel
from lazyllm.tools.rag import DocNode
from lazyllm.tools.rag.rank_fusion.reciprocal_rank_fusion import RRFFusion

class TestToolsCommon(object):

    def test_parallle_rrf(self):
        # this example can be found here : https://docs.zilliz.com/docs/reranking-rrf
        def build_nodes(id_list: list[str]):
            return [DocNode(str_id, text=str_id) for str_id in id_list]

        def retrieve_a():
            nodes = build_nodes(['101', '203', '150', '198', '175'])
            for i in range(0, len(nodes)):
                nodes[i].similarity_score = 100 / (i + 1)
            return nodes

        def retrieve_b():
            nodes = build_nodes(['198', '101', '110', '175', '250'])
            for i in range(0, len(nodes)):
                nodes[i].similarity_score = 1.0 / (i + 1)
            return nodes

        with pipeline() as ppl:
            ppl.prl = parallel(retrieve_a, retrieve_b).aslist
            ppl.rrf = RRFFusion(top_k=10)
        res = ppl()
        new_id_sorts = [ele._uid for ele in res]
        assert new_id_sorts == ['101', '198', '175', '203', '150', '110', '250']

        rrf = RRFFusion(top_k=10)
        nodes1 = retrieve_a()
        nodes2 = retrieve_b()
        res = rrf(nodes1, nodes2)
        new_id_sorts = [ele._uid for ele in res]
        assert new_id_sorts == ['101', '198', '175', '203', '150', '110', '250']
