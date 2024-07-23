from lazyllm import LightEngine

class TestEngine(object):

    def test_engine_subgraph(self):
        nodes = [dict(id='1', kind='LocalModel', name='m1', args=dict(base_model='', deploy_method='dummy'))]
        edges = [dict(iid='__start__', oid='1'), dict(iid='1', oid='__end__')]

        engine = LightEngine()
        engine.start(nodes, edges)
        r = engine.run('1234')
        assert 'reply for You are an AI-Agent developed by LazyLLM' in r
        assert '1234' in r
