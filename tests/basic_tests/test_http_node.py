from lazyllm import LightEngine

class TestEngine(object):

    def test_engine_subgraph(self):
        nodes = [dict(id='1', kind='HttpTool', name='HTTP-1', args={
            "method": "get",
            "url": "https://httpbin.org/ip",  # get public IP of the running server.
            "API_Key": "",
            "headers": {
                "h1": "h1",
                "h2": "h2"
            },
            "params": {
                "p1": "p1",
                "p2": "p2"
            },
            "body": ""
        })]
        edges = [dict(iid='__start__', oid='1'), dict(iid='1', oid='__end__')]

        engine = LightEngine()
        engine.start(nodes, edges)
        r = engine.run()
        assert 'status' in r

if __name__ == '__main__':
    http_node = TestEngine()
    http_node.test_engine_subgraph()
