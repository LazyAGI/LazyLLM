from lazyllm import LightEngine

class TestEngine(object):

    def test_engine_subgraph(self):
        nodes = [dict(id='1', kind='LocalLLM', name='m1', args=dict(base_model='', deploy_method='dummy'))]
        edges = [dict(iid='__start__', oid='1'), dict(iid='1', oid='__end__')]

        nodes = [dict(id='2', kind='SubGraph', name='s1', args=dict(nodes=nodes, edges=edges))]
        edges = [dict(iid='__start__', oid='2'), dict(iid='2', oid='__end__')]

        engine = LightEngine()
        engine.start(nodes, edges)
        r = engine.run('1234')
        assert 'reply for You are an AI-Agent developed by LazyLLM' in r
        assert '1234' in r

    def test_engine_code(self):
        nodes = [dict(id='1', kind='Code', name='m1', args='def test(x: int):\n    return 2 * x\n')]
        edges = [dict(iid='__start__', oid='1'), dict(iid='1', oid='__end__')]

        engine = LightEngine()
        engine.start(nodes, edges)
        assert engine.run(1) == 2
        assert engine.run(2) == 4

    def test_engine_switch(self):
        plus1 = dict(id='1', kind='Code', name='m1', args='def test(x: int):\n    return 1 + x\n')
        double = dict(id='2', kind='Code', name='m2', args='def test(x: int):\n    return 2 * x\n')
        square = dict(id='3', kind='Code', name='m3', args='def test(x: int):\n    return x * x\n')
        switch = dict(id='4', kind='Switch', name='s1', args=dict(judge_on_full_input=True, nodes={
            1: [double],
            2: [plus1, double],
            3: [square]
        }))
        nodes = [switch]
        edges = [dict(iid='__start__', oid='4'), dict(iid='4', oid='__end__')]
        engine = LightEngine()
        engine.start(nodes, edges)
        assert engine.run(1) == 2
        assert engine.run(2) == 6
        assert engine.run(3) == 9

    def test_engine_formatter(self):
        nodes = [dict(id='1', kind='Code', name='m1', args='def test(x: int):\n    return x\n'),
                 dict(id='2', kind='Code', name='m2', args='def test(x: int):\n    return [[x, 2*x], [3*x, 4*x]]\n'),
                 dict(id='3', kind='Code', name='m3', args='def test(x: int):\n    return dict(a=1, b=x * x)\n'),
                 dict(id='4', kind='Code', name='m4', args='def test(x, y, z):\n    return f"{x}{y}{z}"\n')]
        edges = [dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'), dict(iid='__start__', oid='3'),
                 dict(iid='1', oid='4'), dict(iid='2', oid='4', formatter='[:, 1]'),
                 dict(iid='3', oid='4', formatter='[b]'), dict(iid='4', oid='__end__')]

        engine = LightEngine()
        engine.start(nodes, edges)
        assert engine.run(1) == '1[2, 4]1'
        assert engine.run(2) == '2[4, 8]4'
