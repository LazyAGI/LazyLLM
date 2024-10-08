from lazyllm.engine import LightEngine
import pytest
import time
from gradio_client import Client
import lazyllm

class TestEngine(object):

    @pytest.fixture(autouse=True)
    def run_around_tests(self):
        yield
        LightEngine().reset()
        lazyllm.FileSystemQueue().dequeue()
        lazyllm.FileSystemQueue(klass="lazy_trace").dequeue()

    def test_engine_subgraph(self):
        resources = [dict(id='0', kind='LocalLLM', name='m1', args=dict(base_model='', deploy_method='dummy'))]
        nodes = [dict(id='1', kind='SharedLLM', name='s1', args=dict(llm='0', prompt=None))]
        edges = [dict(iid='__start__', oid='1'), dict(iid='1', oid='__end__')]

        nodes = [dict(id='2', kind='SubGraph', name='s1', args=dict(nodes=nodes, edges=edges))]
        edges = [dict(iid='__start__', oid='2'), dict(iid='2', oid='__end__')]

        engine = LightEngine()
        engine.start(nodes, edges, resources)
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

        engine.reset()

        switch = dict(id='4', kind='Switch', name='s1', args=dict(judge_on_full_input=False, nodes={
            'case1': [double],
            'case2': [plus1, double],
            'case3': [square]
        }))
        engine.start([switch], edges)
        assert engine.run('case1', 1) == 2
        assert engine.run('case2', 1) == 4
        assert engine.run('case3', 1) == 1
        assert engine.run('case1', 2) == 4
        assert engine.run('case2', 2) == 6
        assert engine.run('case3', 3) == 9

    def test_engine_ifs(self):
        plus1 = dict(id='1', kind='Code', name='m1', args='def test(x: int):\n    return 1 + x\n')
        double = dict(id='2', kind='Code', name='m2', args='def test(x: int):\n    return 2 * x\n')
        square = dict(id='3', kind='Code', name='m3', args='def test(x: int):\n    return x * x\n')
        ifs = dict(id='4', kind='Ifs', name='i1', args=dict(
            cond='def cond(x): return x < 10', true=[plus1, double], false=[square]))
        nodes = [ifs]
        edges = [dict(iid='__start__', oid='4'), dict(iid='4', oid='__end__')]
        engine = LightEngine()
        engine.start(nodes, edges)
        assert engine.run(1) == 4
        assert engine.run(5) == 12
        assert engine.run(10) == 100

    def test_engine_loop(self):
        nodes = [dict(id='1', kind='Code', name='code', args='def square(x: int): return x * x')]
        edges = [dict(iid='__start__', oid='1'), dict(iid='1', oid='__end__')]

        nodes = [dict(id='2', kind='Loop', name='loop',
                      args=dict(stop_condition='def cond(x): return x > 10', nodes=nodes, edges=edges))]
        edges = [dict(iid='__start__', oid='2'), dict(iid='2', oid='__end__')]

        engine = LightEngine()
        engine.start(nodes, edges)
        assert engine.run(2) == 16

    def test_engine_warp(self):
        nodes = [dict(id='1', kind='Code', name='code', args='def square(x: int): return x * x')]
        edges = [dict(iid='__start__', oid='1'), dict(iid='1', oid='__end__')]

        nodes = [dict(id='2', kind='Warp', name='warp', args=dict(nodes=nodes, edges=edges))]
        edges = [dict(iid='__start__', oid='2'), dict(iid='2', oid='__end__')]

        engine = LightEngine()
        engine.start(nodes, edges)
        assert engine.run(2, 3, 4, 5) == (4, 9, 16, 25)

    def test_engine_formatter(self):
        nodes = [dict(id='1', kind='Formatter', name='f1', args=dict(ftype='python', rule='[:]'))]
        edges = [dict(iid='__start__', oid='1'), dict(iid='1', oid='__end__')]

        engine = LightEngine()
        engine.start(nodes, edges)
        assert engine.run([1, 2]) == [1, 2]

        engine.reset()
        nodes = [dict(id='1', kind='Formatter', name='f1', args=dict(ftype='json', rule='{a, c}'))]
        engine.start(nodes, edges)
        assert engine.run('{"a": 1, "b": 2, "c": 3}') == dict(a=1, c=3)

        engine.reset()
        nodes = [dict(id='1', kind='Formatter', name='f1', args=dict(ftype='yaml', rule='[:]{a}'))]
        engine.start(nodes, edges)
        assert engine.run('- a: 1\n  b: 2\n- a: 3\n  d: 4\n') == [dict(a=1), dict(a=3)]

    def test_engine_edge_formatter(self):
        nodes = [dict(id='1', kind='Code', name='m1', args='def test(x: int):\n    return x\n'),
                 dict(id='2', kind='Code', name='m2', args='def test(x: int):\n    return [[x, 2*x], [3*x, 4*x]]\n'),
                 dict(id='3', kind='Code', name='m3', args='def test(x: int):\n    return dict(a=1, b=x * x)\n'),
                 dict(id='4', kind='Code', name='m4', args='def test(x, y, z):\n    return f"{x}{y}{z}"\n')]
        edges = [dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'), dict(iid='__start__', oid='3'),
                 dict(iid='1', oid='4'), dict(iid='2', oid='4', formatter='[:][1]'),
                 dict(iid='3', oid='4', formatter='[b]'), dict(iid='4', oid='__end__')]

        engine = LightEngine()
        engine.start(nodes, edges)
        assert engine.run(1) == '1[2, 4]1'
        assert engine.run(2) == '2[4, 8]4'

    def test_engine_edge_formatter_start(self):
        nodes = [dict(id='1', kind='Code', name='m1', args='def test(x: int): return x'),
                 dict(id='2', kind='Code', name='m2', args='def test(x: int): return 2 * x'),
                 dict(id='3', kind='Code', name='m3', args='def test(x, y): return x + y')]
        edges = [dict(iid='__start__', oid='1', formatter='[0]'), dict(iid='__start__', oid='2', formatter='[1]'),
                 dict(iid='1', oid='3'), dict(iid='2', oid='3'), dict(iid='3', oid='__end__')]

        engine = LightEngine()
        engine.start(nodes, edges)
        assert engine.run(3, 1) == 5
        assert engine.run(5, 3, 1) == 11

    def test_engine_join_stack(self):
        nodes = [dict(id='0', kind='Code', name='c1', args='def test(x: int): return x'),
                 dict(id='1', kind='JoinFormatter', name='join', args=dict(type='stack'))]
        edges = [dict(iid='__start__', oid='0'), dict(iid='0', oid='1'), dict(iid='1', oid='__end__')]
        engine = LightEngine()
        engine.start(nodes, edges)
        assert engine.run(1) == [1]
        assert engine.run('1') == ['1']
        assert engine.run([1]) == [[1]]

        engine.reset()

        nodes = [dict(id='0', kind='Code', name='c1', args='def test(x: int): return x'),
                 dict(id='1', kind='Code', name='c2', args='def test(x: int): return 2 * x'),
                 dict(id='2', kind='Code', name='c3', args='def test(x: int): return 3 * x'),
                 dict(id='3', kind='JoinFormatter', name='join', args=dict(type='stack'))]
        edges = [dict(iid='__start__', oid='0'), dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'),
                 dict(iid='0', oid='3'), dict(iid='1', oid='3'), dict(iid='2', oid='3'), dict(iid='3', oid='__end__')]
        engine.start(nodes, edges)
        assert engine.run(1) == [1, 2, 3]
        assert engine.run('1') == ['1', '11', '111']
        assert engine.run([1]) == [[1], [1, 1], [1, 1, 1]]

    def test_engine_join_sum(self):
        nodes = [dict(id='0', kind='Code', name='c1', args='def test(x: int): return [x, 2 * x]'),
                 dict(id='1', kind='JoinFormatter', name='join', args=dict(type='sum'))]
        edges = [dict(iid='__start__', oid='0'), dict(iid='0', oid='1'), dict(iid='1', oid='__end__')]
        engine = LightEngine()
        engine.start(nodes, edges)
        assert engine.run(1) == 3
        assert engine.run('1') == '111'
        assert engine.run([1]) == [1, 1, 1]

        engine.reset()

        nodes = [dict(id='0', kind='Code', name='c1', args='def test(x: int): return x'),
                 dict(id='1', kind='Code', name='c2', args='def test(x: int): return 2 * x'),
                 dict(id='2', kind='Code', name='c3', args='def test(x: int): return 3 * x'),
                 dict(id='3', kind='JoinFormatter', name='join', args=dict(type='sum'))]
        edges = [dict(iid='__start__', oid='0'), dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'),
                 dict(iid='0', oid='3'), dict(iid='1', oid='3'), dict(iid='2', oid='3'), dict(iid='3', oid='__end__')]
        engine.start(nodes, edges)
        assert engine.run(1) == 6
        assert engine.run('1') == '111111'
        assert engine.run([1]) == [1, 1, 1, 1, 1, 1]

    def test_engine_join_todict(self):
        nodes = [dict(id='0', kind='Code', name='c1', args='def test(x: int): return x'),
                 dict(id='1', kind='Code', name='c2', args='def test(x: int): return 2 * x'),
                 dict(id='2', kind='Code', name='c3', args='def test(x: int): return 3 * x'),
                 dict(id='3', kind='JoinFormatter', name='join', args=dict(type='to_dict', names=['a', 'b', 'c']))]
        edges = [dict(iid='__start__', oid='0'), dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'),
                 dict(iid='0', oid='3'), dict(iid='1', oid='3'), dict(iid='2', oid='3'), dict(iid='3', oid='__end__')]
        engine = LightEngine()
        engine.start(nodes, edges)
        assert engine.run(1) == dict(a=1, b=2, c=3)
        assert engine.run('1') == dict(a='1', b='11', c='111')
        assert engine.run([1]) == dict(a=[1], b=[1, 1], c=[1, 1, 1])

    def test_engine_join_join(self):
        nodes = [dict(id='0', kind='Code', name='c1', args='def test(x: int): return x'),
                 dict(id='1', kind='Code', name='c2', args='def test(x: int): return 2 * x'),
                 dict(id='2', kind='Code', name='c3', args='def test(x: int): return 3 * x'),
                 dict(id='3', kind='JoinFormatter', name='join', args=dict(type='join'))]
        edges = [dict(iid='__start__', oid='0'), dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'),
                 dict(iid='0', oid='3'), dict(iid='1', oid='3'), dict(iid='2', oid='3'), dict(iid='3', oid='__end__')]
        engine = LightEngine()
        engine.start(nodes, edges)
        assert engine.run('1') == '111111'

        changed_nodes = [dict(id='3', kind='JoinFormatter', name='join', args=dict(type='join', symbol='\n'))]
        engine.update(nodes, changed_nodes, edges)
        assert engine.run('1') == '1\n11\n111'

    def test_engine_server(self):
        nodes = [dict(id='1', kind='Code', name='m1', args='def test(x: int):\n    return 2 * x\n')]
        edges = [dict(iid='__start__', oid='1'), dict(iid='1', oid='__end__')]
        resources = [dict(id='2', kind='server', name='s1', args=dict(port=None)),
                     dict(id='3', kind='web', name='w1', args=dict(port=None, title='网页', history=[], audio=False))
                    ]
        engine = LightEngine()
        engine.start(nodes, edges, resources, gid='graph-1')
        assert engine.run(1) == 2
        time.sleep(3)
        web = engine.build_node('graph-1').func._web
        client = Client(web.url, download_files=web.cach_path)
        chat_history = [['123', None]]
        ans = client.predict(False, chat_history, False, False, api_name="/_respond_stream")
        assert ans[0][-1][-1] == '123123'
        client.close()
        lazyllm.launcher.cleanup()
        web.stop()


class TestEngineRAG(object):

    def test_rag(self):
        resources = [dict(id='0', kind='Document', name='d1', args=dict(dataset_path='rag_master'))]
        nodes = [dict(id='1', kind='Retriever', name='ret1',
                      args=dict(doc='0', group_name='CoarseChunk', similarity='bm25_chinese', topk=3)),
                 dict(id='4', kind='Reranker', name='rek1',
                      args=dict(type='ModuleReranker', output_format='content', join=True,
                                arguments=dict(model="bge-reranker-large", topk=1))),
                 dict(id='5', kind='Code', name='c1',
                      args='def test(nodes, query): return f\'context_str={nodes}, query={query}\''),
                 dict(id='6', kind='LocalLLM', name='m1', args=dict(base_model='', deploy_method='dummy'))]
        edges = [dict(iid='__start__', oid='1'), dict(iid='1', oid='4'), dict(iid='__start__', oid='4'),
                 dict(iid='4', oid='5'), dict(iid='__start__', oid='5'), dict(iid='5', oid='6'),
                 dict(iid='6', oid='__end__')]
        engine = LightEngine()
        engine.start(nodes, edges, resources)
        assert '观天之道，执天之行' in engine.run('何为天道?')

        # test add doc_group
        changed_resources = [dict(id='0', kind='Document', name='d1', args=dict(
            dataset_path='rag_master', node_group=[dict(name='sentence', transform='SentenceSplitter',
                                                        chunk_size=100, chunk_overlap=10)]))]
        changed_nodes = [dict(id='2', kind='Retriever', name='ret2',
                              args=dict(doc='0', group_name='sentence', similarity='bm25', topk=3)),
                         dict(id='3', kind='JoinFormatter', name='c', args=dict(type='sum'))]
        edges = [dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'), dict(iid='1', oid='3'),
                 dict(iid='2', oid='3'), dict(iid='3', oid='4'), dict(iid='__start__', oid='4'),
                 dict(iid='4', oid='5'), dict(iid='__start__', oid='5'), dict(iid='5', oid='6'),
                 dict(iid='6', oid='__end__')]
        engine = LightEngine()
        engine.update(nodes + changed_nodes, changed_nodes, edges, changed_resources)
        assert '观天之道，执天之行' in engine.run('何为天道?')
