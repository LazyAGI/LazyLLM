from lazyllm.engine import LightEngine, NodeMetaHook
import pytest
import time
from gradio_client import Client
import lazyllm
import urllib3
from lazyllm.common.common import TimeoutException
import json
import unittest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

app = FastAPI()


@app.post("/mock_post")
async def receive_json(data: dict):
    return JSONResponse(content=data)


class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        client = TestClient(app)

        def mock_report(self):
            headers = {"Content-Type": "application/json; charset=utf-8"}
            json_data = json.dumps(self._meta_info, ensure_ascii=False)
            try:
                lazyllm.LOG.info(f"meta_info: {self._meta_info}")
                response = client.post(self.URL, data=json_data, headers=headers)
                assert (
                    response.json() == self._meta_info
                ), "mock response should be same as input"
            except Exception as e:
                lazyllm.LOG.warning(f"Error sending collected data: {e}")

        NodeMetaHook.report = mock_report

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
        gid = engine.start(nodes, edges, resources)
        r = engine.run(gid, '1234')
        assert 'reply for You are an AI-Agent developed by LazyLLM' in r
        assert '1234' in r

    def test_engine_code(self):
        nodes = [
            dict(
                id="1",
                kind="Code",
                name="m1",
                args=dict(code="def test(x: int):\n    return 2 * x\n"),
            )
        ]
        edges = [dict(iid='__start__', oid='1'), dict(iid='1', oid='__end__')]

        engine = LightEngine()
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 1) == 2
        assert engine.run(gid, 2) == 4

    def test_engine_switch(self):
        plus1 = dict(
            id="1",
            kind="Code",
            name="m1",
            args=dict(code="def test(x: int):\n    return 1 + x\n"),
        )
        double = dict(
            id="2",
            kind="Code",
            name="m2",
            args=dict(code="def test(x: int):\n    return 2 * x\n"),
        )
        # square = dict(id='3', kind='Code', name='m3', args='def test(x: int):\n    return x * x\n')
        square = dict(
            id="3",
            kind="Code",
            name="m3",
            args=dict(
                code="def test(x: int):\n    return x * x\n",
                _lazyllm_enable_report=True,
            ),
        )
        switch = dict(
            id="4",
            kind="Switch",
            name="s1",
            args=dict(
                judge_on_full_input=True,
                nodes={1: [double], 2: [plus1, double], 3: [square]},
                _lazyllm_enable_report=True,
            ),
        )
        nodes = [switch]
        edges = [dict(iid='__start__', oid='4'), dict(iid='4', oid='__end__')]
        engine = LightEngine()
        engine.set_report_url("mock_post")
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 1) == 2
        assert engine.run(gid, 2) == 6
        assert engine.run(gid, 3) == 9

        engine.reset()
        lazyllm.globals._init_sid("session_test_002")

        switch = dict(
            id="4",
            kind="Switch",
            name="s1",
            args=dict(
                judge_on_full_input=False,
                nodes={"case1": [double], "case2": [plus1, double], "case3": [square]},
                _lazyllm_enable_report=True,
            ),
        )
        gid = engine.start([switch], edges)
        assert engine.run(gid, 'case1', 1) == 2
        assert engine.run(gid, 'case2', 1) == 4
        assert engine.run(gid, 'case3', 1) == 1
        assert engine.run(gid, 'case1', 2) == 4
        assert engine.run(gid, 'case2', 2) == 6
        assert engine.run(gid, 'case3', 3) == 9

    def test_engine_ifs(self):
        # plus1 = dict(id='1', kind='Code', name='m1', args='def test(x: int):\n    return 1 + x\n')
        plus1 = dict(
            id="1",
            kind="Code",
            name="m1",
            args=dict(
                code="def test(x: int):\n    return 1 + x\n",
                _lazyllm_enable_report=True,
            ),
        )
        double = dict(
            id="2",
            kind="Code",
            name="m2",
            args=dict(code="def test(x: int):\n    return 2 * x\n"),
        )
        square = dict(
            id="3",
            kind="Code",
            name="m3",
            args=dict(code="def test(x: int):\n    return x * x\n"),
        )
        ifs = dict(
            id="4",
            kind="Ifs",
            name="i1",
            args=dict(
                cond="def cond(x): return x < 10",
                true=[plus1, double],
                false=[square],
                _lazyllm_enable_report=True,
            ),
        )
        nodes = [ifs]
        edges = [dict(iid='__start__', oid='4'), dict(iid='4', oid='__end__')]
        engine = LightEngine()
        engine.set_report_url("mock_post")
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 1) == 4
        assert engine.run(gid, 5) == 12
        assert engine.run(gid, 10) == 100

    def test_engine_loop(self):
        nodes = [
            dict(
                id="1",
                kind="Code",
                name="code",
                args=dict(code="def square(x: int): return x * x"),
            )
        ]
        edges = [dict(iid='__start__', oid='1'), dict(iid='1', oid='__end__')]

        nodes = [
            dict(
                id="2",
                kind="Loop",
                name="loop",
                args=dict(
                    stop_condition="def cond(x): return x > 10",
                    nodes=nodes,
                    edges=edges,
                    _lazyllm_enable_report=True,
                ),
            )
        ]
        edges = [dict(iid='__start__', oid='2'), dict(iid='2', oid='__end__')]

        engine = LightEngine()
        engine.set_report_url("mock_post")
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 2) == 16

    def test_engine_warp(self):
        nodes = [
            dict(
                id="1",
                kind="Code",
                name="code",
                args=dict(code="def square(x: int): return x * x"),
            )
        ]
        edges = [dict(iid='__start__', oid='1'), dict(iid='1', oid='__end__')]

        nodes = [
            dict(
                id="2",
                kind="Warp",
                name="warp",
                args=dict(
                    nodes=nodes,
                    edges=edges,
                    _lazyllm_enable_report=True,
                ),
            )
        ]
        edges = [dict(iid='__start__', oid='2'), dict(iid='2', oid='__end__')]

        engine = LightEngine()
        engine.set_report_url("mock_post")
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 2, 3, 4, 5) == (4, 9, 16, 25)

    def test_engine_formatter(self):
        nodes = [dict(id='1', kind='Formatter', name='f1', args=dict(ftype='python', rule='[:]'))]
        edges = [dict(iid='__start__', oid='1'), dict(iid='1', oid='__end__')]

        engine = LightEngine()
        gid = engine.start(nodes, edges)
        assert engine.run(gid, [1, 2]) == [1, 2]

        engine.reset()
        nodes = [dict(id='1', kind='Formatter', name='f1', args=dict(ftype='json', rule='{a, c}'))]
        gid = engine.start(nodes, edges)
        assert engine.run(gid, '{"a": 1, "b": 2, "c": 3}') == dict(a=1, c=3)

        engine.reset()
        nodes = [dict(id='1', kind='Formatter', name='f1', args=dict(ftype='yaml', rule='[:]{a}'))]
        gid = engine.start(nodes, edges)
        assert engine.run(gid, '- a: 1\n  b: 2\n- a: 3\n  d: 4\n') == [dict(a=1), dict(a=3)]

        engine.reset()
        nodes = [dict(id='1', kind='Formatter', name='f1', args=dict(ftype='file', rule='decode'))]
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 'hi') == 'hi'
        assert engine.run(gid, '<lazyllm-query>{"query":"aha","files":["path/to/file"]}') == \
               {"query": "aha", "files": ["path/to/file"]}

        engine.reset()
        nodes = [dict(id='1', kind='Formatter', name='f1', args=dict(ftype='file', rule='encode'))]
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 'hi') == 'hi'
        assert engine.run(gid, {"query": "aha", "files": ["path/to/file"]}) == \
               '<lazyllm-query>{"query": "aha", "files": ["path/to/file"]}'

        engine.reset()
        nodes = [dict(id='1', kind='Formatter', name='f1', args=dict(ftype='file', rule='merge'))]
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 'hi') == 'hi'
        assert engine.run(gid, 'hi', '<lazyllm-query>{"query":"aha","files":["path/to/file"]}') == \
               '<lazyllm-query>{"query": "hiaha", "files": ["path/to/file"]}'

    def test_engine_edge_formatter(self):
        nodes = [
            dict(
                id="1",
                kind="Code",
                name="m1",
                args=dict(code="def test(x: int):\n    return x\n"),
            ),
            dict(
                id="2",
                kind="Code",
                name="m2",
                args=dict(
                    code="def test(x: int):\n    return [[x, 2*x], [3*x, 4*x]]\n"
                ),
            ),
            dict(
                id="3",
                kind="Code",
                name="m3",
                args=dict(code="def test(x: int):\n    return dict(a=1, b=x * x)\n"),
            ),
            dict(
                id="4",
                kind="Code",
                name="m4",
                args=dict(code='def test(x, y, z):\n    return f"{x}{y}{z}"\n'),
            ),
        ]
        edges = [dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'), dict(iid='__start__', oid='3'),
                 dict(iid='1', oid='4'), dict(iid='2', oid='4', formatter='[:][1]'),
                 dict(iid='3', oid='4', formatter='[b]'), dict(iid='4', oid='__end__')]

        engine = LightEngine()
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 1) == '1[2, 4]1'
        assert engine.run(gid, 2) == '2[4, 8]4'

    def test_engine_edge_formatter_start(self):
        nodes = [
            dict(
                id="1",
                kind="Code",
                name="m1",
                args=dict(code="def test(x: int): return x"),
            ),
            dict(
                id="2",
                kind="Code",
                name="m2",
                args=dict(code="def test(x: int): return 2 * x"),
            ),
            dict(
                id="3",
                kind="Code",
                name="m3",
                args=dict(code="def test(x, y): return x + y"),
            ),
        ]
        edges = [dict(iid='__start__', oid='1', formatter='[0]'), dict(iid='__start__', oid='2', formatter='[1]'),
                 dict(iid='1', oid='3'), dict(iid='2', oid='3'), dict(iid='3', oid='__end__')]

        engine = LightEngine()
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 3, 1) == 5
        assert engine.run(gid, 5, 3, 1) == 11

    def test_engine_formatter_end(self):
        nodes = [
            dict(
                id="1",
                kind="Code",
                name="m1",
                args=dict(code="def test(x: int):\n    return x\n"),
            ),
            dict(
                id="2",
                kind="Code",
                name="m2",
                args=dict(
                    code="def test1(x: int):\n    return [[x, 2*x], [3*x, 4*x]]\n"
                ),
            ),
            # two unused node
            dict(
                id="3",
                kind="Code",
                name="m3",
                args=dict(code="def test2(x: int):\n    return dict(a=1, b=x * x)\n"),
            ),
            dict(
                id="4",
                kind="Code",
                name="m4",
                args=dict(code='def test3(x, y, z):\n    return f"{x}{y}{z}"\n'),
            ),
        ]
        edges = [dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'), dict(iid='2', oid='__end__'),
                 dict(iid='1', oid='__end__')]

        engine = LightEngine()
        gid = engine.start(nodes, edges)
        r = engine.run(gid, 1)
        print(r, type(r))
        print(isinstance(r, lazyllm.package))

        engine.reset()

        nodes = [
            dict(
                id="1",
                kind="Code",
                name="m1",
                args=dict(code="def test(x: int):\n    return x\n"),
            ),
            dict(
                id="2",
                kind="Code",
                name="m2",
                args=dict(
                    code="def test1(x: int):\n    return [[x, 2*x], [3*x, 4*x]]\n"
                ),
            ),
            dict(
                id="3",
                kind="JoinFormatter",
                name="join",
                args=dict(type="to_dict", names=["a", "b"]),
            ),
        ]
        edges = [dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'), dict(iid='2', oid='3'),
                 dict(iid='1', oid='3'), dict(iid='3', oid='__end__', formatter='*[a, b]')]
        engine = LightEngine()
        gid = engine.start(nodes, edges)
        r = engine.run(gid, 1)
        print(r, type(r))
        print(isinstance(r, lazyllm.package))

    def test_engine_join_stack(self):
        nodes = [
            dict(
                id="0",
                kind="Code",
                name="c1",
                args=dict(code="def test(x: int): return x"),
            ),
            dict(id="1", kind="JoinFormatter", name="join", args=dict(type="stack")),
        ]
        edges = [dict(iid='__start__', oid='0'), dict(iid='0', oid='1'), dict(iid='1', oid='__end__')]
        engine = LightEngine()
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 1) == [1]
        assert engine.run(gid, '1') == ['1']
        assert engine.run(gid, [1]) == [[1]]

        engine.reset()

        nodes = [
            dict(
                id="0",
                kind="Code",
                name="c1",
                args=dict(code="def test(x: int): return x"),
            ),
            dict(
                id="1",
                kind="Code",
                name="c2",
                args=dict(code="def test(x: int): return 2 * x"),
            ),
            dict(
                id="2",
                kind="Code",
                name="c3",
                args=dict(code="def test(x: int): return 3 * x"),
            ),
            dict(id="3", kind="JoinFormatter", name="join", args=dict(type="stack")),
        ]
        edges = [dict(iid='__start__', oid='0'), dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'),
                 dict(iid='0', oid='3'), dict(iid='1', oid='3'), dict(iid='2', oid='3'), dict(iid='3', oid='__end__')]
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 1) == [1, 2, 3]
        assert engine.run(gid, '1') == ['1', '11', '111']
        assert engine.run(gid, [1]) == [[1], [1, 1], [1, 1, 1]]

    def test_engine_join_sum(self):
        nodes = [
            dict(
                id="0",
                kind="Code",
                name="c1",
                args=dict(code="def test(x: int): return [x, 2 * x]"),
            ),
            dict(id="1", kind="JoinFormatter", name="join", args=dict(type="sum")),
        ]
        edges = [dict(iid='__start__', oid='0'), dict(iid='0', oid='1'), dict(iid='1', oid='__end__')]
        engine = LightEngine()
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 1) == 3
        assert engine.run(gid, '1') == '111'
        assert engine.run(gid, [1]) == [1, 1, 1]

        engine.reset()

        nodes = [
            dict(
                id="0",
                kind="Code",
                name="c1",
                args=dict(code="def test(x: int): return x"),
            ),
            dict(
                id="1",
                kind="Code",
                name="c2",
                args=dict(code="def test(x: int): return 2 * x"),
            ),
            dict(
                id="2",
                kind="Code",
                name="c3",
                args=dict(code="def test(x: int): return 3 * x"),
            ),
            dict(id="3", kind="JoinFormatter", name="join", args=dict(type="sum")),
        ]
        edges = [dict(iid='__start__', oid='0'), dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'),
                 dict(iid='0', oid='3'), dict(iid='1', oid='3'), dict(iid='2', oid='3'), dict(iid='3', oid='__end__')]
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 1) == 6
        assert engine.run(gid, '1') == '111111'
        assert engine.run(gid, [1]) == [1, 1, 1, 1, 1, 1]

    def test_engine_join_todict(self):
        nodes = [
            dict(
                id="0",
                kind="Code",
                name="c1",
                args=dict(code="def test(x: int): return x"),
            ),
            dict(
                id="1",
                kind="Code",
                name="c2",
                args=dict(code="def test(x: int): return 2 * x"),
            ),
            dict(
                id="2",
                kind="Code",
                name="c3",
                args=dict(code="def test(x: int): return 3 * x"),
            ),
            dict(
                id="3",
                kind="JoinFormatter",
                name="join",
                args=dict(type="to_dict", names=["a", "b", "c"]),
            ),
        ]
        edges = [dict(iid='__start__', oid='0'), dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'),
                 dict(iid='0', oid='3'), dict(iid='1', oid='3'), dict(iid='2', oid='3'), dict(iid='3', oid='__end__')]
        engine = LightEngine()
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 1) == dict(a=1, b=2, c=3)
        assert engine.run(gid, '1') == dict(a='1', b='11', c='111')
        assert engine.run(gid, [1]) == dict(a=[1], b=[1, 1], c=[1, 1, 1])

    def test_engine_update(self):
        plus1 = dict(
            id="1",
            kind="Code",
            name="m1",
            args=dict(code="def test(x: int):\n    return 1 + x\n"),
        )
        double = dict(
            id="2",
            kind="Code",
            name="m2",
            args=dict(code="def test(x: int):\n    return 2 * x\n"),
        )
        square = dict(
            id="3",
            kind="Code",
            name="m3",
            args=dict(code="def test(x: int):\n    return x * x\n"),
        )
        ifs = dict(id='4', kind='Ifs', name='i1', args=dict(
            cond='def cond(x): return x < 10', true=[plus1, double], false=[square]
        ))
        nodes = [ifs]
        edges = [dict(iid='__start__', oid='4'), dict(iid='4', oid='__end__')]
        engine = LightEngine()
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 1) == 4
        assert engine.run(gid, 5) == 12
        assert engine.run(gid, 10) == 100

        double = dict(
            id="2",
            kind="Code",
            name="m2",
            args=dict(code="def test(x: int):\n    return 3 * x\n"),
        )
        ifs = dict(id='4', kind='Ifs', name='i1', args=dict(
            cond='def cond(x): return x < 10', true=[plus1, double], false=[square]
        ))
        nodes = [ifs]
        engine.update(gid, nodes, edges)

        assert engine.run(gid, 1) == 6
        assert engine.run(gid, 5) == 18
        assert engine.run(gid, 10) == 100

    def test_engine_join_join(self):
        nodes = [
            dict(
                id="0",
                kind="Code",
                name="c1",
                args=dict(code="def test(x: int): return x"),
            ),
            dict(
                id="1",
                kind="Code",
                name="c2",
                args=dict(code="def test(x: int): return 2 * x"),
            ),
            dict(
                id="2",
                kind="Code",
                name="c3",
                args=dict(code="def test(x: int): return 3 * x"),
            ),
            dict(id="3", kind="JoinFormatter", name="join", args=dict(type="join")),
        ]
        edges = [dict(iid='__start__', oid='0'), dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'),
                 dict(iid='0', oid='3'), dict(iid='1', oid='3'), dict(iid='2', oid='3'), dict(iid='3', oid='__end__')]
        engine = LightEngine()
        gid = engine.start(nodes, edges)
        assert engine.run(gid, '1') == '111111'

        nodes[-1] = dict(id='3', kind='JoinFormatter', name='join', args=dict(type='join', symbol='\n'))
        engine.update(gid, nodes, edges)
        assert engine.run(gid, '1') == '1\n11\n111'

    def test_engine_server(self):
        nodes = [
            dict(
                id="1",
                kind="Code",
                name="m1",
                args=dict(code="def test(x: int):\n    return 2 * x\n"),
            )
        ]
        edges = [dict(iid='__start__', oid='1'), dict(iid='1', oid='__end__')]
        resources = [dict(id='2', kind='server', name='s1', args=dict(port=None)),
                     dict(id='3', kind='web', name='w1', args=dict(port=None, title='网页', history=[], audio=False))
                    ]
        engine = LightEngine()
        gid = engine.start(nodes, edges, resources, gid='graph-1')
        assert engine.status(gid) == {'1': 'running', '2': lazyllm.launcher.Status.Running, '3': 'running'}
        assert engine.run(gid, 1) == 2
        time.sleep(3)
        web = engine.build_node('graph-1').func._web
        assert engine.build_node('graph-1').func.api_url is not None
        assert engine.build_node('graph-1').func.web_url == web.url
        client = Client(web.url, download_files=web.cach_path)
        chat_history = [['123', None]]
        ans = client.predict(False, chat_history, False, False, api_name="/_respond_stream")
        assert ans[0][-1][-1] == '123123'
        client.close()
        lazyllm.launcher.cleanup()
        web.stop()

    def test_engine_stop_and_restart(self):
        resources = [dict(id='0', kind='LocalLLM', name='m1', args=dict(base_model='', deploy_method='dummy'))]
        nodes = [dict(id='1', kind='SharedLLM', name='s1', args=dict(llm='0', prompt=None))]
        edges = [dict(iid='__start__', oid='1'), dict(iid='1', oid='__end__')]

        engine = LightEngine()
        assert engine.status('123') == 'unknown'
        gid = engine.start(nodes, edges, resources, gid='123')
        assert gid == '123'

        r = engine.run(gid, '1234')
        assert 'reply for You are an AI-Agent developed by LazyLLM' in r
        assert '1234' in r

        assert engine.status(gid) == {'1': 'running', '0': lazyllm.launcher.Status.Running}
        engine.stop('0')

        assert engine.status(gid) == {'1': 'running', '0': lazyllm.launcher.Status.Cancelled}
        with pytest.raises((TimeoutException, urllib3.exceptions.NewConnectionError, RuntimeError)):
            with lazyllm.timeout(3):
                engine.run(gid, '1234567')

        engine.start('0')
        assert engine.status(gid) == {'1': 'running', '0': lazyllm.launcher.Status.Running}
        r = engine.run(gid, '12345')
        assert 'reply for You are an AI-Agent developed by LazyLLM' in r
        assert '12345' in r
        engine.stop(gid)
        assert engine.status(gid) == {'1': 'running', '0': lazyllm.launcher.Status.Cancelled}

    def test_engine_httptool(self):
        params = {'p1': '{{p1}}', 'p2': '{{p2}}'}
        headers = {'h1': '{{h1}}'}
        url = 'https://postman-echo.com/get'

        nodes = [
            dict(
                id="0",
                kind="Code",
                name="code1",
                args=dict(code='def p1(): return "foo"'),
            ),
            dict(
                id="1",
                kind="Code",
                name="code2",
                args=dict(code='def p2(): return "bar"'),
            ),
            dict(
                id="2",
                kind="Code",
                name="code3",
                args=dict(code='def h1(): return "baz"'),
            ),
            dict(
                id="3",
                kind="HttpTool",
                name="http",
                args=dict(
                    method="GET",
                    url=url,
                    params=params,
                    headers=headers,
                    _lazyllm_arg_names=["p1", "p2", "h1"],
                ),
            ),
        ]
        edges = [dict(iid='__start__', oid='0'), dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'),
                 dict(iid='0', oid='3'), dict(iid='1', oid='3'), dict(iid='2', oid='3'), dict(iid='3', oid='__end__')]

        engine = LightEngine()
        gid = engine.start(nodes, edges, gid='graph-1')
        res = engine.run(gid)
        content = json.loads(res['content'])

        assert content['headers']['h1'] == 'baz'
        assert content['url'] == f'{url}?p1=foo&p2=bar'

    def test_engine_status(self):
        resources = [dict(id='0', kind='LocalLLM', name='m1', args=dict(base_model='', deploy_method='dummy'))]
        llm_node = dict(id='1', kind='SharedLLM', name='s1', args=dict(llm='0', prompt=None))

        plus1 = dict(
            id="2",
            kind="Code",
            name="m1",
            args=dict(code="def test(x: int):\n    return 1 + x\n"),
        )
        double = dict(
            id="3",
            kind="Code",
            name="m2",
            args=dict(code="def test(x: int):\n    return 2 * x\n"),
        )
        square = dict(
            id="4",
            kind="Code",
            name="m3",
            args=dict(code="def test(x: int):\n    return x * x\n"),
        )

        subgraph = dict(id='5', kind='SubGraph', name='subgraph', args=dict(nodes=[double, plus1]))
        ifs = dict(id='6', kind='Ifs', name='i1', args=dict(
            cond='def cond(x): return x % 2 == 0', true=plus1, false=[square]))
        loop = dict(id='7', kind='Loop', name='loop', args=dict(
            stop_condition='def cond(x): return x > 8', nodes=[double]))

        switch = dict(id='8', kind='Switch', name='sw1', args=dict(judge_on_full_input=True, nodes={
            1: [plus1, subgraph], 2: ifs, 3: loop, 5: [ifs]}))

        warp = dict(id='9', kind='Warp', name='w1', args=dict(nodes=[switch, plus1]))
        join = dict(id='10', kind='JoinFormatter', name='join', args=dict(type='join', symbol=', '))
        nodes = [warp, join, llm_node]
        engine = LightEngine()
        gid = engine.start(nodes, [], resources)

        assert '6, 4, 13, 26' in engine.run(gid, 1, 2, 3, 5)
        assert engine.status(gid) == {'9': {'8': {'2': 'running',
                                                  '5': {'3': 'running', '2': 'running'},
                                                  '6': {'2': 'running', '4': 'running'},
                                                  '7': {'3': 'running'}},
                                            '2': 'running'},
                                      '10': 'running',
                                      '1': 'running',
                                      '0': lazyllm.launcher.Status.Running}


class TestEngineRAG(object):

    def test_rag(self):
        resources = [
            dict(id='00', kind='LocalEmbedding', name='e1', args=dict(base_model='bge-large-zh-v1.5')),
            dict(id='0', kind='Document', name='d1', args=dict(dataset_path='rag_master', embed='00'))]
        nodes = [
            dict(
                id="1",
                kind="Retriever",
                name="ret1",
                args=dict(
                    doc="0", group_name="CoarseChunk", similarity="bm25_chinese", topk=3
                ),
            ),
            dict(
                id="4",
                kind="Reranker",
                name="rek1",
                args=dict(
                    type="ModuleReranker",
                    output_format="content",
                    join=True,
                    arguments=dict(model="bge-reranker-large", topk=1),
                ),
            ),
            dict(
                id="5",
                kind="Code",
                name="c1",
                args=dict(
                    code="def test(nodes, query): return f'context_str={nodes}, query={query}'"
                ),
            ),
            dict(
                id="6",
                kind="LocalLLM",
                name="m1",
                args=dict(base_model="", deploy_method="dummy"),
            ),
        ]
        edges = [dict(iid='__start__', oid='1'), dict(iid='1', oid='4'), dict(iid='__start__', oid='4'),
                 dict(iid='4', oid='5'), dict(iid='__start__', oid='5'), dict(iid='5', oid='6'),
                 dict(iid='6', oid='__end__')]
        engine = LightEngine()
        gid = engine.start(nodes, edges, resources)
        r = engine.run(gid, '何为天道?')
        assert '观天之道，执天之行' in r or '天命之谓性，率性之谓道' in r

        # test add doc_group
        resources[-1] = dict(id='0', kind='Document', name='d1', args=dict(
            dataset_path='rag_master', server=True, node_group=[
                dict(name='sentence', transform='SentenceSplitter', chunk_size=100, chunk_overlap=10)]))
        nodes.extend([dict(id='2', kind='Retriever', name='ret2',
                           args=dict(doc='0', group_name='sentence', similarity='bm25', topk=3)),
                      dict(id='3', kind='JoinFormatter', name='c', args=dict(type='sum'))])
        edges = [dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'), dict(iid='1', oid='3'),
                 dict(iid='2', oid='3'), dict(iid='3', oid='4'), dict(iid='__start__', oid='4'),
                 dict(iid='4', oid='5'), dict(iid='__start__', oid='5'), dict(iid='5', oid='6'),
                 dict(iid='6', oid='__end__')]
        engine = LightEngine()
        engine.update(gid, nodes, edges, resources)
        assert '观天之道，执天之行' in engine.run(gid, '何为天道?')
