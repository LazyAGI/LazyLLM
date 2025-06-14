from lazyllm.engine import LightEngine
import pytest
import time
from gradio_client import Client
import lazyllm
import urllib3
from lazyllm.common.common import TimeoutException
import json
import unittest
import subprocess
import socket
import threading
import requests
import os

HOOK_PORT = 33733
HOOK_ROUTE = "mock_post"
fastapi_code = """
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from collections import deque

app = FastAPI()
received_datas = deque(maxlen=100)


@app.post("/{route}")
async def receive_json(data: dict):
    print("Received json data:", data)
    received_datas.append(data)
    return JSONResponse(content=data)

@app.get("/get_last_report")
async def get_last_report():
    if len(received_datas) > 0:
        return received_datas[-1]
    else:
        return {{}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port={port})
""".format(
    port=HOOK_PORT, route=HOOK_ROUTE
)


@pytest.mark.skip_on_win
@pytest.mark.skip_on_mac
class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fastapi_process = subprocess.Popen(
            ["python", "-c", fastapi_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        cls.report_url = f"http://{ip_address}:{HOOK_PORT}/{HOOK_ROUTE}"
        cls.get_url = f"http://{ip_address}:{HOOK_PORT}/get_last_report"

        def read_stdout(process):
            for line in iter(process.stdout.readline, b''):
                print("FastAPI Server Output: ", line.decode(), end='')

        cls.report_print_thread = threading.Thread(
            target=read_stdout, args=(cls.fastapi_process,)
        )
        cls.report_print_thread.daemon = True
        cls.report_print_thread.start()

    @classmethod
    def tearDownClass(cls):
        time.sleep(3)
        cls.fastapi_process.terminate()
        cls.fastapi_process.wait()

    def get_last_report(self):
        r = requests.get(self.get_url)
        json_obj = {}
        try:
            json_obj = json.loads(r.content)
        except Exception as e:
            lazyllm.LOG.warning(str(e))
        return json_obj

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
        nodes = [dict(id='1', kind='Code', name='m1', args=dict(code='def test(x: int):\n    return 2 * x\n'))]
        edges = [dict(iid='__start__', oid='1'), dict(iid='1', oid='__end__')]

        engine = LightEngine()
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 1) == 2
        assert engine.run(gid, 2) == 4

    def test_engine_switch_and_diverter(self):
        plus1 = dict(id='1', kind='Code', name='m1', args=dict(code='def test(x: int):\n    return 1 + x\n'))
        double = dict(id='2', kind='Code', name='m2', args=dict(code='def test(x: int):\n    return 2 * x\n'))
        square = dict(id='3', kind='Code', name='m3',
                      args=dict(code='def test(x: int):\n    return x * x\n', _lazyllm_enable_report=True))
        switch = dict(id="4", kind="Switch", name="s1", args=dict(
            judge_on_full_input=True, nodes={1: [double], 2: [plus1, double], 3: [square]}, _lazyllm_enable_report=True))
        edges = [dict(iid='__start__', oid='4'), dict(iid='4', oid='__end__')]
        engine = LightEngine()
        engine.set_report_url(self.report_url)
        gid = engine.start([switch], edges)
        assert engine.run(gid, 1) == 2
        assert engine.run(gid, 2) == 6
        assert engine.run(gid, 3) == 9

        diverter = dict(id="5", kind="Diverter", name="d1", args=dict(nodes=[[double], [plus1, double], square]))
        edges2 = [dict(iid='__start__', oid='5'), dict(iid='5', oid='__end__')]
        gid = engine.start([diverter], edges2)
        assert engine.run(gid, [1, 2, 3]) == (2, 6, 9)

        engine.reset()

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
        assert "prompt_tokens" in self.get_last_report()

    def test_engine_ifs(self):
        plus1 = dict(id='1', kind='Code', name='m1', args=dict(code='def test(x: int):\n    return 1 + x\n'))
        double = dict(id='2', kind='Code', name='m2', args=dict(code='def test(x: int):\n    return 2 * x\n'))
        square = dict(id='3', kind='Code', name='m3', args=dict(code='def test(x: int):\n    return x * x\n'))
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
        engine.set_report_url(self.report_url)
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 1) == 4
        assert engine.run(gid, 5) == 12
        assert engine.run(gid, 10) == 100
        assert "prompt_tokens" in self.get_last_report()

    def test_data_reflow_in_server(self):
        nodes = [
            {
                "id": "1",
                "kind": "Code",
                "name": "f1",
                "args": {
                    "code": "def main(x): return int(x) + 1",
                    "_lazyllm_enable_report": True,
                },
            },
            {
                "id": "2",
                "kind": "Code",
                "name": "f2",
                "args": {
                    "code": "def main(x): return int(x) + 2",
                    "_lazyllm_enable_report": True,
                },
            },
            {
                "id": "3",
                "kind": "Code",
                "name": "f3",
                "args": {
                    "code": "def main(x): return int(x) + 3",
                    "_lazyllm_enable_report": True,
                },
            },
        ]
        edges = [
            {
                "iid": "__start__",
                "oid": "1",
            },
            {
                "iid": "1",
                "oid": "2",
            },
            {
                "iid": "2",
                "oid": "3",
            },
            {
                "iid": "3",
                "oid": "__end__",
            },
        ]
        resources = [
            {
                "id": "4",
                "kind": "server",
                "name": "s1",
                "args": {},
            }
        ]
        engine = LightEngine()
        engine.set_report_url(self.report_url)
        gid = engine.start(nodes, edges, resources)
        assert engine.run(gid, 1) == 7
        assert "prompt_tokens" in self.get_last_report()

    def test_engine_loop(self):
        nodes = [dict(id='1', kind='Code', name='code', args=dict(code='def square(x: int): return x * x'))]
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
        engine.set_report_url(self.report_url)
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 2) == 16
        assert "prompt_tokens" in self.get_last_report()

    def test_engine_warp(self):
        nodes = [dict(id='1', kind='Code', name='code', args=dict(code='def square(x: int): return x * x'))]
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
        engine.set_report_url(self.report_url)
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 2, 3, 4, 5) == (4, 9, 16, 25)
        assert "prompt_tokens" in self.get_last_report()

    def test_engine_warp_transform(self):
        nodes = [dict(id='1', kind='Code', name='code', args=dict(
            code='def sum(x: int, y: int, z: int): return x + y + z'))]
        edges = [dict(iid='__start__', oid='1'), dict(iid='1', oid='__end__')]

        nodes = [
            dict(
                id="2",
                kind="Warp",
                name="warp",
                args=dict(
                    nodes=nodes,
                    edges=edges,
                    batch_flags=[True, False, True],
                    _lazyllm_enable_report=True,
                ),
            )
        ]
        edges = [dict(iid='__start__', oid='2'), dict(iid='2', oid='__end__')]

        engine = LightEngine()
        engine.set_report_url(self.report_url)
        gid = engine.start(nodes, edges)
        assert engine.run(gid, [2, 3, 4, 5], 1, [1, 2, 3, 1]) == (4, 6, 8, 7)
        assert "prompt_tokens" in self.get_last_report()

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
        nodes = [dict(id='1', kind='Code', name='m1', args=dict(code='def test(x: int):\n    return x\n')),
                 dict(id='2', kind='Code', name='m2',
                      args=dict(code='def test(x: int):\n    return [[x, 2*x], [3*x, 4*x]]\n')),
                 dict(id='3', kind='Code', name='m3',
                 args=dict(code='def test(x: int):\n    return dict(a=1, b=x * x)\n')),
                 dict(id='4', kind='Code', name='m4', args=dict(code='def test(x, y, z):\n    return f"{x}{y}{z}"\n'))]
        edges = [dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'), dict(iid='__start__', oid='3'),
                 dict(iid='1', oid='4'), dict(iid='2', oid='4', formatter='[:][1]'),
                 dict(iid='3', oid='4', formatter='[b]'), dict(iid='4', oid='__end__')]

        engine = LightEngine()
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 1) == '1[2, 4]1'
        assert engine.run(gid, 2) == '2[4, 8]4'

    def test_engine_edge_formatter_from_start(self):
        nodes = [dict(id='1', kind='Code', name='m1', args=dict(code='def test(x: int):\n    return x\n'))]
        edges = [dict(iid='__start__', oid='1', formatter='[1:5]'), dict(iid='1', oid='__end__', formatter='[0:2]')]
        engine = LightEngine()
        gid = engine.start(nodes, edges)
        assert engine.run(gid, [0, 1, 2, 3, 4, 5]) == [1, 2]

    def test_engine_edge_formatter_start(self):
        nodes = [dict(id='1', kind='Code', name='m1', args=dict(code='def test(x: int): return x')),
                 dict(id='2', kind='Code', name='m2', args=dict(code='def test(x: int): return 2 * x')),
                 dict(id='3', kind='Code', name='m3', args=dict(code='def test(x, y): return x + y'))]
        edges = [dict(iid='__start__', oid='1', formatter='[0]'), dict(iid='__start__', oid='2', formatter='[1]'),
                 dict(iid='1', oid='3'), dict(iid='2', oid='3'), dict(iid='3', oid='__end__')]

        engine = LightEngine()
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 3, 1) == 5
        assert engine.run(gid, 5, 3, 1) == 11

    def test_engine_formatter_end(self):
        nodes = [dict(id='1', kind='Code', name='m1', args=dict(code='def test(x: int):\n    return x\n')),
                 dict(id='2', kind='Code', name='m2',
                      args=dict(code='def test1(x: int):\n    return [[x, 2*x], [3*x, 4*x]]\n')),
                 # two unused node
                 dict(id='3', kind='Code', name='m3',
                      args=dict(code='def test2(x: int):\n    return dict(a=1, b=x * x)\n')),
                 dict(id='4', kind='Code', name='m4', args=dict(code='def test3(x, y, z):\n    return f"{x}{y}{z}"\n'))]
        edges = [dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'), dict(iid='2', oid='__end__'),
                 dict(iid='1', oid='__end__')]

        engine = LightEngine()
        gid = engine.start(nodes, edges)
        r = engine.run(gid, 1)
        print(r, type(r))
        print(isinstance(r, lazyllm.package))

        engine.reset()

        nodes = [dict(id='1', kind='Code', name='m1', args=dict(code='def test(x: int):\n    return x\n')),
                 dict(id='2', kind='Code', name='m2',
                      args=dict(code='def test1(x: int):\n    return [[x, 2*x], [3*x, 4*x]]\n')),
                 dict(id='3', kind='JoinFormatter', name='join', args=dict(type='to_dict', names=['a', 'b']))]
        edges = [dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'), dict(iid='2', oid='3'),
                 dict(iid='1', oid='3'), dict(iid='3', oid='__end__', formatter='*[a, b]')]
        engine = LightEngine()
        gid = engine.start(nodes, edges)
        r = engine.run(gid, 1)
        print(r, type(r))
        print(isinstance(r, lazyllm.package))

    def test_engine_join_stack(self):
        nodes = [dict(id='0', kind='Code', name='c1', args=dict(code='def test(x: int): return x')),
                 dict(id='1', kind='JoinFormatter', name='join', args=dict(type='stack'))]
        edges = [dict(iid='__start__', oid='0'), dict(iid='0', oid='1'), dict(iid='1', oid='__end__')]
        engine = LightEngine()
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 1) == [1]
        assert engine.run(gid, '1') == ['1']
        assert engine.run(gid, [1]) == [[1]]

        engine.reset()

        nodes = [dict(id='0', kind='Code', name='c1', args=dict(code='def test(x: int): return x')),
                 dict(id='1', kind='Code', name='c2', args=dict(code='def test(x: int): return 2 * x')),
                 dict(id='2', kind='Code', name='c3', args=dict(code='def test(x: int): return 3 * x')),
                 dict(id='3', kind='JoinFormatter', name='join', args=dict(type='stack'))]
        edges = [dict(iid='__start__', oid='0'), dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'),
                 dict(iid='0', oid='3'), dict(iid='1', oid='3'), dict(iid='2', oid='3'), dict(iid='3', oid='__end__')]
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 1) == [1, 2, 3]
        assert engine.run(gid, '1') == ['1', '11', '111']
        assert engine.run(gid, [1]) == [[1], [1, 1], [1, 1, 1]]

    def test_engine_join_sum(self):
        nodes = [dict(id='0', kind='Code', name='c1', args=dict(code='def test(x: int): return [x, 2 * x]')),
                 dict(id='1', kind='JoinFormatter', name='join', args=dict(type='sum'))]
        edges = [dict(iid='__start__', oid='0'), dict(iid='0', oid='1'), dict(iid='1', oid='__end__')]
        engine = LightEngine()
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 1) == 3
        assert engine.run(gid, '1') == '111'
        assert engine.run(gid, [1]) == [1, 1, 1]

        engine.reset()

        nodes = [dict(id='0', kind='Code', name='c1', args=dict(code='def test(x: int): return x')),
                 dict(id='1', kind='Code', name='c2', args=dict(code='def test(x: int): return 2 * x')),
                 dict(id='2', kind='Code', name='c3', args=dict(code='def test(x: int): return 3 * x')),
                 dict(id='3', kind='JoinFormatter', name='join', args=dict(type='sum'))]
        edges = [dict(iid='__start__', oid='0'), dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'),
                 dict(iid='0', oid='3'), dict(iid='1', oid='3'), dict(iid='2', oid='3'), dict(iid='3', oid='__end__')]
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 1) == 6
        assert engine.run(gid, '1') == '111111'
        assert engine.run(gid, [1]) == [1, 1, 1, 1, 1, 1]

    def test_engine_join_todict(self):
        nodes = [dict(id='0', kind='Code', name='c1', args=dict(code='def test(x: int): return x')),
                 dict(id='1', kind='Code', name='c2', args=dict(code='def test(x: int): return 2 * x')),
                 dict(id='2', kind='Code', name='c3', args=dict(code='def test(x: int): return 3 * x')),
                 dict(id='3', kind='JoinFormatter', name='join', args=dict(type='to_dict', names=['a', 'b', 'c']))]
        edges = [dict(iid='__start__', oid='0'), dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'),
                 dict(iid='0', oid='3'), dict(iid='1', oid='3'), dict(iid='2', oid='3'), dict(iid='3', oid='__end__')]
        engine = LightEngine()
        gid = engine.start(nodes, edges)
        assert engine.run(gid, 1) == dict(a=1, b=2, c=3)
        assert engine.run(gid, '1') == dict(a='1', b='11', c='111')
        assert engine.run(gid, [1]) == dict(a=[1], b=[1, 1], c=[1, 1, 1])

    def test_engine_update(self):
        plus1 = dict(id='1', kind='Code', name='m1', args=dict(code='def test(x: int):\n    return 1 + x\n'))
        double = dict(id='2', kind='Code', name='m2', args=dict(code='def test(x: int):\n    return 2 * x\n'))
        square = dict(id='3', kind='Code', name='m3', args=dict(code='def test(x: int):\n    return x * x\n'))
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

        double = dict(id='2', kind='Code', name='m2', args=dict(code='def test(x: int):\n    return 3 * x\n'))
        ifs = dict(id='4', kind='Ifs', name='i1', args=dict(
            cond='def cond(x): return x < 10', true=[plus1, double], false=[square]
        ))
        nodes = [ifs]
        engine.update(gid, nodes, edges)

        assert engine.run(gid, 1) == 6
        assert engine.run(gid, 5) == 18
        assert engine.run(gid, 10) == 100

    def test_engine_join_join(self):
        nodes = [dict(id='0', kind='Code', name='c1', args=dict(code='def test(x: int): return x')),
                 dict(id='1', kind='Code', name='c2', args=dict(code='def test(x: int): return 2 * x')),
                 dict(id='2', kind='Code', name='c3', args=dict(code='def test(x: int): return 3 * x')),
                 dict(id='3', kind='JoinFormatter', name='join', args=dict(type='join'))]
        edges = [dict(iid='__start__', oid='0'), dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'),
                 dict(iid='0', oid='3'), dict(iid='1', oid='3'), dict(iid='2', oid='3'), dict(iid='3', oid='__end__')]
        engine = LightEngine()
        gid = engine.start(nodes, edges)
        assert engine.run(gid, '1') == '111111'

        nodes[-1] = dict(id='3', kind='JoinFormatter', name='join', args=dict(type='join', symbol='\n'))
        engine.update(gid, nodes, edges)
        assert engine.run(gid, '1') == '1\n11\n111'

    def test_engine_server(self):
        nodes = [dict(id='1', kind='Code', name='m1', args=dict(code='def test(x: int):\n    return 2 * x\n'))]
        edges = [dict(iid='__start__', oid='1'), dict(iid='1', oid='__end__')]
        resources = [dict(id='2', kind='server', name='s1', args=dict(port=None)),
                     dict(id='3', kind='web', name='w1', args=dict(port=None, title='网页', history=[], audio=False))
                    ]
        engine = LightEngine()
        gid = engine.start(nodes, edges, resources, gid='graph-1')
        assert engine.status(gid) == {'1': 'running', '2': lazyllm.launcher.Status.Running, '3': 'running'}
        assert engine.run(gid, 1) == 2
        time.sleep(3)

        server = engine.build_node('graph-1').func._g
        assert isinstance(server, lazyllm.ServerModule)
        m = lazyllm.UrlModule(url=server._url)
        assert m(2) == 4

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
            dict(id='0', kind='Code', name='code1', args=dict(code='def p1(): return "foo"')),
            dict(id='1', kind='Code', name='code2', args=dict(code='def p2(): return "bar"')),
            dict(id='2', kind='Code', name='code3', args=dict(code='def h1(): return "baz"')),
            dict(id='3', kind='HttpTool', name='http', args=dict(
                method='GET', url=url, params=params, headers=headers, _lazyllm_arg_names=['p1', 'p2', 'h1']))
        ]
        edges = [dict(iid='__start__', oid='0'), dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'),
                 dict(iid='0', oid='3'), dict(iid='1', oid='3'), dict(iid='2', oid='3'), dict(iid='3', oid='__end__')]

        engine = LightEngine()
        gid = engine.start(nodes, edges, gid='graph-1')
        res = engine.run(gid)

        assert res['headers']['h1'] == 'baz'
        assert res['url'].endswith(f'{url[5:]}?p1=foo&p2=bar')

    def test_engine_httptool_with_output(self):
        params = {'p1': '{{p1}}', 'p2': '{{p2}}'}
        headers = {'h1': '{{h1}}'}
        url = 'https://postman-echo.com/get'

        nodes = [
            dict(id='0', kind='Code', name='code1', args=dict(code='def p1(): return "foo"')),
            dict(id='1', kind='Code', name='code2', args=dict(code='def p2(): return "bar"')),
            dict(id='2', kind='Code', name='code3', args=dict(code='def h1(): return "baz"')),
            dict(id='3', kind='HttpTool', name='http', args=dict(
                method='GET', url=url, params=params, headers=headers,
                outputs=['headers', 'url'], _lazyllm_arg_names=['p1', 'p2', 'h1']))
        ]
        edges = [dict(iid='__start__', oid='0'), dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'),
                 dict(iid='0', oid='3'), dict(iid='1', oid='3'), dict(iid='2', oid='3'), dict(iid='3', oid='__end__')]

        engine = LightEngine()
        gid = engine.start(nodes, edges, gid='graph-1')
        res = engine.run(gid)

        assert isinstance(res, lazyllm.package) and len(res) == 2
        assert res[0]['h1'] == 'baz'
        assert res[1].endswith(f'{url[5:]}?p1=foo&p2=bar')

        engine.reset()

        nodes[3]['args']['outputs'] = ['output']
        gid = engine.start(nodes, edges)
        res = engine.run(gid)
        assert res['headers']['h1'] == 'baz'

        engine.reset()

        nodes[3]['args']['outputs'] = ['headers']
        nodes[3]['args']['extract_from_result'] = True
        gid2 = engine.start(nodes, edges)
        res = engine.run(gid2)
        assert res['h1'] == 'baz'

    def test_engine_httptool_body(self):
        body = {'b1': '{{b1}}', 'b2': '{{b2}}'}
        headers = {'Content-Type': '{{h1}}'}
        url = 'https://jsonplaceholder.typicode.com/posts'

        nodes = [
            dict(id='0', kind='Constant', name='header', args="application/json"),
            dict(id='1', kind='Constant', name='body1', args="body1"),
            dict(id='2', kind='Constant', name='body2', args="body2"),
            dict(id='3', kind='HttpTool', name='http', args=dict(
                method='POST', url=url, body=body, headers=headers, _lazyllm_arg_names=['h1', 'b1', 'b2']))
        ]
        edges = [dict(iid='__start__', oid='0'), dict(iid='__start__', oid='1'), dict(iid='__start__', oid='2'),
                 dict(iid='0', oid='3'), dict(iid='1', oid='3'), dict(iid='2', oid='3'), dict(iid='3', oid='__end__')]

        engine = LightEngine()
        gid = engine.start(nodes, edges, gid='graph-1')
        res = engine.run(gid)

        assert res['b1'] == 'body1'
        assert res['b2'] == 'body2'

    def test_engine_status(self):
        resources = [dict(id='0', kind='LocalLLM', name='m1', args=dict(base_model='', deploy_method='dummy'))]
        llm_node = dict(id='1', kind='SharedLLM', name='s1', args=dict(llm='0', prompt=None))

        plus1 = dict(id='2', kind='Code', name='m1', args=dict(code='def test(x: int):\n    return 1 + x\n'))
        double = dict(id='3', kind='Code', name='m2', args=dict(code='def test(x: int):\n    return 2 * x\n'))
        square = dict(id='4', kind='Code', name='m3', args=dict(code='def test(x: int):\n    return x * x\n'))

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
        engine.stop('5')  # stop subgraph
        assert '__start__' in engine._nodes and '__end__' in engine._nodes
        engine.release_node(gid)
        assert '__start__' in engine._nodes and '__end__' in engine._nodes

    def test_engine_reader(self):
        resources = [dict(id='file-resource', kind='File', name='file', args=dict(id='file-resource'))]
        nodes = [dict(id='1', kind='Reader', name='m1', args=dict()),
                 dict(id='2', kind='Reader', name='m2', args=dict(file_resource_id='file-resource'))]
        data_root_dir = os.getenv("LAZYLLM_DATA_PATH")
        p = os.path.join(data_root_dir, "rag_master/default/__data/sources/道德经.txt")
        engine = LightEngine()
        gid = engine.start(nodes, [['__start__', '1'], ['1', '__end__']], resources)
        data = engine.run(gid, p)
        assert '道可道' in data

        engine.reset()
        gid = engine.start(nodes, [['__start__', '2'], ['2', '__end__']], resources)
        data = engine.run(gid, p)
        assert '道可道' in data

        engine.reset()
        gid = engine.start(nodes, [['__start__', '2'], ['2', '__end__']], resources)
        file = os.path.join(data_root_dir, "rag_master/default/__data/sources/大学.txt")
        data = engine.run(gid, p, _file_resources={'file-resource': file})
        assert '道可道' in data
        assert '大学之道' in data

@pytest.mark.skip_on_win
@pytest.mark.skip_on_mac
class TestEngineRAG(object):

    def test_rag(self):
        resources = [
            dict(id='0', kind='Document', name='d1', args=dict(
                dataset_path='rag_master', activated_groups=['CoarseChunk', '00'])),
            dict(id='00', kind='LocalEmbedding', name='e1', args=dict(base_model='bge-large-zh-v1.5'))]
        nodes = [dict(id='1', kind='Retriever', name='ret1',
                      args=dict(doc='0', group_name='CoarseChunk', similarity='bm25_chinese', topk=3)),
                 dict(id='4', kind='Reranker', name='rek1',
                      args=dict(type='ModuleReranker', output_format='content', join=True,
                                arguments=dict(model="bge-reranker-large", topk=3))),
                 dict(id='5', kind='Code', name='c1',
                      args=dict(code='def test(nodes, query): return f\'context_str={nodes}, query={query}\'')),
                 dict(id='6', kind='LocalLLM', name='m1', args=dict(base_model='', deploy_method='dummy'))]
        edges = [dict(iid='__start__', oid='1'), dict(iid='1', oid='4'), dict(iid='__start__', oid='4'),
                 dict(iid='4', oid='5'), dict(iid='__start__', oid='5'), dict(iid='5', oid='6'),
                 dict(iid='6', oid='__end__')]
        engine = LightEngine()
        gid = engine.start(nodes, edges, resources)
        r = engine.run(gid, '何为修身?')
        assert '所谓修身在正其心者' in r

        # test add doc_group
        resources[0] = dict(id='0', kind='Document', name='d1', args=dict(
            dataset_path='rag_master', server=True, activated_groups=['CoarseChunk', '00'], node_group=[
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
        r = engine.run(gid, '何为修身?')
        assert '所谓修身在正其心者' in r
