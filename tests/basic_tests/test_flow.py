import lazyllm
from lazyllm import pipeline, parallel, diverter, warp, switch, ifs, loop, graph
from lazyllm import barrier, bind
import time
import pytest
import random

def add_one(x): return x + 1
def xy2z(x, y, z=0): return x + y + 2 * z
def is_1(x): return True if x == 1 else False
def is_2(x): return True if x == 2 else False
def is_3(x): return True if x == 3 else False
def t1(x): return 2 * x
def t2(x): return 3 * x
def t3(x): return x

class TestFlow(object):

    def test_pipeline(self):

        fl = pipeline(add_one, add_one)(1)
        assert fl == 3

        with pipeline() as p:
            p.f1 = add_one
            p.f2 = add_one
            p.f3 = xy2z | bind(y=p.input, z=p.f1)
        # 2 + 4 + 2 * 3
        assert p(2) == 12

    def test_server_with_bind(self):
        with pipeline() as ppl:
            ppl.f1 = lambda x: str(len(x))
            ppl.formatter = (lambda length, query: dict(length=length, query=query)) | bind(query=ppl.input)
            ppl.f2 = lambda y: f"The original query is : {y['query']}, length is : {y['length']}"
        sm = lazyllm.ServerModule(ppl)
        sm.start()
        query = "Hello World"
        assert sm(query) == f"The original query is : {query}, length is : {len(query)}"

    def test_parallel(self):
        fl = parallel(add_one, add_one)(1)
        assert fl == (2, 2)

    def test_parallel_sequential(self):
        fl = parallel.sequential(add_one, add_one)(1)
        assert fl == (2, 2)

    def test_diverter(self):

        fl = diverter(add_one, add_one)(1, 2)
        assert fl == (2, 3)

        div = diverter(lambda x: x + 1, lambda x: x * 2, lambda x: -x)
        assert div(1, 2, 3) == (2, 4, -3)

        div = diverter(a=lambda x: x + 1, b=lambda x: x * 2, c=lambda x: -x).asdict
        assert div(1, 2, 3) == {'a': 2, 'b': 4, 'c': -3}
        assert div(dict(c=3, b=2, a=1)) == {'a': 2, 'b': 4, 'c': -3}

    def test_warp(self):

        fl = warp(add_one)(1, 2, 3)
        assert fl == (2, 3, 4)

    def test_switch(self):

        assert switch({is_1: t1, is_2: t2}, judge_on_full_input=True)(1) == 2
        assert switch({is_1: t1, is_2: t2}, judge_on_full_input=True)(2) == 6
        assert not switch({is_1: t1, is_2: t2}, judge_on_full_input=True)(3)
        assert switch({is_1: t1, is_2: t2, 'default': t3}, judge_on_full_input=True)(3) == 3

        with switch(judge_on_full_input=True) as sw:
            sw.case[is_1::t1]
            sw.case(is_2, t2)
            sw.case[is_3, t3]
        assert sw(1) == 2 and sw(2) == 6 and sw(3) == 3

        with switch(conversion=lambda x: x / 10, judge_on_full_input=True) as sw:
            sw.case[is_1:t1]
            sw.case(is_2, t2)
            sw.case[is_3, t3]
        assert sw(10) == 20 and sw(20) == 60 and sw(30) == 30

        with switch(judge_on_full_input=False) as sw:
            sw.case[is_1:t1]
            sw.case(is_2, t2)
            sw.case[is_3, t3]
        assert sw(1, 30) == 60 and sw(2, 10) == 30 and sw(3, 5) == 5

    def test_ifs(self):

        assert ifs(is_1, t3, t1)(1) == 1
        assert ifs(is_1, t3, t1)(2) == 4

    def test_loop(self):

        assert loop(add_one, count=2)(0) == 2

        with loop(count=2) as lp:
            lp.f1 = add_one
            lp.f2 = add_one

        assert lp(1) == 5

        with loop(stop_condition=lambda x: x > 10) as lp:
            lp.f1 = add_one
            lp.f2 = add_one

        assert lp(1) == 11

    def test_barrier(self):
        res = []

        def get_data(idx):
            res.append(idx)
            return idx + 1

        ppl = pipeline(
            get_data,
            parallel(
                pipeline(
                    get_data,
                    barrier,
                    get_data,
                    barrier,
                    get_data,
                    get_data,
                ),
                pipeline(
                    get_data,
                    barrier,
                    get_data,
                    get_data,
                    get_data,
                    get_data,
                    barrier,
                    get_data,
                ),
            ),
        )
        ppl(0)
        assert res[:3] == [0, 1, 1]
        assert res[3:-3] in ([2, 2, 3, 4, 5], [2, 3, 2, 4, 5], [2, 3, 4, 2, 5], [2, 3, 4, 5, 2])
        assert res[-3:] in ([6, 3, 4], [3, 6, 4], [3, 4, 6])

    def test_graph(self):
        def test1(x):
            time.sleep(2)
            return f'1 get {x};'

        def test2(x): return f'2 get {x};'
        def test3(x): return f'3 get {x};'
        def add(x, y): return x + y
        def concat(x, y): return [x, y]

        with graph() as g:
            g.test1 = test1
            g.test2 = test2
            g.test3 = test3
            g.add = add
            g.concat = concat

        g.add_edge(g.start_node_name, ['test1', 'test2', 'test3'])
        g.add_edge(['test1', 'test2'], 'add')
        g.add_edge(['add', 'test3'], 'concat')
        g.add_edge('concat', g.end_node_name)

        assert g(1) == ['1 get 1;2 get 1;', '3 get 1;']


class TestFlowBind(object):
    def test_bind_pipeline_basic(self):
        with pipeline() as p:
            p.f1 = add_one
            p.f2 = add_one
            p.f3 = xy2z | bind(y=p.input, z=p.f1)
        assert p(2) == 12  # 4 + 2 + 2 * 3

        with pipeline() as p:
            p.f1 = add_one
            p.f2 = add_one
            p.f3 = xy2z | bind(y=p.input, z=p.output('f1'))
        assert p(3) == 16  # 5 + 3 + 2 * 4

    def test_bind_pipeline_unpack(self):
        def func0(x, y, z): return lazyllm.package(x, y, z)
        def func1(x, y, z): return lazyllm.package(x, y, z)
        def func2(x, y, z): return lazyllm.package(x, y, z)
        def func3(x, y, z): return lazyllm.package(x, y, z)

        with lazyllm.pipeline() as ppl:
            ppl.f1 = func1
            with lazyllm.parallel() as ppl.pp:
                ppl.pp.func2 = func2
                ppl.pp.func3 = func3
            ppl.fout = lazyllm.bind(func0, ppl.output('f1')[0], ppl.output('f1')[2], ppl.output('f1')[1])
        assert ppl(1, 2, 3) == (1, 3, 2)

        with lazyllm.pipeline() as ppl:
            ppl.f1 = func1
            with lazyllm.parallel() as ppl.pp:
                ppl.pp.func2 = func2
                ppl.pp.func3 = func3
            ppl.fout = lazyllm.bind(func0, ppl.output('f1')[0], ppl.output('f1', unpack=True)[2:0:-1])
        assert ppl(1, 2, 3) == (1, 3, 2)

        with lazyllm.pipeline() as ppl:
            ppl.f1 = func1
            with lazyllm.parallel() as ppl.pp:
                ppl.pp.func2 = func2
                ppl.pp.func3 = func3
            ppl.fout = lazyllm.bind(func0, ppl.output('f1', unpack=True))
        assert ppl(1, 2, 3) == (1, 2, 3)

    def test_bind_pipeline_nested(self):
        with pipeline() as p:
            p.f1 = add_one
            p.f2 = add_one
            with pipeline() as p.subp:
                p.subp.f3 = xy2z | bind(y=p.input, z=p.output('f1'))

        with pytest.raises(RuntimeError, match='pipeline.input/output can only be bind in direct member of pipeline!'):
            p(3)

        with lazyllm.save_pipeline_result():
            assert p(3) == 16

        with lazyllm.save_pipeline_result():
            with pipeline() as p:
                p.f1 = add_one
                p.f2 = add_one
                p.f3 = add_one
                with parallel().sum as p.subp:
                    p.subp.f3 = xy2z | bind(y=p.input, z=p.output(p.f1))
                    p.subp.f4 = xy2z | bind(y=p.input, z=p.output('f2'))

        assert p(3) == 36  # (6 + 3 + 8) + (6 + 3 + 10)

        with lazyllm.save_pipeline_result(False):
            with pytest.raises(RuntimeError,
                               match='pipeline.input/output can only be bind in direct member of pipeline!'):
                p(3)

    def test_bind_pipeline_in_warp(self):
        num = 5

        with lazyllm.save_pipeline_result():
            with pipeline() as ppl:
                with warp().sum as ppl.wp:
                    with pipeline() as ppl.wp.ppl:
                        ppl.wp.ppl.bug = lambda x: time.sleep(random.randint(0, 30) / 1000)
                        ppl.wp.ppl.for_output = lazyllm.ifs(lambda x, y: y is None,
                                                            lambda x, y: [{'idx': i} for i in range(num)],
                                                            lambda x, y: -1
                                                            ) | bind(ppl.wp.ppl.input["idx"], lazyllm._0)
                        with warp() as ppl.wp.ppl.wp2:
                            with pipeline() as ppl.wp.ppl.wp2.ppl:
                                ppl.wp.ppl.wp2.ppl.bug = lambda x: time.sleep(random.randint(0, 30) / 1000)
                                ppl.wp.ppl.wp2.ppl.for_output = lazyllm.ifs(lambda x, y, z: z is None,
                                                                            lambda x, y, z: x * num + y,
                                                                            lambda x, y, z: -1
                                    ) | bind(ppl.wp.ppl.input["idx"], ppl.wp.ppl.wp2.ppl.input["idx"], lazyllm._0)

        test_data = [{'idx': i} for i in range(num)]
        assert ppl(test_data) == tuple(range(num * num))

    def test_bind_pipeline_nested_kwargs(self):
        with pipeline() as p:
            p.f1 = add_one
            p.f2 = add_one
            with pipeline() as p.subp:
                p.subp.f3 = xy2z | bind(y=p.kwargs['x'], z=p.output('f1'))

        with pytest.raises(RuntimeError, match='pipeline.input/output can only be bind in direct member of pipeline!'):
            p(x=3)

        with lazyllm.save_pipeline_result():
            assert p(x=3) == 16

        def add(x, y): return x + y
        with lazyllm.save_pipeline_result():
            with pipeline() as p:
                p.f1 = add
                p.f2 = add_one
                p.f3 = add_one
                with parallel().sum as p.subp:
                    p.subp.f3 = xy2z | bind(y=p.input, z=p.output(p.f1))
                    p.subp.f4 = xy2z | bind(y=p.kwargs['y'], z=p.output('f2'))

        assert p(1, y=3) == 34  # (6 + 1 + 8) + (6 + 3 + 10)

    def test_bind_pipeline_nested_server(self):
        def add_one(x): return x + 1
        def xy2z(x, y, z=0): return x + y + 2 * z

        with lazyllm.save_pipeline_result():
            with pipeline() as p:
                p.f1 = add_one
                p.f2 = add_one
                p.f3 = add_one
                with parallel().sum as p.subp:
                    p.subp.f3 = xy2z | bind(y=p.input, z=p.output('f1'))
                    p.subp.f4 = xy2z | bind(y=p.input, z=p.output('f2'))

        s = lazyllm.ServerModule(p)
        s.start()
        assert s(3) == 36  # (6 + 3 + 8) + (6 + 3 + 10)
