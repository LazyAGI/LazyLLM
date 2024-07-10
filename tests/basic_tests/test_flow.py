from lazyllm import pipeline, parallel, diverter, warp, switch, ifs, loop, barrier, bind

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

    def test_parallel(self):
        fl = parallel(add_one, add_one)(1)
        assert fl == (2, 2)

    def test_parallel_sequential(self):
        fl = parallel.sequential(add_one, add_one)(1)
        assert fl == (2, 2)

    def test_diverter(self):

        fl = diverter(add_one, add_one)(1, 2)
        assert fl == (2, 3)

    def test_warp(self):

        fl = warp(add_one)(1, 2, 3)
        assert fl == (2, 3, 4)

    def test_switch(self):

        assert switch({is_1: t1, is_2: t2}, judge_on_input=True)(1) == 2
        assert switch({is_1: t1, is_2: t2}, judge_on_input=True)(2) == 6
        assert not switch({is_1: t1, is_2: t2}, judge_on_input=True)(3)
        assert switch({is_1: t1, is_2: t2, 'default': t3}, judge_on_input=True)(3) == 3

        with switch(judge_on_input=True) as sw:
            sw.case[is_1::t1]
            sw.case(is_2, t2)
            sw.case[is_3, t3]
        assert sw(1) == 2 and sw(2) == 6 and sw(3) == 3

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
