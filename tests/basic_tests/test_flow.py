from lazyllm import pipeline, parallel, diverter, warp, switch, ifs, loop, barrier

def add_one(x): return x + 1
def is_1(x): return True if x == 1 else False
def is_2(x): return True if x == 2 else False
def t1(x): return 2 * x
def t2(x): return 3 * x
def t3(x): return x

class TestFlow(object):

    def test_pipeline(self):

        fl = pipeline(add_one, add_one)(1)
        assert fl == 3

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

        assert switch({is_1: t1, is_2: t2})(1) == 2
        assert switch({is_1: t1, is_2: t2})(2) == 6
        assert not switch({is_1: t1, is_2: t2})(3)
        assert switch({is_1: t1, is_2: t2, 'default': t3})(3) == 3

    def test_ifs(self):

        assert ifs(is_1, t3, t1)(1) == 1
        assert ifs(is_1, t3, t1)(2) == 4

    def test_loop(self):

        assert loop(add_one, count=2)(0) == 2
        # assert loop(add_one, stop_condition=is_1)(0)

    def test_barrier(self):
        res = []

        def get_data(idx):
            res.append(str(idx))
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
        print(res, res == ['0', '1', '1', '2', '3', '4', '5', '2', '6', '3', '4'])
