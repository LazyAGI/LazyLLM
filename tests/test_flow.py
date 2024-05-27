from lazyllm import pipeline, parallel, diverter, warp, switch, ifs, loop

def add_one(x):    return x+1
def is_1(x):    return True if x==1 else False       
def is_2(x):    return True if x == 2 else False
def t1(x):    return 2*x
def t2(x):    return 3*x
def t3(x):    return x

class TestFn_Flow(object):
    
    def test_pipeline(self):
        
        fl = pipeline(add_one, add_one)(1)
        assert fl == 3
    
    def test_parallel(self):
        fl = parallel(add_one, add_one)(1)
        assert fl == (2,2)
        
    def test_parallel_sequential(self):
        fl = parallel.sequential(add_one, add_one)(1)
        assert fl == (2,2)
        
    def test_diverter(self):
        
        fl = diverter(add_one, add_one)(1,2)
        assert fl == (2,3)
        
    def test_warp(self):
        
        fl = warp(add_one)(1,2,3)
        assert fl == (2,3,4)
        
    def test_switch(self):

        assert switch({is_1: t1, is_2: t2})(1) == 2
        assert switch({is_1: t1, is_2: t2})(2) == 6
        assert switch({is_1: t1, is_2: t2})(3) == None
        assert switch({is_1: t1, is_2: t2, 'default': t3})(3) == 3
        
    def test_ifs(self):
        
        assert ifs(is_1, t3, t1)(1) == 1
        assert ifs(is_1, t3, t1)(2) == 4
        
    def test_loop(self):
        
        assert loop(add_one, count=2)(0) == 2
        # assert loop(add_one, stop_condition=is_1)(0)
