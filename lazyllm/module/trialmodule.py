from .module import ModuleBase
from lazyllm import OptionIter
import time
import multiprocessing
import copy

def get_options(x):
    if isinstance(x, ModuleBase):
        return x.options
    return []

# TODO(wangzhihong): add process pool to control parallel-number and collect result
class TrialModule(object):
    def __init__(self, m):
        self.m = m

    @staticmethod
    def work(m, q):
        m = copy.deepcopy(m)
        m.update()
        q.put(m.eval_result)

    def update(self):
        options = get_options(self.m)
        q = multiprocessing.Queue()
        ps = []
        for _ in OptionIter(options, get_options):
            p = multiprocessing.Process(target=TrialModule.work, args=(self.m, q))
            ps.append(p)
            p.start()
            time.sleep(1)
        [p.join() for p in ps]
        result = [q.get() for p in ps]
        print(result)