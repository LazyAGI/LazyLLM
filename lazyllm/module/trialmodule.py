from .module import ModuleBase
from lazyllm import OptionIter, ForkProcess, LOG
import time
import copy
import multiprocessing

def get_options(x):
    if isinstance(x, ModuleBase):
        return x.options
    return []

# TODO(wangzhihong): add process pool to control parallel-number and collect result
class TrialModule(object):
    """Parameter grid search module will traverse all its submodules, collect all searchable parameters, and iterate over these parameters for fine-tuning, deployment, and evaluation.

Args:
    m (Callable): The submodule whose parameters will be grid-searched. Fine-tuning, deployment, and evaluation will be based on this module.


Examples:
    >>> import lazyllm
    >>> from lazyllm import finetune, deploy
    >>> m = lazyllm.TrainableModule('b1', 't').finetune_method(finetune.dummy, **dict(a=lazyllm.Option(['f1', 'f2'])))
    >>> m.deploy_method(deploy.dummy).mode('finetune').prompt(None)
    >>> s = lazyllm.ServerModule(m, post=lambda x, ori: f'post2({x})')
    >>> s.evalset([1, 2, 3])
    >>> t = lazyllm.TrialModule(s)
    >>> t.update()
    >>>
    dummy finetune!, and init-args is {a: f1}
    dummy finetune!, and init-args is {a: f2}
    [["post2(reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1})", "post2(reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1})", "post2(reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1})"], ["post2(reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1})", "post2(reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1})", "post2(reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1})"]]
    """
    def __init__(self, m):
        self.m = m

    @staticmethod
    def work(m, q):
        # update option at module.update()
        m = copy.deepcopy(m)
        m.update()
        q.put(m.eval_result)

    def update(self):
        options = get_options(self.m)
        q = multiprocessing.Queue()
        ps = []
        for _ in OptionIter(options, get_options):
            p = ForkProcess(target=TrialModule.work, args=(self.m, q), sync=True)
            ps.append(p)
            p.start()
            time.sleep(1)
        [p.join() for p in ps]
        result = [q.get() for p in ps]
        LOG.info(f'{result}')
