from .flow import FlowBase, Pipeline, Parallel, DPES
import os
import lazyllm
from lazyllm import FlatList
import json
import httpx


class ModuleBase(object):
    def __init__(self):
        self.submodules = []

    def __setattr__(self, name: str, value):
        if isinstance(value, ModuleBase):
            self.submodules.append(value)
        return super().__setattr__(name, value)

    def __call__(self, *args, **kw): return self.forward(*args, **kw)

    # interfaces
    def forward(self, *args, **kw): raise NotImplementedError
    def _get_train_tasks(self): return None
    def _get_deploy_tasks(self): return None
    def _get_eval_tasks(self): return None

    # update module(train or finetune), 
    def update(self, *, mode='all', recursive=True):
        assert mode in ('all', 'server')
        # dfs to get all train tasks
        train_tasks, deploy_tasks, eval_tasks = FlatList(), FlatList(), FlatList()
        stack = [(self, iter(self.submodules if recursive else []))]
        while len(stack) > 0:
            try:
                top = next(stack[-1][1])
                stack.append((top, iter(top.submodules)))
            except StopIteration:
                top = stack.pop()[0]
                train_tasks.absorb(top._get_train_tasks())
                deploy_tasks.absorb(top._get_deploy_tasks())
                eval_tasks.absorb(top._get_eval_tasks())

        if mode == 'all' and len(train_tasks) > 0:
            Parallel(*train_tasks).start()
        if len(deploy_tasks) > 0:
            DPES(*deploy_tasks).start()
        if mode == 'all' and len(eval_tasks) > 0:
            DPES(*eval_tasks).start()

    def update_server(self, *, recursive=True): return self.update(mode='server', recursive=recursive)
    def start(self): return self.update(mode='server', recursive=True)
    def restart(self): return self.start()

    # TODO: add lazyllm.eval
    def eval(self, input_or_path):
        if os.path.exists(input_or_path):
            with open(input_or_path) as f:
                datas = json.load(f)
            output =  [self(item) for item in datas]
        else:
            output =  self(input_or_path)
        return output

    def _overwrote(self, f):
        return getattr(self.__class__, f) is not getattr(__class__, f)


class SequenceModule(ModuleBase):
    def __init__(self, *args):
        super().__init__()
        self.submodules = list(args)

    def forward(self, *args, **kw):
        ppl = Pipeline(*self.submodules)
        return ppl.start(*args, **kw)
    

class UrlModule(ModuleBase):
    def __init__(self, url):
        super().__init__()
        self.url = url
        
    def forward(self, input):
        with httpx.Client(timeout=90) as client:
           response = client.post(self.url, json={'input': input}, headers={'Content-Type': 'application/json'})
        return response.text


class ActionModule(ModuleBase):
    def __init__(self, action):
        super().__init__()
        action.for_each(lambda x: isinstance(x, ModuleBase), lambda x: self.submodules.append(x))
        self.action = action

    def forward(self, *args, **kw):
        handle = self.action.start(*args, **kw)
        return handle


class ServerModule(ModuleBase):
    def __init__(self, m, pre=None, post=None):
        super().__init__()
        self.m = m
        self._url = None
        self._pre_func = pre
        self._post_func = post

    def forward(self, input):
        assert self._url is not None, f'Please start {__class__} first'
        with httpx.Client(timeout=90) as client:
           response = client.post(self._url, json={'input': input}, headers={'Content-Type': 'application/json'})
        return response.text
    
    def url(self, url):
        print('url:', url)
        self._url = url

    def _get_deploy_tasks(self):
        return Pipeline(
            lazyllm.deploy.RelayServer(func=self.m, pre_func=self._pre_func, post_func=self._post_func),
            self.url)

class TrainableModule(ModuleBase):
    def __init__(self, base_model, target_path):
        super().__init__()
        self.base_model = base_model
        self.target_path = target_path
        self._train = lazyllm.train.auto
        self._finetune = lazyllm.finetune.auto
        self._deploy = lazyllm.deploy.auto
    
    def _get_train_tasks(self):
        trainset_getf = lambda : lazyllm.package(self._trainset, None) \
                        if isinstance(self._trainset, str) else self._trainset
        if self.mode == 'train':
            train = self._train(self.base_model, os.path.join(self.target_path, 'train'))
        elif self.mode == 'finetune':
            train = self._finetune(self.base_model, os.path.join(self.target_path, 'finetune'))
        else:
            raise RuntimeError('mode must be train or finetune')
        return Pipeline(trainset_getf, train)

    def _get_deploy_tasks(self):
        return Pipeline(lambda *a: self.target_path,
            self._deploy(pre_func=self._pre_func, post_func=self._post_func))

    def _forward(self, *args, **kw):
        output = curl(self._url, input)
        return output

    def __getattr__(self, key):
        def _setattr(v):
            setattr(self, f'_{key}', v)
            return self
        keys = ['trainset', 'train', 'finetune', 'deploy', 'pre_func', 'post_func', 'mode', 'url', 'evalset']
        if key in keys:
            return _setattr
        elif key.startswith('_') and key[1:] in keys:
            raise ValueError(f'Please call `{key[1:]}()` to set `self.{key}` first.')
        raise AttributeError(f'{__class__} object has no attribute {key}')

    # change to urlmodule when pickling to server process
    def __reduce__(self):
        assert hasattr(self, '_url')
        m = UrlModule(self._url)
        return m.__reduce__()


class Module(object):
    # modules(list of modules) -> SequenceModule
    # action(lazyllm.flow) -> ActionModule
    # url(str) -> UrlModule
    # base_model(str) & target_path(str)-> TrainableModule
    def __new__(self, *args, **kw):
        if len(args) >= 1 and isinstance(args[0], Module):
            return SequenceModule(*args)
        elif len(args) == 1 and isinstance(args[0], list) and isinstance(args[0][0], Module):
            return SequenceModule(*args[0])
        elif len(args) == 0 and 'modules' in kw:
            return SequenceModule(kw['modules'])
        elif len(args) == 1 and isinstance(args[0], FlowBase):
            return ActionModule(args[0])
        elif len(args) == 0 and 'action' in kw:
            return ActionModule(kw['modules'])
        elif len(args) == 1 and isinstance(args[0], str):
            return UrlModule(args[0])
        elif len(args) == 0 and 'url' in kw:
            return UrlModule(kw['url'])
        elif ...:
            return TrainableModule()

    @classmethod
    def sequence(cls, *args, **kw): return SequenceModule(*args, **kw)
    @classmethod
    def action(cls, *args, **kw): return ActionModule(*args, **kw)
    @classmethod
    def url(cls, *args, **kw): return UrlModule(*args, **kw)
    @classmethod
    def trainable(cls, *args, **kw): return TrainableModule(*args, **kw)


# TODO(wangzhihong): remove these examples
# Examples:

m1 = Module.url('1')
m2 = Module.url('2')

seq_m = Module.sequence(m1, m2)
act_m = Module.action(Pipeline(seq_m, m2))

class MyModule(ModuleBase):
    def __init__(self):
        super().__init__()
        self.m1 = act_m
        self.m2 = seq_m 

    def forward(self, *args, **kw):
        ppl = Pipeline(self.m1, self.m2)
        ppl.start()

my_m = MyModule()