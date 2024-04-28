import os
import re
import copy

import requests
import pickle
import codecs
import inspect

import lazyllm
from lazyllm import FlatList, LazyLlmResponse, LazyLlmRequest, Option
from ..flow import FlowBase, Pipeline, Parallel, DPES
import uuid


class ModuleBase(object):
    builder_keys = []  # keys in builder support Option by default

    def __new__(cls, *args, **kw):
        sig = inspect.signature(cls.__init__)
        paras = sig.parameters
        values = list(paras.values())[1:]  # paras.value()[0] is self
        for i, p in enumerate(args):
            if isinstance(p, Option):
                ann = values[i].annotation
                assert ann == Option or (isinstance(ann, (tuple, list)) and Option in ann), \
                    f'{values[i].name} cannot accept Option'
        for k, v in kw.items():
            if isinstance(v, Option):
                ann = paras[k].annotation
                assert ann == Option or (isinstance(ann, (tuple, list)) and Option in ann), \
                    f'{values[i].name} cannot accept Option'
        return object.__new__(cls)

    def __init__(self, *, return_trace=False):
        self.submodules = []
        self._evalset = None
        self._return_trace = return_trace
        self.mode_list  = ('train', 'server', 'eval')
        self._module_id = str(uuid.uuid4().hex)
        self._module_name = None
        self._options = []
        self.eval_result = None

    def __setattr__(self, name: str, value):
        if isinstance(value, ModuleBase):
            self.submodules.append(value)
        elif isinstance(value, Option):
            self._options.append(value)
        elif name.endswith('_args') and isinstance(value, dict):
            for v in value.values():
                if isinstance(v, Option):
                    self._options.append(v)
        return super().__setattr__(name, value)

    def __getattr__(self, key):
        def _setattr(v, **kw):
            if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], dict):
                kw.update(v[1])
                v = v[0]
            if len(kw) > 0:
                setattr(self, f'_{key}_args', kw)
            setattr(self, f'_{key}', v)
            if hasattr(self, f'_{key}_setter_hook'): getattr(self, f'_{key}_setter_hook')()
            return self
        keys =  self.__class__.builder_keys
        if key in keys:
            return _setattr
        elif key.startswith('_') and key[1:] in keys:
            return None
        raise AttributeError(f'{__class__} object has no attribute {key}')


    def __call__(self, *args, **kw): 
        if len(args) == 1 and isinstance(args[0], LazyLlmRequest):
            assert len(kw) == 0, 'Cannot use LazyLlmRequest and kwargs at the same time'
            args[0].kwargs.update(args[0].global_parameters.get(self._module_id, dict()))
        r = self.forward(*args, **kw)
        if self._return_trace:
            if isinstance(r, LazyLlmResponse): r.trace += f'{str(r)}\n'
            else: r = LazyLlmResponse(messages=r, trace=f'{str(r)}\n')
        return r

    # interfaces
    def forward(self, *args, **kw): raise NotImplementedError
    def _get_train_tasks(self): return None
    def _get_deploy_tasks(self): return None

    @property
    def name(self): return self._module_name
    @name.setter
    def name(self, name): self._module_name = name

    def evalset(self, evalset, load_f=None, collect_f=lambda x:x):
        if isinstance(evalset, str) and os.path.exists(evalset):
            with open(evalset) as f:
                assert callable(load_f)
                self._evalset = load_f(f)
        else:
            self._evalset = evalset
        self.eval_result_collet_f = collect_f

    # TODO: add lazyllm.eval
    def _get_eval_tasks(self):
        def set_result(x): self.eval_result = x
        if self._evalset:
            return Pipeline(lambda: [self(**item) if isinstance(item, dict) else self(item)
                                     for item in self._evalset],
                            lambda x: self.eval_result_collet_f(x),
                            set_result)
        return None

    # update module(train or finetune), 
    def _update(self, *, mode=None, recursive=True):
        if not mode:
            mode = list(self.mode_list)
        if type(mode) is not list:
            mode = [mode]
        for item in mode:
            assert item in self.mode_list, f"Cannot find {item} in mode list: {self.mode_list}"
        # dfs to get all train tasks
        train_tasks, deploy_tasks, eval_tasks = FlatList(), FlatList(), FlatList()
        stack = [(self, iter(self.submodules if recursive else []))]
        while len(stack) > 0:
            try:
                top = next(stack[-1][1])
                stack.append((top, iter(top.submodules)))
            except StopIteration:
                top = stack.pop()[0]
                if 'train' in mode:
                    train_tasks.absorb(top._get_train_tasks())
                if 'server' in mode:
                    deploy_tasks.absorb(top._get_deploy_tasks())
                if 'eval' in mode:
                    eval_tasks.absorb(top._get_eval_tasks())

        if 'train' in mode and len(train_tasks) > 0:
            Parallel(*train_tasks).sync_start()
        if 'server' in mode and len(deploy_tasks) > 0:
            DPES(*deploy_tasks).start()
        if 'eval' in mode and len(eval_tasks) > 0:
            DPES(*eval_tasks).start()

    def update(self, *, recursive=True): return self._update(mode=['train', 'server', 'eval'], recursive=recursive)
    def update_server(self, *, recursive=True): return self._update(mode=['server'], recursive=recursive)
    def eval(self, *, recursive=True): return self._update(mode=['eval'], recursive=recursive)
    def start(self): return self._update(mode=['server'], recursive=True)
    def restart(self): return self.start()

    @property
    def options(self):
        options = self._options.copy()
        for m in self.submodules:
            options += m.options
        return options

    def _overwrote(self, f):
        return getattr(self.__class__, f) is not getattr(__class__, f)


class UrlModule(ModuleBase):
    def __init__(self, url, *, stream=False, meta=None, return_trace=False):
        super().__init__(return_trace=return_trace)
        self._url = url
        self._stream = stream
        self._meta = meta if meta else UrlModule
        self.prompt()
        # Set for request by specific deploy:
        self._set_template(template_headers={'Content-Type': 'application/json'})

    def url(self, url):
        print('url:', url)
        self._url = url

    # Cannot modify or add any attrubute of self
    # prompt keys are in __input (ATTENTION: dict, not kwargs)
    # deploy parameters keys are in **kw
    def forward(self, __input=None, **kw):
        assert self._url is not None, f'Please start {self.__class__} first'
        assert len(kw) == 0 or self.template_message is not None, 'kwargs are used in deploy parameters'

        input = __input.input if isinstance(__input, LazyLlmRequest) else __input
        kw = __input.kwargs if isinstance(__input, LazyLlmRequest) else kw
        if self._prompt is not None: # dict input will pass to sub-module if prompt is None
            if not isinstance(input, dict):
                assert len(self._prompt_keys) == 1, f'invalid prompt `{self._prompt}` for input `{input}`'
                input = {self._prompt_keys[0]: input}
            input = self._prompt.format(**input)
        
        if self._meta == ServerModule and isinstance(__input, LazyLlmRequest):
            __input.input = input
            data = codecs.encode(pickle.dumps(__input), 'base64').decode('utf-8')
        elif self.template_message is not None: 
            data = self._modify_parameters(copy.deepcopy(self.template_message), kw)
            data[self.input_key_name] = input
        else:
            data = input

        def _callback(text):
            if isinstance(text, LazyLlmResponse):
                text.messages = text.messages if self._response_split is None else \
                                text.messages.split(self._response_split)[-1]
            else:
                text = text if self._response_split is None else text.split(self._response_split)[-1]
            return text

        # context bug with httpx, so we use requests
        def _impl():
            with requests.post(self._url, json=data, stream=True) as r:
                if r.status_code == 200:
                    for chunk in r.iter_content(None):
                        try:
                            chunk = pickle.loads(codecs.decode(chunk, "base64"))
                            assert isinstance(chunk, LazyLlmResponse)
                        except Exception:
                            chunk = chunk.decode('utf-8')
                        yield(_callback(chunk))
                else:
                    raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))
        if self._stream:
            return _impl()
        else:
            for r in _impl(): pass
            return r
                

    def prompt(self, prompt=None, response_split=None):
        self._prompt, self._response_split = prompt, response_split
        self._prompt_keys = list(set(re.findall(r'\{(\w+)\}', self._prompt))) if prompt else []
        return self
    
    def _set_template(self, template_message=None, input_key_name=None, template_headers=None):
        assert input_key_name is None or input_key_name in template_message.keys()
        self.template_message = template_message
        self.input_key_name = input_key_name
        self.template_headers = template_headers

    def _modify_parameters(self, paras, kw):
        for key, value in paras.items():
            if key == self.input_key_name:
                continue
            elif isinstance(value, dict):
                if key in kw:
                    assert set(kw[key].keys()).issubset(set(value.keys()))
                    value.update(kw.pop(key))
                for k in value.keys():
                    if k in kw: value[k] = kw.pop(k)
            else:
                paras[key] = kw.pop(key)
        return paras

    def set_default_parameters(self, **kw):
        self._modify_parameters(self.template_message, kw)

    def clone(self):
        assert hasattr(self, '_url') and self._url is not None
        m = UrlModule(self._url, stream=self._stream, meta=self._meta, return_trace=self._return_trace).prompt(
            prompt=self._prompt, response_split=self._response_split)
        m._module_id = self._module_id
        m.name = self.name
        m._set_template(
            self.template_message,
            self.input_key_name,
            self.template_headers,
        )
        return m

    def __repr__(self):
        return f'<UrlModule [url: \'{self._url}\']>'

    # change to urlmodule when pickling to server process
    def __reduce__(self):
        if self.__class__ != UrlModule and os.getenv('LAZYLLM_ON_CLOUDPICKLE', False) == 'ON':
            m = self.clone()
            return m.__reduce__()
        else:
            return super(__class__, self).__reduce__()


class ActionModule(ModuleBase):
    def __init__(self, action, *, return_trace=False):
        super().__init__(return_trace=return_trace)
        if not isinstance(action, (tuple, list, FlowBase)):
            # Use flow to assist with input processing
            action = [action]
        if isinstance(action, (tuple, list)):
            self.submodules = [a for a in action if isinstance(a, ModuleBase)]
            self.action = Pipeline(*action)
        elif isinstance(action, FlowBase):
            action.for_each(lambda x: isinstance(x, ModuleBase), lambda x: self.submodules.append(x))
            self.action = action
        else:
            raise TypeError(f'Invalid action type {type(action)}')

    def forward(self, *args, **kw):
        return self.action.start(*args, **kw)

    def __repr__(self):
        representation = f'<ActionModule - {type(self.action).__name__.capitalize()}> ['
        sub_rep = '\n'.join([s for s in repr(self.action).split('\n')][1:-1])
        representation += '\n' + sub_rep + '\n'
        return representation + ']'


class ServerModule(UrlModule):
    def __init__(self, m, pre=None, post=None, stream=False, return_trace=False):
        assert stream is False or return_trace is False, 'Module with stream output has no trace'
        super().__init__(url=None, stream=stream, meta=ServerModule, return_trace=return_trace)
        self.m = ActionModule(m) if isinstance(m, FlowBase) else m
        self._pre_func, self._post_func = pre, post
        assert (post is None) or (stream == False)
        self._set_template(
            copy.deepcopy(lazyllm.deploy.RelayServer.message_format),
            lazyllm.deploy.RelayServer.input_key_name,
            copy.deepcopy(lazyllm.deploy.RelayServer.default_headers),
        )

    def _get_deploy_tasks(self):
        return Pipeline(
            lazyllm.deploy.RelayServer(func=self.m, pre_func=self._pre_func, post_func=self._post_func),
            self.url)
    
    def __repr__(self):
        representation = '<ServerModule> ['
        if isinstance(self.m, (FlowBase, ActionModule, ServerModule)):
            sub_rep = '\n'.join(['    ' + s for s in repr(self.m).split('\n')])
            representation += '\n' + sub_rep + '\n'
        else:
            representation += repr(self.m)
        return representation + ']'


class TrainableModule(UrlModule):
    builder_keys = ['trainset', 'train', 'finetune', 'deploy', 'mode']

    def __init__(self, base_model:Option='', target_path='', *, stream=False, return_trace=False):
        super().__init__(url=None, stream=stream, meta=TrainableModule, return_trace=return_trace)
        # Fake base_model and target_path for dummy
        self.base_model = base_model
        self.target_path = target_path
        self._train = None # lazyllm.train.auto
        self._finetune = lazyllm.finetune.auto
        self._deploy = None # lazyllm.deploy.auto

    def _get_args(self, arg_cls, disable=[]):
        args = getattr(self, f'_{arg_cls}_args', dict())
        if len(set(args.keys()).intersection(set(disable))) > 0:
            raise ValueError(f'Key `{", ".join(disable)}` can not be set in '
                             '{arg_cls}_args, please pass them from Module.__init__()')
        return args

    def _get_train_tasks(self):
        trainset_getf = lambda : lazyllm.package(self._trainset, None) \
                        if isinstance(self._trainset, str) else self._trainset
        if self._mode == 'train':
            args = self._get_args('train', disable=['base_model', 'target_path'])
            train = self._train(base_model=self.base_model, target_path=self.target_path, **args)
        elif self._mode == 'finetune':
            args = self._get_args('finetune', disable=['base_model', 'target_path'])
            train = self._finetune(base_model=self.base_model, target_path=self.target_path, **args)
        else:
            raise RuntimeError('mode must be train or finetune')
        return Pipeline(trainset_getf, train)

    def _get_deploy_tasks(self):
        if os.path.basename(self.target_path) != 'merge':
            target_path = os.path.join(self.target_path, 'merge')

        if not os.path.exists(target_path):
            target_path = self.target_path
        return Pipeline(lambda *a: lazyllm.package(target_path, self.base_model),
                        self._deploy(stream=self._stream, **self._deploy_args), self.url)

    def _deploy_setter_hook(self):
        self._deploy_args = self._get_args('deploy', disable=['target_path'])
        self._set_template(copy.deepcopy(self._deploy.message_format),
            self._deploy.input_key_name, copy.deepcopy(self._deploy.default_headers))

    def __repr__(self):
        mode = '-Train' if self._mode == 'train' else (
               '-Finetune' if self._mode == 'finetune' else '')
        return f'<TrainableModule{mode}> [base-model: "{self.base_model}"]'


class Module(object):
    # modules(list of modules) -> ActionModule
    # action(lazyllm.flow) -> ActionModule
    # url(str) -> UrlModule
    # base_model(str) & target_path(str)-> TrainableModule
    def __new__(self, *args, **kw):
        if len(args) >= 1 and isinstance(args[0], Module):
            return ActionModule(*args)
        elif len(args) == 1 and isinstance(args[0], list) and isinstance(args[0][0], Module):
            return ActionModule(*args[0])
        elif len(args) == 0 and 'modules' in kw:
            return ActionModule(kw['modules'])
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
    def action(cls, *args, **kw): return ActionModule(*args, **kw)
    @classmethod
    def url(cls, *args, **kw): return UrlModule(*args, **kw)
    @classmethod
    def trainable(cls, *args, **kw): return TrainableModule(*args, **kw)
