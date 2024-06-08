import os
import copy
import time
import requests
import pickle
import codecs
import inspect
from concurrent.futures import ThreadPoolExecutor

import lazyllm
from lazyllm import FlatList, LazyLlmResponse, LazyLlmRequest, Option, launchers, LOG
from ..components.prompter import PrompterBase, ChatPrompter, EmptyPrompter
from ..flow import FlowBase, Pipeline, Parallel
import uuid
from ..client import get_redis, redis_client


class ModuleBase(object):
    builder_keys = []  # keys in builder support Option by default
    __enable_request__ = False

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
                    f'{k} cannot accept Option'
        return object.__new__(cls)

    def __init__(self, *, return_trace=False):
        self.submodules = []
        self._evalset = None
        self._return_trace = return_trace
        self.mode_list = ('train', 'server', 'eval')
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
            k = key[:-7] if key.endswith('_method') else key
            if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], dict):
                kw.update(v[1])
                v = v[0]
            if len(kw) > 0:
                setattr(self, f'_{k}_args', kw)
            setattr(self, f'_{k}', v)
            if hasattr(self, f'_{k}_setter_hook'): getattr(self, f'_{k}_setter_hook')()
            return self
        keys = self.__class__.builder_keys
        if key in keys:
            return _setattr
        elif key.startswith('_') and key[1:] in keys:
            return None
        elif key.startswith('_') and key.endswith('_args') and (key[1:-5] in keys or f'{key[1:-4]}method' in keys):
            return dict()
        raise AttributeError(f'{__class__} object has no attribute {key}')

    def __call__(self, *args, **kw):
        if len(args) == 1 and isinstance(args[0], LazyLlmRequest):
            assert len(kw) == 0, 'Cannot use LazyLlmRequest and kwargs at the same time'
            args[0].kwargs.update(args[0].global_parameters.get(self._module_id, dict()))
            if not getattr(getattr(self, '_meta', self.__class__), '__enable_request__', False):
                kw = args[0].kwargs
                args = args[0].input if isinstance(args[0].input, lazyllm.package) else (args[0].input,)
        r = self.forward(*args, **kw)
        if self._return_trace:
            if isinstance(r, LazyLlmResponse): r.trace += f'{str(r)}\n'
            else: r = LazyLlmResponse(messages=r, trace=f'{str(r)}\n')
        return r

    # interfaces
    def forward(self, *args, **kw): raise NotImplementedError
    def _get_train_tasks(self): return None
    def _get_deploy_tasks(self): return None
    def _get_post_process_tasks(self): return None

    @property
    def name(self): return self._module_name
    @name.setter
    def name(self, name): self._module_name = name

    def evalset(self, evalset, load_f=None, collect_f=lambda x: x):
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

        def parallel_infer():
            with ThreadPoolExecutor(max_workers=100) as executor:
                results = list(executor.map(lambda item: self(**item)
                                            if isinstance(item, dict) else self(item), self._evalset))
            return results
        if self._evalset:
            return Pipeline(parallel_infer,
                            lambda x: self.eval_result_collet_f(x),
                            set_result)
        return None

    # update module(train or finetune),
    def _update(self, *, mode=None, recursive=True):  # noqa C901
        if not mode:
            mode = list(self.mode_list)
        if type(mode) is not list:
            mode = [mode]
        for item in mode:
            assert item in self.mode_list, f"Cannot find {item} in mode list: {self.mode_list}"
        # dfs to get all train tasks
        train_tasks, deploy_tasks, eval_tasks, post_process_tasks = FlatList(), FlatList(), FlatList(), FlatList()
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
                    deploy_tasks.absorb(lazyllm.call_once(top._deploy_flag, top._get_deploy_tasks)
                                        if hasattr(top, '_deploy_flag') else top._get_deploy_tasks())
                if 'eval' in mode:
                    eval_tasks.absorb(top._get_eval_tasks())
                post_process_tasks.absorb(top._get_post_process_tasks())

        if 'train' in mode and len(train_tasks) > 0:
            Parallel(*train_tasks).set_sync(True)()
        if 'server' in mode and len(deploy_tasks) > 0:
            if redis_client:
                Parallel(*deploy_tasks).set_sync(False)()
            else:
                Parallel.sequential(*deploy_tasks)()
        if 'eval' in mode and len(eval_tasks) > 0:
            Parallel.sequential(*eval_tasks)()
        Parallel.sequential(*post_process_tasks)()
        return self

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
    __enable_request__ = True

    def __init__(self, *, url='', stream=False, meta=None, return_trace=False):
        super().__init__(return_trace=return_trace)
        self._url = url
        self._stream = stream
        self._meta = meta if meta else UrlModule
        self.prompt()
        # Set for request by specific deploy:
        self._set_template(template_headers={'Content-Type': 'application/json'})
        self._extract_result_func = lambda x: x

    def url(self, url):
        if redis_client:
            redis_client.set(self._module_id, url)
        LOG.debug(f'url: {url}')
        self._url = url

    def maybe_wait_for_url(self):
        if not redis_client:
            return
        try:
            while not self._url:
                self._url = get_redis(self._module_id)
                if self._url:
                    break
                time.sleep(lazyllm.config["redis_recheck_delay"])
        except Exception as e:
            LOG.error(f"Error accessing Redis: {e}")
            raise

    def __call__(self, *args, **kw):
        self.maybe_wait_for_url()
        return super().__call__(*args, **kw)

    # Cannot modify or add any attrubute of self
    # prompt keys (excluding history) are in __input (ATTENTION: dict, not kwargs)
    # deploy parameters keys are in **kw
    def forward(self, __input=None, *, llm_chat_history=None, tools=None, **kw):  # noqa C901
        assert self._url is not None, f'Please start {self.__class__} first'
        assert len(kw) == 0 or not isinstance(__input, LazyLlmRequest), \
            'Cannot provide LazyLlmRequest and kw args at the same time.'

        input, kw = (__input.input, __input.kwargs) if isinstance(__input, LazyLlmRequest) else (__input, kw)
        input = self._prompt.generate_prompt(input, llm_chat_history, tools)

        if self._meta == ServerModule:
            if isinstance(__input, LazyLlmRequest): __input.input = input
            else: __input = LazyLlmRequest(input=input, kwargs=kw)
            data = codecs.encode(pickle.dumps(__input), 'base64').decode('utf-8')
        elif self.template_message is not None:
            data = self._modify_parameters(copy.deepcopy(self.template_message), kw)
            data[self.input_key_name] = input
        else:
            if len(kw) != 0: raise NotImplementedError(f'kwargs ({kw}) are not allowed in UrlModule')
            data = input

        def _callback(text):
            if isinstance(text, LazyLlmResponse):
                text.messages = self._prompt.get_response(self._extract_result_func(text.messages))
            else:
                text = self._prompt.get_response(self._extract_result_func(text))
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
                        yield (_callback(chunk))
                else:
                    raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))
        if self._stream:
            return _impl()
        else:
            for r in _impl(): pass
            return r

    def prompt(self, prompt=None):
        if prompt is None:
            self._prompt = EmptyPrompter()
        elif isinstance(prompt, PrompterBase):
            self._prompt = prompt
        elif isinstance(prompt, str):
            self._prompt = ChatPrompter(prompt)
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
                if key in kw: paras[key] = kw.pop(key)
        return paras

    def set_default_parameters(self, **kw):
        self._modify_parameters(self.template_message, kw)

    def clone(self):
        assert (hasattr(self, '_url') and self._url is not None) or redis_client
        m = UrlModule(url=self._url, stream=self._stream, meta=self._meta, return_trace=self._return_trace)
        m.prompt(prompt=self._prompt)
        m._module_id = self._module_id
        m.name = self.name
        m._set_template(
            self.template_message,
            self.input_key_name,
            self.template_headers,
        )
        return m

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Url', name=self._module_name, url=self._url,
                                 stream=self._stream, return_trace=self._return_trace)

    # change to urlmodule when pickling to server process
    def __reduce__(self):
        if self.__class__ != UrlModule and os.getenv('LAZYLLM_ON_CLOUDPICKLE', False) == 'ON':
            m = self.clone()
            return m.__reduce__()
        else:
            return super(__class__, self).__reduce__()


class ActionModule(ModuleBase):
    __enable_request__ = True

    def __init__(self, *action, return_trace=False):
        super().__init__(return_trace=return_trace)
        if len(action) == 1 and isinstance(action, FlowBase): action = action[0]
        if isinstance(action, (tuple, list)):
            action = Pipeline(*action)
        assert isinstance(action, FlowBase), f'Invalid action type {type(action)}'
        action.for_each(lambda x: isinstance(x, ModuleBase), lambda x: self.submodules.append(x))
        self.action = action

    def forward(self, *args, **kw):
        return self.action(*args, **kw)

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Action', subs=[repr(self.action)],
                                 name=self._module_name, return_trace=self._return_trace)


lazyllm.ReprRule.add_rule('Module', 'Action', 'Flow')


class ServerModule(UrlModule):
    def __init__(self, m, pre=None, post=None, stream=False, return_trace=False, launcher=None):
        assert stream is False or return_trace is False, 'Module with stream output has no trace'
        super().__init__(url=None, stream=stream, meta=ServerModule, return_trace=return_trace)
        self.m = ActionModule(m) if isinstance(m, FlowBase) else m
        self._pre_func, self._post_func = pre, post
        assert (post is None) or (stream is False)
        self._set_template(
            copy.deepcopy(lazyllm.deploy.RelayServer.message_format),
            lazyllm.deploy.RelayServer.input_key_name,
            copy.deepcopy(lazyllm.deploy.RelayServer.default_headers),
        )
        self._deploy_flag = lazyllm.once_flag()
        self._launcher = launcher if launcher else launchers.remote(sync=False)

    def _get_deploy_tasks(self):
        return Pipeline(
            lazyllm.deploy.RelayServer(func=self.m, pre_func=self._pre_func,
                                       post_func=self._post_func, launcher=self._launcher),
            self.url)

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Server', subs=[repr(self.m)], name=self._module_name,
                                 stream=self._stream, return_trace=self._return_trace)

class TrainableModule(UrlModule):
    builder_keys = ['trainset', 'train_method', 'finetune_method', 'deploy_method', 'mode']
    __enable_request__ = False

    def __init__(self, base_model: Option = '', target_path='', *, source=lazyllm.config['model_source'],
                 stream=False, return_trace=False):
        super().__init__(url=None, stream=stream, meta=TrainableModule, return_trace=return_trace)
        # Fake base_model and target_path for dummy
        self.target_path = target_path
        self._train = None  # lazyllm.train.auto
        self._finetune = lazyllm.finetune.auto
        self._deploy = lazyllm.deploy.auto

        self.base_model = base_model
        self._deploy_flag = lazyllm.once_flag()

    # modify default value to ''
    def prompt(self, prompt=''):
        return super(__class__, self).prompt(prompt)

    def _get_args(self, arg_cls, disable=[]):
        args = getattr(self, f'_{arg_cls}_args', dict())
        if len(set(args.keys()).intersection(set(disable))) > 0:
            raise ValueError(f'Key `{", ".join(disable)}` can not be set in '
                             '{arg_cls}_args, please pass them from Module.__init__()')
        return args

    def _get_train_tasks(self):
        trainset_getf = lambda: lazyllm.package(self._trainset, None) \
            if isinstance(self._trainset, str) else self._trainset  # noqa E731
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
        if self._deploy is None: return None
        target_path = self.target_path
        if os.path.basename(self.target_path) != 'merge':
            target_path = os.path.join(self.target_path, 'merge')

        if not os.path.exists(target_path):
            target_path = self.target_path

        def build_deployer(base_model):
            if self._deploy is lazyllm.deploy.AutoDeploy:
                deployer = self._deploy(base_model=base_model, stream=self._stream, **self._deploy_args)
            else:
                deployer = self._deploy(stream=self._stream, **self._deploy_args)
            # For AutoDeploy: class attributes can only be obtained after instantiation
            self._deploy = deployer.__class__
            self._set_template(copy.deepcopy(deployer.message_format), deployer.input_key_name,
                               copy.deepcopy(deployer.default_headers))
            return deployer
        return Pipeline(lambda *a: lazyllm.package(target_path, self.base_model),
                        build_deployer(self.base_model), self.url)

    def _deploy_setter_hook(self):
        self._deploy_args = self._get_args('deploy', disable=['target_path'])
        if hasattr(self._deploy, 'extract_result'): self._extract_result_func = self._deploy.extract_result

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Trainable', mode=self._mode, basemodel=self.base_model,
                                 target=self.target_path, name=self._module_name,
                                 stream=self._stream, return_trace=self._return_trace)


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
            return UrlModule(url=args[0])
        elif len(args) == 0 and 'url' in kw:
            return UrlModule(url=kw['url'])
        elif ...:
            return TrainableModule()

    @classmethod
    def action(cls, *args, **kw): return ActionModule(*args, **kw)
    @classmethod
    def url(cls, *args, **kw): return UrlModule(*args, **kw)
    @classmethod
    def trainable(cls, *args, **kw): return TrainableModule(*args, **kw)


class ModuleRegistryBase(ModuleBase, metaclass=lazyllm.LazyLLMRegisterMetaClass):
    __reg_overwrite__ = 'forward'


register = lazyllm.Register(ModuleRegistryBase, ['forward'])
