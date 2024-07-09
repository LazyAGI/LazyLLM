import os
import copy
import time
import requests
import pickle
import codecs
import inspect
import functools
from concurrent.futures import ThreadPoolExecutor

import lazyllm
from lazyllm import FlatList, LazyLlmResponse, LazyLlmRequest, Option, launchers, LOG
from ..components.prompter import PrompterBase, ChatPrompter, EmptyPrompter
from ..components.utils import ModelManager
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
        self._set_mid()
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
        def _setattr(v, *, _return_value=self, **kw):
            k = key[:-7] if key.endswith('_method') else key
            if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], dict):
                kw.update(v[1])
                v = v[0]
            if len(kw) > 0:
                setattr(self, f'_{k}_args', kw)
            setattr(self, f'_{k}', v)
            if hasattr(self, f'_{k}_setter_hook'): getattr(self, f'_{k}_setter_hook')()
            return _return_value
        keys = self.__class__.builder_keys
        if key in keys:
            return _setattr
        elif key.startswith('_') and key[1:] in keys:
            return None
        elif key.startswith('_') and key.endswith('_args') and (key[1:-5] in keys or f'{key[1:-4]}method' in keys):
            return dict()
        raise AttributeError(f'{self.__class__} object has no attribute {key}')

    def __call__(self, *args, **kw):
        if len(args) == 1 and isinstance(args[0], LazyLlmRequest):
            assert len(kw) == 0, 'Cannot use LazyLlmRequest and kwargs at the same time'
            args[0].kwargs.update(args[0].global_parameters.get(self._module_id, dict()))
            if not getattr(self.__class__, '__enable_request__', False):
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

    def _set_mid(self, mid=None):
        self._module_id = mid if mid else str(uuid.uuid4().hex)
        return self

    _url_id = property(lambda self: self._module_id)

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
        if not mode: mode = list(self.mode_list)
        if type(mode) is not list: mode = [mode]
        for item in mode:
            assert item in self.mode_list, f"Cannot find {item} in mode list: {self.mode_list}"
        # dfs to get all train tasks
        train_tasks, deploy_tasks, eval_tasks, post_process_tasks = FlatList(), FlatList(), FlatList(), FlatList()
        stack, visited = [(self, iter(self.submodules if recursive else []))], set()
        while len(stack) > 0:
            try:
                top = next(stack[-1][1])
                stack.append((top, iter(top.submodules)))
            except StopIteration:
                top = stack.pop()[0]
                if top._module_id in visited: continue
                visited.add(top._module_id)
                if 'train' in mode: train_tasks.absorb(top._get_train_tasks())
                if 'server' in mode: deploy_tasks.absorb(top._get_deploy_tasks())
                if 'eval' in mode: eval_tasks.absorb(top._get_eval_tasks())
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

    def __repr__(self):
        return lazyllm.make_repr('Module', self.__class__, name=self.name)


class UrlTemplate(object):
    def __init__(self, template_message=None, keys_name_handle=None, template_headers=None) -> None:
        self._set_template(template_message, keys_name_handle, template_headers)

    def _set_template(self, template_message=None, keys_name_handle=None, template_headers=None, stop_words=None):
        if isinstance(template_message, UrlTemplate):
            assert keys_name_handle is None and template_headers is None
            self._url_template = template_message._url_template.copy()
        else:
            if template_headers is None: template_headers = {'Content-Type': 'application/json'}
            self._url_template = dict(template_message=template_message, keys_name_handle=keys_name_handle,
                                      template_headers=template_headers)
        if self.keys_name_handle and 'stop' in self.keys_name_handle and stop_words and self.template_message:
            if self.keys_name_handle['stop'] in self.template_message:
                self.template_message[self.keys_name_handle['stop']] = stop_words
            else:
                # stop in sub dict:
                for _, v in self.template_message.items():
                    if isinstance(v, dict) and self.keys_name_handle['stop'] in v:
                        v[self.keys_name_handle['stop']] = stop_words
                        break
                else:
                    raise RuntimeError('No stop symbol found in template_message')

    template_message = property(lambda self: self._url_template['template_message'])
    keys_name_handle = property(lambda self: self._url_template['keys_name_handle'])
    template_headers = property(lambda self: self._url_template['template_headers'])


class UrlModule(ModuleBase, UrlTemplate):
    __enable_request__ = True

    def __init__(self, *, url='', stream=False, return_trace=False):
        super().__init__(return_trace=return_trace)
        self.__url = url
        self._stream = stream
        # Set for request by specific deploy:
        UrlTemplate.__init__(self)
        self._extract_result_func = lambda x: x
        __class__.prompt(self)

    @property
    def _url(self):
        if redis_client:
            try:
                while not self.__url:
                    self.__url = get_redis(self._url_id)
                    if self.__url: break
                    time.sleep(lazyllm.config["redis_recheck_delay"])
            except Exception as e:
                LOG.error(f"Error accessing Redis: {e}")
                raise
        return self.__url

    def _set_url(self, url):
        if redis_client:
            redis_client.set(self._module_id, url)
        LOG.debug(f'url: {url}')
        self.__url = url

    # Cannot modify or add any attrubute of self
    # prompt keys (excluding history) are in __input (ATTENTION: dict, not kwargs)
    # deploy parameters keys are in **kw
    def forward(self, __input=None, *, llm_chat_history=None, tools=None, **kw):  # noqa C901
        assert self._url is not None, f'Please start {self.__class__} first'
        assert len(kw) == 0 or not isinstance(__input, LazyLlmRequest), \
            'Cannot provide LazyLlmRequest and kw args at the same time.'

        input, kw = (__input.input, __input.kwargs) if isinstance(__input, LazyLlmRequest) else (__input, kw)
        input = self._prompt.generate_prompt(input, llm_chat_history, tools)

        if isinstance(self, ServerModule):
            if isinstance(__input, LazyLlmRequest): __input.input = input
            else: __input = LazyLlmRequest(input=input, kwargs=kw)
            data = codecs.encode(pickle.dumps(__input), 'base64').decode('utf-8')
        elif self.template_message:
            data = self._modify_parameters(copy.deepcopy(self.template_message), kw)
            assert 'inputs' in self.keys_name_handle
            data[self.keys_name_handle['inputs']] = input
        else:
            if len(kw) != 0: raise NotImplementedError(f'kwargs ({kw}) are not allowed in UrlModule')
            data = input

        def _callback(text):
            if isinstance(text, LazyLlmResponse):
                text = self._prompt.get_response(self._extract_result_func(text.messages))
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
        elif isinstance(prompt, (str, dict)):
            self._prompt = ChatPrompter(prompt)
        return self

    def _modify_parameters(self, paras, kw):
        for key, value in paras.items():
            if key == self.keys_name_handle['inputs']:
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

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Url', name=self._module_name, url=self._url,
                                 stream=self._stream, return_trace=self._return_trace)


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

def light_reduce(cls):
    def rebuild(mid): return cls()._set_mid(mid)

    def _impl(self):
        assert self._get_deploy_tasks.flag, f'{cls.__name__[1:-4]} shoule be deployed before pickling to another process'
        if os.getenv('LAZYLLM_ON_CLOUDPICKLE', False) == 'ON': return rebuild, (self._module_id,)
        return super(cls, self).__reduce__()
    setattr(cls, '__reduce__', _impl)
    return cls

@light_reduce
class _ServerModuleImpl(ModuleBase):
    def __init__(self, m=None, pre=None, post=None, launcher=None, *, father=None):
        super().__init__()
        self._m = ActionModule(m) if isinstance(m, FlowBase) else m
        self._pre_func, self._post_func = pre, post
        self._launcher = launcher if launcher else launchers.remote(sync=False)
        self._set_url_f = father._set_url if father else None

    @lazyllm.once_wrapper
    def _get_deploy_tasks(self):
        if self._m is None: return None
        return Pipeline(
            lazyllm.deploy.RelayServer(func=self._m, pre_func=self._pre_func,
                                       post_func=self._post_func, launcher=self._launcher),
            self._set_url_f)


class ServerModule(UrlModule):
    def __init__(self, m, pre=None, post=None, stream=False, return_trace=False, launcher=None):
        assert stream is False or return_trace is False, 'Module with stream output has no trace'
        assert (post is None) or (stream is False), 'Stream cannot be true when post-action exists'
        super().__init__(url=None, stream=stream, return_trace=return_trace)
        self._set_template(
            copy.deepcopy(lazyllm.deploy.RelayServer.message_format),
            lazyllm.deploy.RelayServer.keys_name_handle,
            copy.deepcopy(lazyllm.deploy.RelayServer.default_headers),
        )
        self._impl = _ServerModuleImpl(m, pre, post, launcher, father=self)

    _url_id = property(lambda self: self._impl._module_id)

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Server', subs=[repr(self._impl._m)], name=self._module_name,
                                 stream=self._stream, return_trace=self._return_trace)

@light_reduce
class _TrainableModuleImpl(ModuleBase):
    builder_keys = ['trainset', 'train_method', 'finetune_method', 'deploy_method', 'mode']

    def __init__(self, base_model='', target_path='', stream=False, train=None, finetune=None, deploy=None):
        super().__init__()
        # Fake base_model and target_path for dummy
        self._base_model = ModelManager(lazyllm.config['model_source']).download(base_model)
        self._target_path = target_path
        self._train, self._finetune, self._deploy = train, finetune, deploy
        self._stream = stream
        self._father = []

    def _add_father(self, father):
        if father not in self._father: self._father.append(father)

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
            train = self._train(base_model=self._base_model, target_path=self._target_path, **args)
        elif self._mode == 'finetune':
            args = self._get_args('finetune', disable=['base_model', 'target_path'])
            train = self._finetune(base_model=self._base_model, target_path=self._target_path, **args)
        else:
            raise RuntimeError('mode must be train or finetune')
        return Pipeline(trainset_getf, train)

    @lazyllm.once_wrapper
    def _get_deploy_tasks(self):
        if self._deploy is None: return None

        target_path = self._target_path
        if os.path.basename(self._target_path) != 'merge':
            merge_path = os.path.join(self._target_path, 'merge')
            if os.path.exists(merge_path): target_path = merge_path

        if self._deploy is lazyllm.deploy.AutoDeploy:
            deployer = self._deploy(base_model=self._base_model, stream=self._stream, **self._deploy_args)
        else:
            deployer = self._deploy(stream=self._stream, **self._deploy_args)
        template = UrlTemplate(copy.deepcopy(deployer.message_format), deployer.keys_name_handle,
                               copy.deepcopy(deployer.default_headers))
        stop_words = ModelManager.get_model_prompt_keys(self._base_model).get('stop_words')

        for f in self._father:
            f._set_template(template, stop_words=stop_words)
            if hasattr(deployer.__class__, 'extract_result'):
                f._extract_result_func = deployer.__class__.extract_result

        return Pipeline(lambda *a: lazyllm.package(target_path, self._base_model), deployer,
                        lambda url: [f._set_url(url) for f in self._father])

    def _deploy_setter_hook(self):
        self._deploy_args = self._get_args('deploy', disable=['target_path'])


class TrainableModule(UrlModule):
    builder_keys = _TrainableModuleImpl.builder_keys
    __enable_request__ = False

    def __init__(self, base_model: Option = '', target_path='', *, stream=False, return_trace=False):
        super().__init__(url=None, stream=stream, return_trace=return_trace)
        self._impl = _TrainableModuleImpl(base_model, target_path, stream,
                                          None, lazyllm.finetune.auto, lazyllm.deploy.auto)
        self._impl._add_father(self)

    base_model = property(lambda self: self._impl._base_model)
    target_path = property(lambda self: self._impl._target_path)
    _url_id = property(lambda self: self._impl._module_id)

    # modify default value to ''
    def prompt(self, prompt=''):
        if self.base_model != '' and prompt == '' and ModelManager.get_model_type(self.base_model) != 'llm':
            prompt = None
        prompt = super(__class__, self).prompt(prompt)._prompt
        keys = ModelManager.get_model_prompt_keys(self.base_model)
        if keys: prompt._set_model_configs(**keys)
        return self

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Trainable', mode=self._impl._mode, basemodel=self.base_model,
                                 target=self.target_path, name=self._module_name,
                                 stream=self._stream, return_trace=self._return_trace)

    def __getattr__(self, key):
        if key in self.__class__.builder_keys:
            return functools.partial(getattr(self._impl, key), _return_value=self)
        raise AttributeError(f'{__class__} object has no attribute {key}')

    def share(self, prompt=None):
        new = copy.copy(self)
        new._set_mid()
        if prompt is not None: new.prompt(prompt)
        new._impl._add_father(new)
        return new

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
