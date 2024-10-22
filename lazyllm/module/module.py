import os
import re
import copy
import time
import json5 as json
import requests
import pickle
import codecs
import inspect
import functools
from datetime import datetime
from lazyllm import ThreadPoolExecutor, FileSystemQueue
from typing import Dict, List, Any, Union, Optional

import lazyllm
from lazyllm import FlatList, Option, launchers, LOG, package, kwargs, encode_request, globals
from ..components.prompter import PrompterBase, ChatPrompter, EmptyPrompter
from ..components.formatter import FormatterBase, EmptyFormatter
from ..components.utils import ModelManager
from ..flow import FlowBase, Pipeline, Parallel
from ..launcher import LazyLLMLaunchersBase as Launcher
import uuid
from ..client import get_redis, redis_client


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
                    f'{k} cannot accept Option'
        return object.__new__(cls)

    def __init__(self, *, return_trace=False):
        self._submodules = []
        self._evalset = None
        self._return_trace = return_trace
        self.mode_list = ('train', 'server', 'eval')
        self._set_mid()
        self._module_name = None
        self._options = []
        self.eval_result = None

    def __setattr__(self, name: str, value):
        if isinstance(value, ModuleBase):
            self._submodules.append(value)
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
        try:
            kw.update(globals['global_parameters'].get(self._module_id, dict()))
            if (history := globals['chat_history'].get(self._module_id)) is not None: kw['llm_chat_history'] = history
            r = self.forward(**args[0], **kw) if args and isinstance(args[0], kwargs) else self.forward(*args, **kw)
            if self._return_trace:
                lazyllm.FileSystemQueue.get_instance('lazy_trace').enqueue(str(r))
        except Exception as e:
            raise RuntimeError(f'\nAn error occured in {self.__class__} with name {self.name}.\n'
                               f'Args:\n{args}\nKwargs\n{kw}\nError messages:\n{e}\n')
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

    @property
    def submodules(self): return self._submodules

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
    def wait(self): pass

    def stop(self):
        for m in self.submodules:
            m.stop()

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
    def __init__(self, *, url='', stream=False, return_trace=False):
        super().__init__(return_trace=return_trace)
        self.__url = url
        self._stream = stream
        # Set for request by specific deploy:
        UrlTemplate.__init__(self)
        self._extract_result_func = lambda x, inputs: x
        self._stream_parse_parameters = {}
        self._stream_url_suffix = ''
        __class__.prompt(self)
        __class__.formatter(self)

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
    def forward(self, __input=package(), *, llm_chat_history=None, tools=None, stream_output=False, **kw):  # noqa C901
        assert self._url is not None, f'Please start {self.__class__} first'
        stream_output = stream_output or self._stream
        url = self._url

        files = []
        if self.template_message and globals['global_parameters'].get("lazyllm-files"):
            files = globals['global_parameters']["lazyllm-files"]['files']
        query = __input
        __input = self._prompt.generate_prompt(query, llm_chat_history, tools)
        headers = {'Content-Type': 'application/json'}

        if isinstance(self, ServerModule):
            assert llm_chat_history is None and tools is None
            headers['Global-Parameters'] = encode_request(globals._pickle_data)
            headers['Session-ID'] = encode_request(globals._sid)
            data = encode_request((__input, kw))
        elif self.template_message:
            data = self._modify_parameters(copy.deepcopy(self.template_message), kw)
            assert 'inputs' in self.keys_name_handle
            data[self.keys_name_handle['inputs']] = __input
            if 'image' in self.keys_name_handle and files:
                data[self.keys_name_handle['image']] = files
            elif 'audio' in self.keys_name_handle and files:
                data[self.keys_name_handle['audio']] = files
        else:
            if len(kw) != 0: raise NotImplementedError(f'kwargs ({kw}) are not allowed in UrlModule')
            data = __input

        if stream_output:
            if self._stream_url_suffix and not url.endswith(self._stream_url_suffix):
                url += self._stream_url_suffix
            if "stream" in data: data['stream'] = stream_output
        parse_parameters = self._stream_parse_parameters if stream_output else {"delimiter": b"<|lazyllm_delimiter|>"}

        token = getattr(self, "_tool_start_token", '')
        cache = ""

        # context bug with httpx, so we use requests
        with requests.post(url, json=data, stream=True, headers=headers) as r:
            if r.status_code == 200:
                messages = ''
                for line in r.iter_lines(**parse_parameters):
                    if not line: continue
                    try:
                        line = pickle.loads(codecs.decode(line, "base64"))
                    except Exception:
                        line = line.decode('utf-8')
                    chunk = self._prompt.get_response(self._extract_result_func(line, data))
                    if isinstance(chunk, str):
                        if chunk.startswith(messages): chunk = chunk[len(messages):]
                        messages += chunk
                    else:
                        messages = chunk

                    if not stream_output: continue
                    if not cache:
                        if token.startswith(chunk.lstrip('\n') if not token.startswith('\n') else chunk) \
                           or token in chunk: cache = chunk
                        else: FileSystemQueue().enqueue(chunk)
                    elif token in cache:
                        stream_output = False
                        if not cache.startswith(token): FileSystemQueue().enqueue(cache.split(token)[0])
                    else:
                        cache += chunk
                        if not (token.startswith(cache.lstrip('\n') if not token.startswith('\n') else cache)
                                or token in cache):
                            FileSystemQueue().enqueue(cache)
                            cache = ""
            else:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))
            return self._formatter.format(self._extract_and_format(messages))

    def prompt(self, prompt=None):
        if prompt is None:
            self._prompt = EmptyPrompter()
        elif isinstance(prompt, PrompterBase):
            self._prompt = prompt
        elif isinstance(prompt, (str, dict)):
            self._prompt = ChatPrompter(prompt)
        return self

    def _extract_and_format(self, output: str) -> str:
        return output

    def formatter(self, format: FormatterBase = None):
        if isinstance(format, FormatterBase):
            self._formatter = format
        elif format is None:
            self._formatter = EmptyFormatter()
        else:
            raise TypeError("format must be a FormatterBase")
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
    def __init__(self, *action, return_trace=False):
        super().__init__(return_trace=return_trace)
        if len(action) == 1 and isinstance(action, FlowBase): action = action[0]
        if isinstance(action, (tuple, list)):
            action = Pipeline(*action)
        assert isinstance(action, FlowBase), f'Invalid action type {type(action)}'
        self.action = action

    def forward(self, *args, **kw):
        return self.action(*args, **kw)

    @property
    def submodules(self):
        if isinstance(self.action, FlowBase):
            submodule = []
            self.action.for_each(lambda x: isinstance(x, ModuleBase), lambda x: submodule.append(x))
            return submodule
        return super().submodules

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Action', subs=[repr(self.action)],
                                 name=self._module_name, return_trace=self._return_trace)


lazyllm.ReprRule.add_rule('Module', 'Action', 'Flow')

def light_reduce(cls):
    def rebuild(mid): return cls()._set_mid(mid)

    def _impl(self):
        if os.getenv('LAZYLLM_ON_CLOUDPICKLE', False) == 'ON':
            assert self._get_deploy_tasks.flag, f'{cls.__name__[1:-4]} shoule be deployed before used'
            return rebuild, (self._module_id,)
        return super(cls, self).__reduce__()
    setattr(cls, '__reduce__', _impl)
    return cls

@light_reduce
class _ServerModuleImpl(ModuleBase):
    def __init__(self, m=None, pre=None, post=None, launcher=None, port=None, pythonpath=None, *, father=None):
        super().__init__()
        self._m = ActionModule(m) if isinstance(m, FlowBase) else m
        self._pre_func, self._post_func = pre, post
        self._launcher = launcher.clone() if launcher else launchers.remote(sync=False)
        self._set_url_f = father._set_url if father else None
        self._port = port
        self._pythonpath = pythonpath

    @lazyllm.once_wrapper
    def _get_deploy_tasks(self):
        if self._m is None: return None
        return Pipeline(
            lazyllm.deploy.RelayServer(func=self._m, pre_func=self._pre_func, port=self._port,
                                       pythonpath=self._pythonpath, post_func=self._post_func, launcher=self._launcher),
            self._set_url_f)

    def stop(self):
        self._launcher.cleanup()
        self._get_deploy_tasks.flag.reset()

    def __del__(self):
        self.stop()


class ServerModule(UrlModule):
    def __init__(self, m, pre=None, post=None, stream=False, return_trace=False,
                 port=None, pythonpath=None, launcher=None):
        assert stream is False or return_trace is False, 'Module with stream output has no trace'
        assert (post is None) or (stream is False), 'Stream cannot be true when post-action exists'
        super().__init__(url=None, stream=stream, return_trace=return_trace)
        self._set_template(
            copy.deepcopy(lazyllm.deploy.RelayServer.message_format),
            lazyllm.deploy.RelayServer.keys_name_handle,
            copy.deepcopy(lazyllm.deploy.RelayServer.default_headers),
        )
        self._impl = _ServerModuleImpl(m, pre, post, launcher, port, pythonpath, father=self)

    _url_id = property(lambda self: self._impl._module_id)

    def wait(self):
        self._impl._launcher.wait()

    def stop(self):
        self._impl.stop()

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Server', subs=[repr(self._impl._m)], name=self._module_name,
                                 stream=self._stream, return_trace=self._return_trace)

@light_reduce
class _TrainableModuleImpl(ModuleBase):
    builder_keys = ['trainset', 'train_method', 'finetune_method', 'deploy_method', 'mode']

    def __init__(self, base_model='', target_path='', stream=False, train=None, finetune=None, deploy=None):
        super().__init__()
        # TODO(wangzhihong): Update ModelDownloader to support async download, and move it to deploy.
        #                    Then support Option for base_model
        self._base_model = ModelManager(lazyllm.config['model_source']).download(base_model)
        self._target_path = target_path if target_path else os.path.join(os.getcwd(), 'save_ckpt')
        self._train, self._finetune, self._deploy = train, finetune, deploy
        self._stream = stream
        self._father = []
        self._launchers: Dict[str, Dict[str, Launcher]] = dict(default=dict(), manual=dict())
        self._deployer = None
        self._specific_target_path = None

    def _add_father(self, father):
        if father not in self._father: self._father.append(father)

    def _get_train_or_deploy_args(self, arg_cls: str, disable: List[str] = []):
        args = getattr(self, f'_{arg_cls}_args', dict()).copy()
        if len(set(args.keys()).intersection(set(disable))) > 0:
            raise ValueError(f'Key `{", ".join(disable)}` can not be set in '
                             '{arg_cls}_args, please pass them from Module.__init__()')
        if 'url' not in args:
            args['launcher'] = args['launcher'].clone() if args.get('launcher') else launchers.remote(sync=False)
            self._launchers['default'][arg_cls] = args['launcher']
        return args

    def _get_train_tasks_impl(self, mode: Optional[str] = None, **kw):
        mode = mode or self._mode
        assert mode in ('train', 'finetune'), 'mode must be train or finetune'

        trainset_getf = (lambda: lazyllm.package(self._trainset, None)) if isinstance(
            self._trainset, str) else self._trainset
        target_path = self._generate_target_path()
        if not os.path.exists(target_path):
            os.system(f'mkdir -p {target_path}')

        kw = kw or self._get_train_or_deploy_args(mode, disable=['base_model', 'target_path'])
        task = getattr(self, f'_{mode}')(base_model=self._base_model, target_path=target_path, **kw)
        return [trainset_getf, task]

    def _get_train_tasks(self):
        def after_train(real_target_path):
            self._finetuned_model_path = real_target_path
            return real_target_path
        return Pipeline(*self._get_train_tasks_impl(), after_train)

    def _trian(self, name: str, ngpus: int = 1, mode: str = None, batch_size: int = 16,
               micro_batch_size: int = 2, num_epochs: int = 3, learning_rate: float = 5e-4,
               lora_r: int = 8, lora_alpha: int = 32, lora_dropout: float = 0.05, **kw):
        assert name and isinstance(name, str), 'Invalid name: {name}, expect a valid string'
        assert name not in self._launchers['manual'], 'Duplicate name: {name}'
        self._launchers['manual'][name] = kw['launcher'] = launchers.remote(sync=False, ngpus=ngpus)

        Pipeline(*self._get_train_tasks_impl(
            mode=mode, batch_size=batch_size, micro_batch_size=micro_batch_size, num_epochs=num_epochs,
            learning_rate=learning_rate, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, **kw))()

    def _get_all_finetuned_models(self):
        valid_paths = []
        invalid_paths = []
        for root, dirs, files in os.walk(self._target_path):
            if root.endswith('lazyllm_merge'):
                if any(file.endswith(('.bin', '.safetensors')) for file in files):
                    valid_paths.append(os.path.abspath(root))
                else:
                    invalid_paths.append(os.path.abspath(root))
        return valid_paths, invalid_paths

    def _set_specific_finetuned_model(self, model_path):
        valid_paths, invalid_paths = self._get_all_finetuned_models()
        if model_path in valid_paths:
            self._specific_target_path = model_path
        elif model_path in invalid_paths:
            LOG.warning(f'Model Path: {model_path} in list, but the path is invalid. '
                        'Base Model will be used to deploy.')
            self._specific_target_path = None
        else:
            LOG.warning(f'Model Path: {model_path} not in list: {valid_paths}. '
                        'Base Model will be used to deploy.')
            self._specific_target_path = None

    @lazyllm.once_wrapper
    def _get_deploy_tasks(self):
        if self._deploy is None: return None

        if self._deploy is lazyllm.deploy.AutoDeploy:
            self._deployer = self._deploy(base_model=self._base_model, stream=self._stream, **self._deploy_args)
            self._set_template(self._deployer)
        else:
            self._deployer = self._deploy(stream=self._stream, **self._deploy_args)

        def before_deploy(*no_use_args):
            if hasattr(self, '_finetuned_model_path') and self._finetuned_model_path:
                target_path = self._finetuned_model_path
                self._finetuned_model_path = None
            elif self._specific_target_path:
                target_path = self._specific_target_path
            else:
                target_path = ''
            return lazyllm.package(target_path, self._base_model)

        return Pipeline(before_deploy, self._deployer,
                        lambda url: [f._set_url(url) for f in self._father])

    def _set_template(self, deployer):
        template = UrlTemplate(copy.deepcopy(deployer.message_format), deployer.keys_name_handle,
                               copy.deepcopy(deployer.default_headers))
        stop_words = ModelManager.get_model_prompt_keys(self._base_model).get('stop_words')

        for f in self._father:
            f._set_template(template, stop_words=stop_words)
            if hasattr(deployer, 'extract_result'):
                f._extract_result_func = deployer.extract_result

            if hasattr(deployer, 'stream_parse_parameters'):
                f._stream_parse_parameters = deployer.stream_parse_parameters()

            if hasattr(deployer, 'stream_url_suffix'):
                f._stream_url_suffix = deployer.stream_url_suffix()

    def _deploy_setter_hook(self):
        self._deploy_args = self._get_train_or_deploy_args('deploy', disable=['target_path'])
        if self._deploy is not lazyllm.deploy.AutoDeploy:
            self._set_template(self._deploy)
            if url := self._deploy_args.get('url', None):
                assert len(self._deploy_args) == 1, 'Cannot provide other arguments together with url'
                for f in self._father:
                    f._set_url(url)
                self._get_deploy_tasks.flag.set()

    def __del__(self):
        [[launcher.cleanup() for launcher in group.values()] for group in self._launchers.values()]

    def _generate_target_path(self):
        base_model_name = os.path.basename(self._base_model)
        train_set_name = os.path.basename(self._trainset) if isinstance(self._trainset, str) else ''

        def optimize_name(name):
            if len(name) > 10:
                return name[:5] + '_' + name[-4:]
            return name
        base_model_name = optimize_name(base_model_name)
        train_set_name = optimize_name(train_set_name)

        target_path = os.path.join(self._target_path,
                                   f"{base_model_name}-{train_set_name}-"
                                   f"{datetime.now().strftime('%y%m%d%H%M%S%f')[:14]}")
        return target_path


class TrainableModule(UrlModule):
    builder_keys = _TrainableModuleImpl.builder_keys

    def __init__(self, base_model: Option = '', target_path='', *, stream=False, return_trace=False):
        super().__init__(url=None, stream=stream, return_trace=return_trace)
        self._impl = _TrainableModuleImpl(base_model, target_path, stream,
                                          None, lazyllm.finetune.auto, lazyllm.deploy.auto)
        self._impl._add_father(self)
        self.prompt()

    base_model = property(lambda self: self._impl._base_model)
    target_path = property(lambda self: self._impl._target_path)
    _url_id = property(lambda self: self._impl._module_id)

    @property
    def series(self):
        return re.sub(r'\d+$', '', ModelManager.get_model_name(self.base_model).split('-')[0].upper())

    @property
    def type(self):
        return ModelManager.get_model_type(self.base_model).upper()

    def get_all_finetuned_models(self):
        return self._impl._get_all_finetuned_models()

    def set_specific_finetuned_model(self, model_path):
        return self._impl._set_specific_finetuned_model(model_path)

    @property
    def _deploy_type(self):
        if self._impl._deploy is not lazyllm.deploy.AutoDeploy:
            return self._impl._deploy
        elif self._impl._deployer:
            return type(self._impl._deployer)
        else:
            return lazyllm.deploy.AutoDeploy

    def wait(self):
        # TODO(wangzhihong): Split finetune launcher and deploy launcher; Only one deploy launcher is allowed.
        for launcher in self._impl._launchers:
            launcher.wait()

    def stop(self, task_name: Optional[str] = None):
        launcher = self._impl._launchers['manual' if task_name else 'default'][task_name or 'deploy']
        if not task_name: self._impl._get_deploy_tasks.flag.reset()
        launcher.cleanup()

    def status(self, task_name: Optional[str] = None):
        launcher = self._impl._launchers['manual' if task_name else 'default'][task_name or 'deploy']
        return launcher.status

    # modify default value to ''
    def prompt(self, prompt=''):
        if self.base_model != '' and prompt == '' and ModelManager.get_model_type(self.base_model) != 'llm':
            prompt = None
        prompt = super(__class__, self).prompt(prompt)._prompt
        self._tools = getattr(prompt, "_tools", None)
        keys = ModelManager.get_model_prompt_keys(self.base_model)
        if keys:
            prompt._set_model_configs(**keys)
            for key in ["tool_start_token", "tool_args_token", "tool_end_token"]:
                if key in keys: setattr(self, f"_{key}", keys[key])
        return self

    def _loads_str(self, text: str) -> Union[str, Dict]:
        try:
            ret = json.loads(text)
            return self._loads_str(ret) if isinstance(ret, str) else ret
        except Exception:
            LOG.error(f"{text} is not a valid json string.")
            return text

    def _parse_arguments_with_args_token(self, output: str) -> tuple[str, dict]:
        items = output.split(self._tool_args_token)
        func_name = items[0].strip()
        if len(items) == 1:
            return func_name.split(self._tool_end_token)[0].strip() if getattr(self, "_tool_end_token", None)\
                else func_name, {}
        args = (items[1].split(self._tool_end_token)[0].strip() if getattr(self, "_tool_end_token", None)
                else items[1].strip())
        return func_name, self._loads_str(args) if isinstance(args, str) else args

    def _parse_arguments_without_args_token(self, output: str) -> tuple[str, dict]:
        items = output.split(self._tool_end_token)[0] if getattr(self, "_tool_end_token", None) else output
        func_name = ""
        args = {}
        try:
            items = json.loads(items.strip())
            func_name = items.get('name', '')
            args = items.get("parameters", items.get("arguments", {}))
        except Exception:
            LOG.error(f"tool calls info {items} parse error")

        return func_name, self._loads_str(args) if isinstance(args, str) else args

    def _parse_arguments_with_tools(self, output: Dict[str, Any], tools: List[str]) -> bool:
        func_name = ''
        args = {}
        is_tc = False
        tc = {}
        if output.get('name', '') in tools:
            is_tc = True
            func_name = output.get('name', '')
            args = output.get("parameters", output.get("arguments", {}))
            tc = {'name': func_name, 'arguments': self._loads_str(args) if isinstance(args, str) else args}
            return is_tc, tc
        return is_tc, tc

    def _parse_tool_start_token(self, output: str) -> tuple[str, List[Dict]]:
        tool_calls = []
        segs = output.split(self._tool_start_token)
        content = segs[0]
        for seg in segs[1:]:
            func_name, arguments = self._parse_arguments_with_args_token(seg.strip())\
                if getattr(self, "_tool_args_token", None)\
                else self._parse_arguments_without_args_token(seg.strip())
            if func_name:
                tool_calls.append({"name": func_name, "arguments": arguments})

        return content, tool_calls

    def _parse_tools(self, output: str) -> tuple[str, List[Dict]]:
        tool_calls = []
        tools = {tool['function']['name'] for tool in self._tools}
        lines = output.strip().split("\n")
        content = []
        is_tool_call = False
        for idx, line in enumerate(lines):
            if line.startswith("{") and idx > 0:
                func_name = lines[idx - 1].strip()
                if func_name in tools:
                    is_tool_call = True
                    if func_name == content[-1].strip():
                        content.pop()
                    arguments = "\n".join(lines[idx:]).strip()
                    tool_calls.append({'name': func_name, "arguments": arguments})
                    continue
            if "{" in line and 'name' in line:
                try:
                    items = json.loads(line.strip())
                    items = [items] if isinstance(items, dict) else items
                    if isinstance(items, list):
                        for item in items:
                            is_tool_call, tc = self._parse_arguments_with_tools(item, tools)
                            if is_tool_call:
                                tool_calls.append(tc)
                except Exception:
                    LOG.error(f"tool calls info {line} parse error")
            if not is_tool_call:
                content.append(line)
        content = "\n".join(content) if len(content) > 0 else ''
        return content, tool_calls

    def _extract_tool_calls(self, output: str) -> tuple[str, List[Dict]]:
        tool_calls = []
        content = ''
        if getattr(self, "_tool_start_token", None) and self._tool_start_token in output:
            content, tool_calls = self._parse_tool_start_token(output)
        elif self._tools:
            content, tool_calls = self._parse_tools(output)
        else:
            content = output

        return content, tool_calls

    def _build_response(self, content: str, tool_calls: List[Dict[str, str]]) -> str:
        tc = [{'id': str(uuid.uuid4().hex), 'type': 'function', 'function': tool_call} for tool_call in tool_calls]
        if content and tc:
            return globals["tool_delimiter"].join([content, json.dumps(tc, ensure_ascii=False)])
        elif not content and tc:
            return globals["tool_delimiter"] + json.dumps(tc, ensure_ascii=False)
        else:
            return content

    def _extract_and_format(self, output: str) -> str:
        """
        1.extract tool calls information;
            a. If 'tool_start_token' exists, the boundary of tool_calls can be found according to 'tool_start_token',
               and then the function name and arguments of tool_calls can be extracted according to 'tool_args_token'
               and 'tool_end_token'.
            b. If 'tool_start_token' does not exist, the text is segmented using '\n' according to the incoming tools
               information, and then processed according to the rules.
        """
        content, tool_calls = self._extract_tool_calls(output)
        return self._build_response(content, tool_calls)

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Trainable', mode=self._impl._mode, basemodel=self.base_model,
                                 target=self.target_path, name=self._module_name, deploy_type=self._deploy_type,
                                 stream=self._stream, return_trace=self._return_trace)

    def __getattr__(self, key):
        if key in self.__class__.builder_keys:
            return functools.partial(getattr(self._impl, key), _return_value=self)
        raise AttributeError(f'{__class__} object has no attribute {key}')

    def share(self, prompt=None, format=None):
        new = copy.copy(self)
        new._set_mid()
        if prompt is not None: new.prompt(prompt)
        if format is not None: new.formatter(format)
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
