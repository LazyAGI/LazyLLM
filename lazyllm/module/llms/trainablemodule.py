import functools
import json5 as json
from datetime import datetime
from typing import Optional, Dict, List, Any, Union, Tuple
import os
import copy
import uuid
import re
import requests

import lazyllm
from lazyllm import globals, LOG, launchers, Option, package, LazyLLMDeployBase, LazyLLMFinetuneBase
from ...components.formatter import decode_query_with_filepaths, encode_query_with_filepaths
from ...components.formatter.formatterbase import LAZYLLM_QUERY_PREFIX
from ...components.utils import ModelManager
from ...components.utils.file_operate import _base64_to_file, _is_base64_with_mime
from ...launcher import LazyLLMLaunchersBase as Launcher
from .utils import map_kw_for_framework, encode_files
from ...flow import Pipeline
from ..servermodule import ModuleBase, _UrlHelper, UrlModule
from ..utils import light_reduce


class _UrlTemplateStruct(object):
    def __init__(self, template_message=None, keys_name_handle=None, template_headers=None, stop_words=None,
                 extract_result=None, stream_parse_parameters=None, stream_url_suffix=None):
        self.update(template_message, keys_name_handle, template_headers, stop_words,
                    extract_result, stream_parse_parameters, stream_url_suffix)

    def update(self, template_message=None, keys_name_handle=None, template_headers=None, stop_words=None,
               extract_result=None, stream_parse_parameters=None, stream_url_suffix=None):
        self.template_message, self.keys_name_handle = copy.deepcopy(template_message), keys_name_handle or {}
        self.template_headers = template_headers or copy.deepcopy(lazyllm.deploy.RelayServer.default_headers)

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

        self.extract_result_func = extract_result or (lambda x, inputs: x)
        self.stream_parse_parameters = stream_parse_parameters or {}
        self.stream_url_suffix = stream_url_suffix or ''


@light_reduce
class _TrainableModuleImpl(ModuleBase, _UrlHelper):
    builder_keys = ['trainset', 'train_method', 'finetune_method', 'deploy_method', 'mode']

    def __init__(self, base_model: str = '', target_path: str = '', stream: bool = False, train: Optional[type] = None,
                 finetune: Optional[LazyLLMFinetuneBase] = None, deploy: Optional[LazyLLMDeployBase] = None,
                 template: Optional[_UrlTemplateStruct] = None, url_wrapper: Optional[_UrlHelper._Wrapper] = None,
                 trust_remote_code: bool = True):
        super().__init__()
        # TODO(wangzhihong): Update ModelDownloader to support async download, and move it to deploy.
        #                    Then support Option for base_model
        base_model = base_model.rstrip('/\\')
        self._base_model = (ModelManager(lazyllm.config['model_source']).download(base_model) or ''
                            if trust_remote_code else base_model)
        if not self._base_model:
            LOG.warning(f'Cannot get a valid model from {base_model} by ModelManager.')
        self._target_path = os.path.join(lazyllm.config['train_target_root'], target_path)
        self._stream = stream
        self._launchers: Dict[str, Dict[str, Launcher]] = dict(default=dict(), manual=dict())
        self._delimiter = '-LazySplit-'
        self._deployer = None
        self._file_name = None
        self._specific_target_path = target_path or None
        self._train, self._finetune = train, finetune
        self._template = template
        _UrlHelper.__init__(self, url=url_wrapper)
        if base_model and deploy: self.deploy_method(deploy)
        self._prepare_deploy = lambda target_path, base_model: lazyllm.package(target_path, base_model)

    def _get_train_or_deploy_args(self, arg_cls: str, disable: List[str] = []):  # noqa B006
        args = getattr(self, f'_{arg_cls}_args', dict()).copy()
        if len(set(args.keys()).intersection(set(disable))) > 0:
            raise ValueError(f'Key `{", ".join(disable)}` can not be set in '
                             '{arg_cls}_args, please pass them from Module.__init__()')

        if not args.get('url'):
            if arg_cls == 'deploy' and self._deploy is lazyllm.deploy.AutoDeploy:
                self._deploy, args['launcher'], self._deploy_args = lazyllm.deploy.AutoDeploy.get_deployer(
                    base_model=self._base_model, **args)
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
            os.makedirs(target_path, exist_ok=True)

        kw = kw or self._get_train_or_deploy_args(mode, disable=['base_model', 'target_path'])
        task = getattr(self, f'_{mode}')(base_model=self._base_model, target_path=target_path, **kw)
        return [trainset_getf, task]

    def _get_train_tasks(self):
        def after_train(real_target_path):
            self._temp_finetuned_model_path = real_target_path
            self._finetuned_model_path = real_target_path
            return real_target_path
        return Pipeline(*self._get_train_tasks_impl(), after_train)

    def _async_finetune(self, name: str, ngpus: int = 1, **kw):
        assert name and isinstance(name, str), 'Invalid name: {name}, expect a valid string'
        assert name not in self._launchers['manual'], 'Duplicate name: {name}'
        self._launchers['manual'][name] = kw['launcher'] = launchers.remote(sync=False, ngpus=ngpus)
        self._set_file_name(name)

        def after_train(real_target_path):
            self._temp_finetuned_model_path = real_target_path
            self._finetuned_model_path = real_target_path
            return real_target_path
        return Pipeline(*self._get_train_tasks_impl(mode='finetune', **kw), after_train)()

    def _get_all_finetuned_models(self):
        valid_paths = []
        invalid_paths = []
        for root, _, files in os.walk(self._target_path):
            if root.endswith('lazyllm_merge'):
                model_path = os.path.abspath(root)
                model_id = model_path.split(os.sep)[-2].split(self._delimiter)[0]
                if any(file.endswith(('.bin', '.safetensors')) for file in files):
                    valid_paths.append((model_id, model_path))
                else:
                    invalid_paths.append((model_id, model_path))
        return valid_paths, invalid_paths

    def _set_specific_finetuned_model(self, model_path):
        valid_paths, invalid_paths = self._get_all_finetuned_models()
        if model_path in valid_paths:
            self._specific_target_path = model_path
        elif model_path in invalid_paths:
            LOG.warning(f'Model Path: {model_path} in list, but the path is invalid. Base Model will be used to deploy.')
            self._specific_target_path = None
        else:
            LOG.warning(f'Model Path: {model_path} not in list: {valid_paths}. Base Model will be used to deploy.')
            self._specific_target_path = None

    @lazyllm.once_wrapper
    def _get_deploy_tasks(self):
        if self._deploy is None: return None
        if self._deploy is lazyllm.deploy.AutoDeploy:
            raise RuntimeError('No appropriate inference framework was selected, specify it with `.deploy_method()`.')
        kwargs = {'stream': self._stream} if self._deploy is lazyllm.deploy.dummy else {}
        self._deployer = self._deploy(**kwargs, **self._deploy_args)

        def before_deploy(*no_use_args):
            if hasattr(self, '_temp_finetuned_model_path') and self._temp_finetuned_model_path:
                target_path = self._temp_finetuned_model_path
                self._temp_finetuned_model_path = None
            elif self._specific_target_path:
                target_path = self._specific_target_path
            else:
                target_path = ''
            return lazyllm.package(target_path, self._base_model)
        if hasattr(self._deployer, '_prepare_deploy'):
            self._prepare_deploy = self._deployer._prepare_deploy

        return Pipeline(before_deploy, self._prepare_deploy, self._deployer, self._set_url)

    def _deploy_setter_hook(self):
        self._deploy_args = self._get_train_or_deploy_args('deploy', disable=['target_path'])

        if hasattr(self._deploy, 'auto_map') and self._deploy.auto_map:
            self._deploy_args = map_kw_for_framework(self._deploy_args, self._deploy.auto_map)

        stop_words = ModelManager.get_model_prompt_keys(self._base_model).get('stop_words')

        self._template.update(self._deploy.message_format, self._deploy.keys_name_handle,
                              self._deploy.default_headers, extract_result=self._deploy.extract_result,
                              stream_parse_parameters=self._deploy.stream_parse_parameters,
                              stream_url_suffix=self._deploy.stream_url_suffix, stop_words=stop_words)

        if url := self._deploy_args.get('url'):
            assert len(self._deploy_args) == 1, 'Cannot provide other arguments together with url'
            self._set_url(url)
            self._get_deploy_tasks.flag.set()
        self._deploy_args.pop('url', None)

    def __del__(self):
        if hasattr(self, '_launchers'):
            [[launcher.cleanup() for launcher in group.values()] for group in self._launchers.values()]

    def _generate_target_path(self):
        base_model_name = os.path.basename(self._base_model)
        train_set_name = os.path.basename(self._trainset) if isinstance(self._trainset, str) else ''

        def optimize_name(name):
            if len(name) > 10:
                return name[:5] + '_' + name[-4:]
            return name
        base_model_name = optimize_name(base_model_name)
        file_name = base_model_name if not self._file_name else self._file_name
        train_set_name = optimize_name(train_set_name)

        target_path = os.path.join(self._target_path, base_model_name,
                                   f"{file_name}{self._delimiter}{train_set_name}{self._delimiter}"
                                   f"{datetime.now().strftime('%y%m%d%H%M%S%f')[:14]}")
        return target_path

    def _set_file_name(self, name):
        self._file_name = name


class TrainableModule(UrlModule):
    builder_keys = _TrainableModuleImpl.builder_keys

    def __init__(self, base_model: Option = '', target_path='', *, stream: Union[bool, Dict[str, str]] = False,
                 return_trace: bool = False, trust_remote_code: bool = True):
        super().__init__(url=None, stream=stream, return_trace=return_trace, init_prompt=False)
        self._template = _UrlTemplateStruct()
        self._impl = _TrainableModuleImpl(base_model, target_path, stream, None, lazyllm.finetune.auto,
                                          lazyllm.deploy.auto, self._template, self._url_wrapper, trust_remote_code)
        self._stream = stream
        self.prompt()

    template_message = property(lambda self: self._template.template_message)
    keys_name_handle = property(lambda self: self._template.keys_name_handle)
    template_headers = property(lambda self: self._template.template_headers)
    extract_result_func = property(lambda self: self._template.extract_result_func)
    stream_parse_parameters = property(lambda self: self._template.stream_parse_parameters)
    stream_url_suffix = property(lambda self: self._template.stream_url_suffix)

    base_model = property(lambda self: self._impl._base_model)
    target_path = property(lambda self: self._impl._target_path)
    finetuned_model_path = property(lambda self: self._impl._finetuned_model_path)
    _url_id = property(lambda self: self._impl._module_id)

    @property
    def series(self):
        return re.sub(r'\d+$', '', ModelManager._get_model_name(self.base_model).split('-')[0].upper())

    @property
    def type(self):
        return ModelManager.get_model_type(self.base_model).upper()

    def get_all_models(self):
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
        if launcher := self._impl._launchers['default'].get('deploy'):
            launcher.wait()

    def stop(self, task_name: Optional[str] = None):
        try:
            launcher = self._impl._launchers['manual' if task_name else 'default'][task_name or 'deploy']
        except KeyError:
            raise RuntimeError('Cannot stop an unstarted task')
        if not task_name: self._impl._get_deploy_tasks.flag.reset()
        launcher.cleanup()

    def status(self, task_name: Optional[str] = None):
        launcher = self._impl._launchers['manual' if task_name else 'default'][task_name or 'deploy']
        return launcher.status

    # modify default value to ''
    def prompt(self, prompt: Union[str, dict] = '', history: Optional[List[List[str]]] = None):
        if self.base_model != '' and prompt == '' and ModelManager.get_model_type(self.base_model) != 'llm':
            prompt = None
        clear_system = isinstance(prompt, dict) and prompt.get('drop_builtin_system')
        prompter = super(__class__, self).prompt(prompt, history)._prompt
        self._tools = getattr(prompter, "_tools", None)
        keys = ModelManager.get_model_prompt_keys(self.base_model).copy()
        if keys:
            if clear_system: keys['system'] = ''
            prompter._set_model_configs(**keys)
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

    def _decode_base64_to_file(self, content: str) -> str:
        decontent = decode_query_with_filepaths(content)
        files = [_base64_to_file(file_content) if _is_base64_with_mime(file_content) else file_content
                 for file_content in decontent["files"]]
        return encode_query_with_filepaths(query=decontent["query"], files=files)

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
        if isinstance(content, str) and content.startswith(LAZYLLM_QUERY_PREFIX):
            content = self._decode_base64_to_file(content)
        return self._build_response(content, tool_calls)

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Trainable', mode=self._impl._mode, basemodel=self.base_model,
                                 target=self.target_path, name=self._module_name, deploy_type=self._deploy_type,
                                 stream=bool(self._stream), return_trace=self._return_trace)

    def __getattr__(self, key):
        if key in self.__class__.builder_keys:
            return functools.partial(getattr(self._impl, key), _return_value=self)
        raise AttributeError(f'{__class__} object has no attribute {key}')

    def _record_usage(self, text_input_for_token_usage: str, temp_output: str):
        usage = {"prompt_tokens": self._estimate_token_usage(text_input_for_token_usage)}
        usage["completion_tokens"] = self._estimate_token_usage(temp_output)
        self._record_usage_impl(usage)

    def _record_usage_impl(self, usage: dict):
        globals["usage"][self._module_id] = usage
        par_muduleid = self._used_by_moduleid
        if par_muduleid is None:
            return
        if par_muduleid not in globals["usage"]:
            globals["usage"][par_muduleid] = usage
            return
        existing_usage = globals["usage"][par_muduleid]
        if existing_usage["prompt_tokens"] == -1 or usage["prompt_tokens"] == -1:
            globals["usage"][par_muduleid] = {"prompt_tokens": -1, "completion_tokens": -1}
        else:
            for k in globals["usage"][par_muduleid]:
                globals["usage"][par_muduleid][k] += usage[k]

    def forward(self, __input: Union[Tuple[Union[str, Dict], str], str, Dict] = package(),  # noqa B008
                *, llm_chat_history=None, lazyllm_files=None, tools=None, stream_output=False, **kw):
        __input, files = self._get_files(__input, lazyllm_files)
        text_input_for_token_usage = __input = self._prompt.generate_prompt(__input, llm_chat_history, tools)
        url = self._url

        if self.template_message:
            data = self._modify_parameters(copy.deepcopy(self.template_message), kw, optional_keys='modality')
            data[self.keys_name_handle.get('inputs', 'inputs')] = __input
            if files and (keys := list(set(self.keys_name_handle).intersection(LazyLLMDeployBase.encoder_map.keys()))):
                assert len(keys) == 1, 'Only one key is supported for encoder_mapping'
                data[self.keys_name_handle[keys[0]]] = encode_files(files, LazyLLMDeployBase.encoder_map[keys[0]])

            if stream_output:
                if self.stream_url_suffix and not url.endswith(self.stream_url_suffix):
                    url += self.stream_url_suffix
                if "stream" in data: data['stream'] = stream_output
        else:
            data = __input
            if stream_output: LOG.warning('stream_output is not supported when template_message is not set, ignore it')
            assert not kw, 'kw is not supported when template_message is not set'

        with self.stream_output((stream_output := (stream_output or self._stream))):
            return self._forward_impl(data, stream_output=stream_output, url=url, text_input=text_input_for_token_usage)

    def _maybe_has_fc(self, token: str, chunk: str) -> bool:
        return token and (token.startswith(chunk if token.startswith('\n') else chunk.lstrip('\n')) or token in chunk)

    def _forward_impl(self, data: Union[Tuple[Union[str, Dict], str], str, Dict] = package(), *,  # noqa B008
                      url: str, stream_output: Optional[Union[bool, Dict]] = None, text_input: Optional[str] = None):
        headers = self.template_headers or {'Content-Type': 'application/json'}
        parse_parameters = self.stream_parse_parameters if stream_output else {"delimiter": b"<|lazyllm_delimiter|>"}

        # context bug with httpx, so we use requests
        with requests.post(url, json=data, stream=True, headers=headers, proxies={'http': None, 'https': None}) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            messages, cache = '', ''
            token = getattr(self, "_tool_start_token", '')
            color = stream_output.get('color') if isinstance(stream_output, dict) else None

            for line in r.iter_lines(**parse_parameters):
                if not line: continue
                line = self._decode_line(line)

                chunk = self._prompt.get_response(self.extract_result_func(line, data))
                chunk = chunk[len(messages):] if isinstance(chunk, str) and chunk.startswith(messages) else chunk
                messages = chunk if not isinstance(chunk, str) else messages + chunk

                if not stream_output: continue
                if not cache: cache = chunk if self._maybe_has_fc(token, chunk) else self._stream_output(chunk, color)
                elif token in cache:
                    stream_output = False
                    if not cache.startswith(token): self._stream_output(cache.split(token)[0], color)
                else:
                    cache += chunk
                    if not self._maybe_has_fc(token, cache): cache = self._stream_output(cache, color)

            temp_output = self._extract_and_format(messages)
            if text_input: self._record_usage(text_input, temp_output)
            return self._formatter(temp_output)

    def _modify_parameters(self, paras: dict, kw: dict, *, optional_keys: Union[List[str], str] = None):
        for key, value in paras.items():
            if key == self.keys_name_handle['inputs']: continue
            elif isinstance(value, dict):
                if key in kw:
                    assert set(kw[key].keys()).issubset(set(value.keys()))
                    value.update(kw.pop(key))
                else: [setattr(value, k, kw.pop(k)) for k in value.keys() if k in kw]
            elif key in kw: paras[key] = kw.pop(key)

        optional_keys = [optional_keys] if isinstance(optional_keys, str) else (optional_keys or [])
        assert set(kw.keys()).issubset(set(optional_keys)), f'{kw.keys()} is not in {optional_keys}'
        paras.update(kw)
        return paras

    def set_default_parameters(self, *, optional_keys: Optional[List[str]] = None, **kw):
        self._modify_parameters(self.template_message, kw, optional_keys=optional_keys or [])
