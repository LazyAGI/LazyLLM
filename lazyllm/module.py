import os
import re
import copy

import httpx
import requests
from lazyllm.thirdparty import gradio as gr
from pydantic import BaseModel as struct
from typing import Tuple
from types import GeneratorType
import multiprocessing

import lazyllm
from lazyllm import FlatList
from .flow import FlowBase, Pipeline, Parallel, DPES


class ModuleBase(object):
    def __init__(self):
        self.submodules = []
        self._evalset = None
        self.mode_list  = ('train', 'server', 'eval')

    def __setattr__(self, name: str, value):
        if isinstance(value, ModuleBase):
            self.submodules.append(value)
        return super().__setattr__(name, value)

    def __call__(self, *args, **kw): return self.forward(*args, **kw)

    # interfaces
    def forward(self, *args, **kw): raise NotImplementedError
    def _get_train_tasks(self): return None
    def _get_deploy_tasks(self): return None

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
            Parallel(*train_tasks).start().wait()
        if 'server' in mode and len(deploy_tasks) > 0:
            DPES(*deploy_tasks).start()
        if 'eval' in mode and len(eval_tasks) > 0:
            DPES(*eval_tasks).start()

    def update(self, *, recursive=True): return self._update(mode=['train', 'server', 'eval'], recursive=recursive)
    def update_server(self, *, recursive=True): return self._update(mode=['server'], recursive=recursive)
    def eval(self, *, recursive=True): return self._update(mode=['eval'], recursive=recursive)
    def start(self): return self._update(mode=['server'], recursive=True)
    def restart(self): return self.start()

    def _overwrote(self, f):
        return getattr(self.__class__, f) is not getattr(__class__, f)


class ModuleResponse(struct):
    messages: str = ''
    trace: str = ''
    err: Tuple[int, str] = (0, '')


class SequenceModule(ModuleBase):
    def __init__(self, *args):
        super().__init__()
        self.submodules = list(args)

    def forward(self, *args, **kw):
        ppl = Pipeline(*self.submodules)
        return ppl.start(*args, **kw)

    def __repr__(self):
        representation = '<SequenceModule> [\n'
        for m in self.submodules:
            representation += '\n'.join(['    ' + s for s in repr(m).split('\n')]) + '\n'
        return representation + ']'
    

class UrlModule(ModuleBase):
    def __init__(self, url, *, stream=False):
        super().__init__()
        self._url = url
        self._stream = stream
        self.prompt()
        # Set for request by specific deploy:
        self._set_template(template_headers={'Content-Type': 'application/json'})

    def url(self, url):
        print('url:', url)
        self._url = url

    def forward(self, __input=None, **kw):
        assert self._url is not None, f'Please start {self.__class__} first'
        assert (__input is None) ^ (len(kw) == 0), (
            f'Error: Providing args and kwargs at the same time is not allowed in {__class__}')

        def _prepare_data(kw):
            if self.template_message is None: return __input if __input else kw
            data = copy.deepcopy(self.template_message)
            if isinstance(__input, dict): kw = __input
            elif __input is not None: kw[self._prompt_keys[0]] = __input
            assert set(self._prompt_keys).issubset(set(kw.keys())), ('Error: Required keys ['
                f'{",".join(set(self._prompt_keys) - set(kw.keys()))}] are missing from user input')
            data, kw = self._modify_parameters(data, kw)

            if isinstance(kw.get(self.input_key_name, None), dict):
                kw = kw[self.input_key_name]
            data[self.input_key_name] = self._prompt.format(**kw)
            return data

        def _callback(text):
            return text if self._response_split is None else text.split(self._response_split)[-1]

        if self._stream:
            # context bug with httpx, so we use requests
            def _impl():
                with requests.post(self._url, json=_prepare_data(kw), stream=True) as r:
                    for chunk in r.iter_content(None):
                        yield(_callback(chunk.decode('utf-8')))
            return _impl()
        else:
            with httpx.Client(timeout=300) as client:
                response = client.post(self._url, json=_prepare_data(kw), headers=self.template_headers)
                return _callback(response.text)

    def prompt(self, prompt='{input}', response_split=None):
        self._prompt, self._response_split = prompt, response_split
        self._prompt_keys = list(set(re.findall(r'\{(\w+)\}', self._prompt)))
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
        return paras, kw

    def set_default_parameters(self, **kw):
        self._modify_parameters(self.template_message, kw)


class ActionModule(ModuleBase):
    def __init__(self, action):
        super().__init__()
        if isinstance(action, FlowBase):
            action.for_each(lambda x: isinstance(x, ModuleBase), lambda x: self.submodules.append(x))
        self.action = action

    def forward(self, *args, **kw):
        if isinstance(self.action, FlowBase):
            r = self.action.start(*args, **kw).result
        else:
            r = self.action(*args, **kw)
        return r

    def __repr__(self):
        representation = '<ActionModule> ['
        if isinstance(self.action, (FlowBase, ActionModule, ServerModule, SequenceModule)):
            sub_rep = '\n'.join(['    ' + s for s in repr(self.action).split('\n')])
            representation += '\n' + sub_rep + '\n'
        else:
            representation += repr(self.action)
        return representation + ']'


class ServerModule(UrlModule):
    def __init__(self, m, pre=None, post=None, stream=False):
        super().__init__(url=None, stream=stream)
        self.m = m
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
    
    # change to urlmodule when pickling to server process
    def __reduce__(self):
        assert hasattr(self, '_url') and self._url is not None
        m = UrlModule(self._url,  stream=self._stream).prompt(
            prompt=self._prompt, response_split=self._response_split)
        m._set_template(
            self.template_message,
            self.input_key_name,
            self.template_headers,
        )

        return m.__reduce__()

    def __repr__(self):
        representation = '<ServerModule> ['
        if isinstance(self.action, (FlowBase, ActionModule, ServerModule, SequenceModule)):
            sub_rep = '\n'.join(['    ' + s for s in repr(self.action).split('\n')])
            representation += '\n' + sub_rep + '\n'
        else:
            representation += repr(self.action)
        return representation + ']'


css = """
#logging {background-color: #FFCCCB}
"""
class WebModule(ModuleBase):
    def __init__(self, m, *, title='对话演示终端') -> None:
        super().__init__()
        self.m = m
        self.title = title
        self.demo = self.init_web()

    def init_web(self):
        with gr.Blocks(css=css, title=self.title) as demo:
            with gr.Row():
                with gr.Column(scale=3):
                    chat_use_context = gr.Checkbox(interactive=True, value=False, label="使用上下文")
                    stream_output = gr.Checkbox(interactive=True, value=True, label="流式输出")
                    dbg_msg = gr.Textbox(show_label=True, label='处理日志', elem_id='logging', interactive=False, max_lines=10)
                    clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)
                with gr.Column(scale=6):
                    chatbot = gr.Chatbot(height=600)
                    query_box = gr.Textbox(show_label=False, placeholder='输入内容并回车!!!')

            query_box.submit(self._prepare, [query_box, chatbot], [query_box, chatbot], queue=False
                ).then(self._respond_stream, [chat_use_context, chatbot, stream_output], [chatbot, dbg_msg], queue=chatbot
                ).then(lambda: gr.update(interactive=True), None, query_box, queue=False)
            clear_btn.click(self._clear_history, None, outputs=[chatbot, query_box, dbg_msg])
        return demo

    def _prepare(self, query, chat_history):
        if chat_history is None:
            chat_history = []
        return '', chat_history + [[query, None]]
        
    def _respond_stream(self, use_context, chat_history, stream_output):
        try:
            # TODO: move context to trainable module
            input = ('\<eos\>'.join([f'{h[0]}\<eou\>{h[1]}' for h in chat_history]).rsplit('\<eou\>', 1)[0]
                     if use_context else chat_history[-1][0])
            result, log = self.m(input), None
            def get_log_and_message(s, log=''):
                return ((s.messages, s.err[1] if s.err[0] != 0 else s.trace) 
                        if isinstance(s, ModuleResponse) else (s, log))
            if isinstance(result, (ModuleResponse, str)):
                chat_history[-1][1], log = get_log_and_message(result)
            elif isinstance(result, GeneratorType):
                chat_history[-1][1] = ''
                for s in result:
                    if isinstance(s, (ModuleResponse, str)):
                        s, log = get_log_and_message(s, log)
                    chat_history[-1][1] += s
                    if stream_output: yield chat_history, log
            else:
                raise TypeError(f'function result should only be ModuleResponse or str, but got {type(result)}')
        except Exception as e:
            chat_history = None
            log = str(e)
        yield chat_history, log

    def _clear_history(self):
        return [], '', ''

    def _work(self):
        def _impl():
            self.demo.queue().launch(server_name="0.0.0.0", server_port=20570)
        self.p = multiprocessing.Process(target=_impl)
        self.p.start()

    def _get_deploy_tasks(self):
        return Pipeline(self._work)

    def wait(self):
        return self.p.join()


class TrainableModule(UrlModule):
    def __init__(self, base_model='', target_path='', *, stream=False):
        super().__init__(url=None, stream=stream)
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

    def __getattr__(self, key):
        def _setattr(v):
            if isinstance(v, tuple):
                v, kargs = v
                setattr(self, f'_{key}_args', kargs)
            setattr(self, f'_{key}', v)
            if hasattr(self, f'_{key}_setter_hook'): getattr(self, f'_{key}_setter_hook')()
            return self
        keys = ['trainset', 'train', 'finetune', 'deploy', 'mode']
        if key in keys:
            return _setattr
        elif key.startswith('_') and key[1:] in keys:
            return None
        raise AttributeError(f'{__class__} object has no attribute {key}')

    # change to urlmodule when pickling to server process
    def __reduce__(self):
        assert hasattr(self, '_url') and self._url is not None
        m = UrlModule(self._url, stream=self._stream).prompt(
            prompt=self._prompt, response_split=self._response_split)
        m._set_template(
            self.template_message,
            self.input_key_name,
            self.template_headers,
        )
        return m.__reduce__()

    def __repr__(self):
        mode = '-Train' if self._mode == 'train' else (
               '-Finetune' if self._mode == 'finetune' else '')
        return f'<TrainableModule{mode}> [{self.base_model}]'


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