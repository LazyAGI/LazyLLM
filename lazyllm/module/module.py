import os
import inspect
import traceback
from lazyllm import ThreadPoolExecutor

import lazyllm
from lazyllm import FlatList, Option, kwargs, globals, colored_text
from ..flow import FlowBase, Pipeline, Parallel
from ..common.bind import _MetaBind
import uuid
from ..client import redis_client
from ..hook import LazyLLMHook
from lazyllm import FileSystemQueue
from contextlib import contextmanager
from typing import Optional, Union, Dict


# use _MetaBind:
# if bind a ModuleBase: x, then hope: isinstance(x, ModuleBase)==True,
# example: ActionModule.submodules:: isinstance(x, ModuleBase) will add submodule.
class ModuleBase(metaclass=_MetaBind):
    """Module是LazyLLM中的顶层组件，具备训练、部署、推理和评测四项关键能力，每个模块可以选择实现其中的部分或者全部的能力，每项能力都可以由一到多个Component组成。
ModuleBase本身不可以直接实例化，继承并实现 ``forward`` 函数的子类可以作为一个仿函数来使用。
类似pytorch的Moudule，当一个Module A持有了另一个Module B的实例作为成员变量时，会自动加入到submodule中。

如果你需要以下的能力，请让你自定义的类继承自ModuleBase:

1. 组合训练、部署、推理和评测的部分或全部能力，例如Embedding模型需要训练和推理

2. 持有的成员变量具备训练、部署和评测的部分或全部能力，并且想通过Module的根节点的 ``start``,  ``update``, ``eval`` 等方法对其持有的成员进行训练、部署和评测时。

3. 将用户设置的参数从最外层直接传到你自定义的模块中（参考Tools.webpages.WebModule）

4. 希望能被参数网格搜索模块使用（参考TrialModule）


Examples:
    >>> import lazyllm
    >>> class Module(lazyllm.module.ModuleBase):
    ...     pass
    ... 
    >>> class Module2(lazyllm.module.ModuleBase):
    ...     def __init__(self):
    ...         super(__class__, self).__init__()
    ...         self.m = Module()
    ... 
    >>> m = Module2()
    >>> m.submodules
    [<Module type=Module>]
    >>> m.m3 = Module()
    >>> m.submodules
    [<Module type=Module>, <Module type=Module>]
    """
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
        self._used_by_moduleid = None
        self._module_name = None
        self._options = []
        self.eval_result = None
        self._hooks = set()

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
        hook_objs = []
        for hook_type in self._hooks:
            if isinstance(hook_type, LazyLLMHook):
                hook_objs.append(hook_type)
            else:
                hook_objs.append(hook_type(self))
            hook_objs[-1].pre_hook(*args, **kw)
        try:
            kw.update(globals['global_parameters'].get(self._module_id, dict()))
            if (files := globals['lazyllm_files'].get(self._module_id)) is not None: kw['lazyllm_files'] = files
            if (history := globals['chat_history'].get(self._module_id)) is not None: kw['llm_chat_history'] = history

            r = self.forward(**args[0], **kw) if args and isinstance(args[0], kwargs) else self.forward(*args, **kw)
            if self._return_trace:
                lazyllm.FileSystemQueue.get_instance('lazy_trace').enqueue(str(r))
        except Exception as e:
            raise RuntimeError(
                f"\nAn error occured in {self.__class__} with name {self.name}.\n"
                f"Args:\n{args}\nKwargs\n{kw}\nError messages:\n{e}\n"
                f"Original traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
        for hook_obj in hook_objs[::-1]:
            hook_obj.post_hook(r)
        for hook_obj in hook_objs:
            hook_obj.report()
        self._clear_usage()
        return r

    def _stream_output(self, text: str, color: Optional[str] = None, *, cls: Optional[str] = None):
        (FileSystemQueue.get_instance(cls) if cls else FileSystemQueue()).enqueue(colored_text(text, color))
        return ''

    @contextmanager
    def stream_output(self, stream_output: Optional[Union[bool, Dict]] = None):
        if stream_output and isinstance(stream_output, dict) and (prefix := stream_output.get('prefix')):
            self._stream_output(prefix, stream_output.get('prefix_color'))
        yield
        if isinstance(stream_output, dict) and (suffix := stream_output.get('suffix')):
            self._stream_output(suffix, stream_output.get('suffix_color'))

    def used_by(self, module_id):
        self._used_by_moduleid = module_id
        return self

    def _clear_usage(self):
        globals["usage"].pop(self._module_id, None)

    # interfaces
    def forward(self, *args, **kw):
        """定义了每次执行的计算步骤，ModuleBase的所有的子类都需要重写这个函数。


Examples:
    >>> import lazyllm
    >>> class MyModule(lazyllm.module.ModuleBase):
    ...     def forward(self, input):
    ...         return input + 1
    ... 
    >>> MyModule()(1)
    2   
    """
        raise NotImplementedError

    def register_hook(self, hook_type: LazyLLMHook):
        self._hooks.add(hook_type)

    def unregister_hook(self, hook_type: LazyLLMHook):
        if hook_type in self._hooks:
            self._hooks.remove(hook_type)

    def clear_hooks(self):
        self._hooks = set()

    def _get_train_tasks(self):
        """定义训练任务，该函数返回训练的pipeline，重写了此函数的子类可以在update阶段被训练/微调。


Examples:
    >>> import lazyllm
    >>> class MyModule(lazyllm.module.ModuleBase):
    ...     def _get_train_tasks(self):
    ...         return lazyllm.pipeline(lambda : 1, lambda x: print(x))
    ... 
    >>> MyModule().update()
    1
    """
        return None
    def _get_deploy_tasks(self):
        """定义部署任务，该函数返回训练的pipeline，重写了此函数的子类可以在update/start阶段被部署。


Examples:
    >>> import lazyllm
    >>> class MyModule(lazyllm.module.ModuleBase):
    ...     def _get_deploy_tasks(self):
    ...         return lazyllm.pipeline(lambda : 1, lambda x: print(x))
    ... 
    >>> MyModule().start()
    1
    """
        return None
    def _get_post_process_tasks(self): return None

    def _set_mid(self, mid=None):
        self._module_id = mid if mid else str(uuid.uuid4().hex)
        return self

    @property
    def name(self):
        return self._module_name

    @name.setter
    def name(self, name):
        self._module_name = name

    @property
    def submodules(self):
        return self._submodules

    def evalset(self, evalset, load_f=None, collect_f=lambda x: x):
        """为Module设置评测集，设置过评测集的Module在 ``update`` 或 ``eval`` 的时候会进行评测，评测结果会存在eval_result变量中。


Examples:
    >>> import lazyllm
    >>> m = lazyllm.module.TrainableModule().deploy_method(lazyllm.deploy.dummy).finetune_method(lazyllm.finetune.dummy).trainset("").mode("finetune").prompt(None)
    >>> m.evalset([1, 2, 3])
    >>> m.update()
    INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
    >>> print(m.eval_result)
    ["reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1}"]
    """
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
            with ThreadPoolExecutor(max_workers=200) as executor:
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

    def update(self, *, recursive=True):
        """更新模块（及所有的子模块）。当模块重写了 ``_get_train_tasks`` 方法后，模块会被更新。更新完后会自动进入部署和推理的流程。

Args:
    recursive (bool): 是否递归更新所有的子模块，默认为True


Examples:
    >>> import lazyllm
    >>> m = lazyllm.module.TrainableModule().finetune_method(lazyllm.finetune.dummy).trainset("").deploy_method(lazyllm.deploy.dummy).mode('finetune').prompt(None)
    >>> m.evalset([1, 2, 3])
    >>> m.update()
    INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
    >>> print(m.eval_result)
    ["reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1}"]
    """
        return self._update(mode=['train', 'server', 'eval'], recursive=recursive)
    def update_server(self, *, recursive=True): return self._update(mode=['server'], recursive=recursive)
    def eval(self, *, recursive=True):
        """对模块（及所有的子模块）进行评测。当模块通过 ``evalset`` 设置了评测集之后，本函数生效。

Args:
    recursive (bool): 是否递归评测所有的子模块，默认为True


Examples:
    >>> import lazyllm
    >>> class MyModule(lazyllm.module.ModuleBase):
    ...     def forward(self, input):
    ...         return f'reply for input'
    ... 
    >>> m = MyModule()
    >>> m.evalset([1, 2, 3])
    >>> m.eval().eval_result
    ['reply for input', 'reply for input', 'reply for input']
    """
        return self._update(mode=['eval'], recursive=recursive)
    def start(self):
        """部署模块及所有的子模块


Examples:
    >>> import lazyllm
    >>> m = lazyllm.TrainableModule().deploy_method(lazyllm.deploy.dummy).prompt(None)
    >>> m.start()
    <Module type=Trainable mode=None basemodel= target= stream=False return_trace=False>
    >>> m(1)
    "reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}"
    """
        return self._update(mode=['server'], recursive=True)
    def restart(self):
        """重新重启模块及所有的子模块


Examples:
    >>> import lazyllm
    >>> m = lazyllm.TrainableModule().deploy_method(lazyllm.deploy.dummy).prompt(None)
    >>> m.restart()
    <Module type=Trainable mode=None basemodel= target= stream=False return_trace=False>
    >>> m(1)
    "reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}"
    """
        return self.start()
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

    def for_each(self, filter, action):
        for submodule in self.submodules:
            if filter(submodule):
                action(submodule)
            submodule.for_each(filter, action)


class ActionModule(ModuleBase):
    """用于将函数、模块、flow、Module等可调用的对象包装一个Module。被包装的Module（包括flow中的Module）都会变成该Module的submodule。

Args:
    action (Callable|list[Callable]): 被包装的对象，是一个或一组可执行的对象。
"""
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
        try:
            if isinstance(self.action, FlowBase):
                submodule = []
                self.action.for_each(lambda x: isinstance(x, ModuleBase), lambda x: submodule.append(x))
                return submodule
        except Exception as e:
            raise RuntimeError(f"{str(e)}\nOriginal traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
        return super().submodules

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Action', subs=[repr(self.action)],
                                 name=self._module_name, return_trace=self._return_trace)


def flow_start(self):
    ActionModule(self).start()
    return self


lazyllm.ReprRule.add_rule('Module', 'Action', 'Flow')
setattr(lazyllm.LazyLLMFlowsBase, 'start', flow_start)


class ModuleRegistryBase(ModuleBase, metaclass=lazyllm.LazyLLMRegisterMetaClass):
    __reg_overwrite__ = 'forward'


register = lazyllm.Register(ModuleRegistryBase, ['forward'])
