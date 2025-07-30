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
import copy

# use _MetaBind:
# if bind a ModuleBase: x, then hope: isinstance(x, ModuleBase)==True,
# example: ActionModule.submodules:: isinstance(x, ModuleBase) will add submodule.
class ModuleBase(metaclass=_MetaBind):
    """Module is the top-level component in LazyLLM, possessing four key capabilities: training, deployment, inference, and evaluation. Each module can choose to implement some or all of these capabilities, and each capability can be composed of one or more components.
ModuleBase itself cannot be instantiated directly; subclasses that inherit and implement the forward function can be used as a functor.
Similar to PyTorch's Module, when a Module A holds an instance of another Module B as a member variable, B will be automatically added to A's submodules.
If you need the following capabilities, please have your custom class inherit from ModuleBase:

1. Combine some or all of the training, deployment, inference, and evaluation capabilities. For example, an Embedding model requires training and inference.

2. If you want the member variables to possess some or all of the capabilities for training, deployment, and evaluation, and you want to train, deploy, and evaluate these members through the start, update, eval, and other methods of the Module's root node.

3. Pass user-set parameters directly to your custom module from the outermost layer (refer to WebModule).

4. The desire for it to be usable by the parameter grid search module (refer to TrialModule).


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
                hook_objs.append(copy.deepcopy(hook_type))
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
        """Define computation steps executed each time, all subclasses of ModuleBase need to override.


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
        """Define a training task. This function returns a training pipeline. Subclasses that override this function can be trained or fine-tuned during the update phase.


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
        """Define a deployment task. This function returns a deployment pipeline. Subclasses that override this function can be deployed during the update/start phase.


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
        """during update or eval, and the results will be stored in the eval_result variable.


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
        """Update the module (and all its submodules). The module will be updated when the ``_get_train_tasks`` method is overridden.

Args:
    recursive (bool): Whether to recursively update all submodules, default is True.


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
        """Evaluate the module (and all its submodules). This function takes effect after the module has been set with an evaluation set using 'evalset'.

Args:
    recursive (bool): Whether to recursively evaluate all submodules. Defaults to True.


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
        """Deploy the module and all its submodules.


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
        """Re-deploy the module and all its submodules.


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
    """Used to wrap a Module around functions, modules, flows, Module, and other callable objects. The wrapped Module (including the Module within the flow) will become a submodule of this Module.

Args:
    action (Callable|list[Callable]): The object to be wrapped, which is one or a set of callable objects.

**Examples:**

```python
>>> import lazyllm
>>> def myfunc(input): return input + 1
... 
>>> class MyModule1(lazyllm.module.ModuleBase):
...     def forward(self, input): return input * 2
... 
>>> class MyModule2(lazyllm.module.ModuleBase):
...     def _get_deploy_tasks(self): return lazyllm.pipeline(lambda : print('MyModule2 deployed!'))
...     def forward(self, input): return input * 4
... 
>>> class MyModule3(lazyllm.module.ModuleBase):
...     def _get_deploy_tasks(self): return lazyllm.pipeline(lambda : print('MyModule3 deployed!'))
...     def forward(self, input): return f'get {input}'
... 
>>> m = lazyllm.ActionModule(myfunc, lazyllm.pipeline(MyModule1(), MyModule2), MyModule3())
>>> print(m(1))
get 16
>>> 
>>> m.evalset([1, 2, 3])
>>> m.update()
MyModule2 deployed!
MyModule3 deployed!
>>> print(m.eval_result)
['get 16', 'get 24', 'get 32']
```


<span style="font-size: 20px;">**`evalset(evalset, load_f=None, collect_f=<function ModuleBase.<lambda>>)`**</span>

Set the evaluation set for the Module. Modules that have been set with an evaluation set will be evaluated during ``update`` or ``eval``, and the evaluation results will be stored in the eval_result variable. 


<span style="font-size: 18px;">&ensp;**`evalset(evalset, collect_f=lambda x: ...)→ None `**</span>


Args:
    evalset (list) :Evaluation set
    collect_f (Callable) :Post-processing method for evaluation results, no post-processing by default.



<span style="font-size: 18px;">&ensp;**`evalset(evalset, load_f=None, collect_f=lambda x: ...)→ None`**</span>


Args:
    evalset (str) :Path to the evaluation set
    load_f (Callable) :Method for loading the evaluation set, including parsing file formats and converting to a list
    collect_f (Callable) :Post-processing method for evaluation results, no post-processing by default.

**Examples:**

```python
>>> import lazyllm
>>> m = lazyllm.module.TrainableModule().deploy_method(deploy.dummy)
>>> m.evalset([1, 2, 3])
>>> m.update()
INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
>>> m.eval_result
["reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1}"]
```


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
