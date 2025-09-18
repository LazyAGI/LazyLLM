import os
import inspect
import traceback
from lazyllm import ThreadPoolExecutor

import lazyllm
from lazyllm import FlatList, Option, kwargs, globals, colored_text, redis_client
from ..flow import FlowBase, Pipeline, Parallel
from ..common.bind import _MetaBind
import uuid
from ..hook import LazyLLMHook
from lazyllm import FileSystemQueue
from contextlib import contextmanager
from typing import Optional, Union, Dict, List
import copy
from collections import defaultdict
import sqlite3
import pickle
import hashlib
from abc import ABC, abstractmethod
from filelock import FileLock


lazyllm.config.add('cache_dir', str, os.path.join(os.path.expanduser(lazyllm.config['home']), 'cache'), 'CACHE_DIR')
lazyllm.config.add('cache_strategy', str, 'memory', 'CACHE_STRATEGY')
lazyllm.config.add('cache_mode', str, 'RW', 'CACHE_MODE', options=['RW', 'RO', 'WO', 'NONE'])
redis_client = redis_client['module']


class CacheNotFoundError(Exception): pass


class _CacheStorageStrategy(ABC):
    def __init__(self, cache: Optional[bool] = False):
        if cache:
            self._cache_dir = os.path.join(lazyllm.config['cache_dir'], 'module')
            os.makedirs(self._cache_dir, exist_ok=True)
            self._lock = FileLock(os.path.join(self._cache_dir, 'cache.lock'))
        else:
            self._lock = lambda: contextmanager(lambda: (yield))()

    @abstractmethod
    def get(self, key: str, hash_key: str): pass

    @abstractmethod
    def set(self, key: str, hash_key: str, value): pass

    def close(self): pass  # noqa B027


class _MemoryCacheStrategy(_CacheStorageStrategy):
    def __init__(self):
        super().__init__()
        self._cache = defaultdict(dict)

    def get(self, key: str, hash_key: str):
        if key not in self._cache or hash_key not in self._cache[key]:
            raise CacheNotFoundError(f'Cache not found for {key}')
        return self._cache[key][hash_key]

    def set(self, key: str, hash_key: str, value):
        self._cache[key][hash_key] = value

    def close(self):
        self._cache.clear()


class _FileCacheStrategy(_CacheStorageStrategy):
    def __init__(self):
        super().__init__(cache=True)
        self.cache_file = os.path.join(self._cache_dir, 'cache.dat')

    def _load_cache(self):
        if not os.path.exists(self.cache_file): return {}
        try:
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return {}

    def _save_cache(self, cache_data):
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception:
            pass

    def get(self, key: str, hash_key: str):
        with self._lock:
            cache_data = self._load_cache()
            if key not in cache_data or hash_key not in cache_data[key]:
                raise CacheNotFoundError(f'Cache not found for {key}')
            return cache_data[key][hash_key]

    def set(self, key: str, hash_key: str, value):
        with self._lock:
            cache_data = self._load_cache()
            if key not in cache_data:
                cache_data[key] = {}
            cache_data[key][hash_key] = value
            self._save_cache(cache_data)


class _SQLiteCacheStrategy(_CacheStorageStrategy):
    def __init__(self):
        super().__init__(cache=True)
        self.db_path = os.path.join(self._cache_dir, 'cache.db')
        self.conn = None
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT,
                hash_key TEXT,
                value BLOB,
                PRIMARY KEY (key, hash_key)
            )
        ''')
        self.conn.commit()

    def get(self, key: str, hash_key: str):
        with self._lock:
            cursor = self.conn.execute(
                'SELECT value FROM cache WHERE key = ? AND hash_key = ?',
                (key, hash_key)
            )
            row = cursor.fetchone()
            if row is None:
                raise CacheNotFoundError(f'Cache not found for {key}')
            try:
                return pickle.loads(row[0])
            except Exception as e:
                raise CacheNotFoundError(f'Failed to deserialize cache for {key}: {e}')

    def set(self, key: str, hash_key: str, value):
        with self._lock:
            try:
                serialized_value = pickle.dumps(value)
                self.conn.execute(
                    'INSERT OR REPLACE INTO cache (key, hash_key, value) VALUES (?, ?, ?)',
                    (key, hash_key, serialized_value)
                )
                self.conn.commit()
            except Exception:
                pass

    def close(self):
        if self.conn:
            self.conn.close()


class _RedisCacheStrategy(_CacheStorageStrategy):
    def __init__(self, prefix: str = ''):
        if not redis_client: raise RuntimeError('Redis url should be set by `export LAZYLLM_REDIS_URL = xxx`')
        self._client = redis_client[prefix] if prefix else redis_client
        super().__init__()

    def _get_redis_key(self, key: str, hash_key: str):
        return f'{key}:{hash_key}'

    def get(self, key: str, hash_key: str):
        redis_key = self._get_redis_key(key, hash_key)
        value = self._client.get(redis_key)
        if value is None:
            raise CacheNotFoundError(f'Cache not found for {key}')
        try:
            return pickle.loads(value)
        except Exception as e:
            raise RuntimeError(f'Failed to deserialize cache for {key}: {e}')

    def set(self, key: str, hash_key: str, value):
        try:
            redis_key = self._get_redis_key(key, hash_key)
            serialized_value = pickle.dumps(value)
            self._client.set(redis_key, serialized_value)
        except Exception:
            pass


class ModuleCache(object):
    def __init__(self, strategy: Optional[str] = None):
        self._strategy = self._create_strategy(strategy or lazyllm.config['cache_strategy'])

    def _create_strategy(self, strategy: str) -> _CacheStorageStrategy:
        strategy = strategy.lower()
        strategies = {
            'memory': _MemoryCacheStrategy,
            'file': _FileCacheStrategy,
            'sqlite': _SQLiteCacheStrategy,
            'redis': _RedisCacheStrategy,
        }

        if strategy not in strategies:
            raise ValueError(f'Unsupported cache strategy: {strategy}. '
                             f'Available strategies: {list(strategies.keys())}')
        return strategies[strategy]()

    def _hash(self, args, kw):
        content = str(args) + str(sorted(kw.items()) if kw else '')
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, key, args, kw):
        if 'R' not in lazyllm.config['cache_mode']:
            raise CacheNotFoundError('Cannot read cache due to `LAZYLLM_CACHE_MODE = WO`')
        hash_key = self._hash(args, kw)
        return self._strategy.get(key, hash_key)

    def set(self, key, args, kw, value):
        if 'W' not in lazyllm.config['cache_mode']: return
        hash_key = self._hash(args, kw)
        self._strategy.set(key, hash_key, value)

    def close(self):
        self._strategy.close()


module_cache = ModuleCache()


# use _MetaBind:
# if bind a ModuleBase: x, then hope: isinstance(x, ModuleBase)==True,
# example: ActionModule.submodules:: isinstance(x, ModuleBase) will add submodule.
class ModuleBase(metaclass=_MetaBind):
    """ModuleBase is the core base class in LazyLLM, defining the common interface and fundamental capabilities for all modules.  
It abstracts training, deployment, inference, and evaluation logic, while also providing mechanisms for submodule management, hook registration, parameter passing, and recursive updates.  
Custom modules should inherit from ModuleBase and implement the ``forward`` method to define specific inference logic.  

Key Features:
    - Unified management of submodules, automatically tracking held ModuleBase instances.
    - Support for Option type hyperparameters, enabling grid search and automated tuning.
    - Hook system that allows executing custom logic before and after calls.
    - Encapsulated update pipeline covering training, server deployment, and evaluation.
    - Built-in evalset loading and parallel inference evaluation.

Args:
    return_trace (bool): Whether to write inference results into the trace queue for debugging and tracking. Default is ``False``.

Use Cases:
    1. When combining some or all of training, deployment, inference, and evaluation capabilities, e.g., an embedding model requiring both training and inference.
    2. When you want to recursively manage submodules through root-level methods such as ``start``, ``update``, and ``eval``.
    3. When you want user parameters to be automatically propagated from outer modules to inner implementations (see WebModule).
    4. When you want the module to support parameter grid search (see TrialModule).


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
        self._use_cache: Union[bool, str] = False
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

            r = (self._call_impl(**args[0], **kw)
                 if args and isinstance(args[0], kwargs) else self._call_impl(*args, **kw))
            if self._return_trace:
                lazyllm.FileSystemQueue.get_instance('lazy_trace').enqueue(str(r))
        except Exception as e:
            raise RuntimeError(
                f'\nAn error occured in {self.__class__} with name {self.name}.\n'
                f'Args:\n{args}\nKwargs\n{kw}\nError messages:\n{e}\n'
                f'Original traceback:\n{"".join(traceback.format_tb(e.__traceback__))}')
        for hook_obj in hook_objs[::-1]:
            hook_obj.post_hook(r)
        for hook_obj in hook_objs:
            hook_obj.report()
        self._clear_usage()
        return r

    def _call_impl(self, *args, **kw):
        if self._use_cache and 'R' in lazyllm.config['cache_mode']:
            try:
                return module_cache.get(self.__cache_hash__, args, kw)
            except CacheNotFoundError:
                self._cache_miss_handler()
        r = self.forward(**args[0], **kw) if args and isinstance(args[0], kwargs) else self.forward(*args, **kw)
        if self._use_cache and 'W' in lazyllm.config['cache_mode']:
            module_cache.set(self.__cache_hash__, args, kw, r)
        return r

    def _stream_output(self, text: str, color: Optional[str] = None, *, cls: Optional[str] = None):
        (FileSystemQueue.get_instance(cls) if cls else FileSystemQueue()).enqueue(colored_text(text, color))
        return ''

    @contextmanager
    def stream_output(self, stream_output: Optional[Union[bool, Dict]] = None):
        """Context manager for streaming output during inference or execution.  
When a dictionary is provided to ``stream_output``, a prefix and suffix can be specified along with optional colors.

Args:
    stream_output (Optional[Union[bool, Dict]]): Configuration for streaming output.

        - If True, enables default streaming output.
        - If a dictionary, may include:

            - 'prefix' (str): Text to output at the beginning.
            - 'prefix_color' (str, optional): Color of the prefix.
            - 'suffix' (str): Text to output at the end.
            - 'suffix_color' (str, optional): Color of the suffix.
"""
        if stream_output and isinstance(stream_output, dict) and (prefix := stream_output.get('prefix')):
            self._stream_output(prefix, stream_output.get('prefix_color'))
        yield
        if isinstance(stream_output, dict) and (suffix := stream_output.get('suffix')):
            self._stream_output(suffix, stream_output.get('suffix_color'))

    def used_by(self, module_id):
        """Mark which module is using the current module, indicating the calling relationship.  
Supports chaining by returning the module itself.

Args:
    module_id (str): Unique ID of the parent module that uses this module.

**Returns:**

- ModuleBase: Returns the module itself for method chaining.
"""
        self._used_by_moduleid = module_id
        return self

    def _clear_usage(self):
        globals['usage'].pop(self._module_id, None)

    # interfaces
    def forward(self, *args, **kw):
        """Forward computation interface that must be implemented by subclasses.  
This method defines the logic for receiving inputs and returning outputs, and is the core function of the module as a functor.

Args:
    *args: Variable positional arguments, subclass can define the input as needed.
    **kw: Variable keyword arguments, subclass can define the input as needed.
"""
        raise NotImplementedError

    def register_hook(self, hook_type: LazyLLMHook):
        """Register a hook to execute specific logic during module invocation.  
The hook must inherit from ``LazyLLMHook`` and can be used to add custom operations before or after the module's forward computation, such as logging or metrics collection.

Args:
    hook_type (LazyLLMHook): Hook object to register.
"""
        self._hooks.add(hook_type)

    def unregister_hook(self, hook_type: LazyLLMHook):
        """Unregister a previously registered hook.  
If the hook exists in the module, it will be removed and no longer executed during module invocation.

Args:
    hook_type (LazyLLMHook): Hook object to unregister.
"""
        if hook_type in self._hooks:
            self._hooks.remove(hook_type)

    def clear_hooks(self):
        """Clear all hooks registered in the module.  
After calling this, the module will no longer execute any hook logic.
"""
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
        """Set the evaluation set for the module.  
During ``update`` or ``eval``, the module will perform inference on the evaluation set, and the results will be stored in the ``eval_result`` variable.  

Args:
    evalset (Union[list, str]): Evaluation data list or path to an evaluation data file.
    load_f (Optional[Callable]): Function to load and parse the evaluation file into a list if ``evalset`` is a file path, default is None.
    collect_f (Callable): Function to post-process evaluation results, default is ``lambda x: x``.


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
    def _update(self, *, mode: Optional[Union[str, List[str]]] = None, recursive: bool = True):  # noqa C901
        if not mode: mode = list(self.mode_list)
        if type(mode) is not list: mode = [mode]
        for item in mode:
            assert item in self.mode_list, f'Cannot find {item} in mode list: {self.mode_list}'
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

    def update(self, *, recursive: bool = True):
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

    def update_server(self, *, recursive: bool = True):
        """Update the deployment (server) part of the module and its submodules. When a module or submodule implements deployment functionality, the corresponding services will be started.

Args:
    recursive (bool): Whether to recursively update deployment tasks of all submodules, default is True.
"""
        return self._update(mode=['server'], recursive=recursive)
    def eval(self, *, recursive: bool = True):
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
        """Start the deployment services of the module and all its submodules. This ensures that the server functionality of the module and its submodules is executed, suitable for initialization or restarting services.

**Returns:**

- ModuleBase: Returns itself to support method chaining


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
        """Restart the deployment services of the module and its submodules. Internally calls the ``start`` method to reinitialize the services.

**Returns:**

- ModuleBase: Returns itself to support method chaining


Examples:
    >>> import lazyllm
    >>> m = lazyllm.TrainableModule().deploy_method(lazyllm.deploy.dummy).prompt(None)
    >>> m.restart()
    <Module type=Trainable mode=None basemodel= target= stream=False return_trace=False>
    >>> m(1)
    "reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}"
    """
        return self.start()

    def wait(self):
        """Wait for the module or its submodules to finish execution. Currently, this method is a no-op and can be implemented by subclasses according to specific deployment logic.
"""
        pass

    def stop(self):
        """Stop the module and all its submodules. This method recursively calls the ``stop`` method of each submodule, suitable for releasing resources or shutting down services.
"""
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
        """Execute a specified action on all submodules of the module. Recursively traverses all submodules, and if a submodule satisfies the ``filter`` condition, executes the ``action``.

Args:
    filter (Callable): A function that takes a submodule as input and returns a boolean, used to determine whether to perform the action.
    action (Callable): A function to perform on submodules that meet the condition.
"""
        for submodule in self.submodules:
            if filter(submodule):
                action(submodule)
            submodule.for_each(filter, action)

    @property
    def __cache_hash__(self):
        cache_hash = self.__class__.__name__
        if isinstance(self._use_cache, str): cache_hash += f'@{self._use_cache}'
        return cache_hash

    def use_cache(self, flag: Union[bool, str] = True):
        """Enable or disable the caching functionality for the module.

This method controls whether the module uses caching to store and retrieve execution results, 
improving performance and avoiding redundant computations.

Args:
    flag (bool or str, optional): Cache control flag. If True, enables caching; if False, disables caching;
                                 if a string, uses a specific cache identifier. Defaults to True.

**Returns:**

- Returns the module instance itself, supporting method chaining.

"""
        self._use_cache = flag or False
        return self

    def _cache_miss_handler(self): pass


class ActionModule(ModuleBase):
    """Used to wrap a Module around functions, modules, flows, Module, and other callable objects. The wrapped Module (including the Module within the flow) will become a submodule of this Module.

Args:
    action (Callable|list[Callable]): The object to be wrapped, which is one or a set of callable objects.
    return_trace (bool): Whether to enable trace mode to record the execution stack. Defaults to ``False``.

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
        """Executes the wrapped action with the provided input arguments. Equivalent to directly calling the module.

Args:
    args (list of callables or single callable): Positional arguments to be passed to the wrapped action.
    kwargs (dict of callables): Keyword arguments to be passed to the wrapped action.

**Returns:**

- Any: The result of executing the wrapped action.
"""
        return self.action(*args, **kw)

    @property
    def submodules(self):
        """Returns all submodules of type ModuleBase contained in the wrapped action. This automatically traverses any nested modules inside a Pipeline.

**Returns:**

- list[ModuleBase]: List of submodules
"""
        try:
            if isinstance(self.action, FlowBase):
                submodule = []
                self.action.for_each(lambda x: isinstance(x, ModuleBase), lambda x: submodule.append(x))
                return submodule
        except Exception as e:
            raise RuntimeError(f'{str(e)}\nOriginal traceback:\n{"".join(traceback.format_tb(e.__traceback__))}')
        return super().submodules

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Action', subs=[repr(self.action)],
                                 name=self._module_name, return_trace=self._return_trace)


def flow_start(self):
    """Start flow processing execution (deprecated).

This method is deprecated, it is recommended to directly call the flow instance as a function. Executes the flow processing and returns the result.

Args:
    *args: Variable positional arguments passed to the flow processing.
    **kw: Named arguments passed to the flow processing.

**Returns:**

- The result of flow processing.

**Note:**

- This method is marked as deprecated, please use direct invocation of the flow instance instead.
"""
    ActionModule(self).start()
    return self


lazyllm.ReprRule.add_rule('Module', 'Action', 'Flow')
lazyllm.LazyLLMFlowsBase.start = flow_start


class ModuleRegistryBase(ModuleBase, metaclass=lazyllm.LazyLLMRegisterMetaClass):
    __reg_overwrite__ = 'forward'


register = lazyllm.Register(ModuleRegistryBase, ['forward'])
