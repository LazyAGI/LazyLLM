# lazyllm/common AGENTS.md

This directory is LazyLLM's **foundation layer**, depended upon by all upper layers. Changes here have the widest blast radius — proceed with extra caution.

## Mandatory Pre-reading

Before modifying this directory, you must read:
- `lazyllm/AGENTS.md` (global conventions)
- The target file itself
- Upper-layer modules that directly call the target file (verify the change does not break callers)

---

## File Responsibilities

| File | Responsibility |
|------|---------------|
| `registry.py` | Registry system: `LazyLLMRegisterMetaClass`, `Register`, `LazyDict` |
| `bind.py` | Deferred binding: `Bind`, `_0~_9` placeholders, `Placeholder`, `_MetaBind` |
| `globals.py` | Session management: `globals`, `locals`, `init_session`, `teardown_session`, `_GlobalConfig` |
| `common.py` | General utilities: `FlatList`, `ArgsDict`, `LazyLLMCMD`, `once_flag`, `once_wrapper`, etc. |
| `option.py` | Hyperparameter search: `Option`, `OptionIter` |
| `threading.py` | Thread utilities: `Thread`, `ThreadPoolExecutor` |
| `multiprocessing.py` | Process utilities: `SpawnProcess`, `ForkProcess`, `ProcessPoolExecutor` |
| `logger/logger.py` | Logging: `LOG` |
| `exception.py` | Exception handling: `HandledException`, `_trim_traceback` |
| `utils.py` | Serialization utilities: `compile_func`, `obj2str`, `str2obj` |
| `queue.py` | Queues: `RecentQueue`, `FileSystemQueue` |
| `redis_client.py` | Redis client: `redis_client` |

---

## Registry System (`registry.py`)

Reference docs: [`lazyllm/docs/common.py`](../docs/common.py)

### How It Works

`LazyLLMRegisterMetaClass` is the metaclass for all registrable classes. Registration is automatic based on class naming conventions:

```
Class name ends with LazyLLMXxxBase, with metaclass=LazyLLMRegisterMetaClass
    → Creates a LazyDict group
    → Mounts lazyllm.xxx (lowercase) and lazyllm.Xxx (original case) in the lazyllm namespace

Subclass inheriting LazyLLMXxxBase
    → Automatically registered as lazyllm.xxx[subclass_name_lowercase]
    → If __lazyllm_after_registry_hook__ is defined, it is called automatically
      (used for side effects such as registering config entries)
```

Example:
```python
# Define a Base class → automatically creates the lazyllm.deploy namespace
class LazyLLMDeployBase(metaclass=LazyLLMRegisterMetaClass): ...

# Inherit Base → automatically registered as lazyllm.deploy.vllm
class Vllm(LazyLLMDeployBase): ...

# Usage
lazyllm.deploy.vllm(...)
lazyllm.deploy.Vllm(...)  # equivalent
```

**Note:** `LazyLLMRegisterMetaABCClass` is a combined metaclass of `LazyLLMRegisterMetaClass + ABCMeta`, used for Base classes that require abstract methods (e.g. `LazyLLMMemoryBase`).

### LazyDict Features

`LazyDict` is the storage structure for the registry and supports:
- Case-insensitive access: `lazyllm.deploy.Vllm` is equivalent to `lazyllm.deploy.vllm`
- Omitting the group name prefix: if the class name contains the group name, the prefix can be omitted (`lazyllm.deploy.llm` can match `LightLLM`)
- Direct call when there is only one element: `lazyllm.deploy()` is equivalent to `lazyllm.deploy.default()`
- `set_default(key)` to set the default entry

### Register Decorator

`Register` wraps a plain function into a subclass of a Base class and registers it:

```python
register = Register(LazyLLMDeployBase, ['forward'])

@register('deploy.mybackend')
def my_backend_forward(self, ...):
    ...
# Equivalent to creating a MybackendDeploy class that inherits LazyLLMDeployBase
```

### Extension Conventions

- **Adding a registrable component:** inherit the corresponding `LazyLLMXxxBase`; the class name is registered automatically
- **Adding a new registry group:** define a new `LazyLLMXxxBase` class (name must start with `LazyLLM` and end with `Base`, with `metaclass=LazyLLMRegisterMetaClass`)
- **Never** bypass the registry system to instantiate concrete implementation classes directly
- **Never** manually call `LazyLLMRegisterMetaClass.all_clses[...][key] = cls`

---

## Bind Mechanism (`bind.py`)

### Core Components

| Component | Description |
|-----------|-------------|
| `Placeholder (_0~_9)` | Positional argument placeholders; global singletons (one instance per index) |
| `Bind` | Deferred binding container holding function `_f`, positional args `_args`, and keyword args `_kw` |
| `Bind.Args` | Cross-pipeline references (`p.input`, `p.output(step)`) |
| `_MetaBind` | Metaclass that makes `isinstance(bind(module), ModuleBase)` return `True` |

### Placeholder Substitution

`Bind.__call__` replaces placeholders with real arguments at call time:

```python
f = bind(some_func, _0, 'fixed', _1)
f('a', 'b')  # → some_func('a', 'fixed', 'b')
```

### Cross-Pipeline References (`Bind.Args`)

`p.input` is actually `Bind.Args(pipeline_id, 'input')`, resolved at execution time from `locals['bind_args']`:

```python
with pipeline() as p:
    p.step1 = module1
    p.step2 = bind(module2, p.input)              # reference the pipeline's original input
    p.step3 = bind(module3, p.output(p.step1))    # reference step1's output
```

Structure of `locals['bind_args']`:
```python
{
    pipeline_id: {
        'source': pipeline_id,
        'input': <pipeline original input>,
        'kwargs': <pipeline keyword arguments>,
        step1_id: <step1 output>,
        step2_id: <step2 output>,
        ...
    }
}
```

### Transparency of `_MetaBind`

`ModuleBase` uses `_MetaBind` as its metaclass, so:
```python
isinstance(bind(some_module), ModuleBase)  # → True
```
This means the automatic submodule tracking in `ModuleBase.__setattr__` also works for `bind`-wrapped modules.

### Extension Conventions

- **Never** modify `Placeholder._pool` (the singleton pool)
- **Never** resolve placeholders outside `Bind.__call__`
- For new cross-step reference scenarios, reuse the `Bind.Args` mechanism; never invent a new reference approach

---

## Session Management (`globals.py`)

### Two-Layer Architecture

```
globals (Globals / MemoryGlobals / RedisGlobals)
  ├── Session key is thread/coroutine ID (thread: tid-{hex}, async: aid-{hex})
  ├── Global fields:
  │     user_id            Current user ID
  │     chat_history       Conversation history ({module_id: [messages]})
  │     global_parameters  Global parameters ({module_id: {key: value}})
  │     lazyllm_files      File mappings ({module_id: [paths]})
  │     usage              Usage statistics ({module_id: {tokens: n}})
  │     trace              Tracing information
  │     config             Session-level dynamic config (_GlobalConfig read/write entry)
  │     call_stack         Module call stack ([module_id, ...])
  └── Supports Redis for distributed deployments (RedisGlobals, selected automatically)

locals (Locals)
  ├── Fields: bind_args (Flow intermediate results), _lazyllm_agent (Agent workspace)
  └── Falls back to globals when a key is not found
```

### Session Lifecycle

```python
# On the server side when receiving a request
init_session(sid)          # Initialize session, bind to current thread
try:
    result = module(input)
finally:
    teardown_session()     # Clean up session data

# Or use the new_session() context manager
with new_session(sid):
    result = module(input)
```

### Accessing Session Data by `module_id`

```python
# Read the conversation history for this module
history = lazyllm.globals['chat_history'].get(self._module_id, [])

# Write (usually managed by the framework; business code rarely writes directly)
lazyllm.globals['chat_history'][self._module_id] = new_history
```

### `globals.config` (`_GlobalConfig`)

`globals.config` is the read/write entry for **session-level dynamic configuration**, distinct from the global static `lazyllm.config`:

```python
# Read: checks the current session's dynamic config first, falls back to global static config
value = lazyllm.globals.config['qwen_api_key']

# Write: writes to the current session's dynamic config (only affects the current request)
lazyllm.globals.config['qwen_api_key'] = 'sk-xxx'

# Module-ID-grouped dynamic config (ConfigsDict)
lazyllm.globals.config['dynamic_model_configs'] = {
    module_id: {'source': 'qwen', 'model': 'qwen-plus'}
}
```

Lookup path for `_GlobalConfig`:
1. `globals['config'][key]` (current session's dynamic config)
2. If the value is a `ConfigsDict` (grouped by module ID), find the nearest matching module ID in `call_stack`
3. Fall back to `lazyllm.config[key]` (global static config)

`globals.config.add(name, type, default, env, ...)` registers the entry in both the static `config` and the dynamic whitelist, ensuring dynamic config can only modify pre-registered entries.

### `call_stack` Tracking

`ModuleBase._call_impl` maintains the call stack via `globals.stack_enter(self.identities)`:
```python
with globals.stack_enter(self.identities):
    result = self.forward(...)
```
Used by `_GlobalConfig` to look up dynamic config (e.g. `global_parameters`) by module ID.

### Redis Distributed Support

`RedisGlobals` inherits `MemoryGlobals` and serializes session data into Redis. `ServerModule` carries `Global-Parameters` and `Session-ID` in HTTP request headers; the server side restores the session context via `decode_request`.

### Extension Conventions

- **Never** store session state in Module instance variables (causes cross-user contamination under concurrent requests)
- To add new session data shared across modules, add a field to `Globals.__global_attrs__`
- To add new temporary data used only within a Flow, add a field to `Locals.__global_attrs__`
- Never access `globals._data` directly; use `globals['key']`

---

## Complete Utility Class Reference (`common.py`)

The following utilities are available for reuse by all upper layers. **Before adding new functionality, check whether a matching utility already exists.**

Reference docs: [`lazyllm/docs/common.py`](../docs/common.py)

### Data Structures

| Class | Purpose | Typical Usage |
|-------|---------|---------------|
| `FlatList` | Extends `list`; `absorb(item)` auto-flattens a list or appends a single element | Collecting task lists from multiple sources |
| `ArgsDict` | Extends `dict`; `check_and_update(kw)` validates and updates; `parse_kwargs()` generates `--key=value` CLI strings | Component parameter management |
| `CaseInsensitiveDict` | Case-insensitive dict; keys are normalized to lowercase | Registry storage |
| `package` | Extends `tuple`; wraps multiple return values in a Flow; slicing returns `package` | Multi-output in Flow |
| `kwargs` | Extends `dict`; wraps keyword arguments | Keyword argument passing in Flow |
| `arguments` | Combines `package` + `kwargs`; `append(x)` merges arguments | Mixed argument passing in Flow |

### Command Encapsulation

| Class/Function | Purpose | Typical Usage |
|----------------|---------|---------------|
| `LazyLLMCMD` | Command wrapper; supports string or callable; `__str__` auto-redacts API keys; `with_cmd` copies; `get_args` extracts arguments | `cmd` method in Deploy/Finetune backends |

```python
cmd = LazyLLMCMD(
    'python train.py --lr {lr} --epochs {epochs}',
    vars_for_format={'lr': 0.001, 'epochs': 10},
    return_value='model_path',
    checkf=lambda: os.path.exists('model_path'),
)
```

### One-time Execution

| Class/Function | Purpose | Typical Usage |
|----------------|---------|---------------|
| `once_flag` | One-time flag; thread-safe; `set` / `reset` / `__bool__` | Manual control of single execution |
| `call_once(flag, func, ...)` | Ensures a function runs only once using `once_flag` | Resource initialization |
| `once_wrapper(reset_on_pickle)` | Decorator that wraps an instance method to run only once | Model loading, service startup |

```python
# once_wrapper: decorator (most common)
@once_wrapper
def init_resources(self):
    ...  # Runs only once; subsequent calls are no-ops

# once_flag: manual control
flag = once_flag()
call_once(flag, expensive_init)
```

### Wrappers and Descriptors

| Class/Function | Purpose | Typical Usage |
|----------------|---------|---------------|
| `ReadOnlyWrapper` | Read-only wrapper; `__deepcopy__` discards the obj; `isNone()` checks emptiness | Wrapping async job objects |
| `Identity` | Identity function; returns the input unchanged for single input, returns `package` for multiple | Pass-through step in Flow |
| `ResultCollector` | Result collector; `collector(name)(value)` stores and returns; `collector[name]` reads | Collecting intermediate results in Pipeline |
| `DynamicDescriptor` | Descriptor; passes `self` on instance access, `cls` on class access | Properties that work as both instance and class methods |

### Singletons and Lifecycle

| Class/Function | Purpose | Typical Usage |
|----------------|---------|---------------|
| `singleton(cls)` | Function decorator that turns a class into a singleton | Globally unique resources |
| `SingletonMeta` | Singleton metaclass; thread-safe | Metaclass for singleton classes |
| `SingletonABCMeta` | Combined metaclass of `SingletonMeta + ABCMeta` | Used by `Globals` |
| `reset_on_pickle(*fields)` | Class decorator; sets specified fields to `None` on pickle, rebuilds on unpickle | Non-serializable resources (connections, locks) |
| `Finalizer` | Destructor; calls a cleanup function on `__del__` / `__exit__`; supports conditions | Temporary resource cleanup |

### Paths and Files

| Class/Function | Purpose | Typical Usage |
|----------------|---------|---------------|
| `TempPathGenerator` | Context manager; writes string content to temp files and returns path list | Passing large data to subprocesses |
| `is_valid_url(url)` | Checks URL validity | Parameter validation |
| `is_valid_path(path)` | Checks whether a file path exists | Parameter validation |

### Timeout and Retry

| Class/Function | Purpose | Typical Usage |
|----------------|---------|---------------|
| `timeout(duration)` | Context manager; raises `TimeoutException` on expiry | Limiting operation duration |
| `retry(func, stop_after_attempt, delay)` | Retry decorator; supports both functional and decorator usage | Network request retries |

### Repr Utilities

| Class/Function | Purpose | Typical Usage |
|----------------|---------|---------------|
| `make_repr(type, name, *, attrs, subs, items)` | Generates a standardized repr string with nesting support | `__repr__` implementation |
| `ReprRule` | Repr merge rule registry; `add_rule` / `check_combine` | Custom repr merge strategies |

---

## Option (`option.py`)

Hyperparameter placeholder for grid search in `TrialModule`:

```python
lr = Option([0.001, 0.01, 0.1])
model = TrainableModule('qwen2-1.5b').trainset(data).finetune_method(llamafactory)
trial = TrialModule(model, options={'lr': lr})
trial.update()  # Iterates over all lr combinations
```

---

## Common Utility Usage Examples

### ArgsDict

```python
args = ArgsDict({'lr': 0.001, 'epochs': 10})
args.check_and_update({'lr': 0.01})  # Update an existing key
cmd_str = args.parse_kwargs()        # → '--lr=0.01 --epochs=10'
```

### FlatList

```python
tasks = FlatList()
tasks.absorb(module._get_train_tasks())  # Merge elements from another list
tasks.absorb(single_task)               # Append a single element
```

### package / kwargs

```python
# Multiple outputs in Flow
return package(result1, result2)  # Next step receives (result1, result2)

# Keyword argument passing in Flow
return kwargs(key1=val1, key2=val2)  # Next step receives **{key1: val1, key2: val2}
```
