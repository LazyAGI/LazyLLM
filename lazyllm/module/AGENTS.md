# lazyllm/module AGENTS.md

This directory is LazyLLM's **module abstraction layer**, defining the base class and core implementations for all executable units.

Reference docs: [`lazyllm/docs/module.py`](../docs/module.py) (complete API docs for ModuleBase, TrainableModule, ServerModule, etc.)

## Mandatory Pre-reading

Before modifying this directory, you must read:
- `lazyllm/AGENTS.md` (global conventions)
- `lazyllm/common/AGENTS.md` (registry system, Bind, session management)
- `lazyllm/flow/AGENTS.md` (Flow orchestration — Modules can be embedded in Flows)
- `lazyllm/module/module.py` (complete ModuleBase implementation)

---

## File Responsibilities

| File | Responsibility |
|------|---------------|
| `module.py` | `ModuleBase`, `ActionModule`, `register` |
| `servermodule.py` | `ServerModule`, `UrlModule`, `LLMBase`, `StreamCallHelper` |
| `trialmodule.py` | `TrialModule` (hyperparameter grid search) |
| `llms/trainablemodule.py` | `TrainableModule` (local model fine-tuning + deployment) |
| `llms/automodel.py` | `AutoModel` (unified entry point, auto-selects online/local) |
| `llms/online_module.py` | `OnlineModule` (unified online model entry point) |
| `llms/onlinemodule/` | Online model provider implementations (see `onlinemodule/AGENTS.md`) |

---

## ModuleBase Complete Lifecycle

### 1. `__new__`: Option Parameter Validation

At instantiation, checks constructor arguments for `Option` types, ensuring only parameters declared to support `Option` can receive `Option` values.

### 2. `__init__`: Initialization and Submodule Tracking

```python
def __init__(self, *, return_trace=False, id=None, name=None, group_id=None):
    self._submodules = []    # submodule list (auto-tracked)
    self._options = []       # Option hyperparameter list
    self._hooks = []         # Hook list
    self._use_cache = False  # cache switch
    ...
```

**Automatic submodule tracking:** `__setattr__` intercepts all attribute assignments. When the value is a `ModuleBase` instance (or a `bind`-wrapped `ModuleBase`), it is automatically added to `_submodules`:

```python
def __setattr__(self, name, value):
    if isinstance(value, ModuleBase):   # _MetaBind makes bind(module) satisfy this too
        self._submodules.append(value)
    ...
```

### 3. `start()` / `update_server()`: Deploy Services

```python
module.start()  # equivalent to module._update(mode=['server'])
```

`_update` performs a DFS traversal of all submodules, collects deployment tasks, then executes them in parallel via `Parallel.sequential`:

```python
def _update(self, mode, recursive=True):
    deploy_tasks = FlatList()
    # DFS traversal of submodules
    for top in dfs(self.submodules):
        deploy_tasks.absorb(top._get_deploy_tasks())
    Parallel.sequential(*deploy_tasks)()
```

### 4. `__call__`: Session Injection + Hook Execution

```python
def __call__(self, *args, **kw):
    # Inject session data
    kw.update(locals['global_parameters'].get(self._module_id, {}))
    if files := locals['lazyllm_files'].get(self._module_id):
        kw['lazyllm_files'] = files
    if history := locals['chat_history'].get(self._module_id):
        kw['llm_chat_history'] = history

    # Execute hooks + _call_impl
    return execution_with_hooks(self, ...)(self._call_impl)(*args, **kw)
```

### 5. `_call_impl`: Cache Check + Call Stack Management

```python
def _call_impl(self, *args, **kw):
    # Check cache
    if self._use_cache:
        try: return module_cache.get(...)
        except CacheNotFoundError: ...

    # Push call stack, execute forward
    with globals.stack_enter(self.identities):
        return self.forward(*args, **kw)
```

### 6. `forward()`: Subclass Implements Inference Logic

**All subclasses must implement the `forward` method** — it is the only core method that needs to be implemented.

### 7. `update()`: Train + Deploy + Evaluate

```python
module.update()  # triggers the full train → deploy → eval pipeline
```

---

## Builder Pattern (Chained Configuration API)

`builder_keys` defines configuration items that support chained calls:

```python
class TrainableModule(ModuleBase):
    builder_keys = ['trainset', 'train_method', 'finetune_method', 'deploy_method', ...]
```

Corresponding setter methods are auto-generated via `__getattr__`:

```python
model = TrainableModule('qwen2-1.5b') \
    .trainset('/path/to/data') \
    .finetune_method(lazyllm.finetune.llamafactory) \
    .deploy_method(lazyllm.deploy.vllm)
```

**When adding a new Module**, add parameters that need chained configuration to `builder_keys`; never implement setter methods manually.

---

## `_module_id` and Session Isolation

`_module_id` (i.e. `_config_id`) is the module's unique identifier, used to index session data in `globals`:

```python
# Read the conversation history for this module
history = globals['chat_history'].get(self._module_id, [])

# Read global parameters for this module
params = globals['global_parameters'].get(self._module_id, {})
```

`share()` generates a new `_module_id`, giving the shared copy independent session state:

```python
shared_module = original_module.share()  # new ID, independent session
```

**Convention:** Never store session state in instance variables; always access via `globals[key][self._module_id]`.

---

## ServerModule

Deploys any callable as a FastAPI service:

```python
server = ServerModule(my_function)
server.start()       # start the HTTP service
result = server("input")  # call via HTTP
```

### Distributed Data Flow

`ServerModule` carries session context in HTTP request headers:
- `Global-Parameters`: serialized global parameters (`globals` data)
- `Session-ID`: session ID

The server side restores the session context via `decode_request`, ensuring session data consistency in distributed scenarios.

### UrlModule

Wraps the URL of an already-deployed service as a Module:

```python
url_module = UrlModule(url='http://localhost:8080')
result = url_module("input")
```

---

## TrainableModule

An all-in-one Module for local model fine-tuning and deployment.

### Impl Separation Pattern

`TrainableModule` uses `_TrainableModuleImpl` internally to hold the actual deployment state:

```python
class TrainableModule(ModuleBase):
    def __init__(self, ...):
        self._impl = _TrainableModuleImpl(...)  # holds deployment state

    def __deepcopy__(self, memo):
        return self  # deepcopy returns self, avoiding redundant deployments
```

`__deepcopy__` returning `self` is a key design decision: when `TrainableModule` is embedded in a Flow and `deepcopy`-ed, no new deployment instance is created.

### Lifecycle

```python
model = TrainableModule('qwen2-1.5b') \
    .trainset('/data/train.json') \
    .finetune_method(lazyllm.finetune.llamafactory) \
    .deploy_method(lazyllm.deploy.vllm) \
    .mode('finetune')  # 'finetune' / 'deploy' / 'finetune+deploy'

model.start()   # deploy only (skip fine-tuning)
model.update()  # fine-tune + deploy + evaluate
```

---

## ActionModule

Wraps a function, Flow, or Module as a Module:

```python
# Wrap a function
m = ActionModule(lambda x: x.upper())

# Wrap a Flow
m = ActionModule(Pipeline(module1, module2))
```

Use case: when you need to unify non-Module objects under the Module interface.

---

## TrialModule

Hyperparameter grid search — iterates over all `Option` combinations for fine-tuning/deployment/evaluation:

```python
trial = TrialModule(
    TrainableModule('qwen2-1.5b').trainset(data),
    options={'lr': Option([0.001, 0.01]), 'epochs': Option([3, 5])}
)
trial.update()  # iterates over 2×2=4 combinations
```

---

## Conventions for Adding a New Module

1. Inherit `ModuleBase`
2. Implement `forward(self, *args, **kw)`
3. Never store session state in instance variables
4. Add parameters that need chained configuration to `builder_keys`
5. For submodule tracking, assign via `self.xxx = <ModuleBase instance>` (auto-tracked)
6. If deployment is needed, implement `_get_deploy_tasks()`
7. If training is needed, implement `_get_train_tasks()`
8. Refer to `ActionModule` (minimal implementation) or `TrainableModule` (full implementation) for alignment

---

## Prohibited Patterns

- **Never** modify session state outside `forward` (`globals['chat_history']` etc. are managed by the framework)
- **Never** call `start()` inside `__init__` (deployment should be triggered explicitly by the user)
- **Never** override `__call__` (session injection and Hook logic live in `__call__`; subclasses only implement `forward`)
- **Never** create a new deployment instance in `__deepcopy__` (refer to `TrainableModule`'s approach)
