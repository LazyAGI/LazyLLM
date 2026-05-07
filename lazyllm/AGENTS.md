# LazyLLM Framework AGENTS.md

## Mandatory Pre-coding Behavior

**Before modifying any file, you must complete the following reading steps:**

1. Read this file (`lazyllm/AGENTS.md`)
2. Read the target submodule's `AGENTS.md` (if it exists)
3. Read the target file itself and its direct dependencies
4. Find 1–2 existing implementations with similar functionality in the same directory as alignment references

**Prohibited behaviors:**
- Do not write code from memory or external experience without reading the code first
- Do not introduce new design patterns not already used in the codebase
- Do not duplicate logic — repeated capabilities must be extracted into shared modules
- Do not invent new abstraction layers; reuse existing base classes and utilities first

---

## Repository Layer Architecture

```
lazyllm/
├── common/          # Foundation layer: registry, Bind, session management, logging, utilities
├── configs.py       # Config center: lazyllm.config global singleton
├── launcher/        # Task launchers: empty / slurm / sco / k8s
├── flow/            # Orchestration layer: Pipeline / Parallel / Loop / Switch / Graph
├── components/      # Model component layer: Prompter / Formatter / Deploy backends / Finetune backends
├── module/          # Module abstraction layer: ModuleBase / ServerModule / TrainableModule / OnlineModule
├── tools/           # High-level tools (lazy-loaded): RAG / Agent / Memory / MCP / WebModule etc.
├── engine/          # Lightweight engine: LightEngine (internal scheduler, rarely extended)
├── tracing/         # Distributed tracing (lazy-loaded, standalone subsystem)
├── docs/            # Public API doc registration (add_chinese_doc / add_english_doc)
└── prompt_templates/ # Prompt template library
```

**Unidirectional dependency rules (strictly enforced, no reverse references):**

```
configs ← common ← launcher ← components ← module ← tools ← engine
                 ←   flow   ←
```

- `common` must not reference any layer above it except `configs`
- `flow` and `launcher` are peers; they must not reference each other, `module`, or `tools`
- `components` must not reference `module` or `tools`
- `module` may reference `flow` and `components`, but not `tools`
- `tools` may reference all layers below it

---

## Lazy-loading Mechanism

`tools` and `tracing` are lazy-loaded via `__getattr__` in `lazyllm/__init__.py`, avoiding heavy imports at startup:

```python
_LAZY_SUBMODS = ('tracing', 'tools')

def __getattr__(name: str):
    for submod in _LAZY_SUBMODS:
        mod = importlib.import_module(f'.{submod}', package=__package__)
        if name in getattr(mod, '__all__', ()):
            return getattr(mod, name)
```

`tools/__init__.py` also lazy-loads its submodules via `__getattr__`.

**When adding a new `tools` submodule, you must:**
1. Register the submodule name and exported symbols in `_SUBMOD_MAP` in `tools/__init__.py`
2. Add type imports in the `TYPE_CHECKING` block (for IDE completion)
3. Never `import` directly at the top level of `tools/__init__.py`

**When introducing a new third-party dependency, it must go through `lazyllm/thirdparty/` lazy-loading:**

```python
# Packages registered in thirdparty/__init__.py are automatically lazy-loaded
from lazyllm.thirdparty import numpy  # Only imported on first access; gives friendly pip install hint on failure

# To add a new third-party dependency: add a mapping in thirdparty/__init__.py's package_name_map
# e.g. 'yaml' → 'pyyaml', then access via PackageWrapper
```

---

## Core Abstractions Overview

See each submodule's `AGENTS.md` for details. Key conventions are listed here.

### ModuleBase (`module/AGENTS.md`)

Base class for all executable units. Key conventions:
- Implement `forward()` to define inference logic
- `self.xxx = <ModuleBase instance>` automatically registers it as a submodule (`__setattr__` intercepts)
- Never store session state in instance variables; use `lazyllm.globals` indexed by `_module_id`

### Registry System (`common/AGENTS.md`)

- Class names following `LazyLLMXxxBase` **with `metaclass=LazyLLMRegisterMetaClass` or `LazyLLMRegisterMetaABCClass`** → automatically creates the `lazyllm.xxx` namespace
- Subclasses inheriting that Base → automatically registered (no manual registration needed)
- Always inherit the corresponding Base class when extending; never bypass the registry system

### Flow Orchestration (`flow/AGENTS.md`)

Flow is LazyLLM's **application orchestration layer**, suited for complex applications that need one-command deployment (multi-model collaboration, RAG pipelines, Agent loops, etc.). **For simple data processing or single-step calls, prefer plain Python functions — do not overuse Flow.**

Main Flow types:
- `Pipeline`: sequential execution, each step's output feeds the next
- `Parallel`: all components share the input, results are merged after concurrent execution
- `Loop`: repeated execution until a stop condition is met (core of Agent loops)
- `Switch`: conditional routing, selects a branch based on input value
- `Warp`: applies a single module to multiple inputs in parallel (variant of `Parallel`)
- `bind(_0, p.input)`: cross-step reference mechanism

Flow must be stateless; never store execution state outside `_run`.

### LLM Usage (`module/AGENTS.md`, `module/llms/onlinemodule/AGENTS.md`)

LazyLLM provides a unified LLM interface — local and online models share the same calling experience:

**Online models (recommended for rapid prototyping):**
```python
# Automatically routes to Chat/Embedding/MultiModal based on model name
llm = lazyllm.OnlineModule('qwen-plus')                    # Alibaba Qwen
llm = lazyllm.OnlineModule('gpt-4o')                       # OpenAI
llm = lazyllm.OnlineModule('text-embedding-3-small')       # Embedding model

# Unified calling convention
result = llm('Hello, please introduce LazyLLM')
```

**Local models (recommended for production deployment):**
```python
# Deploy a local model with vLLM
llm = lazyllm.TrainableModule('qwen2-7b-instruct').deploy_method(lazyllm.deploy.vllm)
llm.start()  # Start deployment

# Unified calling convention (identical to online models)
result = llm('Hello, please introduce LazyLLM')
```

**Key conventions:**
- Both local and online models inherit `ModuleBase` and share the same `forward()` interface
- `AutoModel` enables seamless switching between local and online
- Never differentiate local vs. online model call patterns in business code

### Prompter (`components/AGENTS.md`)

- Never hardcode prompt strings; inject via `Prompter`
- Special tokens for tool calls are injected via model config, never hardcoded

---

## Minimum Code Principle

One of LazyLLM's design goals is to **implement complex applications with minimal code**. The following patterns are distilled from real code — new code should reuse them first.

### 1. Chained Builder (`builder_keys` mechanism)

`ModuleBase` uses `__getattr__` to turn `builder_keys` entries into setters that return `self`, enabling unlimited chaining:

```python
# Each .method() call returns self and can be chained indefinitely
model = TrainableModule('Qwen2.5-32B-Instruct') \
    .mode('finetune') \
    .trainset('/data/sft.json') \
    .finetune_method((lazyllm.finetune.llamafactory, {'learning_rate': 1e-4})) \
    .deploy_method(lazyllm.deploy.vllm) \
    .prompt(lazyllm.ChatPrompter('You are a helpful assistant.'))
```

`.prompt()`, `.formatter()`, and `.share()` are also chainable methods and can be combined freely.

### 2. `with pipeline() as ppl:` + assignment-as-registration

Inside a `with` block, assigning to `ppl.xxx` automatically registers steps in order. **Plain functions and lambdas can be assigned directly — no need to wrap them in a Module:**

```python
with pipeline() as ppl:
    ppl.retriever = Retriever(doc, 'CoarseChunk', 'bm25_chinese', topk=3)
    ppl.formatter = lambda nodes, q: dict(context_str='\n'.join(n.get_content() for n in nodes), query=q)
    ppl.llm = OnlineChatModule().prompt(ChatPrompter(prompt, extra_keys=['context_str']))
```

### 3. `| bind(key=ppl.input)` to inject extra arguments

The `|` operator combines a function with `bind`, letting a step receive the previous step's output while also referencing the pipeline's original input or another step's output:

```python
with pipeline() as ppl:
    ppl.retriever = Retriever(doc, 'sentences', 'cosine', topk=3)
    # reranker receives retriever's output and also gets the original query via bind
    ppl.reranker = Reranker('ModuleReranker', model=rerank_model, topk=1) | bind(query=ppl.input)
    # formatter gets both reranker output and original query
    ppl.formatter = (lambda nodes, q: dict(context_str=nodes, query=q)) | bind(query=ppl.input)
```

### 4. `parallel().sum` — concurrent execution with automatic merging

`parallel().sum` runs multiple steps concurrently and concatenates the results (`sum` = list concat):

```python
with pipeline() as ppl:
    with parallel().sum as ppl.prl:
        ppl.prl.ret1 = Retriever(doc, 'sentences', 'cosine', topk=3)
        ppl.prl.ret2 = Retriever(doc, 'CoarseChunk', 'bm25_chinese', topk=3)
    ppl.reranker = Reranker(...) | bind(query=ppl.input)
```

Other post-processors: `.asdict` (`{step_name: result}`), `.astuple`, `.aslist`, `.join('sep')`.

### 5. `.share()` — reuse model weights

`share()` creates a new instance sharing the same underlying deployment, with independent prompt/formatter settings, **avoiding redundant deployments of the same model:**

```python
base = TrainableModule('internlm2-chat-7b')
chat_agent   = base.share().prompt(chat_prompt)
painter_agent = base.share().prompt(painter_prompt)
coder_agent  = base.share().prompt(coder_prompt)
# All three agents share one vLLM process, deployed only once
```

### 6. `switch.case` instead of if-else routing

```python
with switch(judge_on_full_input=False).bind(_0, ppl.input) as ppl.sw:
    ppl.sw.case[intent_list[0], chat_module]
    ppl.sw.case[intent_list[1], TrainableModule('SenseVoiceSmall')]
    ppl.sw.case[intent_list[2], pipeline(base.share().prompt(painter_prompt),
                                          TrainableModule('stable-diffusion-3-medium'))]
```

When `judge_on_full_input=False`, the first element of the input is used as the condition; the rest is passed to the matched branch.

### 7. `warp` for batch processing

`warp` turns a single module into a batch-processing version that applies to each element of a list concurrently:

```python
# Apply writer to each outline in outline_list concurrently
ppl.story_generator = warp(base.share().prompt(writer_prompt).formatter())
```

### 8. `ActionModule` — deploy a Flow as a Module

```python
# start() deploys the entire pipeline (auto-deploys all child Modules)
app = ActionModule(ppl).start()
result = app('query')

# Or wrap as a web service
WebModule(ppl, port=range(23466, 24000)).start().wait()
```

### 9. Complete RAG application (~10 lines of core code)

```python
documents = Document('./data', embed=OnlineEmbeddingModule(), manager=False)
documents.create_node_group('sentences', transform=SentenceSplitter, chunk_size=1024)

with pipeline() as ppl:
    with parallel().sum as ppl.prl:
        ppl.prl.ret1 = Retriever(documents, 'sentences', 'cosine', topk=3)
        ppl.prl.ret2 = Retriever(documents, 'CoarseChunk', 'bm25_chinese', topk=3)
    ppl.reranker = Reranker('ModuleReranker', model=rerank_model, topk=1,
                             output_format='content', join=True) | bind(query=ppl.input)
    ppl.llm = OnlineChatModule().prompt(ChatPrompter(prompt, extra_keys=['context_str']))

ActionModule(ppl).start()
```

### 10. Complete Agent (~5 lines)

```python
@fc_register('tool')
def web_search(query: str) -> str:
    '''Search the web. Args: query (str): search query.'''
    return do_search(query)

agent = ReactAgent(OnlineChatModule(), tools=[web_search, calculator])
result = agent('Search for the latest version of LazyLLM and calculate 123*456')
```

**Principle:** If it takes more than 20 lines to implement a complete feature, first check whether there is a reusable existing module or a more concise API. Refer to [`docs/zh/Cookbook/`](../docs/zh/Cookbook/) for complete examples.

---

## Global Code Style Conventions

| Convention | Rule |
|------------|------|
| Quotes | Prefer single quotes `'`; use `'''` for multi-line strings |
| Line length | Maximum 121 characters |
| Docstrings | **Strictly forbidden** inside code (`'''...'''` or `"""..."""`) |
| Documentation | All public API docs must be registered in `lazyllm/docs/` via `add_chinese_doc` / `add_english_doc` |
| Logging | Use `lazyllm.LOG`; never use `print` |
| Config | Register via `lazyllm.config.add(key, type, default, env='LAZYLLM_XXX')`; never read env vars directly |
| Imports | Prefer relative imports within the package; all imports at the top of the file |
| Comments | Only comment non-obvious logic; never comment self-evident operations |

---

## Documentation System Conventions

Public API documentation is **never written inside code**; it is registered in the corresponding file under `lazyllm/docs/`.
Refer to the existing patterns in [`lazyllm/docs/common.py`](docs/common.py), [`lazyllm/docs/module.py`](docs/module.py), and [`lazyllm/docs/flow.py`](docs/flow.py):

```python
# lazyllm/docs/module.py
from lazyllm.docs import add_chinese_doc, add_english_doc, add_example

add_chinese_doc('ClassName', '''Chinese documentation content''')
add_english_doc('ClassName', '''English doc content''')
add_example('ClassName', '''
>>> import lazyllm
>>> m = lazyllm.ClassName(...)
''')
```

Subclasses with unchanged behavior do not repeat their parent's documentation.

---

## Global Prohibited Patterns

1. **Never store session state in Module instance variables**
   - Wrong: `self.history = []` (causes cross-user contamination under concurrent requests)
   - Correct: `lazyllm.globals['chat_history'][self._module_id]`

2. **Never hardcode prompt strings**
   - Wrong: `prompt = "You are a helpful assistant. User: " + query`
   - Correct: inject via `ChatPrompter` or `AlpacaPrompter`

3. **Never bypass the registry system**
   - Wrong: directly `import` and instantiate a concrete backend class
   - Correct: access via `lazyllm.finetune.llamafactory` or `lazyllm.deploy.vllm`

4. **Never introduce new third-party dependencies without lazy-loading**
   - Wrong: `import heavy_library` at module top level
   - Correct: register in `lazyllm/thirdparty/` and access via `PackageWrapper`

5. **Never duplicate existing logic**
   - When duplicate logic is found, extract it into a shared utility in `common/`

6. **Never write docstrings inside `lazyllm/` package code**
   - All documentation is maintained in `lazyllm/docs/`

---

## Submodule AGENTS.md Navigation

| Submodule | AGENTS.md Location | Core Content |
|-----------|-------------------|--------------|
| common | `lazyllm/common/AGENTS.md` | Registry system, Bind mechanism, globals/locals session management |
| flow | `lazyllm/flow/AGENTS.md` | Pipeline/Parallel/Loop/Switch/Graph data flow and extension |
| module | `lazyllm/module/AGENTS.md` | ModuleBase lifecycle, ServerModule, TrainableModule |
| onlinemodule | `lazyllm/module/llms/onlinemodule/AGENTS.md` | Online model provider extension conventions |
| components | `lazyllm/components/AGENTS.md` | Prompter, Formatter, Deploy backends, Finetune backends |
| tools | `lazyllm/tools/AGENTS.md` | Lazy-loading mechanism, submodule navigation, ModuleTool, Memory, MCP |
| rag | `lazyllm/tools/rag/AGENTS.md` | Full RAG pipeline: Reader/Transform/Store/Retriever/Reranker |
| agent | `lazyllm/tools/agent/AGENTS.md` | Agent types, tool registration, FunctionCall execution flow |
