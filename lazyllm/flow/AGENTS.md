# lazyllm/flow AGENTS.md

This directory implements LazyLLM's **orchestration layer**, providing Flow types including `Pipeline`, `Parallel`, `Loop`, `Switch`, `IFS`, `Warp`, and `Graph`.

Reference docs: [`lazyllm/docs/flow.py`](../docs/flow.py)

## Mandatory Pre-reading

Before modifying this directory, you must read:
- `lazyllm/AGENTS.md` (global conventions)
- `lazyllm/common/AGENTS.md` (Bind mechanism, session management)
- `lazyllm/flow/flow.py` (complete implementation)

---

## When to Use Flow vs. Plain Python

**Prefer plain Python** (regular functions/classes) when:
- Simple data processing or single-step calls
- No need for distributed deployment or multi-model collaboration
- Logic is clear and does not require visualization

**Use Flow** when:
- You need **one-command deployment** of a complex application (`pipeline.start()` auto-deploys all child modules)
- Multi-model collaboration (RAG pipelines, Agent loops, multi-path retrieval fusion)
- You need `bind` to reference intermediate results across steps
- You need concurrent execution of multiple branches (`Parallel`)

```python
# Plain Python (simple scenario, preferred)
def process(query):
    result = llm(query)
    return formatter(result)

# Flow (complex scenario, needs deployment)
with pipeline() as ppl:
    ppl.llm = llm
    ppl.formatter = formatter
ppl.start()  # auto-deploys llm
```

---

## Class Hierarchy

```
FlowBase (SessionConfigableBase, metaclass=_MetaBind)
  └── LazyLLMFlowsBase (+ LazyLLMRegisterMetaClass)
        ├── Pipeline
        │     └── Loop
        ├── Parallel
        │     ├── Diverter  (_scatter=True)
        │     └── Warp      (_scatter=True, maps a single module over multiple inputs)
        ├── Switch
        ├── IFS
        └── Graph
```

- `FlowBase`: manages the `_items` list, context manager (`with` syntax), `__setattr__` interception
- `LazyLLMFlowsBase`: adds `__call__`, `invoke`, `post_action`, Hook support
- Concrete Flow classes: implement the `_run` method

---

## Data Flow Models

### Pipeline

```
input → step1 → step2 → ... → stepN → output
                                    ↘ post_action (optional, does not affect output)
```

- Each step's output feeds the next step's input
- Each step's result is stored in `locals['bind_args'][pipeline_id][step_id]` for `bind` references
- Intermediate results are only saved when `save_flow_result=True` (or `config['save_flow_result']=True`)

### Parallel

```
         ┌→ module1 → out1 ┐
input ───┼→ module2 → out2 ├→ package(out1, out2, out3)
         └→ module3 → out3 ┘
```

- Executes concurrently by default (`ThreadPoolExecutor`); set `_concurrent=False` for sequential
- Post-processing strategies (set via attribute):
  - `.asdict`: `{name: value}` dict
  - `.astuple`: `(out1, out2, ...)` tuple
  - `.aslist`: `[out1, out2, ...]` list
  - `.sum`: concatenates all outputs (list concat, or numeric/string sum)
  - `.join(sep)`: joins string outputs with a separator

### Diverter (variant of Parallel)

```
(in1, in2, in3) → module1(in1), module2(in2), module3(in3) → package(out1, out2, out3)
```

Multiple inputs are routed to different modules respectively; the number of inputs must match the number of modules.

### Warp (variant of Parallel)

```
(in1, in2, in3) → module(in1), module(in2), module(in3) → package(out1, out2, out3)
```

A single module is applied to multiple inputs in parallel.

### Loop

Inherits from `Pipeline`; controls iteration via `_loop_count` and `_stop_condition`:

```python
Loop(module, stop_condition=lambda x: isinstance(x, str), count=10)
```

- `stop_condition`: checked after each iteration; stops when it returns `True`
- `count`: maximum number of iterations (default `sys.maxsize`)
- When `_judge_on_full_input=False`: output is `(judge_value, actual_output)`; only `judge_value` is checked against the stop condition

### Switch

```python
Switch({
    cond1: module1,
    cond2: module2,
    'default': module_default,
})
```

- `conversion` parameter: transforms the input before comparing against conditions
- Supports three construction styles: dict, alternating positional args, or `switch.case[cond::func]` dynamic addition

### IFS

```python
IFS(condition_func, true_module, false_module)
```

Two-way conditional branch: `condition_func(input)` returning `True` routes to `true_module`, otherwise to `false_module`.

### Graph

Directed acyclic graph supporting complex dependency relationships:

```python
with Graph() as g:
    g.node1 = module1
    g.node2 = module2
    g.node3 = module3
    Graph.edge(g.node1, g.node3)
    Graph.edge(g.node2, g.node3)
```

---

## Construction Patterns

### Positional arguments

```python
p = Pipeline(module1, module2, module3)
```

### Keyword arguments (named steps)

```python
p = Pipeline(step1=module1, step2=module2)
```

### Context manager (recommended — supports `bind` references)

```python
with pipeline() as p:
    p.step1 = module1
    p.step2 = module2
    p.step3 = bind(module3, p.input)  # reference the pipeline's original input
```

Inside the `with` block, `p.xxx = value` triggers `__setattr__` interception, which automatically calls `_add(name, value)` to append the step to `_items`.

### Nested Flows

```python
with pipeline() as outer:
    with parallel() as outer.prl:
        outer.prl.branch1 = module1
        outer.prl.branch2 = module2
    outer.step2 = module3
```

When nested, the inner Flow is automatically registered into the outer Flow's `_items` when its `with` block exits.

---

## How `bind` Is Resolved in a Flow

```
bind(module, p.input, p.output(p.step1))
    ↓ at call time
invoke(it, __input, bind_args_source=bind_args_source)
    ↓
Bind.Args in it._args are resolved from locals['bind_args'][pipeline_id]
    ↓
p.input → bind_args_source['input']  (pipeline's original input)
p.output(p.step1) → bind_args_source[step1_id]  (step1's output)
```

**Key conventions:**
- `p.input`: references the pipeline's original input
- `p.output(step)`: references a step's output (`step` can be the step object or its name string)
- `p.kwargs`: references the pipeline's keyword arguments
- `_0, _1`: positional placeholders, replaced by position at call time

---

## `_FuncWrap` and Hooks

Plain functions (lambdas, function objects) are wrapped in `_FuncWrap` when added to a Flow:

```python
def _make_step_item(self, v):
    if _is_function(v) or v in self._items:
        return _FuncWrap(v)
    return v
```

`_FuncWrap` supports the Hook mechanism (`execution_with_hooks`), consistent with `ModuleBase`'s Hook mechanism.

`builtins.isinstance` is monkey-patched so that `isinstance(_FuncWrap(f), type(f))` returns `True`, preserving type transparency.

---

## Extension Conventions

### Adding a New Flow Type

1. Inherit `LazyLLMFlowsBase`
2. Implement `_run(self, __input, **kw)`
3. **Flow must be stateless**: never store execution state outside `_run`; all intermediate data is passed via `locals['bind_args']`
4. Refer to `Pipeline._run` as the implementation reference

```python
class MyFlow(LazyLLMFlowsBase):
    def _run(self, __input, **kw):
        output = __input
        for it in self._items:
            output = self.invoke(it, output, **kw)
        return output
```

### Adding a New Post-processing Strategy

Add a new enum value to `Parallel.PostProcessType` and handle it in `_post_process`.

### Prohibited Patterns

- **Never** modify `self._items` inside `_run` (a Flow is immutable after construction)
- **Never** store execution state on the Flow instance (e.g. `self.last_output = ...`)
- **Never** call `invoke` outside `_run` (`invoke` contains error handling and bind resolution logic)
- **Never** directly manipulate `locals['bind_args']` (managed automatically by `Pipeline._run`)

---

## Common Usage Patterns

### RAG Retrieval + Reranking

```python
with pipeline() as ppl:
    with parallel().sum as ppl.prl:
        ppl.prl.ret1 = Retriever(doc, 'sentences', 'cosine', topk=3)
        ppl.prl.ret2 = Retriever(doc, 'CoarseChunk', 'bm25_chinese', topk=3)
    ppl.reranker = bind(Reranker(...), query=ppl.input)
    ppl.llm = OnlineChatModule().prompt(ChatPrompter(...))
```

### Agent Loop

```python
agent_loop = Loop(
    FunctionCall(llm=llm, tools=tools),
    stop_condition=lambda x: isinstance(x, str),
    count=10,
)
```

### Conditional Routing

```python
router = Switch(
    conversion=classify_intent,
    cases={
        'rag': rag_pipeline,
        'chat': chat_module,
        'default': fallback_module,
    }
)
```
