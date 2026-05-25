# lazyllm/components AGENTS.md

This directory is LazyLLM's **model component layer**, providing four types of components: Prompter (prompt management), Formatter (output formatting), Deploy backends (inference deployment), and Finetune backends (fine-tuning).

Reference docs: [`lazyllm/docs/components.py`](../docs/components.py) (complete API docs for Prompter, Formatter, etc.)

## Mandatory Pre-reading

Before modifying this directory, you must read:
- `lazyllm/AGENTS.md` (global conventions)
- `lazyllm/common/AGENTS.md` (registry system)
- The existing implementation in the target subdirectory (e.g. read `deploy/vllm.py` before modifying a Deploy backend)

---

## Directory Structure

```
components/
├── prompter/          # Prompt components
│   ├── prompter.py        # PrompterBase (base class)
│   ├── chatPrompter.py    # ChatPrompter
│   ├── alpacaPrompter.py  # AlpacaPrompter
│   └── builtinPrompt.py   # Built-in prompt utility functions
├── formatter/         # Formatting components
│   ├── formatterbase.py   # LazyLLMFormatterBase
│   ├── jsonformatter.py   # JsonFormatter / JsonLikeFormatter
│   └── yamlformatter.py   # YamlFormatter
├── deploy/            # Inference deployment backends
│   ├── base.py            # LazyLLMDeployBase
│   ├── vllm.py            # Vllm
│   ├── lmdeploy.py        # LMDeploy
│   ├── lightllm.py        # LightLLM
│   ├── infinity.py        # Infinity (Embedding)
│   ├── embed.py           # Embedding (transformers)
│   └── ...
├── finetune/          # Fine-tuning backends
│   ├── base.py            # LazyLLMFinetuneBase
│   ├── llamafactory.py    # LLamaFactory
│   ├── alpacalora.py      # AlpacaLora
│   ├── collie.py          # Collie
│   └── flagembedding.py   # FlagEmbedding
└── auto/              # Auto-selection
    ├── autodeploy.py      # AutoDeploy
    └── autofinetune.py    # AutoFinetune
```

---

## Prompter Component

### Class Hierarchy

```
PrompterBase (base class, not used directly)
  ├── EmptyPrompter    # Passes input through unchanged
  ├── ChatPrompter     # General conversation template; supports multi-turn history and tool calls
  └── AlpacaPrompter   # Alpaca format; suited for local model fine-tuning; no history support
```

### Template Variable Conventions (`ChatPrompter`)

`ChatPrompter` uses a fixed template string; model-specific special tokens are injected via `_set_model_configs`:

```
{sos}           system prompt start token
{eos}           system prompt end token
{soh}           user message start token
{eoh}           user message end token
{soa}           assistant message start token
{eoa}           assistant message end token
{system}        system prompt content
{instruction}   instruction content
{tools}         tool descriptions (JSON)
{history}       conversation history
{input}         current user input
```

**Tool call tokens are injected via model config — never hardcode them:**
```python
# Correct: inject via _set_model_configs
prompter._set_model_configs(
    system='<|system|>',
    tool_start_token='<tool_call>',
    tool_end_token='</tool_call>',
)

# Wrong: hardcoded tokens
prompt = '<tool_call>' + json.dumps(tool) + '</tool_call>'
```

### Three Output Formats

The `format` parameter of `generate_prompt(input, history, tools, format)`:

| format | Output type | Use case |
|--------|-------------|----------|
| `None` | String | Local deployment (direct concatenation) |
| `'openai'` | OpenAI messages list | Online model API |
| `'anthropic'` | Anthropic format (separate `system` field) | Claude API |

### Usage Conventions

```python
# Correct: inject via Prompter
module = TrainableModule('qwen2-1.5b').prompt(
    ChatPrompter('You are a helpful assistant. {instruction}')
)

# Wrong: hardcoded prompt
def forward(self, query):
    return self.llm(f'You are a helpful assistant.\nUser: {query}\nAssistant:')
```

### Adding a New Prompter

Inherit `PrompterBase` and implement `_generate_prompt_impl`:

```python
class MyPrompter(PrompterBase):
    def __init__(self, instruction=None):
        super().__init__(instruction=instruction)
        self._init_prompt('{instruction}\n{input}', instruction)

    def _generate_prompt_impl(self, input, history=None, tools=None, label=None):
        ...
```

---

## Formatter Component

### Class Hierarchy

```
LazyLLMFormatterBase (metaclass=LazyLLMRegisterMetaClass)
  ├── EmptyFormatter         # Pass-through
  ├── JsonLikeFormatter      # General JSON/Python data slicing
  │     ├── PythonFormatter  # Same as JsonLikeFormatter
  │     └── JsonFormatter    # Additionally parses string → JSON
  ├── FunctionCallFormatter  # Extracts role/content/tool_calls
  ├── FileFormatter          # encode/decode/merge file paths
  └── PipelineFormatter      # Combines multiple formatters
```

### `JsonLikeFormatter` Slice Syntax

```python
JsonFormatter('[0]')           # data[0]
JsonFormatter('[0:3]')         # data[0:3]
JsonFormatter('[0,2,4]')       # [data[0], data[2], data[4]]
JsonFormatter('[{key1,key2}]') # {key1: data[key1], key2: data[key2]}
JsonFormatter('*[0,1]')        # package(data[0], data[1]) (for multi-input pipeline)
```

### `|` Operator Composition

```python
formatter = JsonFormatter('[0]') | JsonFormatter('[{name}]')
# equivalent to PipelineFormatter(Pipeline(f1, f2))
```

### `__ror__` Integration with Pipeline

A Formatter can be appended directly to the end of a Pipeline:

```python
pipeline | JsonFormatter('[0]')
# equivalent to adding the formatter as a step at the end of the pipeline
```

### Adding a New Formatter

Inherit `LazyLLMFormatterBase` and implement `_parse_py_data_by_formatter`:

```python
class MyFormatter(LazyLLMFormatterBase):
    def _load(self, msg: str):
        # Parse string into a Python object (optional; returns string by default)
        return json.loads(msg)

    def _parse_py_data_by_formatter(self, py_data):
        # Process the Python object
        return py_data['result']
```

---

## Deploy Backend

### Base Class Interface (`LazyLLMDeployBase`)

```python
class LazyLLMDeployBase(ComponentBase):
    keys_name_handle = None    # Request field mapping (input field name → API field name)
    message_format = None      # Request body template
    default_headers = {'Content-Type': 'application/json'}
    stream_url_suffix = ''     # URL suffix for streaming requests

    def __init__(self, *, launcher=launchers.remote()):
        ...

    def cmd(self, *args, **kw) -> str:
        # Returns the command-line string to start the inference service (subclass must implement)
        raise NotImplementedError
```

### Adding a New Deploy Backend

1. Inherit `LazyLLMDeployBase`
2. Implement `cmd(self, *args, **kw) -> str` returning the startup command
3. Set `keys_name_handle` (request field mapping) and `message_format` (request body template)
4. Refer to `deploy/vllm.py` or `deploy/lmdeploy.py` for alignment

```python
class MyDeploy(LazyLLMDeployBase):
    keys_name_handle = {'inputs': 'prompt'}
    message_format = {'prompt': '', 'max_tokens': 512}

    def __init__(self, launcher=launchers.remote(), **kw):
        super().__init__(launcher=launcher)
        self.kw = kw

    def cmd(self, finetuned_model=None, base_model=None) -> str:
        model = finetuned_model or base_model
        return f'python -m myframework.server --model {model}'
```

### `AutoDeploy` Auto-selection Logic

`AutoDeploy.get_deployer` automatically selects a backend based on model type:

| Model type | Default backend |
|------------|----------------|
| `embed` / `rerank` | `Infinity` or `Embedding` (depending on available dependencies) |
| `sd` | `StableDiffusionDeploy` |
| `stt` | `SenseVoiceDeploy` |
| `tts` | `TTSDeploy` |
| `vlm` | `LMDeploy` |
| `llm` (small model ≤7B) | `LightLLM` |
| `llm` (large model >7B) | `vLLM` or `LMDeploy` |

**When adding a new backend**, if it should be included in `AutoDeploy`'s auto-selection, add the corresponding logic to `get_deployer` in `auto/autodeploy.py`.

---

## Finetune Backend

### Base Class Interface (`LazyLLMFinetuneBase`)

```python
class LazyLLMFinetuneBase(ComponentBase):
    __reg_overwrite__ = 'cmd'  # method name overwritten by the registry system

    def __init__(self, base_model, target_path, *, launcher=launchers.remote()):
        ...

    def cmd(self, *args, **kw) -> str:
        # Returns the command-line string to start fine-tuning (subclass must implement)
        raise NotImplementedError
```

### Adding a New Finetune Backend

1. Inherit `LazyLLMFinetuneBase`
2. Implement `cmd(self, *args, **kw) -> str` returning the fine-tuning command
3. Refer to `finetune/llamafactory.py` for alignment

```python
class MyFinetune(LazyLLMFinetuneBase):
    def __init__(self, base_model, target_path, *, launcher=launchers.remote(), **kw):
        super().__init__(base_model, target_path, launcher=launcher)
        self.kw = kw

    def cmd(self, trainset, valset=None) -> str:
        return (f'python -m myframework.train '
                f'--model {self.base_model} '
                f'--output {self.target_path} '
                f'--data {trainset}')
```

### `AutoFinetune` Auto-selection Logic

`AutoFinetune` automatically selects a fine-tuning framework based on model type and available dependencies. Priority: `llamafactory` > `alpacalora` > `collie`.

---

## Prohibited Patterns

- **Never** hardcode model-specific special tokens in a Prompter
- **Never** read environment variables directly in the `cmd` method (access via `lazyllm.config`)
- **Never** implement `forward` in a Deploy/Finetune backend (managed uniformly by `ComponentBase`)
- **Never** skip the `super().__init__()` call
