# lazyllm/module/llms/onlinemodule AGENTS.md

This directory implements LazyLLM's **online model provider** support, covering three interface types: Chat, Embedding, and MultiModal.

## Mandatory Pre-reading

Before modifying this directory, you must read:
- `lazyllm/AGENTS.md` (global conventions)
- `lazyllm/module/AGENTS.md` (ModuleBase lifecycle)
- `lazyllm/module/llms/onlinemodule/base/onlineChatModuleBase.py` (complete Chat Base implementation)
- `lazyllm/module/llms/onlinemodule/base/utils.py` (registry hook, automatic config registration)
- An existing provider implementation of the same type (e.g. `supplier/openai.py` or `supplier/qwen.py`)

---

## Directory Structure

```
onlinemodule/
├── base/
│   ├── onlineChatModuleBase.py      # LazyLLMOnlineChatModuleBase (Chat Base)
│   ├── onlineEmbeddingModuleBase.py # LazyLLMOnlineEmbedModuleBase (Embedding Base)
│   ├── onlineMultiModalBase.py      # LazyLLMOnlineMultiModalBase (MultiModal Base)
│   └── utils.py                     # LazyLLMOnlineBase (shared base), check_and_add_config
├── supplier/                        # Provider implementations (one file per provider)
│   ├── __init__.py                  # Auto-scans and imports all provider files (pkgutil.iter_modules)
│   ├── openai.py                    # OpenAIChat, OpenAIEmbed
│   ├── qwen.py                      # QwenChat, QwenEmbed
│   ├── claude.py                    # ClaudeChat
│   ├── deepseek.py                  # DeepseekChat
│   ├── glm.py                       # GlmChat
│   ├── kimi.py                      # KimiChat
│   ├── doubao.py                    # DoubaoChat, DoubaoEmbed
│   └── ...
├── chat.py                          # OnlineChatModule (unified Chat entry), dynamic_chat_config
├── embedding.py                     # OnlineEmbeddingModule (unified Embedding entry)
├── multimodal.py                    # OnlineMultiModalModule (unified MultiModal entry)
├── dynamic_router.py                # _DynamicSourceRouterMixin (dynamic routing)
├── map_model_type.py                # MODEL_MAPPING (model name → type mapping)
└── fileHandler.py                   # FileHandlerBase (file upload handling)
```

---

## Automatic Registration Mechanism

**Provider classes are automatically registered via the metaclass — no manual registry maintenance is needed.** The full flow:

1. `supplier/__init__.py` uses `pkgutil.iter_modules` to auto-scan and `import` all provider files
2. Each provider class (e.g. `QwenChat`) inherits `LazyLLMOnlineChatModuleBase`, which uses `LazyLLMRegisterMetaClass` as its metaclass
3. The metaclass automatically calls `__lazyllm_after_registry_hook__` at class definition time. This hook:
   - Infers `_model_series` from the class name (e.g. `QwenChat` → `qwen`)
   - Automatically calls `check_and_add_config` to register `qwen_api_key`, `qwen_model_name`, etc. into `globals.config`
   - Registers `default_source`, `default_key`, and other global config entries into `lazyllm.config`

**Therefore, when adding a new provider:**
- **No need** to manually call `lazyllm.config.add` (the metaclass hook handles it)
- **No need** to add a mapping to any registry table (inheriting = registering)
- **No need** to modify any code in `chat.py` / `embedding.py`

---

## Three Base Classes

### OnlineChatModuleBase (`base/onlineChatModuleBase.py`)

Inheritance chain: `LazyLLMOnlineChatModuleBase` → `LazyLLMOnlineBase` + `LLMBase`

**Methods that must be implemented:**

| Method | Description |
|--------|-------------|
| `_get_system_prompt(self)` | Returns the provider's default system prompt |
| `_convert_msg_format(self, msg)` | Converts standard OpenAI-format messages to the provider's format (return `msg` directly if identical) |
| `_str_to_json(self, msg, stream_output)` | Parses SSE streaming or non-streaming responses into the standard format |

**Optional methods:**

| Method | Description |
|--------|-------------|
| `_get_tools_call_format(self)` | Request format for tool calls (omit if tool calls are not supported) |
| `_prepare_request_data(self, data)` | Additional processing of the request body before sending |

**Class attributes that must be set:**

| Attribute | Description |
|-----------|-------------|
| `TRAINABLE_MODEL_LIST` | List of models that support fine-tuning (empty list `[]` if none) |
| `VLM_MODEL_PREFIX` | List of VLM model name prefixes (for automatic VLM model detection) |
| `NO_PROXY` | Whether to disable proxy (default `True`) |

### OnlineEmbeddingModuleBase (`base/onlineEmbeddingModuleBase.py`)

**Methods that must be implemented:**

| Method | Description |
|--------|-------------|
| `_get_embed_url(self)` | Returns the Embedding API URL |
| `_convert_response(self, response)` | Converts the API response to a list of vectors |

### OnlineMultiModalBase (`base/onlineMultiModalBase.py`)

**Methods that must be implemented:**

| Method | Description |
|--------|-------------|
| `_get_system_prompt(self)` | Returns the default system prompt |
| `_convert_msg_format(self, msg)` | Converts the message format |

---

## Complete Steps for Adding a New Provider

### 1. Create a new file under `supplier/`

Use the provider name in lowercase as the filename, e.g. `supplier/newprovider.py`.

Refer to `supplier/openai.py` (most complete implementation) or `supplier/deepseek.py` (minimal implementation).

```python
# supplier/newprovider.py
from ..base import OnlineChatModuleBase

class NewProviderChat(OnlineChatModuleBase):
    TRAINABLE_MODEL_LIST = []
    VLM_MODEL_PREFIX = []
    NO_PROXY = True

    def __init__(self, base_url=None, model=None, api_key=None,
                 stream=True, return_trace=False, skip_auth=False, **kw):
        base_url = base_url or 'https://api.newprovider.com/v1/'
        model = model or 'newprovider-default-model'
        super().__init__(
            api_key=api_key or self._default_api_key(),
            base_url=base_url,
            model_name=model,
            stream=stream,
            return_trace=return_trace,
            skip_auth=skip_auth,
            **kw,
        )

    def _get_system_prompt(self):
        return 'You are a helpful assistant.'

    def _convert_msg_format(self, msg):
        return msg  # return directly if identical to OpenAI format

    def _str_to_json(self, msg, stream_output):
        # Parse SSE response; refer to openai.py for implementation
        ...
```

**The metaclass hook automatically handles the following (no manual action needed):**
- Infers `_model_series = 'newprovider'` from the class name `NewProviderChat`
- Registers `globals.config['newprovider_api_key']` (env var `NEWPROVIDER_API_KEY`)
- Registers `globals.config['newprovider_model_name']` (env var `NEWPROVIDER_MODEL_NAME`)
- Registers the class into the `lazyllm.llm.newprovider` namespace

### 2. Add model mappings to `map_model_type.py`

```python
MODEL_MAPPING = {
    ...
    # ===== NewProvider =====
    'newprovider-chat-v1': 'llm',
    'newprovider-embed-v1': 'embed',
    'newprovider-vision-v1': 'vlm',
}
```

Model type values: `'llm'` (text chat), `'embed'` (embedding), `'vlm'` (vision-language), `'sd'` (image generation), `'tts'` (text-to-speech), `'stt'` (speech-to-text)

### 3. Done

`supplier/__init__.py` auto-scans and imports the new file — no other changes needed.

---

## Dynamic Routing Mechanism (`dynamic_router.py`)

Dynamic routing allows **runtime** switching of provider, model, and API key without recreating module instances.

### Use Cases

- Multi-tenant scenarios: different users use different API keys or models
- A/B testing: randomly route the same request to different providers
- Fallback strategy: switch to a backup provider when the primary is unavailable

### Method 1: Context Manager (recommended)

```python
from lazyllm.module.llms.onlinemodule.chat import dynamic_chat_config

# Applies to all modules (modules=None)
with dynamic_chat_config(source='qwen', model='qwen-plus'):
    result = my_chat_module('hello')

# Applies to a specific module instance
with dynamic_chat_config(modules=my_module, source='openai', model='gpt-4o',
                          url='https://api.openai.com/v1/'):
    result = my_module('hello')

# Dynamic API key (skip_auth=True skips authentication, for private deployments)
with dynamic_chat_config(source='qwen', model='qwen-plus', skip_auth=True):
    result = my_module('hello')
```

Parameters of `dynamic_chat_config`:
- `modules`: specifies which module instances are affected (`None` means all modules)
- `source`: provider name (e.g. `'qwen'`, `'openai'`, `'deepseek'`)
- `model`: model name
- `url`: API base URL (overrides the default)
- `skip_auth`: whether to skip authentication

### Method 2: Dynamic API Key (specified at creation time)

Pass `api_key='auto'` at creation time; the key is read dynamically from `globals.config` at runtime:

```python
# Create without binding an API key
llm = OnlineChatModule('qwen-plus', api_key='auto')

# Inject the API key dynamically at runtime (can differ per request)
lazyllm.globals.config['qwen_api_key'] = 'sk-user-specific-key'
result = llm('hello')
```

### Method 3: `source='dynamic'` (fully dynamic routing)

Pass `source='dynamic'` at creation time; the full config is read from `globals.config['dynamic_model_configs']` at runtime:

```python
# Create without binding any provider
llm = OnlineChatModule(source='dynamic')

# Configure dynamically at runtime (ConfigsDict grouped by module ID)
lazyllm.globals.config['dynamic_model_configs'] = {
    llm._module_id: {
        'chat': {
            'source': 'qwen',
            'model': 'qwen-plus',
            'url': 'https://dashscope.aliyuncs.com/compatible-mode/v1/',
        }
    }
}
result = llm('hello')
```

---

## Automatic Model Type Detection

When no type is specified, `OnlineModule` uses `map_model_type.get_model_type(model_name)` to automatically detect the model type and route to the corresponding Chat/Embedding/MultiModal module:

```python
OnlineModule('qwen-plus')                    # → OnlineChatModule
OnlineModule('text-embedding-3-small')       # → OnlineEmbeddingModule
OnlineModule('gpt-4o')                       # → OnlineChatModule (VLM model)
```

`_OnlineModuleMeta`'s `__instancecheck__` makes `isinstance(x, OnlineModule)` return `True` for instances of all three base classes.

**When adding a new model, you must add a mapping to `MODEL_MAPPING`**; otherwise `OnlineModule` cannot auto-route.

---

## Prohibited Patterns

- **Never** manually call `lazyllm.config.add` to register provider config (the metaclass hook handles it)
- **Never** hardcode API URLs in provider classes (pass via `__init__` parameter with a default value)
- **Never** modify `_model_name`, `_base_url`, or other core attributes outside `__init__`
- **Never** skip the `super().__init__()` call (the Base class initialization contains critical logic)
- **Never** implement `forward` in a provider class (implemented uniformly by the Base class)
