# lazyllm/tools AGENTS.md

This directory is LazyLLM's **high-level tools layer**, imported on demand via lazy-loading. It contains advanced capabilities including RAG, Agent, Memory, MCP, WebModule, and more.

## Mandatory Pre-reading

Before modifying this directory, you must read:
- `lazyllm/AGENTS.md` (global conventions, especially the lazy-loading mechanism)
- `lazyllm/module/AGENTS.md` (ModuleBase — all Modules in the tools layer inherit from it)
- The target subdirectory's `AGENTS.md` (if it exists)

---

## Lazy-loading Mechanism

`tools/__init__.py` implements lazy-loading via `__getattr__`, avoiding heavy imports at startup:

```python
_SUBMOD_MAP = {
    'rag': ['Document', 'Retriever', 'Reranker', ...],
    'agent': ['ToolManager', 'ReactAgent', 'FunctionCall', ...],
    'sandbox': ['LazyLLMSandboxBase', ...],
    ...
}

def __getattr__(name: str):
    if name in _SUBMOD_MAP_REVERSE:
        module = import_module(f'.{_SUBMOD_MAP_REVERSE[name]}', package=__package__)
        globals()[name] = value = getattr(module, name)
        return value
```

### When adding a new submodule, you must:

1. Register the submodule name and exported symbol list in `_SUBMOD_MAP` in `tools/__init__.py`
2. Add type imports in the `TYPE_CHECKING` block (for IDE completion)
3. **Never** `import` directly at the top level of `tools/__init__.py` (breaks lazy-loading)

```python
# Correct: register in _SUBMOD_MAP
_SUBMOD_MAP = {
    ...
    'mymodule': ['MyClass', 'my_function'],
}

# Correct: add type imports in TYPE_CHECKING block
if TYPE_CHECKING:
    from .mymodule import MyClass, my_function

# Wrong: top-level direct import
from .mymodule import MyClass  # triggers loading when `import lazyllm` runs
```

---

## Submodule Navigation

| Subdirectory | Responsibility | AGENTS.md |
|--------------|---------------|-----------|
| `rag/` | Full RAG pipeline: document management, retrieval, reranking | `rag/AGENTS.md` |
| `agent/` | Agent types, tool registration, FunctionCall | `agent/AGENTS.md` |
| `memory/` | Conversation memory management (MemU, Mem0, PowerMem) | See this document |
| `mcp/` | MCP protocol client; converts MCP tools to ModuleTool | See this document |
| `fs/` | Cloud file system adapters: Feishu, Confluence, Notion, Google Drive, S3, etc. | See this document |
| `webpages/` | `WebModule`: provides a web conversation interface | — |
| `sandbox/` | Code sandbox execution (DummySandbox, SandboxFusion) | — |
| `classifier/` | `IntentClassifier`: intent classification | — |
| `sql/` | `SqlManager`, `MongoDBManager`: database management | — |
| `sql_call/` | `SqlCall`: natural language to SQL execution | — |
| `tools/` | Built-in tools: `HttpTool`, `calculator`, `weather`, search engines | — |
| `git/` | Git platform adapters: GitHub, GitLab, Gitee, GitCode | — |
| `http_request/` | `HttpRequest`: general-purpose HTTP request tool | — |
| `eval/` | RAG evaluation: `ResponseRelevancy`, `Faithfulness`, etc. | — |
| `data/` | Data processing operators and pipelines | — |
| `actors/` | `ParameterExtractor`, `QuestionRewrite`, `CodeGenerator` | — |
| `review/` | `ChineseCorrector`: Chinese text correction | — |
| `servers/` | GraphRAG, MinerU, and other server-side modules | — |

---

## Cloud File System (`fs/`)

`fs/` provides a unified cloud file system interface supporting Feishu, Confluence, Notion, Google Drive, OneDrive, S3, Obsidian, and more, as well as the local file system.

### Core Interface

`FS` is the unified routing entry point (singleton) that automatically routes to the corresponding FS implementation based on URI format:

```python
from lazyllm.tools.fs import FS

# URI format: protocol(@space_id):/path
fs = FS

# Read a file
content = fs.cat('feishu:/docs/my-document')
content = fs.cat('feishu@wiki123:/knowledge-base/page')  # Feishu wiki (requires space_id)
content = fs.cat('s3://my-bucket/data/file.txt')
content = fs.cat('/local/path/to/file.txt')  # local file

# List a directory
files = fs.ls('feishu:/docs/')

# Write a file
fs.pipe('feishu:/docs/new-doc.md', b'# Hello World')  # .md auto-converted to Feishu Docx

# Create a directory
fs.mkdir('feishu:/docs/new-folder')

# Delete a file
fs.rm('feishu:/docs/old-doc')
```

### Supported Protocols

| Protocol | Class | Description |
|----------|-------|-------------|
| `feishu` | `FeishuFS` | Feishu cloud drive (supports OAuth user auth and App credentials) |
| `feishu@{space_id}` | `FeishuWikiFS` | Feishu wiki (requires space_id) |
| `confluence` | `ConfluenceFS` | Confluence knowledge base |
| `notion` | `NotionFS` | Notion pages |
| `googledrive` | `GoogleDriveFS` | Google Drive |
| `onedrive` | `OneDriveFS` | OneDrive |
| `yuque` | `YuqueFS` | Yuque |
| `ones` | `OnesFS` | ONES project management |
| `s3` | `S3FS` | AWS S3 |
| `obsidian` | `ObsidianFS` | Obsidian local knowledge base |
| No protocol prefix | `LocalFileSystem` | Local file system |

### Authentication Configuration

**Static token (recommended for development):**

```python
# Configure via lazyllm.config (environment variable or ~/.lazyllm/config.json)
# Feishu: LAZYLLM_FEISHU_APP_ID + LAZYLLM_FEISHU_APP_SECRET
# Or: LAZYLLM_FEISHU_USER_ACCESS_TOKEN

# Pass token directly
from lazyllm.tools.fs import FeishuFS
feishu = FeishuFS(token='user_access_token_xxx')
```

**Dynamic token (recommended for multi-tenant production environments):**

```python
from lazyllm.tools.fs import dynamic_fs_config

# Context manager for temporarily injecting a dynamic token
with dynamic_fs_config({'feishu': 'user_token_for_this_request'}):
    content = FS.cat('feishu:/docs/my-doc')
```

**Dynamic auth mode (`dynamic_auth=True`):**

```python
# Create without binding a token; reads from globals.config at runtime
feishu = FeishuFS(dynamic_auth=True)
lazyllm.globals.config['feishu_token'] = 'runtime_token'
content = feishu.cat('/docs/my-doc')
```

### Feishu Special Features

`FeishuFS` supports the following additional features:
- Writing `.md` files automatically converts them to Feishu Docx format (via `mistune` Markdown AST parsing)
- `user_refresh_token='auto'` automatically persists the refresh_token to `~/.lazyllm/tokens.txt`
- Supports webhook registration

`FeishuWikiFS` (Feishu wiki) additionally supports:
- `get_doc_blocks(doc_id)`: retrieves the document block structure
- `update_doc_block_text(doc_id, block_id, text)`: fine-grained document editing

### Adding a New FS Provider

1. Inherit `LazyLLMFSBase` (in `fs/base.py`), using `LazyLLMRegisterMetaABCClass` as the metaclass
2. The class name must end with `FS` (the protocol name is inferred automatically from the class name, e.g. `MyCloudFS` → `mycloud`)
3. Implement the abstract methods of `fsspec.AbstractFileSystem` (`ls`, `_open`, `info`, etc.)
4. Export the new class in `fs/__init__.py`
5. Refer to `fs/supplier/feishu.py` for alignment

```python
# fs/supplier/mycloud.py
from ..base import LazyLLMFSBase
from lazyllm.common import LazyLLMRegisterMetaABCClass

class MyCloudFS(LazyLLMFSBase, metaclass=LazyLLMRegisterMetaABCClass):
    protocol = 'mycloud'  # optional; inferred from class name by default

    def __init__(self, token=None, dynamic_auth=False, **kwargs):
        super().__init__(token=token, dynamic_auth=dynamic_auth, **kwargs)

    def ls(self, path, detail=True, **kwargs):
        # List directory contents
        ...

    def _open(self, path, mode='rb', **kwargs):
        # Open a file; return a CloudFSBufferedFile instance
        ...

    def info(self, path, **kwargs):
        # Return a file/directory info dict
        ...
```

---

## Memory Interface (`memory/`)

### Abstract Base Class (`LazyLLMMemoryBase`)

```python
class LazyLLMMemoryBase(ABC, metaclass=LazyLLMRegisterMetaABCClass):
    def add(self, query, output=None, history=None, user_id=None, agent_id=None):
        # Store a conversation into memory (auto-formatted as a messages list)
        ...

    def get(self, query=None, user_id=None, agent_id=None):
        # Retrieve relevant memories
        ...

    def __call__(self, query=None):
        # Automatically reads user_id / agent_id from globals
        return self.get(query, globals.get('user_id'), globals.get('agent_id'))

    @abstractmethod
    def _add(self, message, user_id, agent_id): ...

    @abstractmethod
    def _get(self, query, user_id, agent_id): ...
```

### Memory Factory Class

```python
Memory(source='memu')      # MemU (default, in-memory)
Memory(source='mem0')      # Mem0 (requires Mem0 API key)
Memory(source='powermem')  # PowerMem (requires PowerMem service)
```

### `memory_hook`

Automatically stores memories before and after LLM calls via the Hook mechanism:

```python
from lazyllm.tools.memory import memory_hook

llm = OnlineChatModule().hook(memory_hook)
```

### Adding a New Memory Provider

1. Inherit `LazyLLMMemoryBase`
2. Implement the `_add` and `_get` abstract methods
3. Register in the `Memory.SUPPLIERS` dict in `memory/memory.py`

---

## MCPClient Interface (`mcp/`)

### Two Transport Protocols

```python
# HTTP SSE (remote MCP service)
client = MCPClient('http://localhost:8080/sse')

# stdio (local command-line MCP service)
client = MCPClient('python', args=['-m', 'my_mcp_server'])
```

### Getting the Tool List

`get_tools()` returns a standard `ModuleTool` list that can be passed directly to an Agent:

```python
client = MCPClient('http://localhost:8080/sse')
tools = client.get_tools()  # returns [ModuleTool, ...]

agent = ReactAgent(llm, tools=tools)
```

### Tool Filtering

```python
tools = client.get_tools(allowed_tools=['search', 'calculator'])
```

---

## General Tool Conventions (`tools/tools/`)

Built-in tools follow these conventions:

1. Inherit `ModuleTool` or register with the `@fc_register('tool')` decorator
2. docstring must include parameter descriptions (required for schema generation)
3. Tool functions must have type annotations

```python
@fc_register('tool')
def my_search_tool(query: str, max_results: int = 5) -> str:
    '''Search for information on the web.

    Args:
        query (str): The search query.
        max_results (int): Maximum number of results to return.

    Returns:
        str: Search results as formatted text.
    '''
    ...
```

---

## WebModule (`webpages/`)

Wraps any Module as a web conversation interface:

```python
web = WebModule(chat_module, port=8080)
web.start().wait()
```

Supports multi-turn conversation, file upload, and streaming output.

---

## Prohibited Patterns

- **Never** `import` submodules directly at the top level of `tools/__init__.py` (breaks lazy-loading)
- **Never** omit docstrings from tool functions (schema generation depends on them)
- **Never** omit type annotations from tool functions (schema generation depends on them)
- **Never** store user session data in instance variables in a Memory implementation (causes cross-user contamination under concurrent requests)
- **Never** hardcode tokens in FS implementations (inject via `lazyllm.config` or `dynamic_fs_config`)
