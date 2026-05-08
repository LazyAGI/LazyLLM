# lazyllm/tools/rag AGENTS.md

This directory implements LazyLLM's **complete RAG (Retrieval-Augmented Generation) system**, covering the full pipeline from document reading, parsing, splitting, and vectorization to retrieval and reranking.

Reference docs: [`lazyllm/docs/tools/tool_rag.py`](../../docs/tools/tool_rag.py)

## Mandatory Pre-reading

Before modifying this directory, you must read:
- `lazyllm/AGENTS.md` (global conventions)
- `lazyllm/tools/AGENTS.md` (lazy-loading mechanism)
- `lazyllm/module/AGENTS.md` (ModuleBase — all RAG components inherit from it)
- The target file itself and its direct dependencies

---

## Full RAG Pipeline Data Flow

```
Document (document management)
  ↓ file scanning + parsing
Reader (document reading)
  ↓ raw text
Transform (text splitting)
  ↓ list of DocNodes
Store (storage)
  ├── segment_store (text storage: MapStore / Elasticsearch / OpenSearch)
  └── vector_store (vector storage: Milvus / Chroma / in-memory)
  ↓
Retriever (retrieval)
  ↓ list of DocNodes
Reranker (reranking, optional)
  ↓ ranked list of DocNodes
LLM (generation)
```

---

## Core Data Structure: DocNode

`DocNode` is the fundamental unit of all data in the RAG system:

```python
class DocNode:
    uid: str           # unique identifier
    text: str          # text content
    metadata: dict     # metadata (file path, page number, custom fields, etc.)
    embedding: dict    # vectors (key = embed_key, value = vector list)
    parent: DocNode    # parent node (used in hierarchical splitting)
    children: dict     # child node dict (group_name → [DocNode])
```

### Node Groups

`Document` has three built-in node groups:

| Group name | Chunk size | Use case |
|------------|-----------|----------|
| `FineChunk` | ~128 tokens | Fine-grained retrieval |
| `MediumChunk` | ~256 tokens | Balanced retrieval |
| `CoarseChunk` | ~1024 tokens | Coarse-grained retrieval |
| `sentences` | By sentence | Semantic retrieval |

Custom groups:

```python
document.create_node_group(
    name='MyChunk',
    transform=SentenceSplitter(chunk_size=512),
    parent='CoarseChunk',  # further split from this group
)

# Use LLM for summarization / keyword extraction
document.create_node_group(
    name='Summary',
    transform=LLMParser(llm, prompt='Summarize: {text}'),
    parent='CoarseChunk',
)
```

---

## Document (Document Management)

```python
Document(
    dataset_path='./data',           # document directory or file path
    embed=TrainableModule('bge-m3'), # embedding model (or dict for multi-embedding)
    manager=False,                   # service mode (see below)
    store_conf={                     # vector store configuration
        'type': 'milvus',
        'kwargs': {'uri': 'http://localhost:19530'}
    },
    doc_fields={...},                # custom metadata fields (DocField)
)
```

Reference docs: [`lazyllm/docs/tools/tool_rag.py`](../../docs/tools/tool_rag.py)

### Service Modes

The `manager` parameter controls how the document service is deployed:

| Value | Behavior | Use case |
|-------|----------|----------|
| `False` (default) | Pure local mode, no DocServer | Development / testing |
| `True` | Automatically starts DocServer + parsing service locally | Single-machine production |
| `'ui'` | Same as `manager=True`, also creates a management UI | Single-machine deployment with UI |
| `DocServer(url='http://host:8080')` | Connects to an existing remote DocServer | Distributed production |
| `DocServer(port=8080, parser_url='...')` | Starts a new DocServer and connects to it | Single-machine with custom port |
| `DocumentProcessor(url='...')` | Connects to a parsing service only, without DocServer | Parsing capability only |

**Auto-trigger rule:** When `manager=False` but `store_conf` is a persistent store (e.g. Milvus/Chroma) and `dataset_path` is a directory, `manager` is automatically set to `True` (to prevent multi-process race conditions).

**Important constraint:** When using `manager=True` or connecting to a DocServer, `store_conf` must point to a network endpoint (Milvus/Chroma, etc.); embedded (filesystem-bound) vector stores are not supported.

### Distributed Deployment (Offline Parsing + Online Retrieval Separated)

For production environments, it is recommended to deploy **offline parsing** and **online retrieval** on separate machines:

```
Machine A (parsing service)                Machine B (document service)
DocumentProcessor                          DocServer
  - executes document parsing tasks          - receives document upload/add requests
  - callbacks to DocServer                   - manages KBs, task status
  - port: 9966                               - port: 8080
                                                   ↓
                                           Application process (online retrieval)
                                           Document(manager=DocServer(url=...))
                                           + Milvus (vector store)
```

**Step 1: Start the parsing service (Machine A)**

```python
from lazyllm.tools.rag.parsing_service import DocumentProcessor

proc = DocumentProcessor(
    db_config={'db_type': 'sqlite', 'db_name': '/shared/parser.db'},
    callback_url='http://machine-b:8080/v1/callback',
    num_workers=4,
)
proc.start()  # exposes HTTP port 9966
```

**Step 2: Start the document service (Machine B)**

```python
from lazyllm.tools.rag.doc_service import DocServer

doc_server = DocServer(
    port=8080,
    parser_url='http://machine-a:9966',   # required; points to the parsing service
    storage_dir='/shared/uploads',
    db_config={'db_type': 'sqlite', 'db_name': '/shared/doc_service.db'},
)
doc_server.start()
```

**Step 3: Application process connects (online retrieval)**

```python
doc = Document(
    dataset_path=None,
    embed=my_embed_model,
    manager=DocServer(url='http://machine-b:8080'),  # connect to existing DocServer
    store_conf={'type': 'milvus', 'kwargs': {'uri': 'http://milvus:19530'}},
)
retriever = Retriever(doc, 'CoarseChunk', 'cosine', topk=3)
```

**Notes:**
- `DocServer` and `DocumentProcessor` share the same DB (SQLite file must be shared, or use MySQL/PostgreSQL)
- `parser_url` is a required parameter for `DocServer`; there is no longer a built-in mock parsing server
- Distributed deployments must use network-endpoint vector stores (Milvus/Chroma, etc.)

### DocImpl

`DocImpl` is the internal implementation of `Document`, responsible for managing document storage. **Never operate on `DocImpl` directly** — access it through `Document`'s public interface.

---

## Reader (Document Reading)

Built-in Readers:

| Reader | Supported formats |
|--------|------------------|
| `PDFReader` | PDF |
| `DocxReader` | DOCX |
| `PPTXReader` | PPTX |
| `PandasCSVReader` | CSV |
| `PandasExcelReader` | Excel |
| `MarkdownReader` | Markdown |
| `ImageReader` | Images (JPG/PNG, etc.) |
| `MineruPDFReader` | PDF (uses MinerU for parsing; supports complex layouts) |

`SimpleDirectoryReader` automatically selects the appropriate Reader based on file extension.

### Adding a New Reader

Inherit `LazyLLMReaderBase` (in `readers/readerBase.py`) and implement `load_data`:

```python
from lazyllm.tools.rag.readers.readerBase import LazyLLMReaderBase

class MyReader(LazyLLMReaderBase):
    def load_data(self, file_path: str, extra_info: dict = None) -> list:
        # Returns a list of DocNodes
        text = open(file_path).read()
        return [DocNode(text=text, metadata={'file_path': str(file_path)})]
```

---

## Transform (Text Splitting)

Built-in Transforms:

| Transform | Splitting method |
|-----------|----------------|
| `SentenceSplitter` | Splits at sentence boundaries (recommended) |
| `CharacterSplitter` | Splits by character count |
| `RecursiveSplitter` | Recursively splits by separators |
| `MarkdownSplitter` | Splits by Markdown headings |
| `CodeSplitter` | Splits by code structure |
| `LLMParser` | Uses an LLM for summarization / keyword extraction |

### Adding a New Transform

Inherit `NodeTransform` (in `transform/base.py`) and implement `transform`:

```python
from lazyllm.tools.rag.transform.base import NodeTransform

class MyTransform(NodeTransform):
    def __init__(self, chunk_size=512):
        super().__init__()
        self._chunk_size = chunk_size

    def transform(self, node: DocNode, **kwargs) -> list:
        # Split one DocNode into multiple DocNodes
        chunks = split_text(node.text, self._chunk_size)
        return [DocNode(text=chunk, metadata=node.metadata.copy()) for chunk in chunks]
```

---

## Store (Storage)

### Responsibilities of the Two Store Types

| Store type | Responsibility | Implementations |
|------------|---------------|----------------|
| `segment_store` | Stores text content and metadata | `MapStore` (in-memory) / `ElasticsearchStore` / `OpenSearchStore` |
| `vector_store` | Stores vectors; supports vector retrieval | In-memory / Milvus / Chroma |

`MapStore` is the default in-memory implementation and serves as both `segment_store` and `vector_store`.

### Store Configuration

```python
# Default (in-memory MapStore)
store_conf = None

# Milvus
store_conf = {
    'type': 'milvus',
    'kwargs': {'uri': 'http://localhost:19530', 'db_name': 'my_db'}
}

# Chroma
store_conf = {
    'type': 'chroma',
    'kwargs': {'host': 'localhost', 'port': 8000}
}
```

### Adding a New Store Backend

Inherit `LazyLLMStoreBase` (in `store/store_base.py`) and implement the abstract methods.

---

## Retriever (Retrieval)

```python
Retriever(
    doc=document,
    group_name='CoarseChunk',          # node group to retrieve from
    similarity='cosine',                # similarity function
    similarity_cut_off=0.5,            # similarity threshold (results below this are filtered out)
    topk=6,                            # return top-k results
    output_format='content',           # output format
    join='\n',                         # merge output (only valid when output_format='content')
    weight=0.7,                        # RRF fusion weight (mutually exclusive with priority)
)
```

### Output Formats

| output_format | Return type | Description |
|---------------|-------------|-------------|
| `None` | `List[DocNode]` | Raw node list (default) |
| `'content'` | `List[str]` or `str` | Text list (merged into a string when `join` is set) |
| `'dict'` | `List[dict]` | Dict list (contains text and metadata) |

### Similarity Functions

Built-in similarities: `'cosine'`, `'euclidean'`, `'dot'`, `'bm25_chinese'`, `'bm25'`

Register a custom similarity:

```python
from lazyllm.tools.rag.similarity import register_similarity

@register_similarity(mode='embedding', batch=True)
def my_similarity(query_embedding, node_embeddings, **kwargs):
    # Returns a list of similarity scores
    ...
```

### Multi-path Retrieval Fusion

```python
with parallel().sum as ppl.prl:
    ppl.prl.ret1 = Retriever(doc, 'sentences', 'cosine', topk=3, weight=0.6)
    ppl.prl.ret2 = Retriever(doc, 'CoarseChunk', 'bm25_chinese', topk=3, weight=0.4)
```

---

## Reranker (Reranking)

```python
Reranker(
    name='ModuleReranker',
    model='bge-reranker-v2-m3',  # reranking model
    topk=3,                       # keep top-k after reranking
    output_format='content',
    join='\n',
)
```

Register a custom Reranker:

```python
from lazyllm.tools.rag.rerank import register_reranker

@register_reranker
class MyReranker:
    def rerank(self, nodes, query, **kwargs):
        # Returns a sorted list of DocNodes
        ...
```

---

## Typical RAG Pipeline

```python
documents = Document('./data', embed=TrainableModule('bge-m3'))

with pipeline() as ppl:
    with parallel().sum as ppl.prl:
        ppl.prl.ret1 = Retriever(documents, 'sentences', 'cosine', topk=3)
        ppl.prl.ret2 = Retriever(documents, 'CoarseChunk', 'bm25_chinese', topk=3)
    ppl.reranker = bind(Reranker('ModuleReranker', model='bge-reranker-v2-m3', topk=3,
                                  output_format='content', join='\n'),
                        query=ppl.input)
    ppl.llm = OnlineChatModule().prompt(ChatPrompter(
        'Based on the following context, answer the question.\nContext: {context}\nQuestion: {input}',
        extro_keys=['context']
    ))

ppl.start()
result = ppl('What is LazyLLM?')
```

---

## Prohibited Patterns

- **Never** operate on `DocImpl` directly (access via `Document`'s public interface)
- **Never** operate on `Store` directly (access via `Document` and `Retriever`)
- **Never** modify the `metadata` of an input `DocNode` inside a Transform (create a new `DocNode` instead)
- **Never** cache retrieval results in instance variables inside a Retriever (causes cross-user contamination under concurrent requests)
- **Never** skip the `super().__init__()` call
