# Multi-Document & Multi-Source Routing RAG Agent

This article demonstrates the implementation of a **multi-document, multi-source routing** Retrieval-Augmented Generation (RAG) intelligent Q&A agent, built using the **LazyLLM** framework. It integrates query rewriting, multi-route retrieval, result reranking, context merging, and final answer generation.  
The agent can handle multiple document collections and different data sources simultaneously, automatically selecting the most suitable retrieval route at runtime for efficient cross-domain information access.

!!! abstract "By the end of this section, you will learn the following LazyLLM key points"

    - How to use [OnlineEmbeddingModule][lazyllm.module.OnlineEmbeddingModule] to create retrieval and reranking models;
    - How to build multiple Node Groups (block / coarse / QA-pair) and reuse retriever instances;
    - How to implement multi-document and multi-source routing strategies within a single Agent;
    - How to register multi-functional retrieval and answering tools with [fc_register][lazyllm.tools.fc_register];
    - How to use [pipeline][lazyllm.flow.pipeline] and [parallel][lazyllm.flow.parallel] for multi-route retrieval and result fusion;
    - How to use [ReactAgent][lazyllm.tools.ReactAgent] to run a multi-tool reasoning chain.

---

## Design Concept

### Core Objectives

- **Multi-document support**  
  Load multiple document collections from different domains or formats into a unified document manager, and build corresponding retrieval indexes for each collection.

- **Multi-source routing**  
  In the retrieval stage, route queries to the most suitable data source and indexing strategy (e.g., coarse chunk retrieval, fine-grained block retrieval, QA-pair retrieval).

- **High-quality output**  
  Improve recall rate through query rewriting, increase coverage via multi-route retrieval, and ensure answer quality through reranking and context merging.

---

### Workflow Design

1. **Query Rewriting**  
   Use an LLM to rewrite the user's natural language question into a form more suitable for retrieval, optimizing matching performance.

2. **Multi-route Retrieval (Multi-document & Multi-source Routing)**  
   - **CoarseChunk Retrieval** (coarse-grained, wide coverage)  
   - **Block Retrieval** (fine-grained, high precision)  
   - **QA-pair Retrieval** (structured Q&A, direct answers)

   All retrievers work across multiple document collections, with the agent executing retrieval requests in parallel and aggregating results.

3. **Reranking & Result Fusion**  
   Use a `Reranker` model to score candidate content from different sources and documents, then deduplicate and merge into a single context string.

4. **Context-based Answer Generation**  
   Provide the user question and the merged context to an LLM to generate the final natural language answer.

---

## Implementation

### Prompt Design

```python
rewriter_prompt = (
    "You are a query rewriting assistant. Your task is to transform user queries to improve retrieval."
    "Note: You do not need to answer, only rewrite the question to make it more retrievable."
    "Example:"
    "Input: What is RAG?"
    "Output: What is the definition of RAG?"
)
rag_prompt = (
    "You will play the role of an AI Q&A assistant and complete a dialogue task."
    " In this task, you need to provide your answer based on the given context and question."
)
```

* **rewriter\_prompt** rewrites queries to ensure high recall in cross-document and multi-source retrieval.
* **rag\_prompt** guides the model to answer based on the provided context.

---

### Model & Multi-Document Index Initialization

1. **Create Embedding & Reranking Models** (`OnlineEmbeddingModule`)
2. **Load Documents**
   Extendable to multiple paths (multiple `lazyllm.Document` instances for different folders/data sources).
3. **Create Node Groups**

   * `block` (line split)
   * `CoarseChunk` (coarse split)
   * `qapair` (LLM-based Q\&A pair parsing)
4. **Build Retriever Instances** (`Retriever`) for multi-route routing.

---

### Tool Registration

Register functions with `@fc_register("tool")` so they can be called by the Agent:

* `rewrite_query`: Rewrite the original user question
* `kb_search`: Multi-route retrieval (multi-document, multi-source) + reranking + context fusion
* `answer_with_context`: Generate an answer based on the given context

`kb_search` uses `pipeline` + `parallel` to concurrently run multiple retrievers, and optionally QA-pair retrieval.

---

### Multi-Document & Multi-Source Routing Strategy

In `_retrieve_and_rerank`, the retrieval stage:

* Calls different granularity retrievers (`RETR_COARSE` and `RETR_BLOCK`) in parallel
* Optionally uses `RETR_QAPAIR` based on requirements
* Uses the `Reranker` model to merge scores and unify ranking

This design ensures that regardless of which document collection or data source the data comes from, the final answer is generated from a unified merged context.

---

### Building ReactAgent

```python
TOOLS = ["rewrite_query", "kb_search", "answer_with_context"]

agent = ReactAgent(
    llm=llm,
    tools=TOOLS,
    max_retries=5,
    return_trace=True,
    stream=False
)
```

The ReactAgent automatically calls the appropriate tools to complete the **“query rewriting → retrieval → answering”** reasoning chain.

---

### Deploying the Application

```python
WebModule(agent, port=23466).start().wait()
```

This deploys the agent as a web service accessible via HTTP API.

---

### Demo

The image below shows the visualization of the multi-document, multi-source routing RAG Agent during the retrieval and answering process.

![multirag](../assets/multirag.png)

---

## Full Code

<details>
<summary>Click to view full code</summary>

```python
# rag_agent_demo.py
import os
import json
from typing import List, Dict, Any
import lazyllm
from lazyllm import bind
from lazyllm.tools import fc_register
from lazyllm import WebModule, ChatPrompter
from lazyllm.module import OnlineChatModule
from lazyllm.tools import ReactAgent

# ---------------- Prompts ----------------
rewriter_prompt = (
    "You are a query rewriting assistant. Your task is to transform user queries to improve retrieval."
    "Note: You do not need to answer, only rewrite the question to make it more retrievable."
    "Example:"
    "Input: What is RAG?"
    "Output: What is the definition of RAG?"
)
rag_prompt = (
    "You will play the role of an AI Q&A assistant and complete a dialogue task."
    " In this task, you need to provide your answer based on the given context and question."
)

# ---------------- Model & Multi-Document Index Initialization ----------------
online_embed = lazyllm.OnlineEmbeddingModule(source='qwen')
online_rerank = lazyllm.OnlineEmbeddingModule(source='qwen', type="rerank")

llm = OnlineChatModule(source="qwen", stream=False)

docs = lazyllm.Document(
    "Your-Own-Path",
    embed=online_embed
)

# block: line split
docs.create_node_group(name='block', transform=(lambda d: d.split('\n')))

try:
    from lazyllm.tools.rag.chunk import CoarseChunker
    docs.create_node_group(
        name="CoarseChunk",
        transform=CoarseChunker(chunk_size=800, chunk_overlap=100)
    )
except Exception:
    def _coarse(d: str):
        parts, buf = [], []
        for line in d.splitlines():
            if line.strip():
                buf.append(line)
            else:
                if buf:
                    parts.append("\n".join(buf))
                    buf = []
        if buf:
            parts.append("\n".join(buf))
        return parts
    docs.create_node_group(name="CoarseChunk", transform=_coarse)

# qapair: Q&A pair parsing with LLM
qa_parser = lazyllm.LLMParser(llm, language="zh", task_type="qa")
docs.create_node_group(name='qapair', transform=qa_parser)

# ---- Reuse retriever singletons (support multi-source routing) ----
RETR_COARSE = lazyllm.Retriever(doc=docs, group_name="CoarseChunk",
                                similarity="cosine", topk=3)
RETR_BLOCK  = lazyllm.Retriever(doc=docs, group_name="block",
                                similarity="bm25_chinese", topk=3)
RETR_QAPAIR = lazyllm.Retriever(doc=docs, group_name="qapair",
                                similarity="cosine", topk=3)

# Warm up
try:
    _ = RETR_BLOCK("warmup")
    _ = RETR_COARSE("warmup")
    _ = RETR_QAPAIR("warmup")
except Exception:
    pass

# =============== Tool Registration ===============
def _rewrite_with_llm(query: str) -> str:
    prompter = ChatPrompter(instruction=rewriter_prompt)
    return llm.share(prompter)(query)

@fc_register("tool")
def rewrite_query(query: str) -> str:
    return _rewrite_with_llm(query)

def _retrieve_and_rerank(query: str, topk: int = 3):
    with lazyllm.pipeline() as ppl:
        with lazyllm.parallel().sum as ppl.prl:
            ppl.prl.retriever1 = RETR_COARSE
            ppl.prl.retriever2 = RETR_BLOCK
        ppl.reranker = lazyllm.Reranker("ModuleReranker", model=online_rerank, topk=topk) | bind(query=ppl.input)
    return ppl(query)

@fc_register("tool")
def kb_search(query: str, use_qapair: bool = True, topk: int = 3) -> str:
    nodes_a = _retrieve_and_rerank(query, topk=topk)
    nodes_b = RETR_QAPAIR(query) if use_qapair else []

    contents_seen = set()
    merged_nodes: List[Any] = []
    for n in list(nodes_a) + list(nodes_b):
        c = n.get_content()
        if c not in contents_seen:
            contents_seen.add(c)
            merged_nodes.append(n)

    context_str = "\n".join([n.get_content() for n in merged_nodes])
    return context_str

@fc_register("tool")
def answer_with_context(query: str, context_str: str) -> str:
    prompter = ChatPrompter(instruction=rag_prompt, extra_keys=['context_str'])
    return llm.share(prompter)(dict(context_str=context_str, query=query))

# =============== Build Agent ===============
TOOLS = ["rewrite_query", "kb_search", "answer_with_context"]

agent = ReactAgent(
    llm=llm,
    tools=TOOLS,
    max_retries=5,
    return_trace=True,
    stream=False
)

# =============== Minimal Runnable Example ===============
if __name__ == "__main__":
    WebModule(agent, port=23466).start().wait()
```

</details>