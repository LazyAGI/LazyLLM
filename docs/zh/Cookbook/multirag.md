# 多文档·多数据源路由 RAG Agent

本文我们将实现一个支持 **多文档、多数据源路由** 的智能检索问答 Agent，基于 **LazyLLM** 框架构建，集成了查询重写、多路检索、结果重排序、上下文融合以及基于上下文的答案生成。该 Agent 能够同时对接多个文档集合与不同数据源，并在运行时自动选择合适的检索路径，从而实现高效的跨域信息检索与问答。

!!! abstract "通过本节您将学习到 LazyLLM 的以下要点"

```
- 如何使用 [OnlineEmbeddingModule][lazyllm.module.OnlineEmbeddingModule] 创建检索与重排序模型；
- 如何构建多种 Node Group（block / coarse / QA-pair）并复用检索器实例；
- 如何在同一 Agent 内实现多文档、多数据源路由策略；
- 如何通过 [fc_register][lazyllm.tools.fc_register] 注册多功能检索与回答工具；
- 如何使用 [pipeline][lazyllm.flow.pipeline] 与 [parallel][lazyllm.flow.parallel] 实现多路检索与结果融合；
- 如何使用 [ReactAgent][lazyllm.tools.ReactAgent] 实现多工具推理链路；
```

---

## 设计思路

### 核心目标

* **多文档支持**
  将多个不同领域或格式的文档集合加载到统一的文档管理器中，并为每个文档集合建立对应的检索索引。

* **多数据源路由**
  在检索阶段根据不同的需求路由到最合适的数据源与索引策略（例如：粗粒度分块检索、精粒度行检索、QA-pair 检索）。

* **高质量结果输出**
  通过查询重写提升召回率，通过多路检索增加覆盖率，再通过重排序与上下文融合确保答案质量。

---

### 流程设计

1. **查询重写**
   使用 LLM 将用户原始问题改写为更适合检索的查询，以优化匹配效果。

2. **多路检索（多文档 & 多数据源路由）**

   * **CoarseChunk 检索**（粗粒度，覆盖面大）
   * **Block 检索**（精粒度，匹配度高）
   * **QA-pair 检索**（结构化问答对，直击答案）

   三类检索器同时对接多个文档集合，Agent 会并行执行检索请求并聚合结果。

3. **重排序与结果融合**
   利用 `Reranker` 模型对来自不同数据源与文档的候选内容统一打分，并去重合并成上下文字符串。

4. **基于上下文生成答案**
   将用户问题与融合后的上下文送入 LLM，生成最终的自然语言答案。

---

### 效果展示

下图展示了多文档、多数据源路由 RAG Agent 在检索与问答过程中的可视化效果。

![multirag](../assets/multirag.png)

## 代码实现

### 提示词设计

```python
rewriter_prompt = (
    "你是一个查询重写助手，负责给用户查询进行模板切换。"
    "注意，你不需要进行回答，只需要对问题进行重写，使更容易进行检索。"
    "下面是一个简单的例子："
    "输入：RAG是啥？"
    "输出：RAG的定义是什么？"
)
rag_prompt = (
    "You will play the role of an AI Q&A assistant and complete a dialogue task."
    " In this task, you need to provide your answer based on the given context and question."
)
```

* **rewriter\_prompt** 用于改写查询，保证跨文档、多源检索时的召回率；
* **rag\_prompt** 用于根据上下文生成最终答案。

---

### 模型与多文档索引初始化

1. **创建嵌入与重排序模型**（`OnlineEmbeddingModule`）
2. **加载文档**
   可扩展为多路径（多个 `lazyllm.Document` 实例对应不同文件夹 / 数据源）
3. **创建 Node Group**

   * `block`（行切分）
   * `CoarseChunk`（粗分块）
   * `qapair`（LLM 解析 QA 对）
4. **构建检索器实例**（`Retriever`），支持多路路由

---

### 工具注册

通过 `@fc_register("tool")` 实现可由 Agent 调用的多功能工具：

* `rewrite_query`：改写用户原始问题
* `kb_search`：多路检索（多文档、多数据源）+ 重排序 + 上下文融合
* `answer_with_context`：基于上下文回答用户问题

`kb_search` 使用 `pipeline` + `parallel` 并发执行多个检索器，并支持是否额外使用 `QA-pair` 检索。

---

### 多文档、多数据源路由策略

在 `_retrieve_and_rerank` 中，检索阶段会：

* 并行调用不同粒度的检索器（如 `RETR_COARSE` 与 `RETR_BLOCK`）
* （可选）根据业务逻辑启用 `RETR_QAPAIR`
* 使用 `Reranker` 模型融合评分，统一排序

此设计使得无论数据来自哪一个文档集合或数据源，都能在统一上下文中融合并生成答案。

---

### 构建 ReactAgent

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

ReactAgent 会自动调用合适的工具链完成 **“查询重写 → 检索 → 回答”** 的多步推理。

---

### 启动应用

```python
WebModule(agent, port=23466).start().wait()
```

部署为 Web 服务，支持 HTTP API 调用。

---

## 完整代码

<details>
<summary>点击查看完整代码</summary>

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

# ---------------- 提示词 ----------------
rewriter_prompt = (
    "你是一个查询重写助手，负责给用户查询进行模板切换。"
    "注意，你不需要进行回答，只需要对问题进行重写，使更容易进行检索。"
    "下面是一个简单的例子："
    "输入：RAG是啥？"
    "输出：RAG的定义是什么？"
)
rag_prompt = (
    "You will play the role of an AI Q&A assistant and complete a dialogue task."
    " In this task, you need to provide your answer based on the given context and question."
)

# ---------------- 模型与多文档索引初始化 ----------------
online_embed = lazyllm.OnlineEmbeddingModule(source='qwen')
online_rerank = lazyllm.OnlineEmbeddingModule(source='qwen', type="rerank")

llm = OnlineChatModule(source="qwen", stream=False)

docs = lazyllm.Document(
    "Your-Own-Path",
    embed=online_embed
)

# block：按行拆分
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

# qapair：用 LLM 解析 QA 对
qa_parser = lazyllm.LLMParser(llm, language="zh", task_type="qa")
docs.create_node_group(name='qapair', transform=qa_parser)

# ---- 复用多路检索器单例（支持多数据源路由）----
RETR_COARSE = lazyllm.Retriever(doc=docs, group_name="CoarseChunk",
                                similarity="cosine", topk=3)
RETR_BLOCK  = lazyllm.Retriever(doc=docs, group_name="block",
                                similarity="bm25_chinese", topk=3)
RETR_QAPAIR = lazyllm.Retriever(doc=docs, group_name="qapair",
                                similarity="cosine", topk=3)

# 预热
try:
    _ = RETR_BLOCK("warmup")
    _ = RETR_COARSE("warmup")
    _ = RETR_QAPAIR("warmup")
except Exception:
    pass

# =============== 工具注册 ===============
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

# =============== 构建 Agent ===============
TOOLS = ["rewrite_query", "kb_search", "answer_with_context"]

agent = ReactAgent(
    llm=llm,
    tools=TOOLS,
    max_retries=5,
    return_trace=True,
    stream=False
)

# =============== 最小可运行示例 ===============
if __name__ == "__main__":
    WebModule(agent, port=23466).start().wait()
```

</details>