# 构建带聊天历史的问答应用

问答系统的一个关键挑战是处理对话上下文。本教程展示如何为 RAG 应用添加记忆功能，使其能够处理依赖于聊天历史的后续问题。

我们将使用 LazyLLM 框架构建一个能够有效处理多轮对话的会话式 RAG 系统。

## 环境配置

### 组件

我们将使用 LazyLLM 的内置组件来构建会话式 RAG 应用：

- **Document**：用于加载和管理带有嵌入的文档
- **Retriever**：基于相似度检索相关文档
- **Reranker**：对检索到的文档进行重新排序以提高相关性
- **OnlineChatModule**：用于基于 LLM 的问题重构和答案生成
- **ReactAgent**：用于创建可以进行多次检索迭代的智能体

### 依赖

```python
import os
import tempfile
from typing import List, Dict, Any
import lazyllm
from lazyllm import pipeline, parallel, bind, fc_register
from lazyllm import Document, Retriever, Reranker, SentenceSplitter, OnlineEmbeddingModule
```

## 链式处理

会话式 RAG 的一个关键挑战在于，用户消息和 LLM 的响应并不是唯一的上下文形式。用户消息可能会引用之前对话的部分内容，这使得单独理解它们变得具有挑战性。

例如，下面的第二个问题依赖于第一个问题的上下文：

```
Human: 任务分解的标准方法是什么？
AI: 任务分解的标准方法是思维链（Chain of Thought，CoT）提示...
Human: 这种方法有哪些常见的扩展？
```

第二个问题"这种方法有哪些常见的扩展？"如果没有第一个问题的上下文是模糊的。我们的系统需要理解"这种方法"指的是"思维链提示"。

### 聊天历史的状态管理

为了处理这种情况，我们需要基于聊天历史来理解问题上下文。让我们构建我们的会话式 RAG 系统：

```python
class ConversationalRAGPipeline:
    """基于 LazyLLM 组件的多轮会话 RAG 管道"""
    
    def __init__(self, docs_path: str = None):
        self.chat_history = []
        self.docs_path = docs_path or self._create_sample_docs()
        self.pipeline = None
        self.init_pipeline()
```

首先，我们设置文档和嵌入：

```python
def init_pipeline(self):
    """初始化 RAG 管道"""
    # 创建文档和嵌入
    documents = Document(
        dataset_path=self.docs_path, 
        embed=OnlineEmbeddingModule(
            source="qwen",
            type="embed",
            embed_model_name="text-embedding-v1"
        ),
        manager=False
    )
    documents.create_node_group(
        name="sentences", 
        transform=SentenceSplitter, 
        chunk_size=512, 
        chunk_overlap=100
    )
```

接下来，我们创建一个上下文化提示，用于基于聊天历史重新表述问题：

```python
contextualize_prompt = """你是一个帮助重写问题的助手。你的任务是：
1. 分析用户的聊天历史
2. 理解最新的问题
3. 如果问题依赖于历史上下文，将其重写为独立可理解的形式
4. 如果问题已经是独立的，返回原始问题
5. 只返回重写后的问题，不要添加任何解释

示例：
历史记录："用户：告诉我关于小明的信息\n助手：小明是一名工程师..."
问题："他的工作是什么？"
重写为："小明的工作是什么？"

历史记录：{chat_history}
问题：{question}

重写后的问题："""

self.contextualizer = lazyllm.OnlineChatModule(
    source="deepseek",
    timeout=30
).prompt(lazyllm.ChatPrompter(contextualize_prompt))
```

然后我们构建具有并行检索和重排序的主要 RAG 管道：

```python
with pipeline() as ppl:
    # Parallel retrieval
    with parallel().sum as ppl.prl:
        ppl.prl.retriever1 = Retriever(
            documents, 
            group_name="sentences", 
            similarity="cosine", 
            topk=3
        )
        ppl.prl.retriever2 = Retriever(
            documents, 
            "CoarseChunk", 
            "bm25_chinese", 
            0.003, 
            topk=2
        )
    
    # Reranking
    ppl.reranker = Reranker(
        "ModuleReranker", 
        model=OnlineEmbeddingModule(type="rerank", source="qwen"), 
        topk=3, 
        output_format='content', 
        join=True
    ) | bind(query=ppl.input)
    
    # LLM generates answer
    ppl.llm = lazyllm.OnlineChatModule(
        source="deepseek",
        stream=False,
        timeout=60
    ).prompt(lazyllm.ChatPrompter(rag_prompt, extra_keys=["context_str", "chat_history"]))

self.pipeline = ppl
```

聊天方法处理对话流程：

```python
def chat(self, question: str) -> str:
    """处理单轮对话"""
    # 1. 重构问题
    contextualized_question = self.contextualize_question(question)
    
    # 2. 构建历史对话文本
    history_text = self._format_chat_history(self.chat_history, max_turns=4)
    
    # 3. 通过 RAG 管道生成答案
    response = self.pipeline(
        contextualized_question,
        chat_history=history_text
    )
    
    # 4. 更新聊天历史
    self.chat_history.append({"role": "user", "content": question})
    self.chat_history.append({"role": "assistant", "content": response})
    
    return response
```

让我们测试我们的会话式 RAG：

```python
rag = ConversationalRAGPipeline()

# 第一个问题
response1 = rag.chat("任务分解的标准方法是什么？")
print(f"回复 1: {response1}")

# 依赖上下文的后续问题
response2 = rag.chat("这种方法有哪些常见的扩展？")
print(f"回复 2: {response2}")
```

注意，模型在第二个问题中生成的问题包含了对话上下文，将"这种方法"转换为"思维链提示的扩展"。

## 智能体

智能体利用 LLM 的推理能力在执行过程中做出决策。使用智能体允许你将更多的检索过程决策权交给系统。虽然它们的行为比上述"链式"处理更难预测，但它们能够执行多个检索步骤来服务于一个查询，或者对单个搜索进行迭代。

下面我们使用 LazyLLM 的 ReactAgent 组装一个最小的 RAG 智能体：

```python
@fc_register("tool")
def conversational_rag_chat(question: str) -> str:
    """
    Multi-turn conversational RAG tool
    
    Args:
        question (str): User question
    
    Returns:
        str: RAG system's answer
    """
    global rag_system
    if rag_system is None:
        rag_system = ConversationalRAGPipeline()
    
    return rag_system.chat(question)

def create_rag_agent():
    """Create RAG agent"""
    agent = lazyllm.ReactAgent(
        llm=lazyllm.OnlineChatModule(source="deepseek", timeout=60),
        tools=["conversational_rag_chat"],
        max_retries=3,
        return_trace=False,
        stream=False
    )
    
    return agent
```

与我们之前的实现的主要区别在于，这里不是以最终生成步骤结束运行，而是工具调用可以循环回去收集更多信息。然后智能体可以使用检索到的上下文回答问题，或者生成另一个工具调用以获取更多信息。

让我们用一个通常需要迭代序列检索步骤的问题来测试：

```python
agent = create_rag_agent()

response = agent(
    "任务分解的标准方法是什么？"
    "获得答案后，查找该方法的常见扩展。"
)
print(response)
```

注意智能体：
1. 生成查询以搜索任务分解的标准方法
2. 收到答案后，生成第二个查询以搜索其常见扩展
3. 在收到所有必要的上下文后，回答问题

## 运行应用

启动 Web 界面：

```bash
python agent_n_rag.py
```

然后访问 `http://localhost:8849` 通过 Web 界面与会话式 RAG 系统进行交互。

## 下一步

我们已经介绍了构建基本会话式问答应用的步骤：

* 我们使用链来构建一个可预测的应用，该应用基于聊天历史来理解问题上下文
* 我们使用智能体来构建一个可以对查询序列进行迭代的应用

要探索不同类型的检索器和检索策略，请访问 LazyLLM 的检索组件文档。

要了解更高级的智能体架构，请查看 LazyLLM 的智能体文档和示例。
