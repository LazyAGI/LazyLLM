# 知识库问答助手

本文我们将实现一个知识库问答助手应用。

> 通过本节您将学习到 LazyLLM 的以下要点：
>
> - RAG 相关模块的使用：
>      - `Document`
>      - `Retriever`
>      - `Reranker`


## 设计思路

要设计一个知识库文档助手，首先需要有一个知识库（Documents），在构建好知识库后需要一些召回器（Retriever）用于匹配相似度高的文段。
在召回文档后一般还需要结合输入，使用 Reranker 模型再次进行重排序以获得顺序更好的文段。最后将召回内容和用户提问送给大模型来回答。

整体设计如下所示：

![Great Writer](../../assets/5_rag_1.svg)

这里设计了两个召回器，分别对文档从不同的颗粒度进行相似度匹配。

## 代码实现

让我们基于 LazyLLM 来实现上述设计思路吧。

### 设计提示词

根据设计，我们需要让大模型结合搜索到的文档和用户的输入来回答问题。此时就需要给大模型设计一个提示词模板。

```python
prompt = '你将扮演一个人工智能问答助手的角色，完成一项对话任务。在这个任务中，你需要根据给定的上下文以及问题，给出你的回答。'
```

### 设置模型

首先我们需要一个知识库：

```python
documents = Document(dataset_path='/file/to/yourpath', embed=lazyllm.TrainableModule('bge-large-zh-v1.5'))
```

这里使用到了 LazyLLM 里的 `Document`，它接收一个包含有文档的路径作为参数，
另外还需要指定一个词嵌入模型（这里用的是 `bge-large-zh-v1.5` ），该模型可以把文档进行向量化，
同时也可以把用户的请求进行向量化。这样就可以用两个向量化后的内容进行相似度计算来召回匹配的知识库文段内容了。

然后我们定义两个召回器，从不同文档的划分颗粒度来进行匹配：

```python
documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
retriever1 = Retriever(documents, group_name="sentences", similarity="cosine", topk=3)
retriever2 = Retriever(documents, "CoarseChunk", "bm25_chinese", 0.003, topk=3)
```

这里用到了LazyLLM里的召回器 `Retriever`。

- 召回器要求第一个参数要指定用哪个数据库，这里用到了我们刚加载好的知识库。
- 召回器的第二个参数 `group_name` 需要指定一个组名。
    - 第一个召回器我们在实例化的 `documents` 中，自定义了一个叫做 `sentences` 的组，这个组用到的节点变换方法是句子划分 `SentenceSplitter`，句子块大小是1024，重叠大小是100。在实例化 `Retriever` 的时候用到了这个组，并且使用余弦相似度来计算相似度，且取前3个最相似的句段。
    - 第二个召回器用了 LazyLLM 内置的 `CoarseChunk`, 其背后用到节点变换方法也是 `SentenceSplitter`，只是它的句子块大小更大，为1024，重叠大小也是100。这里同时还指定了相似度计算的方法是中文的 BM25，同时相似度低于 0.003 的就会被丢弃，默认是负无穷大是不丢弃的。并且最后也是取前3个最相似的句段。

接下来我们定义一个重排器Reranker:

```python
reranker = Reranker("ModuleReranker", model="bge-reranker-large", topk=1)
```

这里用到了 LazyLLM 里的 `Reranker`，它可以对召回的内容进行重新排序。这里取重排后的最相似的内容作为输出。

最后我们再设置一下 LLM 模型：

```python
llm = lazyllm.TrainableModule("internlm2-chat-7b").prompt(lazyllm.ChatPrompter(prompt, extro_keys=["context_str"]))
```
这里对 `TrainableModule` 设置了 `prompt`, 详细描述可参见：[绘画大师](painting_master.md)

### 组装应用

现在让我们把上面的模块都组装起来吧：

```python
with pipeline() as ppl:
    with parallel().sum as ppl.prl:
        prl.retriever1 = Retriever(documents, group_name="sentences", similarity="cosine", topk=3)
        prl.retriever2 = Retriever(documents, "CoarseChunk", "bm25_chinese", 0.003, topk=3)

    ppl.reranker = Reranker("ModuleReranker", model="bge-reranker-large", topk=1) | bind(query=ppl.input)
    ppl.formatter = (lambda nodes, query: dict(context_str="".join([node.get_content() for node in nodes]), query=query)) | bind(query=ppl.input)
    ppl.llm = lazyllm.TrainableModule("internlm2-chat-7b").prompt(lazyllm.ChatPrompter(prompt, extro_keys=["context_str"]))
```

上面代码中 `parallel().sum` 将其所有并行元素的输出都加载一起。这里我们指定了两类召回器进行召回，
分别都有3个结果，再经过 `.sum` 设置就可将结果相加起来，获得6个召回的结果。

```python
ppl.formatter = (lambda nodes, query: dict(context_str="".join([node.get_content() for node in nodes]), query=query)) | bind(query=ppl.input)
```

这里定义了一个匿名函数来进行格式化，将格式化后的内容喂给大模型。
需要注意的是这里 `bind` 上了用户的输入（`bind`可参考：[大作家](great_writer.md)）。对应了设计图中的蓝色线条。
另外 `reranker`也 `bind` 上了用户的输入，对应了设计图中的红色线条。

### 启动应用

最后，我们将控制流 `ppl` 套入一个客户端，并启动部署（`start()`），在部署完后保持客户端不关闭（`wait()`）。

```python
lazyllm.WebModule(ppl, port=23456).start().wait()
```

## 完整代码

<details>
<summary>点击获取import和prompt</summary>

```python
import os
import lazyllm
from lazyllm import pipeline, parallel, bind, _0, Document, Retriever, Reranker

prompt = '你将扮演一个人工智能问答助手的角色，完成一项对话任务。在这个任务中，你需要根据给定的上下文以及问题，给出你的回答。'
```
</details>

```python
documents = Document(dataset_path='/file/to/yourpath', embed=lazyllm.TrainableModule('bge-large-zh-v1.5'))
documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)

with pipeline() as ppl:
    with parallel().sum as ppl.prl:
        prl.retriever1 = Retriever(documents, group_name="sentences", similarity="cosine", topk=3)
        prl.retriever2 = Retriever(documents, "CoarseChunk", "bm25_chinese", 0.003, topk=3)

    ppl.reranker = Reranker("ModuleReranker", model="bge-reranker-large", topk=1) | bind(query=ppl.input)
    ppl.formatter = (lambda nodes, query: dict(context_str="".join([node.get_content() for node in nodes]), query=query)) | bind(query=ppl.input)
    ppl.llm = lazyllm.TrainableModule("internlm2-chat-7b").prompt(lazyllm.ChatPrompter(prompt, extro_keys=["context_str"]))

lazyllm.WebModule(ppl, port=23456).start().wait()
```
