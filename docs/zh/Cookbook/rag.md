# 知识库问答助手

本文将展示如何实现一个知识库问答助手。在开始本节之前，建议先阅读 [RAG 最佳实践](../Best%20Practice/rag.md)。

!!! abstract "通过本节您将学习到 LazyLLM 的以下要点"

    - RAG 相关模块：[Document][lazyllm.tools.Document]，[Retriever][lazyllm.tools.Retriever] 和 [Reranker][lazyllm.tools.Reranker]
    - 内置的 ChatPrompter 模块
    - 内置的流程模块：[Pipeline][lazyllm.flow.Pipeline] 和 [Parallel][lazyllm.flow.Parallel]

## 版本-1

从 [RAG 最佳实践](../Best%20Practice/rag.md) 可知，RAG 的核心在于基于特定的文档集合的回答用户提出的问题。根据这个目标，我们设计了下面的流程：

![rag-cookbook-1](../assets/rag-cookbook-1.svg)

`Query` 是用户输入的查询；`Retriever` 根据用户的查询从文档集合 `Document` 中找到匹配的文档；大模型 `LLM` 基于 `Retriever` 传过来的文档，结合用户的查询，给出最终的回答。

下面的例子 `rag.py` 实现了上述功能：

```python
# Part0

import lazyllm

# Part1

documents = lazyllm.Document(dataset_path="/path/to/your/doc/dir",
                             embed=lazyllm.OnlineEmbeddingModule(),
                             manager=False)

# Part2

retriever = lazyllm.Retriever(doc=documents,
                              group_name="CoarseChunk",
                              similarity="bm25_chinese",
                              similarity_cut_off=0.003,
                              topk=3)

# Part3

llm = lazyllm.OnlineChatModule()

# Part4

prompt = '你将扮演一个人工智能问答助手的角色，完成一项对话任务。在这个任务中，你需要根据给定的上下文以及问题，给出你的回答。'
llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))

# Part5

query = input("query(enter 'quit' to exit): ")
if query == "quit":
    exit(0)

# Part6

doc_node_list = retriever(query=query)

res = llm({
    "query": query,
    "context_str": "".join([node.get_content() for node in doc_node_list]),
})

# Part7

print(f"answer: {res}")
```

我们来简单说明下各部分的代码。

1. Part0 导入了 `lazyllm`。

2. Part1 从本地加载知识库目录，并使用内置的 `OnlineEmbeddingModule` 作为向量模型。

3. Part2 创建一个用于检索文档的 `Retriever`，并使用内置的 `CoarseChunk`（参考 [CoarseChunk 的定义][llm.tools.Retriever]）将文档按指定的大小分块，然后使用内置的 `bm25_chinese` 作为相似度计算函数，并且丢弃相似度小于 0.003 的结果，最后取最相近的 3 篇文档。

4. Part3 创建用来回答问题的大模型实例。

5. Part4 由于需要大模型基于我们提供的文档回答问题，我们在提问的时候需要告诉大模型哪些是参考资料，哪个是我们的问题。这里使用内置的 `ChatPrompter` 将 `Retriever` 返回的文档内容作为参考资料告诉大模型。这里用到的 `ChatPrompter` 两个参数含义如下：

    * `instruction`：提供给大模型的指引内容；
    * `extra_keys`：从传入的 dict 中的哪个字段获取参考资料。

6. Part5 打印提示信息，等待用户输入要查询的内容。

7. Part6 是主流程：接收用户的输入，使用 `Retriever` 根据用户输入的 `query` 检索出相关的文档，然后把 `query` 和参考资料 `context_str` 打包成一个 dict 传给大模型，并等待结果返回。

8. Part7 把结果打印到屏幕上。

运行前修改 Part1 中 `Document` 的 `dataset_path` 参数指向需要检索的目录，还有在命令行中设置好申请到的商汤日日新模型的鉴权信息（参考 [快速入门](../index.md#hello-world)），然后运行：

```python
python3 rag.py
```

输入要查询的内容，等待大模型给我们返回结果。

在这个例子中，我们在检索相似文档的时候只用了内置的算法 `CoarseChunk`，将文档按照固定的长度分块，对每个分块计算相似度。这种方法在某些场景下可能有不错的效果；但是在大部分情况下，这种简单粗暴的分块可能会将重要信息拆成两段，造成信息损坏，从而得到的文档并不是和用户输入的查询相关的。

## 版本-2：添加召回策略和排序

为了从不同的维度来衡量文档和查询的相关性，我们在 Part1 增加一个从句子粒度对文档进行拆分的名为 `sentences` 的 `Node Group`：

```python
documents.create_node_group(name="sentences",
                            transform=lambda d: '。'.split(d))
```

并且在 Part2 中使用 `cosine` 来计算句子和用户查询的相似度，然后筛选最相关的 3 篇文档：

```python
retriever2 = Retriever(doc=documents,
                       group_name="sentences",
                       similarity="cosine",
                       topk=3)
```

我们现在有两个不同的检索函数返回了不同顺序的结果，并且这些结果可能是有重复的，这时就需要用 `Reranker` 对多个结果重新排序。我们在 Part2 和 Part3 之间插入 Part8，新增一个 `ModuleReranker`，使用指定模型来排序并取最符合条件的 1 篇文章：

```python
# Part8

reranker = Reranker(name="ModuleReranker",
                    model="bge-reranker-large",
                    topk=1)
```

`ModuleReranker` 是 `LazyLLM` 内置的一个通用排序模块，它可以简化使用模型来对文档排序的场景。可以查看 [最佳实践中关于 Reranker 的介绍](../Best%20Practice/rag.md#Reranker)。

同时我们将 Part6 改写如下：

```python
# Part6-2

doc_node_list_1 = retriever1(query=query)
doc_node_list_2 = retriever2(query=query)

doc_node_list = reranker(nodes=doc_node_list_1+doc_node_list_2, query=query)

res = llm({
    "query": query,
    "context_str": "".join([node.get_content() for node in doc_node_list]),
})
```

改进后的流程如下：

![rag-cookbook-2](../assets/rag-cookbook-2.svg)

<details>

<summary>附完整代码（点击展开）：</summary>

```python
# Part0

import lazyllm

# Part1

documents = lazyllm.Document(dataset_path="rag_master",
                             embed=lazyllm.OnlineEmbeddingModule(),
                             manager=False)

documents.create_node_group(name="sentences",
                            transform=lambda s: '。'.split(s))

# Part2

retriever1 = lazyllm.Retriever(doc=documents,
                               group_name="CoarseChunk",
                               similarity="bm25_chinese",
                               similarity_cut_off=0.003,
                               topk=3)

retriever2 = lazyllm.Retriever(doc=documents,
                               group_name="sentences",
                               similarity="cosine",
                               topk=3)

# Part8

reranker = lazyllm.Reranker(name='ModuleReranker',
                            model=lazyllm.OnlineEmbeddingModule(type="rerank"),
                            topk=1)

# Part3

llm = lazyllm.OnlineChatModule()

# Part4

prompt = '你将扮演一个人工智能问答助手的角色，完成一项对话任务。在这个任务中，你需要根据给定的上下文以及问题，给出你的回答。'
llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))

# Part5

query = input("query(enter 'quit' to exit): ")
if query == "quit":
    exit(0)

# Part6

doc_node_list_1 = retriever1(query=query)
doc_node_list_2 = retriever2(query=query)

doc_node_list = reranker(nodes=doc_node_list_1+doc_node_list_2, query=query)

res = llm({
    "query": query,
    "context_str": "".join([node.get_content() for node in doc_node_list]),
})

# Part7

print(f"answer: {res}")
```

</details>

## 版本-3：使用 flow

从 版本-2 的流程图可以看到整个流程已经比较复杂了。我们注意到两个（或多个）`Retriever` 的检索过程互不影响，它们可以并行执行。同时整个流程上下游之间也有着明确的依赖关系，需要保证上游执行完成之后才可以进行下一步。

`LazyLLM` 提供了一系列的辅助工具来简化执行流程的编写。我们可以用 [parallel](../Best%20Practice/flow.md#parallel) 把两个检索过程封装起来，让其可以并行执行；同时使用 [pipeline](../Best%20Practice/flow.md#pipeline) 把整个流程封装起来。重写后的整个程序如下：

```python
import lazyllm

# Part0

documents = lazyllm.Document(dataset_path="rag_master",
                             embed=lazyllm.OnlineEmbeddingModule(),
                             manager=False)

documents.create_node_group(name="sentences",
                            transform=lambda s: '。'.split(s))

prompt = '你将扮演一个人工智能问答助手的角色，完成一项对话任务。在这个任务中，你需要根据给定的上下文以及问题，给出你的回答。'

# Part1

with lazyllm.pipeline() as ppl:
    with lazyllm.parallel().sum as ppl.prl:
        prl.retriever1 = lazyllm.Retriever(doc=documents,
                                           group_name="CoarseChunk",
                                           similarity="bm25_chinese",
                                           topk=3)
        prl.retriever2 = lazyllm.Retriever(doc=documents,
                                           group_name="sentences",
                                           similarity="cosine",
                                           topk=3)

    ppl.reranker = lazyllm.Reranker(name='ModuleReranker',
                                    model=lazyllm.OnlineEmbeddingModule(type="rerank"),
                                    topk=1) | bind(query=ppl.input)

    ppl.formatter = (
        lambda nodes, query: dict(
            context_str = "".join([node.get_content() for node in nodes]),
            query = query,
        )
    ) | bind(query=ppl.input)

    ppl.llm = lazyllm.OnlineChatModule().prompt(
        lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))

# Part2

rag = lazyllm.ActionModule(ppl)
rag.start()

query = input("query(enter 'quit' to exit): ")
if query == "quit":
    exit(0)

res = rag(query)
print(f"answer: {res}")
```

在 Part1 中，外层的 `with` 语句所构造的 `pipeline` 即是我们上面的整个处理流程：先用 `Retriever` 检索出文档，接着交给 `Reranker` 排序选出合适的文档，再传给 `LLM` 参考，最后得到答案。其中分别实现两种不同检索策略的两个 `Retriever` 被封装进了一个 `parallel` 中，内层的 `with` 语句构造的 `parallel.sum`，表示把所有模块的输出合并之后作为 `parallel` 模块的输出。因为大模型 `llm` 需要接收的参数需要放在一个 dict 中，我们新增了一个 `formatter` 的组件，用于把用户查询的内容 `query` 和由 `Reranker` 选出的参考文档拼接而成的提示内容打包，作为 `llm` 的输入。

图示如下：

![rag-cookbook-3](../assets/rag-cookbook-3.svg)

`pipeline` 和 `parallel` 只是方便流程搭建以及可能的一些性能优化，并不会改成程序的逻辑。另外用户查询作为重要的提示内容，基本是中间每个模块都会用到，我们在这里还用了 `LazyLLM` 提供的 [bind()](../Best%20Practice/flow.md#use-bind) 函数将用户查询 `query` 作为参数传给 `ppl.reranker` 和 `ppl.formatter`。

## 版本-4：自定义检索和排序策略

上面的例子使用的都是 `LazyLLM` 内置的组件。现实中总有我们覆盖不到的用户需求，为了满足这些需求，`Retriever` 和 `Reranker` 提供了插件机制，用户可以自定义检索和排序策略，通过 `LazyLLM` 提供的注册接口添加到框架中。

为了简化说明和体现我们编写的策略的效果，这里不考虑策略是否实用。

首先我们来实现一个相似度计算的函数，并使用 `LazyLLM` 提供的 [register_similarity()](../Best%20Practice/rag.md#Retriever) 函数把这个函数注册到框架中：

```python
@lazyllm.tools.rag.register_similarity(mode='text', batch=True)
def MySimilarityFunc(query: str, nodes: List[DocNode], **kwargs) -> List[Tuple[DocNode, float]]:
    return [(node, 0.0) for node in nodes]
```

这个函数给每个文档都打了零分，并且按照输入的文档列表的顺序返回新的结果列表。也就是说，每次检索的结果取决于读取文档的顺序。

类似地，我们使用 `LazyLLM` 提供的 [register_reranker()](../Best%20Practice/rag.md#Reranker) 函数来注册一个按照输入的文档列表顺序返回结果的函数：

```python
@lazyllm.tools.rag.register_reranker(batch=True)
def MyReranker(nodes: List[DocNode], **kwargs) -> List[DocNode]:
    return nodes
```

之后就可以像内置的组件一样通过函数名称来使用这些扩展，例如下面的代码片段分别生成了使用我们上面定义的相似度计算和检索策略的 `Retriever` 和 `Reranker`：

```python
my_retriever = lazyllm.Retriever(doc=documents,
                                 group_name="sentences",
                                 similarity="MySimilarityFunc",
                                 topk=3)

my_reranker = Reranker(name="MyReranker")
```

当然返回的结果可能会很奇怪 :)

这里只是简单介绍了怎么使用 `LazyLLM` 注册扩展的机制。可以参考 [Retriever](../Best%20Practice/rag.md#Retriever) 和 [Reranker](../Best%20Practice/rag.md#Reranker) 的文档，在遇到不能满足需求的时候通过编写自己的相似度计算和排序策略来实现自己的应用。

## 版本-5：自定义存储后端

在定义好 Node Group 的转换规则之后，`LazyLLM` 会把检索过程中用到的转换得到的 Node Group 内容保存起来，这样后续使用的时候可以避免重复执行转换操作。为了方便用户存取不同种类的数据，`LazyLLM` 支持用户自定义存储后端。

如果没有指定，`LazyLLM` 默认使用 `MapStore` （基于 dict 的 key/value 存储）作为存储后端。
用户可以通过 `Document` 的参数 `store_conf` 来指定其它存储后端。例如想使用 Milvus 作为存储后端，我们可以这样配置：

```python
milvus_store_conf = {
    'type': 'milvus',
    'kwargs': {
        'uri': '/path/to/milvus/dir/milvus.db',
        'index_kwargs': {
            'index_type': 'HNSW',
            'metric_type': 'COSINE',
        }
    },
}
```

其中 `type` 为后端类型，`kwargs` 时需要传递给后端的参数。各字段含义可见 [RAG 最佳实践](../Best%20Practice/rag.md#存储和索引)。

!!! 注意

    在最新版本的 `LazyLLM` 中，推荐传入`segment_store` 和 `vector_store` 两个字段，分别指定切片存储和向量存储的配置。对于原先传入 `type` 字段的情况，`LazyLLM` 会自动将 `type` 字段自动映射，如果为向量存储，则默认切片存储为 `MapStore`。

如果用户期望在本地存储切片数据，可以传入 `segment_store` 字段，指定 `type` 为 `map`，并传入 `uri` 字段，指定切片存储的目录。
```python
store_conf = {
    'segment_store': {
        'type': 'map',
        'kwargs': {
            'uri': '/path/to/segment/dir/sqlite3.db',
        },
    },
    'vector_store': {
        'type': 'milvus',
        'kwargs': {
            'uri': '/path/to/milvus/dir/milvus.db',
            'index_kwargs': {
                'index_type': 'HNSW',
                'metric_type': 'COSINE',
            }
        },
    },
}
```

如果使用 Milvus，还可以给 `Document` 传递 `doc_fields` 参数，用于指定需要存储的字段及类型等信息。例如下面的配置：

```python
doc_fields = {
    'comment': DocField(data_type=DataType.VARCHAR, max_size=65535, default_value=' '),
    'signature': DocField(data_type=DataType.VARCHAR, max_size=32, default_value=' '),
}
```

配置了两个字段 `comment` 和 `signature` 两个字段。其中 `comment` 是一个字符串，最大长度是 65535，默认值为空；`signature` 类型是一个字符串，最大长度是 32，默认值为空。

下面是一个使用 Milvus 作为存储后端的完整例子：

<details>

<summary>附完整代码（点击展开）：</summary>

```python
# -*- coding: utf-8 -*-

import os
import lazyllm
from lazyllm import bind, config
from lazyllm.tools.rag import DocField, DataType
import shutil

class TmpDir:
    def __init__(self):
        self.root_dir = os.path.expanduser(os.path.join(config['home'], 'rag_for_example_ut'))
        self.rag_dir = os.path.join(self.root_dir, 'rag_master')
        os.makedirs(self.rag_dir, exist_ok=True)
        self.store_file = os.path.join(self.root_dir, "milvus.db")

    def __del__(self):
        shutil.rmtree(self.root_dir)

tmp_dir = TmpDir()

milvus_store_conf = {
    'type': 'milvus',
    'kwargs': {
        'uri': tmp_dir.store_file,
        'index_kwargs': {
            'index_type': 'HNSW',
            'metric_type': 'COSINE',
        }
    },
}

doc_fields = {
    'comment': DocField(data_type=DataType.VARCHAR, max_size=65535, default_value=' '),
    'signature': DocField(data_type=DataType.VARCHAR, max_size=32, default_value=' '),
}

prompt = 'You will play the role of an AI Q&A assistant and complete a dialogue task.'\
    ' In this task, you need to provide your answer based on the given context and question.'

documents = lazyllm.Document(dataset_path=tmp_dir.rag_dir,
                             embed=lazyllm.TrainableModule("bge-large-zh-v1.5"),
                             manager=False,
                             store_conf=milvus_store_conf,
                             doc_fields=doc_fields)

documents.create_node_group(name="block", transform=lambda s: s.split("\n") if s else '')

with lazyllm.pipeline() as ppl:
    ppl.retriever = lazyllm.Retriever(doc=documents, group_name="block", topk=3)

    ppl.reranker = lazyllm.Reranker(name='ModuleReranker',
                                    model="bge-reranker-large",
                                    topk=1,
                                    output_format='content',
                                    join=True) | bind(query=ppl.input)

    ppl.formatter = (
        lambda nodes, query: dict(context_str=nodes, query=query)
    ) | bind(query=ppl.input)

    ppl.llm = lazyllm.TrainableModule('internlm2-chat-7b').prompt(
        lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))

if __name__ == '__main__':
    filters = {
        'signature': ['sig_value'],
    }
    rag = lazyllm.ActionModule(ppl)
    rag.start()
    res = rag('何为天道？', filters=filters)
    print(f'answer: {res}')
```

</details>

## 版本-6：离在线分离，接入远程部署的 `Document`

RAG系统往往包含文档解析与在线问答两阶段，其中文档解析阶段耗时较长，但可以离线执行，而问答阶段则需要快速响应。为了满足这一需求，`LazyLLM` 提供了 `Document` 的远程部署与接入功能，支持用户将 `Document` 部署在远程服务器上，并使用 url 的方式接入。

### 使用服务模式启动 `Document`
```python
docs = lazyllm.Document(dataset_path="rag_master",
                        name="doc_server",
                        embed=lazyllm.TrainableModule("bge-large-zh-v1.5"),
                        server=9200,
                        store_conf=milvus_store_conf,
                        doc_fields=doc_fields)
docs.create_node_group(name="sentences", transform=lambda s: '。'.split(s))
docs.activate_groups(["sentences", "CoarseChunk"]) # 使用 docs.activate_group('sentences', embed_keys=['xxx']) 激活单个节点组

docs.start()
```

其中 `server` 参数指定为 `9200`，表示使用服务模式启动 `Document`，在调用 `docs.start()` 之后，`Document` 会启动一个服务，并指定为 `9200` 端口。（也可以指定为 `True`，表示随机分配一个端口）

在启动前，确保文档服务已经创建了所需的所有节点组，并根据需求执行 `docs.activate_groups()` 激活相应的节点组，只有激活的节点组才会被服务发现，并在后续的文档解析和检索中使用。此处我们激活了 `sentences` 和 `CoarseChunk` 两个节点组，默认激活所有向量模型，如果`docs`传入了字典格式的多个向量模型，传参时仅需传入对应向量模型的 key 即可。


### 使用 url 接入 `Document`

启动后，假设文档服务部署在 `127.0.0.1` 的 `9200` 端口，则可以通过 `http://127.0.0.1:9200/` 访问文档服务。我们使用 `lazyllm.UrlDocument` 来接入文档服务，并指定文档服务的名称 `doc_server`。
```python
docs2 = lazyllm.Document(url="http://127.0.0.1:9200/", name="doc_server")
retriever = lazyllm.Retriever(doc=docs2, group_name="sentences", topk=3)

query = "何为天道？"
res = retriever(query=query)
print(f"answer: {res}")
```

此时，`docs2` 的 `url` 参数指向了文档服务的地址，`name` 参数指定为 `doc_server`，表示文档服务的名称。用户便可以直接使用 `docs2` 进行检索，而无需关心文档服务部署在何处。
