# RAG (检索增强生成)

RAG (Retrieval-Augmented Generation) 是 LazyLLM 的核心功能之一，用于构建基于知识库的问答系统。

## 核心概念

LazyLLM的RAG功能实现由以下组件构成：

| 组件名称 | 组件功能 | 参考文档 |
|---------|---------|---------|
| Document | 文档处理的主入口 | [Document使用示例](../assets/rag/document.md) |
| Reader | 文档解析 | [Reader使用示例](../assets/rag/reader.md) |
| Transform | 文档切分 | [Transform使用示例](../assets/rag/transform.md) |
| Store | 切分后数据存储 | [Store存储使用示例](../assets/rag/store.md) |
| Retriever | 数据召回 | [Retriever使用示例](../assets/rag/retriever.md) |
| Reranker | 数据重排 | [Reranker使用示例](../assets/rag/reranker.md) |

### 1. Document (文档管理)

Document 是 LazyLLM 的文档管理组件，是所有文档和处理配置的主入口。支持本地目录加载及通过API上传两种方式实现文档解析。

#### 参数说明

    1.dataset_path：指定从哪个文件目录构建
    2.embed：期望在语义检索中使用的向量模型。 如果需要对文本生成多个向量模型，此处需要通过字典的方式指定，key 标识 embedding 的名字，value 为对应的 embedding 模型
    3.manager：是否使用 ui 界面会影响 Document 内部的处理逻辑，默认为 False
    4.launcher：启动服务的方式，集群应用会用到这个参数，单机应用可以忽略
    5.store_conf：配置使用哪种存储引擎保存文档解析结果
    6.doc_fields：配置需要存储和检索的字段及对应的类型（当前在使用内存存储、Chroma以及Milvus向量数据库时支持该功能）

#### 基本用法

```python
import lazyllm

documents = lazyllm.Document(
    dataset_path="/path/to/your/doc/dir",
    embed=lazyllm.OnlineEmbeddingModule(),
    manager=False
)
```

#### 节点与节点组（文档切分）

Document 中加载并解析完成的文件，会按照指定的规则（在 LazyLLM 中被称为 变换（Transform）），被进一步细分成不同粒度切片节点（Node），并存至对于的节点组（Node Group）当中，用户可以通过 Document.create_node_group() 来创建自己的 Node Group。
create_node_group() 支持以下传参:

1.name：节点组的名字
2.transform：期望当前节点组按照什么规则分割节点，支持传入基于NodeTransform的变换类，同时支持传入一个lambda函数，用于操作传入节点的实际内容
3.parent：当前定义节点组的父节点组，用于指定本次转换是基于哪个节点组进行的，默认是整篇文档（名为 lazyllm-root 的根节点组）。

创建节点组使用示例
```python
docs = Document()
docs.create_node_group(name='block',
                       transform=lambda d: d.split('\n'))

docs.create_node_group(name='block-summary',
                       transform=lambda b: summary_llm(b),
                       parent='block')
```

#### 存储和索引

LazyLLM 提供了多种存储配置选项，以满足不同的存储和检索需求。用户可以通过配置store_conf参数，选择不同配置的存储引擎： 配置项参数 store_conf 是一个字典，支持同时配置切片存储（segment）与向量存储（vector）两个配置，即：
```python
store_conf = {"segment_store": {}, "vector_store": {}}
```

存储配置示例
```python
store_conf = {
    'segment_store': {
        'type': 'map',
        'kwargs': {
            'uri': '/path/to/segment/dir/sqlite3.db',
        },
    },
    'vector_store': {
        'type': 'chroma',
        'kwargs': {
            'dir': '/path/to/vector/dir',
            'index_kwargs': {
                'hnsw': {
                    'space': 'cosine',
                    'ef_construction': 200,
                }
            }
        },
    },
}
```

需要存储额外信息可以通过doc_fields进行添加
```python
doc_fields = {
    'author': DocField(data_type=DataType.VARCHAR, max_size=128, default_value=' '),
    'public_year': DocField(data_type=DataType.INT32),
}
```

具体使用示例:[Document使用示例](../assets/rag/document.md)

### 2. Reader（读取器）

reader负责对检索阶段获取的相关文档进行恰当的处理，以保证后面可以生成高质量的回答响应。
可以通过注册的方式注册进document实现指定文件阅读器，支持局部和全局注册两种注册方式。
支持自定义reader和继承基类实现reader的注册，并且提供了高性能开源工具的使用。

具体使用示例参考: [Reader使用示例](../assets/rag/reader.md)

### 3. Transform（切分器）

transform负责将reader读取器解析出来的文档内容进行转换，包括切分，聚合，增强等操作。
目前transform内置了CharacterSplitter，RecursiveSplitter，SentenceSplitter，MarkdownSplitter, XMLSplitter, HTMLSPlitter, JSONSplitter, YAMLSplitter，GeneralCodeSplitter和CodeSplitter切分类。

也支持自定义transform用于文档的切分。

具体使用示例参考: [Transform使用示例](../assets/rag/transform.md)

### 4. Store（后端存储）

在定义好 Node Group 的转换规则之后，LazyLLM 会把检索过程中用到的转换得到的 Node Group 内容保存起来，这样后续使用的时候可以避免重复执行转换操作。为了方便用户存取不同种类的数据，LazyLLM 支持用户自定义存储后端。

如果没有指定，LazyLLM 默认使用 MapStore （基于 dict 的 key/value 存储）作为存储后端。 用户可以通过 Document 的参数 store_conf 来指定其它存储后端。例如想使用 Milvus 作为存储后端，我们可以这样配置：

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

在最新版本的 LazyLLM 中，推荐传入segment_store 和 vector_store 两个字段，分别指定切片存储和向量存储的配置。对于原先传入 type 字段的情况，LazyLLM 会自动将 type 字段自动映射，如果为向量存储，则默认切片存储为 MapStore。
如果用户期望在本地存储切片数据，可以传入 segment_store 字段，指定 type 为 map，并传入 uri 字段，指定切片存储的目录。

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

其余后端存储使用示例，参考[Store存储使用示例](../assets/rag/store.md)

### 5. Retriever (检索器)

Retriever 负责根据查询从文档中检索相关内容。

#### 基本用法

```python
retriever = lazyllm.Retriever(
    doc=documents,
    group_name="CoarseChunk",
    similarity="bm25_chinese",
    similarity_cut_off=0.003,
    topk=3
)

# 检索
doc_node_list = retriever(query="用户问题")
```

#### 相似度计算方法

LazyLLM 内置的相似度计算方法：
- `bm25_chinese` - 中文 BM25
- `bm25_english` - 英文 BM25
- `cosine` - 余弦相似度
- `euclidean` - 欧氏距离
- `manhattan` - 曼哈顿距离
- `dot` - 点积

详细使用示例参考: [Retriever使用示例](../assets/rag/retriever.md)

### 6. Reranker (排序器)

Reranker 用于对多个检索结果重新排序，提高相关性。

#### 基本用法

```python
reranker = lazyllm.Reranker(
    name='ModuleReranker',
    model=lazyllm.OnlineEmbeddingModule(type="rerank"),
    topk=1
)

# 对多个检索结果排序
doc_node_list = reranker(
    nodes=doc_node_list_1 + doc_node_list_2,
    query="用户问题"
)
```

具体使用示例参考: [Reranker使用示例](../assets/rag/reranker.md)

### 7. 完整 RAG 流程

#### 基础 RAG 流程

```python
import lazyllm

# Part1: 创建文档对象
documents = lazyllm.Document(
    dataset_path="/path/to/your/doc/dir",
    embed=lazyllm.OnlineEmbeddingModule(),
    manager=False
)

# Part2: 创建检索器
retriever = lazyllm.Retriever(
    doc=documents,
    group_name="CoarseChunk",
    similarity="bm25_chinese",
    similarity_cut_off=0.003,
    topk=3
)

# Part3: 创建大模型
llm = lazyllm.OnlineChatModule()

# Part4: 设置提示词
prompt = '你将扮演一个人工智能问答助手的角色，完成一项对话任务。在这个任务中，你需要根据给定的上下文以及问题，给出你的回答。'
llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))

# Part5: 查询和生成
query = input("query(enter 'quit' to exit): ")
doc_node_list = retriever(query=query)
res = llm({
    "query": query,
    "context_str": "".join([node.get_content() for node in doc_node_list]),
})
print(f"answer: {res}")
```

#### 使用 Flow 的 RAG 流程

```python
import lazyllm
from lazyllm import bind

documents = lazyllm.Document(
    dataset_path="/path/to/your/doc/dir",
    embed=lazyllm.OnlineEmbeddingModule(),
    manager=False
)

documents.create_node_group(
    name="sentences",
    transform=lambda s: '。'.split(s)
)

prompt = '你将扮演一个人工智能问答助手的角色，完成一项对话任务。在这个任务中，你需要根据给定的上下文以及问题，给出你的回答。'

# 使用 pipeline 构建流程
with lazyllm.pipeline() as ppl:
    with lazyllm.parallel().sum as ppl.prl:
        prl.retriever1 = lazyllm.Retriever(
            doc=documents,
            group_name="CoarseChunk",
            similarity="bm25_chinese",
            topk=3
        )
        prl.retriever2 = lazyllm.Retriever(
            doc=documents,
            group_name="sentences",
            similarity="cosine",
            topk=3
        )

    ppl.reranker = lazyllm.Reranker(
        name='ModuleReranker',
        model=lazyllm.OnlineEmbeddingModule(type="rerank"),
        topk=1
    ) | bind(query=ppl.input)

    ppl.formatter = (
        lambda nodes, query: dict(
            context_str="".join([node.get_content() for node in nodes]),
            query=query,
        )
    ) | bind(query=ppl.input)

    ppl.llm = lazyllm.OnlineChatModule().prompt(
        lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str'])
    )

rag = lazyllm.ActionModule(ppl)
rag.start()

query = input("query(enter 'quit' to exit): ")
res = rag(query)
print(f"answer: {res}")
```

#### 综合优化后的RAG实现

```python
import lazyllm

# 定义嵌入模型和重排序模型
# embedding_model = lazyllm.OnlineEmbeddingModule()
embedding_model = lazyllm.TrainableModule("bge-large-zh-v1.5").start()

# 如果要使用在线重排模型
# 目前LazyLLM仅支持 qwen和glm 在线重排模型，请指定相应的 API key。
# online_rerank = lazyllm.OnlineEmbeddingModule(type="rerank")
# 本地重排序模型
offline_rerank = lazyllm.TrainableModule('bge-reranker-large').start()

docs = lazyllm.Document("/mnt/lustre/share_data/dist/cmrc2018/data_kb", embed=embedding_model)
docs.create_node_group(name='block', transform=(lambda d: d.split('\n')))

# 定义检索器
retriever1 = lazyllm.Retriever(docs, group_name="CoarseChunk", similarity="cosine", topk=3) 
retriever2 = lazyllm.Retriever(docs, group_name="block", similarity="bm25_chinese", topk=3)

# 定义重排器
reranker = lazyllm.Reranker('ModuleReranker', model=offline_rerank, topk=3)

# 定义大模型
llm = lazyllm.TrainableModule('internlm2-chat-20b').deploy_method(lazyllm.deploy.Vllm).start()

# prompt 设计
prompt = '你是一个友好的 AI 问答助手，你需要根据给定的上下文和问题提供答案。\
          根据以下资料回答问题：\
          {context_str} \n '
llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))

# 执行推理
query = "2008年有哪些赛事？"
result1 = retriever1(query=query)
result2 = retriever2(query=query)
result = reranker(result1+result2, query=query)

# 将query和召回节点中的内容组成dict，作为大模型的输入
res = llm({"query": query, "context_str": "".join([node.get_content() for node in result])})

print(f'Answer: {res}')
```

## 高级功能

### 自定义文档解析器（MinerU）

```python
from lazyllm.tools.rag.readers import MineruPDFReader

# 注册 PDF 解析器
documents.add_reader(
    "*.pdf",
    MineruPDFReader(url="http://127.0.0.1:8888")
)
```

### 自定义存储后端

```python
from lazyllm.tools.rag import DocField, DataType

# 使用 Milvus 作为存储后端
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

# 自定义字段
doc_fields = {
    'comment': DocField(
        data_type=DataType.VARCHAR,
        max_size=65535,
        default_value=' '
    ),
    'signature': DocField(
        data_type=DataType.VARCHAR,
        max_size=32,
        default_value=' '
    )
}

documents = lazyllm.Document(
    dataset_path="/path/to/your/doc/dir",
    embed=lazyllm.OnlineEmbeddingModule(),
    store_conf=store_conf,
    doc_fields=doc_fields
)
```

### 自定义相似度计算

```python
import lazyllm
from typing import List, Tuple

@lazyllm.tools.rag.register_similarity(mode='text', batch=True)
def MySimilarityFunc(query: str, nodes: List, **kwargs) -> List[Tuple]:
    return [(node, 0.0) for node in nodes]

my_retriever = lazyllm.Retriever(
    doc=documents,
    group_name="sentences",
    similarity="MySimilarityFunc",
    topk=3
)
```

### 自定义 Reranker

```python
import lazyllm
from typing import List

@lazyllm.tools.rag.register_reranker(batch=True)
def MyReranker(nodes: List, **kwargs) -> List:
    return nodes

my_reranker = lazyllm.Reranker(name="MyReranker")
```

### 远程部署 Document

```python
# 服务端启动
docs = lazyllm.Document(
    dataset_path="rag_master",
    name="doc_server",
    embed=lazyllm.TrainableModule("bge-large-zh-v1.5"),
    server=9200,
    store_conf=store_conf,
    doc_fields=doc_fields
)
docs.create_node_group(name="sentences", transform=lambda s: '。'.split(s))
docs.activate_groups(["sentences", "CoarseChunk"])
docs.start()

# 客户端接入
docs2 = lazyllm.Document(
    url="http://127.0.0.1:9200/",
    name="doc_server"
)
retriever = lazyllm.Retriever(doc=docs2, group_name="sentences", topk=3)
```

## 使用场景

- 知识库问答系统
- 企业文档检索
- 学术论文问答
- 技术文档助手
- 领域知识专家系统
