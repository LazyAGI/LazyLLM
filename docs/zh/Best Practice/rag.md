# RAG

检索增强生成（Retrieval-augmented Generation，RAG）是当前备受关注的大模型前沿技术之一。其工作原理是，当模型需要生成文本或回答问题时，首先会从一个庞大的文档集合中检索出相关的信息。这些检索到的信息随后会被用于指导生成过程，从而显著提高生成文本的质量和准确性。通过这种方式，RAG 能够在处理复杂问题时提供更加精确和有意义的回答，是自然语言处理领域的重要进展之一。这种方法的优越性在于它结合了检索和生成的优势，使得模型不仅能够生成流畅的文本，还能基于真实数据提供有依据的回答。

一般来说 RAG 的流程如下图所示：

![RAG intro](../assets/rag-intro.svg)

## 设计

根据上面的描述，我们将 RAG 抽象为如下流程：

![RAG modules relationship](../assets/rag-modules-relationship.svg)

### Document

从原理介绍中可以看到，文档集合中有各种各样的文档格式：可以是结构化的存储在数据库中的一条条记录，可以是 DOCX，PDF，PPT 等富文本或者 Markdown 这样的纯文本，也可能是从某个 API 获取的内容（如通过搜索引擎检索得到的相关信息）等等。由于集合内的文档格式各异，针对这些不同格式的文档，我们需要特定的解析器来提取其中有用的文本，图片，表格，音频和视频等内容。在 `LazyLLM` 中，这些用于提取特定内容的解析器被抽象成 `DataLoader`。目前 `LazyLLM` 内置的 `DataLoader` 可以支持 DOCX，PDF，PPT，EXCEL 等常见的富文本内容提取。使用 `DataLoader` 提取到的文档内容会被存储在 `Document` 中。

`Document` 支持本地目录加载及通过API上传两种方式实现文档解析。用户可以直接用以下语句从本地目录构建一个文档集合 `docs`。
```python
docs = Document(dataset_path='/path/to/doc/dir', embed=MyEmbeddingModule(), manager=False)
```

其中 `Document` 的构造函数有以下参数：
* `dataset_path`：指定从哪个文件目录构建；
* `embed`：期望在语义检索中使用的向量模型。 如果需要对文本生成多个向量模型，此处需要通过字典的方式指定，key 标识 embedding 的名字，value 为对应的 embedding 模型；
* `manager`：是否使用 ui 界面会影响 `Document` 内部的处理逻辑，默认为 `False`；
* `launcher`：启动服务的方式，集群应用会用到这个参数，单机应用可以忽略；
* `store_conf`：配置使用哪种存储引擎保存文档解析结果；
* `doc_fields`：配置需要存储和检索的字段及对应的类型（当前在使用内存存储、Chromadb以及Milvus向量数据库时支持该功能）
* 更多参数说明请前往[Document API][lazyllm.Document]查看。

#### 节点与节点组

`Document` 中加载并解析完成的文件，会按照指定的规则（在 `LazyLLM` 中被称为 `变换（Transform）`），被进一步细分成不同粒度切片节点（`Node`），并存至对于的节点组（`Node Group`）当中。

用户可以通过 `Document.create_node_group()` 来创建自己的 `Node Group`。

`create_node_group()` 支持以下传参：
* `name`：节点组的名字；
* `transform`：期望当前节点组按照什么规则分割节点，支持传入基于[NodeTransform][lazyllm.tool.rag.NodeTransform]的变换类，同时支持传入一个`lambda`函数，用于操作传入节点的实际内容。
* `parent`：当前定义节点组的父节点组，用于指定本次转换是基于哪个节点组进行的，默认是整篇文档（名为 `lazyllm-root` 的根节点组）。

!!! 注意

    `LazyLLM` 中提供了三种内置的节点组：
    * `FineChunk`：长度为 128 个 token，overlap 为 12 的节点组；
    * `MediumChunk`：长度为 256 个 token，overlap 为 25 的节点组；
    * `CoarseChunk`：长度为 1024 个 token，overlap 为 100 的节点组。
    对于以上三种节点组，用户无需手动创建，即可在后续的检索中使用。


我们可以通过以下例子，了解节点组是如何创建的：
```python
docs = Document()

# (1) 创建按换行符拆分的段落节点组
docs.create_node_group(name='block',
                       transform=lambda d: d.split('\n'))

# (2) 创建按照文档粒度总结的节点组
docs.create_node_group(name='doc-summary',
                       transform=lambda d: summary_llm(d))

# (3) 在block节点组的基础上，创建按句号分割的句子节点组
docs.create_node_group(name='sentence',
                       transform=lambda b: b.split('。'),
                       parent='block')

# (4) 创建针对block节点组调用大模型进行总结的节点组
docs.create_node_group(name='block-summary',
                       transform=lambda b: summary_llm(b),
                       parent='block')

# (4) 创建针对block节点组提取关键词的关键词节点组
docs.create_node_group(name='keyword',
                       transform=lambda b: keyword_llm(b),
                       parent='block')

# (6) 创建针对句子节点组，提取句子长度的长度节点组
docs.create_node_group(name='sentence-len',
                       transform=lambda s: len(s),
                       parent='sentence')
```

语句 1 以换行符为分割符，将所有文档都拆成了一个个的段落块，每个块就是 1 个 `Node`，这些 `Node` 构成了名为 `block` 的 `Node Group`。

语句 2 使用一个可以提取摘要的大模型把每个文档的摘要作为一个名为 `doc-summary` 的 `Node Group`，这个 `Node Group` 中只有一个 `Node`，内容就是整个文档的摘要。

由于 `block` 和 `doc-summary` 都是从 `lazyllm_root` 这个根节点经过不同的规则转换得到的，所以它俩都是 `lazyllm_root` 的子节点。

语句 3 在 `block` 这个 `Node Group` 的基础上进一步转换，使用中文句号作为分割符而得到一个个句子，每个句子都是一个 `Node`，共同构成了 `sentence` 这个 `Node Group`。

语句 4 在 `block` 这个 `Node Group` 的基础上，使用可以抽取摘要的大模型对其中的每个 `Node` 做处理，从而得到的每个段落摘要的 `Node`，组成了 `block-summary`。

语句 5 也是在 `block` 这个 `Node Group` 的基础上，在可以抽取关键词的大模型的帮助下，将每个段落都抽取出来一些关键词，每个段落的关键词是一个个的 `Node`，共同组成了 `keyword` 这个 `Node Group`。

最后语句 6 在 `sentence` 的基础上，统计了每个句子的长度，得到了一个包含每个句子长度的名为 `sentence-len` 的 `Node Group`。

语句 2，4，5 用到的提供摘要（summary）和关键词（keywords）抽取的功能，可以使用 `LazyLLM` 内置的 `LLMParser`。用法可以参考 [LLMParser 的使用说明][lazyllm.tools.LLMParser]。


这些 `Node Group` 的关系如下图所示：

![relationship of node groups](../assets/rag-relationship-of-node-groups.svg)


`Node Group` 的拆分粒度和规则各不相同，反映了文档不同方面的特征。在后续的处理中，我们通过在不同的场合使用这些特征，从而更好地判断文档和用户输入的查询内容的相关性。

#### 存储和索引

`LazyLLM` 提供了多种存储配置选项，以满足不同的存储和检索需求。用户可以通过配置`store_conf`参数，选择不同配置的存储引擎：
配置项参数 `store_conf` 是一个字典，支持同时配置切片存储（segment）与向量存储（vector）两个配置，即：
```python
store_conf = {"segment_store": {}, "vector_store": {}}
```

内部每个字典包含字段如下：

* `type`：存储后端类型。切片与向量存储目前支持的类型如下：
    - 切片存储（`segment_store`）:
        - `map`：内存 key/value 存储，可配置本地路径，以使用基于`sqlite3`的本地切片存储；
        - `opensearch`：使用 OpenSearch 引擎存储数据；
    - 向量存储（`vector_store`）:
        - `chromadb`：使用 ChromaDB 存储数据；
        - `milvus`：使用 Milvus 存储数据。
* `kwargs`：存储引擎所需客户端配置针对不同的存储引擎，`kwargs` 包含不同的参数，具体如下：
    - `map`：
        - `uri`（可选）：本地切片存储路径，基于`sqlite3`的本地切片存储。
    - `opensearch`：
        - `uris`（必填）：OpenSearch 存储地址（分布式支持传入多个地址），如 `ip:port` 格式的 url；
        - `client_kwargs`（必填）：OpenSearch 客户端配置，用于配置 OpenSearch 的连接参数，如 `user`、`password` 等，详情见[OpenSearch 官方文档](https://opensearch-project.github.io/opensearch-py/api-ref/clients/opensearch_client.html)；
        - `index_kwargs`（可选）：OpenSearch 索引配置，用于配置 OpenSearch 的索引参数，规定了索引类型、切片保存字段等；
    - `chromadb`：
        - `uri`（可选）：ChromaDB 远端存储地址，如 `ip:port` 格式的 url；
        - `dir`（可选）：本地存储数据的目录，与 `uri` 二选一；
        - `index_kwargs`（可选）：ChromaDB 索引配置，用于配置 ChromaDB 的索引参数，规定了向量索引类型，相似度计算方式等，详情见[ChromaDB 官方文档](https://docs.trychroma.com/docs/collections/configure)；
        - `client_kwargs`（可选）：ChromaDB 客户端配置，用于配置 ChromaDB 的连接参数；
    - `milvus`：
        - `uri`（必填）：Milvus 存储地址，可以是一个db文件路径或者如 `ip:port` 格式的 url；
        - `db_name`（可选）：Milvus 数据库层级隔离的名称，默认为 `lazyllm`；
        - `client_kwargs`（可选）：Milvus 客户端配置，用于配置 Milvus 的连接参数；
        - `index_kwargs`（可选）：Milvus 索引配置，可以是一个 dict 或者 list。如果是一个 dict 表示所有的 embedding index 使用同样的配置；如果是一个 list，list 中的元素是 dict，表示由 `embed_key` 所指定的 embedding 所使用的配置。当前只支持 `floaing point embedding` 和 `sparse embedding` 两种 embedding 类型，分别支持的参数如下：
            - `floating point embedding`：[https://milvus.io/docs/index-vector-fields.md?tab=floating](https://milvus.io/docs/index-vector-fields.md?tab=floating)
            - `sparse embedding`：[https://milvus.io/docs/index-vector-fields.md?tab=sparse](https://milvus.io/docs/index-vector-fields.md?tab=sparse)

下面是一个使用 内存 key/value 存储作为切片存储，ChromaDB 作为向量存储的配置样例：

```python
store_conf = {
    'segment_store': {
        'type': 'map',
        'kwargs': {
            'uri': '/path/to/segment/dir/sqlite3.db',
        },
    },
    'vector_store': {
        'type': 'chromadb',
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
如需对Milvus向量数据库进行多组参数配置，可参考如下格式，此处`embed_key`需要与Document多embedding的key一一对应:
```python
{
    ...
    'index_kwargs' = [
        {
            'embed_key': 'vec1',
            'index_type': 'HNSW',
            'metric_type': 'COSINE',
        },{
            'embed_key': 'vec2',
            'index_type': 'SPARSE_INVERTED_INDEX',
            'metric_type': 'IP',
        }
    ]
}
```

注意：使用 ChromaDB 或 Milvus 作为向量存储，如果期望进行某一字段的标量过滤作为检索条件，还需要提供可能作为搜索条件的特殊字段说明，通过 `doc_fields` 这个参数传入。`doc_fields` 是一个 dict，其中 key 为需要存储或检索的字段名称，value 是一个 `DocField` 类型的结构体，包含字段类型等信息。

例如，如果需要存储文档的作者信息和发表年份可以这样配置：

```python
doc_fields = {
    'author': DocField(data_type=DataType.VARCHAR, max_size=128, default_value=' '),
    'public_year': DocField(data_type=DataType.INT32),
}
```

### Retriever

文档集合中的文档不一定都和用户要查询的内容相关，因此接下来我们要使用 `Retriever` 从 `Document` 中筛选出和用户查询相关的文档。

例如，用户可以这样创建一个 `Retriever` 实例：

```python
retriever = Retriever(documents, group_name="sentence", similarity="cosine", topk=3)  # or retriever = Retriever([document1, document2, ...], group_name="sentence", similarity="cosine", topk=3)
```

表示在 `sentence` 这个节点组中执行向量检索，计算用户查询的内容 `query` 和每个节点（`Node`）的相似度。`topk` 表示最多取最相近的多少个节点。

`Retriever` 的构造函数有以下参数：

* `doc`：要从哪个 `Document` 中检索文档，或者要从哪些 `Document` 列表中检索文档；
* `group_name`：要使用文档的哪个 `Node Group` 来检索，使用 `LAZY_ROOT_NAME` 表示在原始文档内容中进行检索；
* `similarity`：指定用来计算 `Node` 和用户查询内容之间的相似度的函数名称，`LazyLLM` 内置的相似度计算函数有 `bm25`，`bm25_chinese` 和 `cosine`，用户也可以自定义自己的计算函数，如果不指定则默认使用向量检索；
* `similarity_cut_off`：丢弃相似度小于指定值的结果，默认为 `-inf`，表示不丢弃。 在多 embedding 场景下，如果需要对不同的 embedding 指定不同的值，则该参数需要以字典的方式指定，key 表示指定的是哪个 embedding， value 表示相应的阈值。如果所有 embedding 使用同一个阈值，则此参数只传一个数值即可；
* `index`：在哪个索引上进行查找，目前只支持 `default` 和 `smart_embedding_index`；
* `topk`：表示返回最相关的文档数，默认值为 6；
* `embed_keys`：表示通过哪些 embedding 做检索，不指定表示用全部 embedding 进行检索；
* `similarity_kw`：需要透传给 `similarity` 函数的参数。

用户可以通过使用 `LazyLLM` 提供的 `register_similarity()` 函数来注册自己的相似度计算函数。`register_similarity()` 有以下参数：

* `func`：用于计算相似度的函数；
* `mode`：计算模式，支持 `text` 和 `embedding` 两种，会影响传给 `func` 的参数；
* `decend`：是否降序排列，默认为 `True`；
* `batch`：是否多 batch，会影响传给 `func` 的参数和返回值。

!!! 注意

    外接存储引擎一般不支持自定义相似度计算函数，只有不使用外部存储引擎时，注册的`similarity` 参数才有效。

当 `mode` 的取值为 `text` 时表示使用 `Node` 的内容，计算函数的参数 `query` 的类型为 `str`，即需要和 `Node` 比较的文本内容，`Node` 的内容可以通过 `node.get_text()` 获取；若为 `embedding` 则表示使用 `Document` 初始化时指定的 `embed` 函数转换得到的向量来计算，此时 `query` 的类型为 `List[float]`，`Node` 的向量可以通过 `node.embedding` 来获取。返回值中的 `float` 表示文档的得分。

当 `batch` 为 `True` 时，计算函数的参数有 `nodes`，类型为 `List[DocNode]`，返回值类型为 `List[(DocNode, float)]`；若为 `False` 时，计算函数的参数有 `node`，类型为 `DocNode`，返回值类型为 `float`，表示文档的得分。

根据 `mode` 和 `batch` 不同的取值，用户自定义的相似度计算函数的原型有以下几种形式：

```python
# (1)
@lazyllm.tools.rag.register_similarity(mode='text', batch=True)
def dummy_similarity_func(query: str, nodes: List[DocNode], **kwargs) -> List[Tuple[DocNode, float]]:

# (2)
@lazyllm.tools.rag.register_similarity(mode='text', batch=False)
def dummy_similarity_func(query: str, nodes: List[DocNode], **kwargs) -> float:

# (3)
@lazyllm.tools.rag.register_similarity(mode='embedding', batch=True)
def dummy_similarity_func(query: List[float], nodes: List[DocNode], **kwargs) -> List[Tuple[DocNode, float]]:

# (4)
@lazyllm.tools.rag.register_similarity(mode='embedding', batch=False)
def dummy_similarity_func(query: List[float], node: DocNode, **kwargs) -> float:
```

`Retriever` 实例使用时需要传入要查询的 `query` 字符串，还有可选的过滤器 `filters` 用于字段过滤。`filters` 是一个 dict，其中 key 是要过滤的字段，value 是一个可取值列表，表示只要该字段的值匹配列表中的任意一个值即可。只有当所有的条件都满足该 node 才会被返回。

`filters` 的用法如下方代码所示：

```python
filters = {
    "author": ["A", "B", "C"],
    "publish_year": [2002, 2003, 2004],
}
doc_list = retriever(query=query, filters=filters)
```

其中 `filters` 的键值可以在初始化`Document`时传入`doc_fields`参数进行自定义（具体可参考 [Document 用法介绍](../Best%20Practice/rag.md#Document)）。除此之外，`filters`也支持通过以下内置元数据进行过滤，分别是：file_name（文件名）、file_type（文件类型）、file_size（文件大小）、creation_date（创建日期）、last_modified_date（最终修改日期）、last_accessed_date（最后访问日期）。

### Reranker

当我们从最初的文档集合筛选出和用户查询相关性比较高的文档后，接下来就可以进一步对这些文档进行排序，选出更贴合用户查询内容的文档。这一步工作由 `Reranker` 来完成。

例如，我们可以使用

```python
reranker = Reranker('ModuleReranker', model='bge-reranker-large', topk=1)
```

来创建一个 `Reranker` 对所有 `Retriever` 返回的文档再做一次排序。

`Reranker` 的构造函数有以下参数：

* `name`：指定用来排序的函数名称，`LazyLLM` 内置的函数有 `ModuleReranker` 和 `KeywordFilter`；
* `kwargs`：透传给排序函数的参数。

其中内置的 `ModuleReranker` 是一个支持使用指定模型来排序的通用函数。其函数原型是：

```python
def ModuleReranker(
    nodes: List[DocNode],
    model: str,
    query: str,
    topk: int = -1,
    **kwargs
) -> List[DocNode]:
```

表示使用指定的模型 `model`，结合用户输入的内容 `query`，对文档列表 `nodes` 进行排序，返回相似度最高的 `topk` 篇文档。`kwargs` 就是 `Reranker` 构造函数中透传过来的参数。

内置的 `KeywordFilter` 用于过滤包含或不包含指定关键词的文档。其函数原型是：

```python
def KeywordFilter(
    node: DocNode,
    required_keys: List[str],
    exclude_keys: List[str],
    language: str = "en",
    **kwargs
) -> Optional[DocNode]:
```

如果 `node` 中包含 `required_keys` 中的所有关键词，并且不包含 `exclude_keys` 中的任意一个关键词，就返回 `node` 本身；否则返回 `None`。参数 `language` 表示文档的语言种类；`kwargs` 就是 `Reranker` 构造函数中头传过来的参数。

用户还可以通过 `LazyLLM` 提供的 `register_reranker()` 来注册自己的排序函数。`register_reranker()` 有以下参数：

* `func`：用于排序的函数；
* `batch`：是否是多 batch。

当 `batch` 为 `True` 时，`func` 的参数 `nodes` 是一个 `DocNode` 列表，表示需要排序的所有文档，返回值也是一个 `DocNode` 列表，表示排序后的文档列表；当 `batch` 为 `False` 时，参数是一个待处理的 `DocNode`，返回值是一个 `Optional[DocNode]`，此时 `Reranker` 可作为过滤器使用，如果传入的文档符合要求，可返回传入的 `DocNode`，否则返回 `None` 表示丢弃该 `Node`。

根据 `batch` 不同的取值，相应的 `func` 函数原型有以下 2 种：

```python
# (1)
@lazyllm.tools.rag.register_reranker(batch=True)
def dummy_reranker(nodes: List[DocNode], **kwargs) -> List[DocNode]:

# (2)
@lazyllm.tools.rag.register_reranker(batch=False)
def dummy_reranker(node: DocNode, **kwargs) -> Optional[DocNode]:
```

`Reranker` 实例可以这样使用：

```python
doc_list = reranker(doc_list, query=query)
```

表示使用 `Reranker` 创建时指定的模型排序并返回排序后的结果。

## 示例

关于 RAG 的示例可以参考 [CookBook 中的 RAG 例子](../Cookbook/rag.md)。
