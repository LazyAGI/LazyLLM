# RAG

检索增强生成（Retrieval-augmented Generation，RAG）是当前备受关注的大模型前沿技术之一。其工作原理是，当模型需要生成文本或回答问题时，首先会从一个庞大的文档集合中检索出相关的信息。这些检索到的信息随后会被用于指导生成过程，从而显著提高生成文本的质量和准确性。通过这种方式，RAG 能够在处理复杂问题时提供更加精确和有意义的回答，是自然语言处理领域的重要进展之一。这种方法的优越性在于它结合了检索和生成的优势，使得模型不仅能够生成流畅的文本，还能基于真实数据提供有依据的回答。

一般来说 RAG 的流程如下图所示：

![RAG intro](../assets/rag-intro.svg)

## 设计

根据上面的描述，我们将 RAG 抽象为如下流程：

![RAG modules relationship](../assets/rag-modules-relationship.svg)

### Document

从原理介绍中可以看到，文档集合中有各种各样的文档格式：可以是结构化的存储在数据库中的一条条记录，可以是 DOCX，PDF，PPT 等富文本或者 Markdown 这样的纯文本，也可能是从某个 API 获取的内容（如通过搜索引擎检索得到的相关信息）等等。由于集合内的文档格式各异，针对这些不同格式的文档，我们需要特定的解析器来提取其中有用的文本，图片，表格，音频和视频等内容。在 `LazyLLM` 中，这些用于提取特定内容的解析器被抽象成 `DataLoader`。目前 `LazyLLM` 内置的 `DataLoader` 可以支持 DOCX，PDF，PPT，EXCEL 等常见的富文本内容提取。使用 `DataLoader` 提取到的文档内容会被存储在 `Document` 中。

目前 `Document` 只支持从本地目录中提取文档内容，用户可以用以下语句

```python
docs = Document(dataset_path='/path/to/doc/dir', embed=MyEmbeddingModule(), manager=False)
```

从本地目录构建一个文档集合 `docs`。其中 `Document` 的构造函数有以下参数：

* `dataset_path`：指定从哪个文件目录构建；
* `embed`：使用指定的模型来对文本进行 embedding；
* `manager`：是否使用 ui 界面会影响 `Document` 内部的处理逻辑，默认为 `False`；
* `launcher`：启动服务的方式，集群应用会用到这个参数，单机应用可以忽略。

一个 `Document` 实例可能会按照指定的规则（在 `LazyLLM` 中被称为 `Transformer`），被进一步细分成若干粒度不同的被称为 `Node` 的节点集合（`Node Group`）。这些 `Node` 除了包含的文档内容外，还记录了自己是从哪一个`Node` 拆分而来，以及本身又被拆分成哪些更细粒度的 `Node`这些信息。用户可以通过 `Document.create_node_group()` 来创建自己的 `Node Group`。

下面我们通过例子来介绍 `Node` 和 `Node Group`：

```python
docs = Document()

# (1)
docs.create_node_group(name='block',
                       transform=lambda d: '\n'.split(d))

# (2)
docs.create_node_group(name='doc-summary',
                       transform=lambda d: summary_llm(d))

# (3)
docs.create_node_group(name='sentence',
                       transform=lambda b: '。'.split(b),
                       parent='block')

# (4)
docs.create_node_group(name='block-summary',
                       transform=lambda b: summary_llm(b),
                       parent='block')

# (5)
docs.create_node_group(name='keyword',
                       transform=lambda b: keyword_llm(b),
                       parent='block')

# (6)
docs.create_node_group(name='sentence-len',
                       transform=lambda s: len(s),
                       parent='sentence')
```

首先语句 1 以换行符为分割符，将所有文档都拆成了一个个的段落块，每个块就是 1 个 `Node`，这些 `Node` 构成了名为 `block` 的 `Node Group`。

语句 2 使用一个可以提取摘要的大模型把每个文档的摘要作为一个名为 `doc-summary` 的 `Node Group`，这个 `Node Group` 中只有一个 `Node`，内容就是整个文档的摘要。

由于 `block` 和 `doc-summary` 都是从 `lazyllm_root` 这个根节点经过不同的规则转换得到的，所以它俩都是 `lazyllm_root` 的子节点。

语句 3 在 `block` 这个 `Node Group` 的基础上进一步转换，使用中文句号作为分割符而得到一个个句子，每个句子都是一个 `Node`，共同构成了 `sentence` 这个 `Node Group`。

语句 4 在 `block` 这个 `Node Group` 的基础上，使用可以抽取摘要的大模型对其中的每个 `Node` 做处理，从而得到的每个段落摘要的 `Node`，组成了 `block-summary`。

语句 5 也是在 `block` 这个 `Node Group` 的基础上，在可以抽取关键词的大模型的帮助下，将每个段落都抽取出来一些关键词，每个段落的关键词是一个个的 `Node`，共同组成了 `keyword` 这个 `Node Group`。

最后语句 6 在 `sentence` 的基础上，统计了每个句子的长度，得到了一个包含每个句子长度的名为 `sentence-len` 的 `Node Group`。

语句 2，4，5 用到的提供摘要（summary）和关键词（keywords）抽取的功能，可以使用 `LazyLLM` 内置的 `LLMParser`。用法可以参考 [LLMParser 的使用说明][lazyllm.tools.LLMParser]。

这些 `Node Group` 的关系如下图所示：

![relationship of node groups](../assets/rag-relationship-of-node-groups.svg)

!!! Note "注意"

    `Document.create_node_group()` 有一个名为 `parent` 的参数，用于指定本次转换是基于哪个 `Node Group` 进行的。如果不指定则默认是整篇文档，也就是名为 `lazyllm-root` 的根 `Node`。另外，`Document` 的构造函数中有一个 `embed` 参数，是用来把 `Node` 的内容转换成向量的函数。

这些 `Node Group` 的拆分粒度和规则各不相同，反映了文档不同方面的特征。在后续的处理中，我们通过在不同的场合使用这些特征，从而更好地判断文档和用户输入的查询内容的相关性。

### Retriever

文档集合中的文档不一定都和用户要查询的内容相关，因此接下来我们要使用 `Retriever` 从 `Document` 中筛选出和用户查询相关的文档。

例如，用户可以这样创建一个 `Retriever` 实例：

```python
retriever = Retriever(documents, group_name="sentence", similarity="cosine", topk=3)
```

表示在 `sentence` 这个 `Node Group` 中使用 `cosine` 作为相似度计算函数，计算用户查询的内容 `query` 和每个 `Node` 的相似度。`topk` 表示最多取最相近的多少篇文档。

`Retriever` 的构造函数有以下参数：

* `doc`：要从哪个 `Document` 中检索文档；
* `group_name`：要使用文档的哪个 `Node Group` 来检索，使用 `LAZY_ROOT_NAME` 表示在原始文档内容中进行检索；
* `similarity`：指定用来计算 `Node` 和用户查询内容之间的相似度的函数名称，`LazyLLM` 内置的相似度计算函数有 `bm25`，`bm25_chinese` 和 `cosine`，用户也可以自定义自己的计算函数；
* `similarity_cut_off`：丢弃相似度小于指定值的结果，默认为 `-inf`，表示不丢弃；
* `index`：在哪个索引上进行查找，目前只支持 `default`；
* `topk`：表示返回最相关的文档数，默认值为 6；
* `similarity_kw`：需要透传给 `similarity` 函数的参数。

用户可以通过使用 `LazyLLM` 提供的 `register_similarity()` 函数来注册自己的相似度计算函数。`register_similarity()` 有以下参数：

* `func`：用于计算相似度的函数；
* `mode`：计算模式，支持 `text` 和 `embedding` 两种，会影响传给 `func` 的参数；
* `decend`：是否降序排列，默认为 `True`；
* `batch`：是否多 batch，会影响传给 `func` 的参数和返回值。

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

`Retriever` 实例可以这样使用：

```python
doc_list = retriever(query=query)
```

来检索和 `query` 相关的文档。

### Reranker

当我们从最初的文档集合筛选出和用户查询相关性比较高的文档后，接下来就可以进一步对这些文档进行排序，选出更贴合用户查询内容的文档。这一步工作由 `Reranker` 来完成。

例如，我们可以使用

```python
reranker = Reranker('ModuleReranker', model='bg-reranker-large', topk=1)
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
