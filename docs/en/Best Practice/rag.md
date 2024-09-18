# RAG

Retrieval-augmented Generation (RAG) is one of the cutting-edge technologies in large models that is currently receiving a lot of attention. Its working principle is that when the model needs to generate text or answer questions, it first retrieves relevant information from a vast collection of documents. This retrieved information is then used to guide the generation process, significantly improving the quality and accuracy of the generated text. In this way, RAG is able to provide more precise and meaningful responses when dealing with complex questions, making it one of the significant advancements in the field of natural language processing. The superiority of this method lies in its combination of the strengths of retrieval and generation, allowing the model to not only produce fluent text but also to provide evidence-based answers based on real data.

Generally, the process of RAG can be illustrated as follows in the diagram below:

![RAG intro](../assets/rag-intro.svg)

## Design

Based on the above description, we abstract the RAG process as follows:

![RAG modules relationship](../assets/rag-modules-relationship.svg)

### Document

From the principle introduction, it can be seen that the document collection contains various document formats: it can be structured records stored in a database, rich text formats such as DOCX, PDF, PPT, or plain text like Markdown, or even content obtained from an API (such as information retrieved through a search engine), etc. Due to the diverse document formats within the collection, we need specific parsers to extract useful content such as text, images, tables, audio, and video from these different formats. In `LazyLLM`, these parsers used to extract specific content are abstracted as `DataLoader`. Currently, the `DataLoader` built into `LazyLLM` can support the extraction of common rich text content such as DOCX, PDF, PPT, and EXCEL. The document content extracted using `DataLoader` is stored in a `Document`.

Currently, `Document` only supports extracting document content from a local directory, and users can build a document collection docs from a local directory using the following statement:

```python
docs = Document(dataset_path='/path/to/doc/dir', embed=MyEmbeddingModule(), manager=False)
```

The Document constructor has the following parameters:

* `dataset_path`: Specifies which file directory to build from.
* `embed`: Uses the specified model to perform text embedding.
* `manager`: Whether to use the UI interface, which will affect the internal processing logic of Document; the default is True.
* `launcher`: The method of launching the service, which is used in cluster applications; it can be ignored for single-machine applications.

A `Document` instance may be further subdivided into several sets of nodes with different granularities, known as `Node` sets (the `Node Group`), according to specified rules (referred to as `Transformer` in `LazyLLM`). These `Node`s not only contain the document content but also record which `Node` they were split from and which finer-grained `Node`s they themselves were split into. Users can create their own `Node Group` by using the `Document.create_node_group()` method.

Below, we will introduce `Node` and `Node Group` through an example:

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

First, statement 1 splits all documents into individual paragraph blocks using line breaks as delimiters, with each block being a single `Node`. These `Node`s constitute a `Node Group` named `block`.

Statement 2 uses a large model capable of extracting summaries to treat each document's summary as a `Node Group` named `doc-summary`. This `Node Group` contains only one `Node`, which is the summary of the entire document.

Since both `block` and `doc-summary` are derived from the root node `lazyllm_root` through different transformation rules, they are both child nodes of `lazyllm_root`.

Statement 3 further transforms the `Node Group` named `block` by using Chinese periods as delimiters to obtain individual sentences, with each sentence being a `Node`. Together, they form the `Node Group` named `sentence`.

Statement 4, based on the `Node Group` named `block`, uses a large model that can extract summaries to process each `Node`, resulting in a `Node Group` named `block-summary` that consists of paragraph summaries.

Statement 5, also based on the `Node Group` named `block`, with the help of a large model that can extract keywords, extracts keywords for each paragraph. The keywords of each paragraph are individual `Node`s, which together form the `Node Group` named `keyword`.

Finally, statement 6, based on the `Node Group` named `sentence`, counts the length of each sentence, resulting in a `Node Group` named `sentence-len` that contains the length of each sentence.

The functionality for extracting summaries and keywords used in statements 2, 4, and 5 can be achieved with `LazyLLM`'s built-in `LLMParser`. Usage instructions can be found in the [LLMParser documentation][lazyllm.tools.LLMParser].

The relationship of these `Node Group`s is shown in the diagram below:

![relationship of node groups](../assets/rag-relationship-of-node-groups.svg)

!!! Note

    The `Document.create_node_group()` method has a parameter named `parent` which is used to specify which `Node Group` the transformation is based on. If not specified, it defaults to the entire document, which is the root `Node` named `lazyllm-root`. Additionally, the `Document` constructor has an `embed` parameter, which is a function used to convert the content of a `Node` into a vector.

These `Node Group`s have different granularities and rules, reflecting various characteristics of the document. In subsequent processing, we use these characteristics in different contexts to better judge the relevance between the document and the user's query content.

### Retriever

The documents in the document collection may not all be relevant to the content the user wants to query. Therefore, next, we will use the `Retriever` to filter out documents from the `Document` that are relevant to the user's query.

For example, a user can create a `Retriever` instance like this:

```python
retriever = Retriever(documents, group_name="sentence", similarity="cosine", topk=3)
```

This indicates that within the `Node Group` named `sentence`, the `cosine` similarity function will be used to calculate the similarity between the user's query content `query` and each `Node`. The `topk` parameter specifies that the top k most similar nodes should be selected, in this case, the top 3.

The constructor of the `Retriever` has the following parameters:

* `doc`: Specifies which `Document` to retrieve documents from.
* `group_name`: Specifies which `Node Group` of the document to use for retrieval. Use `LAZY_ROOT_NAME` to indicate that the retrieval should be performed on the original document content.
* `similarity`: Specifies the name of the function to calculate the similarity between a `Node` and the user's query content. The similarity calculation functions built into `LazyLLM` include `bm25`, `bm25_chinese`, and `cosine`. Users can also define their own calculation functions.
* `similarity_cut_off`: Discards results with a similarity less than the specified value. The default is `-inf`, which means no results are discarded.
* `index`: Specifies on which index to perform the search. Currently, only `default` is supported.
* `topk`: Specifies the number of most relevant documents to return. The default value is 6.
* `similarity_kw`: Parameters that need to be passed through to the `similarity` function.

Users can register their own similarity calculation functions by using the `register_similarity()` function provided by `LazyLLM`. The `register_similarity()` function has the following parameters:

* `func`: The function used to calculate similarity.
* `mode`: The calculation mode, which supports two types: `text` and `embedding`. This will affect the parameters passed to `func`.
* `descend`: Whether to sort in descending order. The default is `True`.
* `batch`: Whether to process in multiple batches. This will affect the parameters passed to func and the return value.

When the `mode` parameter is set to `text`, it indicates that the content of the `Node` should be used for calculation. The type of the `query` parameter for the calculation function is `str`, which is the text content to be compared with the `Node`. The content of the `Node` can be obtained using `node.get_text()`. If `mode` is set to `embedding`, it indicates that the vectors obtained by converting with the `embed` function specified during the initialization of the `Document` should be used for calculation. In this case, the type of query is `List[float]`, and the vector of the `Node` can be accessed through `node.embedding`. The `float` in the return value represents the score of the document.

When `batch` is `True`, the calculation function has a parameter named `nodes`, which is of type `List[DocNode]`, and the type of return value is `List[(DocNode, float)]`. If `batch` is `False`, the calculation function has a parameter named `node`, which is of type `DocNode`, and the type of return value is `float`, which represents the score of the document.

Depending on the different values of `mode` and `batch`, the prototype of the user-defined similarity calculation function can have several forms:

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

An instance of `Retriever` can be used as follows to retrieve documents related to the `query`:

```python
doc_list = retriever(query=query)
```

### Reranker

After filtering out documents from the initial document collection that are relatively relevant to the user's query, the next step is to further sort these documents to select the ones that are more aligned with the user's query content. This step is performed by the `Reranker`.

For example, you can create a `Reranker` to perform another sorting on all documents returned by the `Retriever` using:

```python
reranker = Reranker('ModuleReranker', model='bg-reranker-large', topk=1)
```

The constructor of the `Reranker` has the following parameters:

* `name`: Specifies the name of the function to be used for sorting. The functions built into `LazyLLM` include `ModuleReranker` and `KeywordFilter`.
* `kwargs`: Parameters to be passed through to the sorting function.

The built-in `ModuleReranker` is a general function that supports sorting using a specified model. Its prototype is:

```python
def ModuleReranker(
    nodes: List[DocNode],
    model: str,
    query: str,
    topk: int = -1,
    **kwargs
) -> List[DocNode]:
```

This indicates that the `ModuleReranker` function uses the specified `model`, in combination with the user's input `query`, to sort the list of document nodes `nodes` and return the top `topk` documents with the highest similarity. The `kwargs` are the parameters passed through from the `Reranker` constructor.

The built-in `KeywordFilter` function is used to filter documents that do or do not contain specified keywords. Its prototype is:

```python
def KeywordFilter(
    node: DocNode,
    required_keys: List[str],
    exclude_keys: List[str],
    language: str = "en",
    **kwargs
) -> Optional[DocNode]:
```

This function checks if the `node` contains all the keywords in `required_keys` and none of the keywords in `exclude_keys`. If the `node` meets these criteria, it returns the `node` itself; otherwise, it returns `None`. The `language` parameter specifies the language of the document, and `kwargs` are the parameters passed through from the `Reranker` constructor.

Users can register their own sorting functions through the `register_reranker()` function provided by `LazyLLM`. The `register_reranker()` function has the following parameters:

* `func`: The function used for sorting.
* `batch`: Indicates whether the function processes multiple batches.

When `batch` is `True`, the func is expected to take a list of `DocNode` objects as the parameter `nodes`, which represents all the documents that need to be sorted. The return value should also be a list of `DocNode` objects, representing the sorted list of documents.

When `batch` is `False`, the `func` is expected to take a single `DocNode` object as the parameter, which represents the document to be processed. The return value should be an `Optional[DocNode]`, meaning that the `Reranker` can be used as a filter. If the input document meets the criteria, the function can return the input `DocNode`; otherwise, return `None` to indicate that the `Node` should be discarded.

Based on the different values of `batch`, the corresponding `func` function prototypes are as follows:

```python
# (1)
@lazyllm.tools.rag.register_reranker(batch=True)
def dummy_reranker(nodes: List[DocNode], **kwargs) -> List[DocNode]:

# (2)
@lazyllm.tools.rag.register_reranker(batch=False)
def dummy_reranker(node: DocNode, **kwargs) -> Optional[DocNode]:
```

An instance of `Reranker` can be used as follows:

```python
doc_list = reranker(doc_list, query=query)
```

which means using the model specified at the time of creation of the `Reranker` to sort and return the sorted results.

## Examples

For an example of RAG, you can refer to [RAG examples in the CookBook](../Cookbook/rag.md).
