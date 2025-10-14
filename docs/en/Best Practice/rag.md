# RAG

Retrieval-augmented Generation (RAG) is one of the cutting-edge technologies in large models that is currently receiving a lot of attention. Its working principle is that when the model needs to generate text or answer questions, it first retrieves relevant information from a vast collection of documents. This retrieved information is then used to guide the generation process, significantly improving the quality and accuracy of the generated text. In this way, RAG is able to provide more precise and meaningful responses when dealing with complex questions, making it one of the significant advancements in the field of natural language processing. The superiority of this method lies in its combination of the strengths of retrieval and generation, allowing the model to not only produce fluent text but also to provide evidence-based answers based on real data.

Generally, the process of RAG can be illustrated as follows in the diagram below:

![RAG intro](../assets/rag-intro.svg)

## Design

Based on the above description, we abstract the RAG process as follows:

![RAG modules relationship](../assets/rag-modules-relationship.svg)

### Document

From the principle introduction, it can be seen that the document collection contains various document formats: it can be structured records stored in a database, rich text formats such as DOCX, PDF, PPT, or plain text like Markdown, or even content obtained from an API (such as information retrieved through a search engine), etc. Due to the diverse document formats within the collection, we need specific parsers to extract useful content such as text, images, tables, audio, and video from these different formats. In `LazyLLM`, these parsers used to extract specific content are abstracted as `DataLoader`. Currently, the `DataLoader` built into `LazyLLM` can support the extraction of common rich text content such as DOCX, PDF, PPT, and EXCEL. The document content extracted using `DataLoader` is stored in a `Document`.

`Document`  not only supports extracting document content from a local directory, but also supports API. Users can build a document collection `docs` from a local directory using the following statement:

```python
docs = Document(dataset_path='/path/to/doc/dir', embed=MyEmbeddingModule(), manager=False)
```

The Document constructor has the following parameters:

* `dataset_path`: Specifies which file directory to build from.
* `embed`: Uses the specified model to perform text embedding. If you need to generate multiple embeddings for the text, you need to specify them in a dictionary, where the key identifies the name of the embedding and the value is the corresponding embedding model.
* `manager`: Whether to use the UI interface, which will affect the internal processing logic of Document; the default is True.
* `launcher`: The method of launching the service, which is used in cluster applications; it can be ignored for single-machine applications.
* `store_conf`: Configure which storage backend to use.
* `doc_fields`: Configure the fields and corresponding types that need to be stored and retrieved (currently only used by the ChromaDB and Milvus backend).

#### Node and NodeGroup

A `Document` instance may be further subdivided into several sets of nodes with different granularities, known as `Node` sets (the `Node Group`), according to specified rules (referred to as `Transformer` in `LazyLLM`). These `Node`s not only contain the document content but also record which `Node` they were split from and which finer-grained `Node`s they themselves were split into. Users can create their own `Node Group` by using the `Document.create_node_group()` method.

`create_node_group` method has the following parameters:

* `name`: Specifies the name of the `Node Group`.
* `transform`: Specifies the transformation rule of the `Node Group`, which can be a subclass of [NodeTransformer][lazyllm.tools.rag.NodeTransformer], or a function that takes content as input.
* `parent`: Specifies the parent `Node Group`, if not specified, it defaults to the entire document, which is the root `Node` named `lazyllm-root`.

!!! Note

    `LazyLLM` provides three built-in `Node Group`s:
    * `FineChunk`: A `Node Group` with a length of 128 tokens and an overlap of 12.
    * `MediumChunk`: A `Node Group` with a length of 256 tokens and an overlap of 25.
    * `CoarseChunk`: A `Node Group` with a length of 1024 tokens and an overlap of 100.

    For these three `Node Group`s, users do not need to manually create them, and they can be used directly in subsequent retrieval.

Below, we will introduce `Node` and `Node Group` through an example:

```python
docs = Document()

# (1) Split the document into individual paragraph blocks using line breaks as delimiters, with each block being a single `Node`.
docs.create_node_group(name='block',
                       transform=lambda d: d.split('\n'))

# (2) Use a llm capable of extracting summaries to treat each document's summary as a `Node Group` named `doc-summary`. This `Node Group` contains only one `Node`, which is the summary of the entire document.
docs.create_node_group(name='doc-summary',
                       transform=lambda d: summary_llm(d))

# (3) Further transform the `Node Group` named `block` by using Chinese periods as delimiters to obtain individual sentences, with each sentence being a `Node`. Together, they form the `Node Group` named `sentence`.
docs.create_node_group(name='sentence',
                       transform=lambda b: b.split('。'),
                       parent='block')

# (4) Based on the `Node Group` named `block`, use a large model that can extract summaries to process each `Node`, resulting in a `Node Group` named `block-summary` that consists of paragraph summaries.
docs.create_node_group(name='block-summary',
                       transform=lambda b: summary_llm(b),
                       parent='block')

# (5) Based on the `Node Group` named `block`, with the help of a llm that can extract keywords, extracts keywords for each paragraph. The keywords of each paragraph are individual `Node`s, which together form the `Node Group` named `keyword`.
docs.create_node_group(name='keyword',
                       transform=lambda b: keyword_llm(b),
                       parent='block')

# (6) Based on the `Node Group` named `sentence`, count the length of each sentence, resulting in a `Node Group` named `sentence-len` that contains the length of each sentence.
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


These `Node Group`s have different granularities and rules, reflecting various characteristics of the document. In subsequent processing, we use these characteristics in different contexts to better judge the relevance between the document and the user's query content.

#### Store and Index

`LazyLLM` offers the functionality of configurable storage backends, which can meet various storage and retrieval needs.

The configuration parameter `store_conf` is a `dict` type that includes both `segment_store` and `vector_store` with the following fields:
```python
store_conf = {"segment_store": {}, "vector_store": {}}
```

In each of the `segment_store` and `vector_store`, the `type` field specifies the type of storage backend, and the `kwargs` field specifies the configuration parameters for the storage backend.

* `type`: This is the type of storage backend. Currently supported storage backends include:
    - `segment_store`:
        - `map`: In-memory key/value storage, which can be given a `uri` parameter to specify the directory where data is stored (using sqlite3 as the underlying storage engine).
        - `opensearch`: Use the OpenSearch backend for data storage.
    - `vector_store`:
        - `chromadb`: Uses ChromaDB for data storage.
        - `milvus`: Uses Milvus for data storage.
* `kwargs`: This is a dictionary that contains the configuration parameters for the storage backend, different storage backends have different configuration parameters:
    - `map`:
        - `uri` (optional): The directory where data is stored (using sqlite3 as the underlying storage engine).
    - `opensearch`:
        - `uris` (required): The OpenSearch storage address (support multiple addresses), which can be a list of URL in the format of `ip:port`.
        - `client_kwargs` (required): The configuration parameters for the OpenSearch client, e.g. `user`, `password`, etc, can be found in [OpenSearch official documentation](https://opensearch-project.github.io/opensearch-py/api-ref/clients/opensearch_client.html).
        - `index_kwargs` (optional): The configuration parameters for the OpenSearch index and slice storage.
    - `chromadb`:
        - `uri` (optional): The ChromaDB storage address, which can be a URL in the format of `ip:port`.
        - `dir` (optional): The directory where data is stored, which is used when `uri` is not specified.
        - `index_kwargs` (optional): The configuration parameters for the ChromaDB index, setting the index type and similarity calculation method, can be found in [ChromaDB official documentation](https://docs.trychroma.com/docs/collections/configure).
        - `client_kwargs` (optional): The configuration parameters for the ChromaDB client.
    - `milvus`:
        - `uri` (required): The Milvus storage address, which can be a db file path or a URL in the format of `ip:port`.
        - `db_name` (optional): The name of the Milvus database, which is used to isolate the database layer.
        - `client_kwargs` (optional): The configuration parameters for the Milvus client.
        - `index_kwargs` (optional): The configuration parameters for the Milvus index, which can be a dictionary or a list. If it is a dictionary, it means that all embedding indexes use the same configuration; if it is a list, the elements in the list are dictionaries, representing the configuration used by the embeddings specified by `embed_key`. Currently, only `floating point embedding` and `sparse embedding` are supported for the two types of embeddings, with the following supported parameters respectively:
            - `floating point embedding`: [https://milvus.io/docs/index-vector-fields.md?tab=floating](https://milvus.io/docs/index-vector-fields.md?tab=floating)
            - `sparse embedding`: [https://milvus.io/docs/index-vector-fields.md?tab=sparse](https://milvus.io/docs/index-vector-fields.md?tab=sparse)


Here is an example configuration using Chroma as the storage backend and Milvus as the retrieval backend:

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
Also you can configure multi index type for Milvus backend as follow, where the `embed_key` should match the key of multi embeddings passed to Document:

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

Note: If ChromaDB or Milvus is used as a vector storage, if you want to perform scalar filtering on a specific field as a search condition, you also need to provide a description of special fields that may be used as search conditions, passed in through the `doc_fields` parameter. `doc_fields` is a dictionary where the key is the name of the field that needs to be stored or retrieved, and the value is a `DocField` type structure containing information such as the field type.

For example, if you need to store the author information and publication year of documents, you can configure it as follows:

```python
doc_fields = {
    'author': DocField(data_type=DataType.VARCHAR, max_size=128, default_value=' '),
    'public_year': DocField(data_type=DataType.INT32),
}
```

### Retriever

The documents in the document collection may not all be relevant to the content the user wants to query. Therefore, next, we will use the `Retriever` to filter out documents from the `Document` that are relevant to the user's query.

For example, a user can create a `Retriever` instance like this:

```python
retriever = Retriever(documents, group_name="sentence", similarity="cosine", topk=3)  # retriever = Retriever([document1, document2, ...], group_name="sentence", similarity="cosine", topk=3)
```

This indicates that within the `Node Group` named `sentence`, the `cosine` similarity function will be used to calculate the similarity between the user's query content `query` and each `Node`. The `topk` parameter specifies that the top k most similar nodes should be selected, in this case, the top 3.

The constructor of the `Retriever` has the following parameters:

* `doc`: Specifies which `Document` to retrieve documents from. Or which `Document` list to retrieve documents from.
* `group_name`: Specifies which `Node Group` of the document to use for retrieval. Use `LAZY_ROOT_NAME` to indicate that the retrieval should be performed on the original document content.
* `similarity`: Specifies the name of the function to calculate the similarity between a `Node` and the user's query content. The similarity calculation functions built into `LazyLLM` include `bm25`, `bm25_chinese`, and `cosine`. Users can also define their own calculation functions. If not specified, the vector retrieval will be used by default.
* `similarity_cut_off`: Discards results with a similarity less than the specified value. The default is `-inf`, which means no results are discarded. In a multi-embedding scenario, if you need to specify different values for different embeddings, this parameter needs to be specified in a dictionary format, where the key indicates which embedding is specified and the value indicates the corresponding threshold. If all embeddings use the same threshold, this parameter only needs to pass a single value.
* `index`: On which index to search, currently only `default` and `smart_embedding_index` are supported.
* `topk`: Specifies the number of most relevant documents to return. The default value is 6.
* `embed_keys`: Indicates which embeddings to use for retrieval. If not specified, all embeddings will be used for retrieval.
* `similarity_kw`: Parameters that need to be passed through to the `similarity` function.

Users can register their own similarity calculation functions by using the `register_similarity()` function provided by `LazyLLM`. The `register_similarity()` function has the following parameters:

* `func`: The function used to calculate similarity.
* `mode`: The calculation mode, which supports two types: `text` and `embedding`. This will affect the parameters passed to `func`.
* `descend`: Whether to sort in descending order. The default is `True`.
* `batch`: Whether to process in multiple batches. This will affect the parameters passed to func and the return value.

!!! Note

    External storage engines generally do not support custom similarity calculation functions, only when not using external storage engines, the registered `similarity` parameter is effective.

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

The `Retriever` instance requires the `query` string to be passed in when used, along with optional `filters` for field filtering. `filters` is a dictionary where the key is the field to be filtered on, and the value is a list of acceptable values, indicating that the node will be returned if the field’s value matches any one of the values in the list. Only when all conditions are met will the node be returned.

Here is an example of using `filters`:

```python
filters = {
    "author": ["A", "B", "C"],
    "publish_year": [2002, 2003, 2004],
}
doc_list = retriever(query=query, filters=filters)
```

You can customize the filtering keys by passing the parameter `doc_fields` when initializing `Document`(refer to [Document](../Best%20Practice/rag.md#Document)). Or you can choose the builtin global metadata from the given list: ["file_name", "file_type", "file_size", "creation_date", "last_modified_date", "last_accessed_date"].

### Reranker

After filtering out documents from the initial document collection that are relatively relevant to the user's query, the next step is to further sort these documents to select the ones that are more aligned with the user's query content. This step is performed by the `Reranker`.

For example, you can create a `Reranker` to perform another sorting on all documents returned by the `Retriever` using:

```python
reranker = Reranker('ModuleReranker', model='bge-reranker-large', topk=1)
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
