# Knowledge Base Q&A Assistant

This article will demonstrate how to implement a knowledge base Q&A assistant. Before starting this section, it is recommended to read the [RAG Best Practices](../Best%20Practice/rag.md) first.

!!! abstract "Through this section, you will learn about the following key points of LazyLLM"

    - RAG related modules: [Document][lazyllm.tools.Document], [Retriever][lazyllm.tools.Retriever] and [Reranker][lazyllm.tools.Reranker]
    - Built-in ChatPrompter module
    - Built-in process modules: [Pipeline][lazyllm.flow.Pipeline] and [Parallel][lazyllm.flow.Parallel]

## Version-1

As known from [RAG Best Practices](../Best%20Practice/rag.md), the core of RAG is to answer the questions raised by users based on a specific collection of documents. With this goal in mind, we have designed the following process:

![rag-cookbook-1](../assets/rag-cookbook-1.svg)

`Query` is the user's input query; `Retriever` finds matching documents from the document collection `Document` based on the user's query; the large language model `LLM` provides the final answer based on the documents passed over by `Retriever` and combined with the user's query.

The following example rag.py implements the aforementioned functionality:

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
                              topk=3)

# Part3

llm = lazyllm.OnlineChatModule()

# Part4

prompt = 'You will act as an AI question-answering assistant and complete a dialogue task. In this task, you need to provide your answers based on the given context and questions.'
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

Let's briefly explain the code in each part.

1. Part0 imports `lazyllm`.

2. Part1 loads the local knowledge base directory and uses the built-in `OnlineEmbeddingModule` as the embedding function.

3. Part2 creates a `Retriever` for document retrieval, and uses the built-in `CoarseChunk` (refer to [definition of CoarseChunk][llm.tools.Retriever]) to chunk the documents into specified sizes, then uses the built-in `bm25_chinese` as the similarity calculation function, discards results with a similarity less than 0.003, and finally takes the closest 3 documents.

4. Part3 creates an instance of the large model used for answering questions.

5. Part4 since we need the large model to answer questions based on the documents we provide, we need to tell the large model what are the reference materials and what is our question when we ask. Here, we use the built-in `ChatPrompter` to tell the large model the document content returned by `Retriever` as reference materials. The meanings of the two parameters used in `ChatPrompter` are as follows:

    * instruction: the guidance content provided to the large model;
    * extra_keys: from which field in the passed-in dict to get the reference materials.

6. Part5 prints prompt information, waiting for the user to input the content they want to query.

7. Part6 is the main process: receive the user's input, use `Retriever` to retrieve relevant documents based on the user's `query`, then package the `query` and reference material `context_str` into a dict and pass it to the large model, waiting for the result to return.

8. Part7 prints the result to the screen.

Before running, modify the `dataset_path` parameter of `Document` in Part1 to point to the directory that needs to be retrieved. Also, set the authentication information for the applied SenseTime DailyNew model in the command line (refer to [Quick Start](../index.md)), and then run the following command:

```python
python3 rag.py
```

Enter the content you want to query and wait for the large model to return the results.

In this example, we only used the built-in algorithm `CoarseChunk` for retrieving similar documents, which chunks the document into fixed lengths and calculates the similarity for each chunk. This method may have decent results in some scenarios; however, in most cases, this simple and rough chunking may split important information into two segments, causing information loss, and the resulting documents may not be relevant to the user's input query.

## Version-2: Adding Recall Strategy and Sorting

To measure the relevance between documents and queries from different dimensions, we add a `Node Group` named `sentences` in Part1 that splits the document at the sentence level:

```python
documents.create_node_group(name="sentences",
                            transform=lambda d: '。'.split(d))
```

And in Part2, we use `cosine` to calculate the similarity between sentences and the user's query, then filter out the top 3 most relevant documents:

```python
retriever2 = Retriever(doc=documents,
                       group_name="sentences",
                       similarity="cosine",
                       topk=3)
```

Now that we have two different retrieval functions returning results in different orders, and these results may be duplicated, we need to use a `Reranker` to re-rank the multiple results. We insert a new Part8 between Part2 and Part4, adding a `ModuleReranker`, which uses a specified model to sort and select the most eligible single article:

```python
# Part8

reranker = Reranker(name="ModuleReranker",
                    model=lazyllm.OnlineEmbeddingModule(type="rerank"),
                    topk=1)
```

`ModuleReranker` is a general sorting module built into `LazyLLM` that simplifies the scenario of using a model to sort documents. You can check [the introduction to Reranker in the Best Practice guide](../Best%20Practice/rag.md#Reranker) for more information.

At the same time, we will rewrite Part 6 as follows:

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

The improved process is as follows:

![rag-cookbook-2](../assets/rag-cookbook-2.svg)

<details>

<summary>Here is the complete code (click to expand):</summary>

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

prompt = 'You will act as an AI question-answering assistant and complete a dialogue task. In this task, you need to provide your answers based on the given context and questions.'
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

## Version-3: using the Flow

From the flowchart of version-2, it can be seen that the entire process is already quite complex. We noticed that the retrieval processes of two (or more) `Retriever`s do not affect each other; they can be executed in parallel. At the same time, there are clear dependencies between the upstream and downstream processes, which require the upstream to be completed before the next step can proceed.

`LazyLLM` provides a set of auxiliary tools to simplify the writing of the execution flow. We can use [parallel](../Best%20Practice/flow.md#parallel) to encapsulate the two retrieval processes, allowing them to execute in parallel; and use [pipeline](../Best%20Practice/flow.md#pipeline) to encapsulate the entire process. The entire rewritten program is as follows:


```python
import lazyllm

# Part0

documents = lazyllm.Document(dataset_path="rag_master",
                             embed=lazyllm.OnlineEmbeddingModule(),
                             manager=False)

documents.create_node_group(name="sentences",
                            transform=lambda s: '。'.split(s))

prompt = 'You will act as an AI question-answering assistant and complete a dialogue task. In this task, you need to provide your answers based on the given context and questions.'

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

In Part1, the outer `with` statement constructs the `pipeline` which is our entire processing flow above: first, the `Retriever` retrieves documents, then it hands them over to the `Reranker` to sort and select suitable documents, which are then passed to the `LLM` for reference, and finally, an answer is obtained. The two `Retriever`s that implement different retrieval strategies are encapsulated within a `parallel`. The `parallel.sum` constructed by the inner `with` statement indicates that the outputs of all modules are merged and used as the output of the `parallel` module. Since the large model `llm` requires its parameters to be placed in a dict, we added a `formatter` component to package the user's query content `query` and the prompt content composed of the reference documents selected by the `Reranker` as the input for `llm`.

The illustration is as follows:

![rag-cookbook-3](../assets/rag-cookbook-3.svg)

`pipeline` and `parallel` are merely convenient for process setup and may offer some performance optimizations, but they do not alter the logic of the program. Additionally, the user query is an important prompt content that is essentially used by every intermediate module. Here, we also utilize the [bind()](../Best%20Practice/flow.md#use-bind) function provided by `LazyLLM` to pass the user query `query` as a parameter to `ppl.reranker` and `ppl.formatter`.

## Version-4: Customizing Retrieval and Sorting Strategies

The examples above all use components built into `LazyLLM`. However, in reality, there are always user needs that we cannot cover. To meet these needs, `Retriever` and `Reranker` provide a plugin mechanism that allows users to define their own retrieval and sorting strategies, adding them to the framework through the registration interface provided by `LazyLLM`.

To simplify the explanation and demonstrate the effect of the strategies we write, we will not consider whether the strategies are practical.

First, let's implement a function for similarity calculation and use the [register_similarity()](../Best%20Practice/rag.md#Retriever) function provided by `LazyLLM` to register this function with the framework:

```python
@lazyllm.tools.rag.register_similarity(mode='text', batch=True)
def MySimilarityFunc(query: str, nodes: List[DocNode], **kwargs) -> List[Tuple[DocNode, float]]:
    return [(node, 0.0) for node in nodes]
```

This function assigns a score of zero to each document and returns a new results list in the order of the input document list. That is to say, the retrieval results each time depend on the order in which the documents are read.

Similarly, we use the [register_reranker()](../Best%20Practice/rag.md#Reranker) function provided by LazyLLM to register a function that returns results in the order of the input document list:

```python
@lazyllm.tools.rag.register_reranker(batch=True)
def MyReranker(nodes: List[DocNode], **kwargs) -> List[DocNode]:
    return nodes
```

After that, these extensions can be used just like the built-in components, by referring to them by their function names. For example, the following code snippet creates a `Retriever` and a `Reranker` using the similarity calculation and retrieval strategy we defined above, respectively:


```python
my_retriever = lazyllm.Retriever(doc=documents,
                                 group_name="sentences",
                                 similarity="MySimilarityFunc",
                                 topk=3)

my_reranker = Reranker(name="MyReranker")
```

Certainly, the results returned might be a little wired :)

Here, we've simply introduced how to use the `LazyLLM` extension registration mechanism. You can refer to the documentation for [Retriever](../Best%20Practice/rag.md#Retriever) and [Reranker](../Best%20Practice/rag.md#Reranker) for more information. When you encounter scenarios where the built-in functionalities do not meet your needs, you can implement your own applications by writing custom similarity calculation and sorting strategies.

## Version-5: Customizing Storage Backend

After defining the transformation rules for the Node Group, `LazyLLM` will save the content of the Node Group obtained during the retrieval process, so as to avoid repeating the transformation operation when it is used subsequently. To facilitate users’ access to different types of data, `LazyLLM` supports custom storage backends.

If not specified, `LazyLLM` uses a dict-based key/value as the default storage backend. Users can specify other storage backends through the `Document` parameter `store_conf`. For example, if you want to use Milvus as the storage backend, you can configure it like this:

```python
milvus_store_conf = {
    'type': 'milvus',
    'kwargs': {
        'uri': store_file,
        'index_kwargs': {
            'index_type': 'HNSW',
            'metric_type': 'COSINE',
        }
    },
}
```

The `type` parameter is the backend type, and `kwargs` are the parameters that need to be passed to the backend. The meanings of each field are as follows:

* `type`: The backend type to be used. Currently supported:
    - `map`: In-memory key/value storage;
    - `chroma`: Use Chroma to store data;
        - `dir` (required): The directory where data is stored.
    - `milvus`: Use Milvus to store data.
        - `uri` (required): The Milvus storage address, which can be a file path or a URL in the format of `ip:port`;
        - `index_kwargs` (optional): Milvus index configuration, which can be a dict or a list. If it is a dict, it means all embedding indexes use the same configuration; if it is a list, the elements in the list are dict, indicating the configuration used by the embedding specified by `embed_key`. Currently, only `floating point embedding` and `sparse embedding` are supported for embedding types, with the following supported parameters respectively:
            - `floating point embedding`: [https://milvus.io/docs/index-vector-fields.md?tab=floating](https://milvus.io/docs/index-vector-fields.md?tab=floating)
            - `sparse embedding`: [https://milvus.io/docs/index-vector-fields.md?tab=sparse](https://milvus.io/docs/index-vector-fields.md?tab=sparse)

If using Milvus, we also need to pass the `doc_fields` parameter to `Document`, which is used to specify the fields and types of information that need to be stored. For example, the following configuration:

```python
doc_fields = {
    'comment': DocField(data_type=DataType.VARCHAR, max_size=65535, default_value=' '),
    'signature': DocField(data_type=DataType.VARCHAR, max_size=32, default_value=' '),
}
```

Two fields are configured: `comment` and `signature`. The `comment` is a string with a maximum length of 65535 and a default value of empty; the `signature` is a string type with a maximum length of 32 and a default value of empty.

Here is a complete example using Milvus as the storage backend:

<details>

<summary>Here is the complete code (click to expand):</summary>

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
    res = rag('What is the way of heaven?', filters=filters)
    print(f'answer: {res}')
```

</details>

## Version-6: Customizing Index Backend

To accelerate data retrieval and meet various retrieval needs, `LazyLLM` also supports specifying index backends for different storage backends. This can be done through the indices field in the `store_conf` parameter of `Document`. The index types configured in `indices` can be used in `Retriever` (by specifying the `index` parameter).

For instance, if you want to use a key/value store based on dict and use Milvus as the retrieval backend for this storage, you can configure it as follows:

```python
milvus_store_conf = {
    'type': 'map',
    'indices': {
        'smart_embedding_index': {
            'backend': 'milvus',
            'kwargs': {
                'uri': store_file,
                'index_kwargs': {
                    'index_type': 'HNSW',
                    'metric_type': 'COSINE',
                }
            },
        },
    },
}
```

The parameter `type` has been introduced in Version-5 and will not be repeated here. `indices` is a dict where the key is the index type, and the value is a dict whose content varies depending on the different index types.

Currently, indices only supports `smart_embedding_index`, with the following parameters:

* `backend`: Specifies the type of index backend used for embedding retrieval. Only milvus is supported at the moment;
* `kwargs`: The parameters that need to be passed to the index backend. In this case, the parameters passed to the milvus backend are the same as those introduced for the milvus storage backend in the section of Version-5.

Here is a complete example using milvus as the index backend:

<details>

<summary>Here is the complete code (click to expand):</summary>

```python
# -*- coding: utf-8 -*-

import os
import lazyllm
from lazyllm import bind
import tempfile

def run(query):
    _, store_file = tempfile.mkstemp(suffix=".db")

    milvus_store_conf = {
        'type': 'map',
        'indices': {
            'smart_embedding_index': {
                'backend': 'milvus',
                'kwargs': {
                    'uri': store_file,
                    'index_kwargs': {
                        'index_type': 'HNSW',
                        'metric_type': 'COSINE',
                    }
                },
            },
        },
    }

    documents = lazyllm.Document(dataset_path="rag_master",
                                 embed=lazyllm.TrainableModule("bge-large-zh-v1.5"),
                                 manager=False,
                                 store_conf=milvus_store_conf)

    documents.create_node_group(name="sentences",
                                transform=lambda s: '。'.split(s))

    prompt = 'You will play the role of an AI Q&A assistant and complete a dialogue task.'\
        ' In this task, you need to provide your answer based on the given context and question.'

    with lazyllm.pipeline() as ppl:
        ppl.retriever = lazyllm.Retriever(doc=documents, group_name="sentences", topk=3,
                                          index='smart_embedding_index')

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

        rag = lazyllm.ActionModule(ppl)
        rag.start()
        res = rag(query)

    os.remove(store_file)

    return res

if __name__ == '__main__':
    res = run('What is the way of heaven?')
    print(f'answer: {res}')
```

</details>
