# 使用 LazyLLM 构建 PDF 问答系统

本文介绍如何基于 **LazyLLM 框架**，实现一个从 PDF 文档中检索内容并回答问题的 RAG（Retrieval-Augmented Generation）系统。

------

### 1. 安装依赖

确保您已经安装了 LazyLLM：

```bash
pip install lazyllm
```

然后导入所需模块：

```python
import lazyllm
from lazyllm.tools.rag import Document
from lazyllm import pipeline, Retriever, Reranker, bind
```

### 2. 初始化模型

在 RAG 系统中，我们需要：

- **Embedding 模型**：将 PDF 文本向量化，用于相似度检索。
- **Reranker 模型**：对检索到的候选文本进行重排序。
- **LLM 模型**：根据上下文和问题生成最终回答。

在 LazyLLM 中，可以用 `TrainableModule` 启动这些模型：

```python
# 启动向量化模型
embed_model = lazyllm.TrainableModule("bge-large-zh-v1.5").start()

# 启动重排序模型
rerank_model = lazyllm.TrainableModule("bge-reranker-large").start()
```

> 这里使用了 **bge 系列模型**，您也可以替换为其他可用模型。

### 3. 加载 PDF 文档

我们都知道 Retriever 是需要从知识库中检索和问题相关的信息，而知识库中的内容会是各种各样的，有结构化存储的，也有非结构化存储的。为了让 Retriever 中的检索模块和知识库中知识的存储格式解耦，就需要 Reader 模块来大显身手了。Reader 的功能就是负责把以各种形式存储的知识读取出来，然后以统一的格式给检索模块使用。

LazyLLM 的`Reader`模块是在 `Document` 内部使用的， 所以我们基于PDF文档创建一个 `Document` 实例，然后我们直接通过 `Document` 实例来调用 `Reader` 的模块，并传入文档路径。LazyLLM 默认使用 `pypdf` 包来解析PDF格式文件。

这里我们将使用[DeepSeek-R1](https://arxiv.org/pdf/2501.12948)的这篇论文来作为要加载的PDF，您也可以随意使用您选择的PDF。

```python
documents = Document(
    dataset_path="data_path",
    embed=embed_model,
    manager=False
)

# 加载 PDF 文件
documents._impl._reader.load_data(
    input_files=["data_path/2501.12948v1.pdf"]
)
```

> LazyLLM 为其他数据源提供了[许多其他文档加载器](https://docs.lazyllm.ai/zh-cn/latest/API%20Reference/tools/#lazyllm.tools.rag.readers)，或者您可以创建[自定义文档加载器](https://docs.lazyllm.ai/zh-cn/latest/Tutorial/5/)。

### 4. 文本切分

将 PDF 内容按段落或换行分块，以便后续检索。

```python
documents.create_node_group(
    name="block",
    transform=lambda s: s.split("\n") if s else ''
)
```

### 5. 构建 RAG Pipeline

通过LazyLLM 提供的 `pipeline` ，我们可以将 **Retriever → Reranker → Formatter → LLM** 串联成一个端到端流程。

```python
prompt = (
    "You will play the role of an AI Q&A assistant and complete a dialogue task. "
    "In this task, you need to provide your answer based on the given context and question."
)

with lazyllm.pipeline() as ppl:
    # 1. 检索
    ppl.retriever = Retriever(
        doc=documents,
        group_name="block",
        topk=3,
        similarity='bm25'
    )

    # 2. 重排序
    ppl.reranker = Reranker(
        name='ModuleReranker',
        model=rerank_model,
        topk=1,
        output_format='content',
        join=True
    ) | bind(query=ppl.input)

    # 3. 格式化
    ppl.formatter = (
        lambda nodes, query: dict(context_str=nodes, query=query)
    ) | bind(query=ppl.input)

    # 4. LLM
    ppl.llm = lazyllm.TrainableModule('internlm2-chat-7b').prompt(
        lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str'])
    ).start()
```

### 6. 启动 Web 服务

最后，使用 `WebModule` 启动交互界面。

```python
lazyllm.WebModule(ppl, port=23456).start().wait()
```

启动后访问 `http://localhost:23456`，即可通过 Web 界面与会话式 RAG 系统进行交互，您可以输入问题并从 PDF 文档中得到答案。

### 效果展示

![rag-demo](../../assets/rag_pdf_qa_1.png)


### 完整代码

```python
import lazyllm
from lazyllm.tools.rag import Document
from lazyllm import pipeline, Retriever, Reranker, bind

# 初始化模型
embed_model = lazyllm.TrainableModule("bge-large-zh-v1.5").start()
rerank_model = lazyllm.TrainableModule("bge-reranker-large").start()

# 加载 PDF
documents = Document(
    dataset_path="/home/mnt/pansihan/ProjectLazyLLM/test/pdf_data",
    embed=embed_model,
    manager=False
)
documents._impl._reader.load_data(
    input_files=["/home/mnt/pansihan/ProjectLazyLLM/test/pdf_data/2501.12948v1.pdf"]
)
documents.create_node_group(name="block", transform=lambda s: s.split("\n") if s else '')

prompt = (
    "You will play the role of an AI Q&A assistant and complete a dialogue task. "
    "In this task, you need to provide your answer based on the given context and question."
)

# 构建 RAG Pipeline
with lazyllm.pipeline() as ppl:
    # Parallel retrieval
    ppl.retriever = Retriever(
        doc=documents, 
        group_name="block", topk=3,
        similarity='bm25')
    
    # Reranking
    ppl.reranker = Reranker(
        name='ModuleReranker',
        model=rerank_model,
        topk=1,
        output_format='content',
        join=True
    ) | bind(query=ppl.input)

    ppl.formatter = (
        lambda nodes, query: dict(context_str=nodes, query=query)
    ) | bind(query=ppl.input)

    ppl.llm = lazyllm.TrainableModule('internlm2-chat-7b').prompt(
        lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str'])
    ).start()

# 启动 Web 服务
lazyllm.WebModule(ppl, port=23456).start().wait()
```
