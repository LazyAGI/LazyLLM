# Retriever的作用

Retriever 负责根据查询从文档中检索相关内容。

## 内置的Retrievr类型

### Retriever
```python
retriever = lazyllm.Retriever(
    doc=documents,
    group_name="CoarseChunk",
    similarity="bm25_chinese",
    similarity_cut_off=0.003,
    output_format = 'content',
    join = True,
    topk=3
)

# 检索
doc_node_list = retriever(query="用户问题")
```

参数说明
- `doc`: Document 对象
- `group_name`: 使用的节点组名称
- `similarity`: 相似度计算方法
- `similarity_cut_off`: 相似度阈值，低于此值的结果会被过滤
- `output_format`:有效取值是 content 、dict 和 None，content 表示 Retriever 检索输出的是 Node 的内容，即 str 类型。dict 表示 Retriever 检索输出的是字典类型，即把 Node 中的内容转换成字典进行输出。None 表示不做任何后处理，直接以 Node 类型输出。
- `join`:有效取值是布尔值、字符串值。当 join 为 False 时，只会对 output_format 为 content 有影响，即不对输出内容进行拼接，是以 List[str] 格式进行输出。当 join 为 True 时，会给 join 赋值空字符串。当 join 为字符串时同时 output_format 为 content，则使用 join 对 nodes中的文本进行拼接输出，即以 str 形式输出。
- `topk`: 返回前 k 个结果

### TempDocRetriever

临时文档检索器，继承自TempRetriever，用于快速处理临时文件并执行检索任务。

参数：

- embed (Callable, default: None ) – 嵌入函数。
- output_format (Optional[str], default: None ) – 结果输出格式(如json),可选默认为None
- join (Union[bool, str], default: False ) – 是否合并多段结果(True或用分隔符如"\n")

```python
import lazyllm
from lazyllm.tools import TempDocRetriever, Document, SentenceSplitter

retriever = TempDocRetriever(output_format="text", join="---------------")
retriever.create_node_group(transform=lambda text: [s.strip() for s in text.split("。") if s] )
retriever.add_subretriever(group=Document.MediumChunk, topk=3)
files = ["/path/to/file.txt"]
results = retriever.forward(files, "什么是机器学习?")
print(results)
```

## 进阶策略

### 大小块策略
当召回到某个节点时，不仅可以取出该节点的内容，还可以取出该节点的父节点。即通过小块召回大块。

```python
import lazyllm
from lazyllm import bind
from lazyllm import Document, Retriever, TrainableModule

llm = lazyllm.OnlineChatModule(source='qwen', model='qwen-turbo')
prompt = '你是一个友好的 AI 问答助手，你需要根据给定的上下文和问题提供答案。\
          根据以下资料回答问题：\
          {context_str} \n '

robot = llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))

# 文档加载
docs = Document("/mnt/lustre/share_data/dist/cmrc2018/data_kb")
docs.create_node_group(name='sentences', transform=(lambda d: d.split('\n') if d else ''), parent=Document.CoarseChunk)

# 定义两个不同的检索器在同一个节点组使用不同相似度方法进行检索
retriever = Retriever(docs, group_name="sentences", similarity="bm25_chinese", topk=3)

# 执行查询
query = "都有谁参加了2008年的奥运会？"

# 原节点检索结果
doc_node_list = retriever(query=query)
doc_node_res = "".join([node.get_content() for node in doc_node_list])
print(f"原节点检索结果：\n{doc_node_res}")
print('='*100)

# 父节点对应结果
parent_list = [node.parent.get_text() for node in doc_node_list]
print(f"父节点检索结果：\n{''.join(parent_list)}")
print('='*100)

# 将query和召回节点中的内容组成dict，作为大模型的输入
res = robot({"query": query, "context_str": "".join([node_text for node_text in parent_list])})

print("系统答案：\n", res)
```
### 特殊节点组

除了transform外，还提供LLMParser可以实现基于LLM对文档进行解析的操作，支持三种模式：
- 摘要提取（summary）：对文段内容进行分析，提炼出核心信息，生成简洁且能代表全文主旨的摘要，帮助用户快速获取关键信息。
- 关键词提取（keyword）：自动从文段中识别出最具代表性的关键词，以便后续检索、分类或分析。
- QA对提取：从文段中自动抽取多个问答对，匹配与用户查询相似的问题，从而提供预设答案。这些问答对可以用于问答系统、知识库建设或生成式 AI 组件的参考，以提高信息获取的效率和准确性。

```python
import lazyllm

# 请在运行脚本前将要使用的在线模型 Api-key 抛出为环境变量，或更改为本地模型
llm = OnlineChatModule()

# LLMParser 是 LazyLLM 内置的基于 LLM 进行节点组构造的类，支持 summary，keywords和qa三种
summary_llm = LLMParser(llm, language="zh", task_type="summary") # 摘要提取LLM
keyword_llm = LLMParser(llm, language="zh", task_type="keywords") # 关键词提取LLM
qapair_llm = LLMParser(llm, language="zh", task_type="qa") # 问答对提取LLM
# 可以通过下方代码查看新的节点信息 
# nodes = qa_parser(DocNode(text=file_text)) 

# 利用 LLMParser 创建节点组
docs = Document("/path/to/your/doc/")

docs.create_node_group(name='summary', transform=lambda d: summary_llm(d), trans_node=True)
docs.create_node_group(name='keyword', transform=lambda d: keyword_llm(d), trans_node=True)
docs.create_node_group(name='qapair', transform=lambda d: qapair_llm(d), trans_node=True)

# 查看节点组内容，此处我们通过一个检索器召回一个节点并打印其中的内容，后续都通过这个方式实现
group_names = ["CoarseChunk", "summary", "keyword", "qapair"]
for group_name in group_names:
    retriever = Retriever(docs, group_name=group_name, similarity="bm25_chinese", topk=1)
    node = retriever("亚硫酸盐有什么作用？")
    print(f"======= {group_name} =====")
    print(node[0].get_content())
```

### 构建复杂节点组

```python
from lazyllm import OnlineChatModule, Document, LLMParser

# 请在运行脚本前将要使用的在线模型 Api-key 抛出为环境变量，或更改为本地模型
llm = OnlineChatModule()

# LLMParser 是 LazyLLM 内置的基于 LLM 进行节点组构造的类，支持 summary，keywords和qa三种
summary_llm = LLMParser(llm, language="zh", task_type="summary") # 摘要提取LLM
keyword_llm = LLMParser(llm, language="zh", task_type="keywords") # 关键词提取LLM
qapair_llm = LLMParser(llm, language="zh", task_type="qa") # 问答对提取LLM

docs = Document("/path/to/your/doc/")

# 以换行符为分割符，将所有文档都拆成了一个个的段落块，每个块是1个Node，这些Node构成了名为"block"的NodeGroup
docs.create_node_group(name='block', transform=lambda d: d.split('\n'))

# 使用一个可以提取问答对的大模型把每个文档的摘要作为一个名为"qapair"的NodeGroup，内容是针对文档的问题和答案对
docs.create_node_group(name='qapair', transform=lambda d: qapair_llm(d), trans_node=True)

# 使用一个可以提取摘要的大模型把每个文档的摘要作为一个名为"doc-summary"的NodeGroup，内容是整个文档的摘要
docs.create_node_group(name='doc-summary', transform=lambda d: summary_llm(d), trans_node=True)

# 在"block"的基础上，通过关键词抽取大模型从每个段落都抽取出一些关键词，每个段落的关键词是一个个的Node，共同组成了"keyword"这个NodeGroup
docs.create_node_group(name='keyword', transform=lambda b: keyword_llm(b), parent='block', trans_node=True)

# 在"block"的基础上进一步转换，使用中文句号作为分割符而得到一个个句子，每个句子都是一个Node，共同构成了"sentence"这个NodeGroup
docs.create_node_group(name='sentence', transform=lambda d: d.split('。'), parent='block')

# 在"block"的基础上，使用可以抽取摘要的大模型对其中的每个Node做处理，从而得到的每个段落摘要的Node，组成"block-summary"
docs.create_node_group(name='block-summary', transform=lambda b: summary_llm(b), parent='block', trans_node=True)

# 在"sentence"的基础上，统计每个句子的长度，得到了一个包含每个句子长度的名为"sentence-len"的NodeGroup
docs.create_node_group(name='sentence-len', transform=lambda s: len(s), parent='sentence')
```

### 多步骤查询

```python
import lazyllm
from lazyllm import Document, ChatPrompter, Retriever

# prompt设计
rewrite_prompt = "你是一个查询改写助手，将用户的查询改写的更加清晰。\
          注意，你不需要对问题进行回答，只需要对原问题进行改写.\
          下面是一个简单的例子：\
          输入：RAG\
          输出：为我介绍下RAG\
          用户输入为："

judge_prompt = "你是一个判别助手，用于判断某个回答能否解决对应的问题。如果回答可以解决问题，则输出True，否则输出False。 \
            注意，你的输出只能是True或者False。不要带有任何其他输出。 \
            当前回答为{context_str} \n"

robot_prompt = '你是一个友好的 AI 问答助手，你需要根据给定的上下文和问题提供答案。\
                根据以下资料回答问题：\
                {context_str} \n '

# 加载文档库，定义检索器在线大模型，
documents = Document(dataset_path="/mnt/lustre/share_data/dist/cmrc2018/data_kb")
retriever = Retriever(doc=documents, group_name="CoarseChunk", similarity="bm25_chinese", topk=3)
llm = lazyllm.OnlineChatModule(source='qwen', model='qwen-turbo')

# 重写查询的LLM
rewrite_robot = llm.share(ChatPrompter(instruction=rewrite_prompt))

# 根据问题和查询结果进行回答的LLM
robot = llm.share(ChatPrompter(instruction=robot_prompt, extra_keys=['context_str']))

# 用于判断当前回复是否满足query要求的LLM
judge_robot = llm.share(ChatPrompter(instruction=judge_prompt, extra_keys=['context_str']))

# 推理
query = "MIT OpenCourseWare是啥？"

LLM_JUDGE = False
while LLM_JUDGE is not True:
    query_rewrite = rewrite_robot(query)                # 执行查询重写
    print('\n重写的查询：', query_rewrite)

    doc_node_list = retriever(query_rewrite)            # 得到重写后的查询结果
    res = robot({"query": query_rewrite, "context_str": "\n".join([node.get_content() for node in doc_node_list])})

    # 判断当前回复是否能满足query要求
    LLM_JUDGE = bool(judge_robot({"query": query, "context_str": res}))
    print(f"\nLLM判断结果：{LLM_JUDGE}")

# 打印结果
print('\n最终回复: ', res)
```


## 召回评测指标

LazyLLM提供了上下文召回率，上下文相关性两种评测算法

使用示例：

```python
import lazyllm
from lazyllm.tools.eval import LLMContextRecall, NonLLMContextRecall, ContextRelevance

# 检索组件要求准备满足如下格式要求的数据进行评估
data = [{'question': '非洲的猴面包树果实的长度约是多少厘米？',
         # 当使用基于LLM的评估方法时要求答案是标注的正确答案
         'answer': '非洲猴面包树的果实长约15至20厘米。',
         # context_retrieved 为召回器召回的文档，按段落输入为列表
         'context_retrieved': ['非洲猴面包树是一种锦葵科猴面包树属的大型落叶乔木，原产于热带非洲，它的果实长约15至20厘米。',
                              '钙含量比菠菜高50％以上，含较高的抗氧化成分。',],
         # context_reference 为标注的应当被召回的段落
         'context_reference': ['非洲猴面包树是一种锦葵科猴面包树属的大型落叶乔木，原产于热带非洲，它的果实长约15至20厘米。']
}]
# 返回召回文档的命中率，例如上述data成功召回了标注的段落，因此召回率为1
m_recall = NonLLMContextRecall()
res = m_recall(data) # 1.0

# 返回召回文档中的上下文相关性分数，例如上述data召回的两个句子中只有一个是相关的
m_cr = ContextRelevance()
res = m_cr(data) # 0.5

# 返回基于LLM计算的召回率，LLM基于answer和context_retrieved判断是否召回了所有相关文档
# 适用于没有标注的情况下使用，比较耗费 token ，请根据需求谨慎使用
m_lcr = LLMContextRecall(lazyllm.OnlineChatModule())
res = m_lcr(data) # 1.0
```

## Similarity
LazyLLM 原生支持 cosine 相似度和 bm25 相似度，只需在检索时传入对应方法的名称指定要使用哪种相似度计算方式即可。

### Embedding的使用

Embedding 是一种将文档映射为一个保留原文语义的高维向量的技术，基于这种高维向量可以高效进行语义检索。目前通常使用基于 BERT 的 Embedding 模型对文档进行向量化得到稠密的向量表示。LazyLLM 支持调用线上和线下 Embedding 模型，其中线上模型通过 OnlineEmbeddingModule 调用，线下模型通过 TrainableModule 调用。

OnlineEmbeddingModule 用来管理创建目前市面上的在线Embedding服务模块，可配置参数有：

- source (str) – 指定要创建的模块类型，可选为 openai / sensenova / glm / qwen / doubao
- embed_url (str) – 指定要访问的平台的基础链接，默认是官方链接
- embed_model_name (str) – 指定要访问的模型，默认值为 text-embedding-ada-002(openai) / nova-embedding-stable(sensenova) / embedding-2(glm) / text-embedding-v1(qwen) / doubao-embedding-text-240715(doubao)

```python
from lazyllm import OnlineEmbeddingModule, TrainableModule, deploy

# 将在线模型 API key 抛出为环境变量，以 sensenova 为例
#     export LAZYLLM_SENSENOVA_API_KEY=     
#     export LAZYLLM_SENSENOVA_SECRET_KEY=
online_embed = OnlineEmbeddingModule("sensenova")

# 将本地模型地址抛出为环境变量：
#     export LAZYLLM_MODEL_PATH=/path/to/your/models
offline_embed = TrainableModule('bge-large-zh-v1.5').start()
# 或者使用绝对路径：
offline_embed = TrainableModule('/path/to/your/bge-large-zh-v1.5').start()

# 启动稀疏嵌入模型，目前仅支持 bge-m3
offline_sparse_embed = TrainableModule('bge-m3').deploy_method(
            (deploy.AutoDeploy, 
            {'embed_type': 'sparse'})).start()

print("online embed: ", online_embed("hello world"))
print("offline embed: ", offline_embed("hello world"))
print("offline sparse embed: ",  offline_sparse_embed("hello world"))
```

### 指定相似度

LazyLLM 原生支持 cosine 相似度进行语义检索。为文档指定 Embedding 后，Retriever 传入参数 similarity=“cosine” 即可。LazyLLM.Retriever 可传入的值有：“cosine”，“bm25”和“bm25_chinese”，其中只有“cosine”需要为文档指定Embedding。需要注意的是，哪怕为文档指定了 Embedding，但检索时使用 “bm25” 和 “bm25_chinese” 相似度时并不会进行嵌入计算。下面这段代码展示了使用原生相似度进行召回的例子，并简单对比了 cosine 和 bm25 的的召回特点：

```python
from lazyllm import Document, Retriever, TrainableModule

# 定义 embedding 模型
embed_model = TrainableModule("bge-large-zh-v1.5").start()

# 文档加载
docs = Document("/path/to/your/document", embed=embed_model)
docs.create_node_group(name='block', transform=(lambda d: d.split('\n')))

# 定义两个不同的检索器在同一个节点组使用不同相似度方法进行检索
retriever1 = Retriever(docs, group_name="block", similarity="cosine", topk=3)
retriever2 = Retriever(docs, group_name="block", similarity="bm25_chinese", topk=3)

# 执行查询
query = "都有谁参加了2008年的奥运会？"
result1 = retriever1(query=query)
result2 = retriever2(query=query)

print("余弦相似度召回结果：")
print("\n\n".join([res.get_content() for res in result1]))
print("bm25召回结果：")
print("\n\n".join([res.get_content() for res in result2]))
```

### 多Embedding召回

```python
from lazyllm import Document, Retriever, TrainableModule

# 定义多个 embedding 模型
bge_m3_embed = TrainableModule('bge-m3').start()
bge_large_embed = TrainableModule('bge-large-zh-v1.5').start()
embeds = {'vec1': bge_m3_embed, 'vec2': bge_large_embed}

# 文档加载
docs = Document("/mnt/lustre/share_data/dist/cmrc2018/data_kb", embed=embeds)
docs.create_node_group(name='block', transform=(lambda d: d.split('\n')))

# 定义两个不同的检索器分别对相同节点组的不同 embedding 进行检索 
retriever1 = Retriever(docs, group_name="block", embed_keys=['vec1'], similarity="cosine", topk=3)
retriever2 = Retriever(docs, group_name="block", embed_keys=['vec2'], similarity="cosine", topk=3)

# 执行检索
query = "都有谁参加了2008年的奥运会？"
result1 = retriever1(query=query)
result2 = retriever2(query=query)

print("使用bge-m3进行余弦相似度召回结果：")
print("\n\n".join([res.get_content() for res in result1]))
print("使用bge-large-zh-v1.5进行余弦相似度召回结果：")
print("\n\n".join([res.get_content() for res in result2]))
```
