# Flow组件组合使用

通过组合多个Flow组件进而实现复杂数据流管理

## 组合流程

### Pipeline + Parallel

```python
from lazyllm import pipeline, parallel
with pipeline() as ppl:
    # 第一步：并行处理
    with parallel().sum as ppl.p1:
        p1.task1 = lambda x: x * 2
        p1.task2 = lambda x: x + 3

    # 第二步：处理并行结果
    ppl.step2 = lambda x: x / 2

result = ppl(1)  # ((1*2) + (1+3)) / 2 = 3
```

### Pipeline + Diverter

```python
from lazyllm import pipeline
from lazyllm.flow import Diverter

def process_input(input):
    return input.strip()

diverter = Diverter()
diverter.add_branch(
    lambda x: f"搜索: {x}",
    condition=lambda x: x.startswith('search')
)
diverter.add_branch(
    lambda x: f"计算: {x}",
    condition=lambda x: x.startswith('calc')
)
diverter.add_branch(
    lambda x: f"其他: {x}",
    default=True
)

with pipeline() as ppl:
    ppl.preprocess = process_input
    ppl.divert = diverter

result = ppl("search AI")  # 结果: '搜索: search AI'
```

### 复杂流程示例

```python
from lazyllm import pipeline, parallel
from lazyllm.flow import Diverter

# 定义处理函数
def data_preprocessing(input):
    return input.lower().strip()

def text_processing(input):
    return f"文本处理: {input}"

def image_processing(input):
    return f"图像处理: {input}"

def audio_processing(input):
    return f"音频处理: {input}"

def result_aggregation(results):
    return f"聚合结果: {' | '.join(results)}"

# 构建复杂流程
with pipeline() as ppl:
    # 第一步：数据预处理
    ppl.preprocess = data_preprocessing

    # 第二步：根据类型分支
    with Diverter() as ppl.diverter:
        ppl.diverter.add_branch(
            text_processing,
            condition=lambda x: x.startswith('text')
        )
        ppl.diverter.add_branch(
            image_processing,
            condition=lambda x: x.startswith('image')
        )
        ppl.diverter.add_branch(
            audio_processing,
            condition=lambda x: x.startswith('audio')
        )

    # 第三步：并行处理多种输出
    with parallel().sum as ppl.parallel:
        ppl.parallel.analysis1 = lambda x: len(x)
        ppl.parallel.analysis2 = lambda x: x.split()

    # 第四步：结果聚合
    ppl.aggregate = result_aggregation

result = ppl("Text example")
# 结果: '聚合结果: 11 | [text, example]'
```

## 与 RAG 结合

```python
import lazyllm
from lazyllm import pipeline, parallel, bind

documents = lazyllm.Document(
    dataset_path="/path/to/docs",
    embed=lazyllm.OnlineEmbeddingModule()
)

prompt = '根据上下文回答问题：'

with pipeline() as ppl:
    # 并行检索多个策略
    with parallel().sum as ppl.prl:
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

    # 重排序
    ppl.reranker = lazyllm.Reranker(
        name='ModuleReranker',
        model=lazyllm.OnlineEmbeddingModule(type="rerank"),
        topk=1
    ) | bind(query=ppl.input)

    # 格式化输入
    ppl.formatter = (
        lambda nodes, query: dict(
            context_str="".join([node.get_content() for node in nodes]),
            query=query,
        )
    ) | bind(query=ppl.input)

    # 生成回答
    ppl.llm = lazyllm.OnlineChatModule().prompt(
        lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str'])
    )

rag = lazyllm.ActionModule(ppl)
result = rag("用户问题")
```

## 与 Agent 结合

```python
from lazyllm import pipeline, parallel
from lazyllm.tools import FunctionCallAgent

# 定义工具
@fc_register('tool')
def search_tool(query: str) -> str:
    return f"搜索结果: {query}"

@fc_register('tool')
def calc_tool(num: float) -> float:
    return num * 2

# 创建多个 Agent
llm1 = lazyllm.OnlineChatModule()
agent1 = FunctionCallAgent(llm1, tools=['search_tool'])

llm2 = lazyllm.OnlineChatModule()
agent2 = FunctionCallAgent(llm2, tools=['calc_tool'])

def aggregator(results):
    return f"Agent1: {results[0]}\nAgent2: {results[1]}"

# 组合流程
with pipeline() as ppl:
    # 并行执行多个 Agent
    with parallel() as ppl.prl:
        ppl.prl.search_agent = agent1
        ppl.prl.calc_agent = agent2

    # 聚合结果
    ppl.aggregate = aggregator

result = ppl("搜索并计算")
```
