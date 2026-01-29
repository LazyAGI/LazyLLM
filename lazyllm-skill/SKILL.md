---
name: lazyllm-skill
description: >
  LazyLLM framework for building multi-agent AI applications. Use when task mentioned LazyLLM or AI program for:
  (1) Flow orchestration - linear, branching, parallel, loop workflows for complex data pipelines,
  (2) Model fine-tuning and acceleration - finetuning LLMs with LLaMA-Factory/Alpaca-LoRA/Collie and acceleration with vLLM/LMDeploy/LightLLM. Includes comprehensive code examples for all components,
  (3) RAG systems - knowledge-based QA with document retrieval, vectorization, and generation,
  (4) Agent development - single/multi-agent systems with tools, memory, planning, and web interfaces.
---

# LazyLLM 框架

LazyLLM 是构建和优化多 Agent 应用的一站式开发工具，为应用开发、数据准备、模型部署、模型微调、评测和可视化提供了大量工具。
其基础能力的使用参考文档: [references/basic.md](references/basic.md)

## 组件选择指南

根据任务需求选择合适的组件：

| 任务类型 | 使用组件 | 参考文档 |
|---------|---------|---------|
| 复杂流程编排 | Flow (pipeline, parallel, diverter) | [references/flow.md](references/flow.md) |
| 微调加速和部署 | AutoFinetune, AutoDeploy | [references/finetune.md](references/finetune.md) |
| 知识库问答系统 | RAG (Document, Retriever, Reranker) | [references/rag.md](references/rag.md) |
| 智能体应用 | Agent (ReactAgent, ReWOOAgent) | [references/agent.md](references/agent.md) |

## Flow (数据流编排)

Flow 是 LazyLLM 中用于编排复杂数据处理流程的核心组件体系。
当一个任务不再是“单函数调用”，而是由多个处理步骤、条件分支、并行任务或循环逻辑构成时，应优先使用 Flow 进行组织。

### 什么时候应该使用 Flow

当满足以下任一情况时，适合使用 Flow:

    1.任务需要拆分为多个处理阶段，并按顺序执行
    2.不同步骤之间存在数据依赖关系
    3.需要并行执行多个独立任务并汇总结果
    4.需要根据条件选择不同处理路径
    5.需要对同一处理逻辑批量作用于多条输入
    6.需要循环执行某组步骤直到满足条件
    7.需要用 DAG（有向无环图）描述复杂依赖关系

### 什么时候不需要使用 Flow

以下场景通常不需要 Flow:

    1.只有一个简单函数调用
    2.无分支、无并行、无组合的直线逻辑
    3.仅做一次性的轻量处理

### Flow组件

Flow 由多种可组合的流程组件构成，不同组件对应不同控制结构:

- Pipeline：顺序执行
- Parallel：并行执行多个任务
- Diverter：多分支并行路由
- Warp：单模块并行应用于多输入
- IFS：if-else 条件结构
- Switch：多路条件选择
- Loop：循环执行
- Graph：DAG 形式的复杂依赖图
- Bind：在流程中显式绑定和传递数据

这些组件可以单独使用，也可以相互嵌套组合。

### 详细文档

[references/flow.md](references/flow.md)

## Finetune(微调加速和部署)

模型微调和推理加速，并且快速将模型进行部署，支持Apacalora, Collie, LlamaFactory, Flagembedding, Dummy, Auto多种微调框架。LazyLLM支持Lightllm, VLLM, LMDeploy, Dummy, Auto多种推理方式。

### 内置的微调方法

- AlpacaloraFinetune: 基于 alpaca-lora 项目提供的 LoRA 微调能力，用于对大语言模型进行 LoRA 微调。
- CollieFinetune: 基于 Collie 框架提供的 LoRA 微调能力，用于对大语言模型进行 LoRA 微调。
- LlamafactoryFinetune: 基于 LLaMA-Factory 框架提供的训练能力，用于对大语言模型(或视觉语言模型)进行训练。
- FlagembeddingFinetune: 基于 FlagEmbedding 框架提供的训练能力，用于训练嵌入和重排模型。
- AutoFinetune: 可根据输入的参数自动选择合适的微调框架和参数，以对大语言模型进行微调。
- DummyFinetune: 用于占位实现微调逻辑。 此类主要用于演示或测试目的，因为它不执行任何实际的微调操作。

### 内置的推理加速

- Lightllm: 基于 LightLLM 框架提供的推理能力，用于对大语言模型进行推理。
- Vllm: 基于 VLLM 框架提供的推理能力，用于大语言模型的部署与推理。
- LMDeploy: 基于 LMDeploy 框架，用于启动并管理大语言模型的推理服务。
- DummyDeploy: 一个用于测试的模拟部署类，实现了一个简单的流水线风格部署服务，该类主要用于内部测试和示例用途。
- AutoDeploy: 根据输入的参数自动选择合适的推理框架和参数，以对大语言模型进行推理。

### 详细文档

[references/finetune.md](references/finetune.md)

## RAG (检索增强生成)

构建基于知识库的问答系统，包括文档加载、切分、向量化、检索和生成。

### 基础 RAG

```python
import lazyllm

documents = lazyllm.Document(
    dataset_path="/path/to/docs",
    embed=lazyllm.OnlineEmbeddingModule(),
    manager=False
)

retriever = lazyllm.Retriever(
    doc=documents,
    group_name="CoarseChunk",
    similarity="bm25_chinese",
    topk=3
)

llm = lazyllm.OnlineChatModule()
prompt = '根据上下文回答问题：'
llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))

query = "用户问题"
doc_node_list = retriever(query=query)
res = llm({
    "query": query,
    "context_str": "".join([node.get_content() for node in doc_node_list]),
})
```

### 高级 RAG

使用 Flow 编排多策略检索和重排序：

```python
import lazyllm
from lazyllm import bind

documents = lazyllm.Document(
    dataset_path="/path/to/docs",
    embed=lazyllm.OnlineEmbeddingModule()
)

documents.create_node_group(name="sentences", transform=lambda s: '。'.split(s))

prompt = '根据上下文回答问题：'

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
res = rag("用户问题")
```

### 详细文档

[references/rag.md](references/rag.md)

## Agent (智能体)

构建能够自主规划、调用工具、执行任务的智能体系统。

### 基础 Agent

```python
from lazyllm.tools import ReactAgent
from lazyllm import OnlineChatModule
@fc_register('tool')
def query_db(user_query: str, db_name: str = DB_NAME, tables_info: dict = TABLE_INFO) -> str:
    '''
    General SQL Query Tool for any database and tables.

    Args:
        user_query (str): User's natural language query.
        db_name (str, optional): SQLite database file. Defaults to DB_NAME.
        tables_info (dict, optional): Table structure info. Defaults to TABLE_INFO.

    Returns:
        str: SQL query result.
    '''
    sql_manager = SqlManager(
        'sqlite', None, None, None, None,
        db_name=db_name, tables_info_dict=tables_info
    )
    sql_call = SqlCall(sql_llm, sql_manager, use_llm_for_sql_result=False)

    return sql_call(user_query)

llm = OnlineChatModule()
tools = ['query_db']

if __name__ == '__main__':
    init_db(DB_NAME, SAMPLE_DATA)

    user_input = 'Show the total profit for each product category, sorted from highest to lowest.'
    agent = ReactAgent(llm, tools)
    answer = agent(user_input)
    print('Answer:\n', answer)
```

### 详细文档

[references/agent.md](references/agent.md)

## 最佳实践

1. 先使用 Basic 能力构建模块: 优先使用 ModuleBase / ActionModule 封装能力，所有模型、工具、Flow 都应以 Module 形式存在。 参考文档[references/basic.md](references/basic.md)

2. 模型统一通过 AutoModel 或 OnlineModule 创建: 不直接实例化具体模型实现类，便于后续切换在线/本地与部署方式。 参考文档[Model使用示例](./assets/basic/model.md)

3. Prompt 必须通过 Prompter 注入: 不在代码中硬编码 prompt，使提示词与逻辑解耦、可复用、可配置。 参考文档[Prompter使用示例](./assets/basic/prompter.md)

4. 能力暴露统一走 AutoRegistry: 新能力通过继承 Base 类或 Register 装饰器注册，通过 lazyllm.<group>.<key> 访问。 参考文档[AutoRegistry使用示例](./assets/basic/registry.md)

5. 配置与代码解耦: API Key(需提醒用户配置)、模型名、参数等放入 Config 或环境变量，不写死在代码中。参考文档[Config使用示例](./assets/basic/config.md)

6. 先搭建最小可运行版本（MVP）: 先保证单模型/单流程跑通，再逐步增加并行、分支、重排等能力。

7. 复杂控制逻辑统一使用 Flow: 多阶段、并行、条件、循环、DAG 依赖场景下使用 pipeline / parallel / diverter / loop 等组件。参考文档[references/flow.md](references/flow.md)

8. RAG 优先保证检索质量: 先优化切分、Embedding、Retriever、Reranker，再调整 Prompt。参考文档[references/rag.md](references/rag.md)

9. Agent 中规划与执行分离: Agent 负责规划，具体执行逻辑放在 Tool / Module 中。参考文档[references/agent.md](references/agent.md)

10. 增量调试: 先测单 Module，再测子 Flow，最后测试完整系统。

11. 优先使用 Auto 系列能力: 如 AutoModel、AutoFinetune、AutoDeploy，在必要时再指定具体实现。

12. 第三方依赖通过 lazyllm.thirdparty 引入: 避免直接 import，保证懒加载与可选依赖管理。参考文档[Thirdparty使用示例](./assets/basic/thirdparty.md)

代码示例合集:
[基础使用代码合集](./scripts/README.md)

## 官方文档

完整文档: https://docs.lazyllm.ai/zh-cn/latest/
