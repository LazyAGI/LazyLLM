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
基础能力参考: [references/basic.md](references/basic.md)

---

## 安装与环境

- **安装核心包**: `pip install lazyllm`
- **按需安装功能组件**: `lazyllm install embedding chat finetune`（分别对应嵌入、对话、微调相关依赖，详见 [CLI 使用](#cli-使用)）
- **使用在线模型前**: 需配置对应平台的 API Key（见 [在线模型 API Key 配置](#在线模型-api-key-配置)）
- 更多环境与依赖说明见 [环境依赖](assets/basic/environment.md)

---

## 模型调用概览（本地 vs 线上）

LazyLLM 中所有模型能力统一通过 **AutoModel**、**OnlineModule**、**TrainableModule** 三类入口使用，优先推荐 **AutoModel** 以便自动选择在线或本地。

| 使用场景       | 推荐方式 | 说明 |
|----------------|----------|------|
| **线上模型**   | `lazyllm.OnlineModule(source=..., model=...)` 或 `lazyllm.AutoModel(source=..., model=...)` | 调用厂商 API，需配置对应 API Key。`source` 可为 `qwen` / `openai` / `glm` / `deepseek` / `sensenova` / `kimi` / `doubao` 等。 |
| **本地模型**   | `lazyllm.TrainableModule(model_name).start()` 或 `lazyllm.AutoModel(source='local', model=...)` | 本地加载与推理，可配合 `deploy_method(lazyllm.deploy.vllm)` 等指定 vLLM / LightLLM / LMDeploy。 |
| **统一入口**   | `lazyllm.AutoModel(model=..., source=...)` | 根据 `source` 与 config 自动选择在线或本地；不写 `source` 时由配置或模型名推断。 |

**简要示例**:

```python
import lazyllm

# 线上：需配置 LAZYLLM_<PLATFORM>_API_KEY
chat = lazyllm.OnlineModule(source='qwen', model='qwen-plus')
print(chat('你好'))

# 本地：先 start 再调用
local = lazyllm.TrainableModule('qwen2-1.5b').start()
print(local('你好'))

# 统一入口（自动选在线/本地）
llm = lazyllm.AutoModel(source='sensenova', model='SenseNova-V6-5-Pro')
print(llm('你好'))
```

详细参数、Embedding/Reranker/多模态类型及环境变量见 [Model 使用示例](assets/basic/model.md)。

---

## 在线模型 API Key 配置

使用 **OnlineModule** 或 **AutoModel** 调用线上模型前，需配置对应平台 API Key（推荐环境变量，避免写死在代码中）：

- **通用格式**: `export LAZYLLM_<平台名>_API_KEY=<你的 key>`
- **部分平台**（如 SenseNova）需同时配置: `LAZYLLM_SENSENOVA_API_KEY` 与 `LAZYLLM_SENSENOVA_SECRET_KEY`

常用平台与变量名示例：

| 平台       | 环境变量 |
|------------|----------|
| 通义千问 Qwen | `LAZYLLM_QWEN_API_KEY` |
| OpenAI     | `LAZYLLM_OPENAI_API_KEY` |
| 智谱 GLM   | `LAZYLLM_GLM_API_KEY` |
| DeepSeek   | `LAZYLLM_DEEPSEEK_API_KEY` |
| 日日新 SenseNova | `LAZYLLM_SENSENOVA_API_KEY`（可选 `LAZYLLM_SENSENOVA_SECRET_KEY`） |
| Kimi       | `LAZYLLM_KIMI_API_KEY` |
| 豆包 Doubao | `LAZYLLM_DOUBAO_API_KEY` |
| 硅基流动 SiliconFlow | `LAZYLLM_SILICONFLOW_API_KEY` |

完整平台列表见 [平台与 API Key](assets/basic/api_key_platforms.md)。配置方式也可通过 `lazyllm.config` 管理，见 [Config 使用示例](assets/basic/config.md)。

---

## 组件选择指南

根据任务需求选择组件：

| 任务类型         | 使用组件 | 参考文档 |
|------------------|----------|----------|
| 复杂流程编排     | Flow (pipeline, parallel, diverter) | [references/flow.md](references/flow.md) |
| 微调与部署       | AutoFinetune, AutoDeploy | [references/finetune.md](references/finetune.md) |
| 知识库问答       | RAG (Document, Retriever, Reranker) | [references/rag.md](references/rag.md) |
| 智能体应用       | Agent (ReactAgent, ReWOOAgent 等) + lazyllm.tools | [references/agent.md](references/agent.md) |

---

## CLI 使用

LazyLLM 提供命令行入口，用于安装依赖、部署模型与运行服务：

| 命令 | 用途 |
|------|------|
| `lazyllm install [embedding\|chat\|finetune\|...]` | 安装功能组件组或指定 Python 包（如 `lazyllm install embedding chat`） |
| `lazyllm deploy <model> [--framework vllm\|lightllm\|...] [--chat=true]` | 部署本地模型或启动 MCP 服务；`deploy mcp_server ...` 可启动 MCP 服务器 |
| `lazyllm run chatbot\|rag\|workflow.json\|...` | 运行聊天服务、RAG 服务、JSON 工作流或训练/推理服务 |

示例:

```bash
lazyllm install embedding chat
lazyllm deploy llama3-chat --framework vllm --chat=true
lazyllm run chatbot --model chatglm3-6b --framework vllm
lazyllm run rag --model bge-base --documents /path/to/docs
```

详细参数与子命令见 [CLI 使用](assets/basic/cli.md)。

---

## Flow（数据流编排）

Flow 用于编排多步骤、分支、并行或循环的数据处理流程。

### 何时使用 Flow

- 任务需拆分为多阶段并按序或按依赖执行  
- 需要并行执行多任务再汇总  
- 需要条件分支、循环或 DAG 依赖  

### 何时不必用 Flow

- 单次简单函数调用、无分支无并行的直线逻辑  

### Flow 组件

- **Pipeline**: 顺序执行  
- **Parallel**: 并行执行多任务  
- **Diverter**: 多分支并行路由  
- **Warp**: 单模块并行作用于多输入  
- **IFS / Switch**: 条件分支  
- **Loop**: 循环  
- **Graph**: DAG 依赖  
- **Bind**: 显式绑定与传递数据  

分布式或集群任务可通过 `lazyllm.launchers`（如 `remote`、`sco`、`slurm`）指定 Launcher，详见 [Launcher](assets/basic/launcher.md)。  
详细文档: [references/flow.md](references/flow.md)

---

## Finetune（微调与部署）

支持多种微调框架（AlpacaLoRA、Collie、LLaMA-Factory、FlagEmbedding、Auto 等）与推理部署（LightLLM、vLLM、LMDeploy、Auto 等）。  
微调与部署时可通过 `launcher` 参数指定单机/多机或调度器（如 `lazyllm.launchers.remote(ngpus=4)`），详见 [references/finetune.md](references/finetune.md) 与 [deploy_framework](assets/finetune/deploy_framework.md)。  
详细文档: [references/finetune.md](references/finetune.md)

---

## RAG（检索增强生成）

基于知识库的问答：文档加载、切分、向量化、检索与生成。  
基础用法见下；多策略检索与重排可用 Flow 编排，见 [references/rag.md](references/rag.md)。

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
llm.prompt(lazyllm.ChatPrompter(instruction='根据上下文回答问题：', extra_keys=['context_str']))

query = "用户问题"
doc_node_list = retriever(query=query)
res = llm({"query": query, "context_str": "".join([n.get_content() for n in doc_node_list])})
```

详细文档: [references/rag.md](references/rag.md)

---

## Agent（智能体）

构建能规划、调用工具、执行任务的智能体。Agent 与工具均来自 **lazyllm.tools**：`fc_register` 注册自定义工具，内置 Agent（ReactAgent、PlanAndSolveAgent、ReWOOAgent、FunctionCallAgent）可搭配内置或 MCP 工具使用。  
**Web 界面**：将 Agent 或 Flow 包装为 `lazyllm.WebModule(m)` 可快速得到 Web 对话界面；文档管理场景可用 `DocWebModule`，见 [内置 Module](assets/basic/modules.md)。

### 基础 Agent 示例

```python
import lazyllm
from lazyllm.tools import ReactAgent, fc_register

@fc_register('tool')
def my_tool(query: str) -> str:
    """Tool description for the agent."""
    return f"Result: {query}"

llm = lazyllm.OnlineModule(source='deepseek', model='deepseek-chat')
agent = ReactAgent(llm, tools=['my_tool'])
print(agent('用户问题'))
```

内置工具（搜索、Http、SQL、CodeGenerator 等）见 [内置工具的使用](assets/agent/tools.md)。  
详细文档: [references/agent.md](references/agent.md)

---

## lazyllm.tools 能力概览

**lazyllm.tools** 是 LazyLLM 的工具与能力模块，统一从 `lazyllm.tools` 或子模块按需导入，用于 RAG、Agent、数据与评测等场景。

| 类别 | 能力 | 典型用法 / 文档 |
|------|------|------------------|
| **RAG** | Document, Retriever, Reranker, SentenceSplitter, TempDocRetriever, GraphDocument, GraphRetriever, LLMParser | 知识库构建与检索，见 [references/rag.md](references/rag.md) |
| **Agent** | fc_register, ReactAgent, PlanAndSolveAgent, ReWOOAgent, FunctionCallAgent, ToolManager, SkillManager, MCPClient | 工具注册、多类 Agent、MCP 工具，见 [references/agent.md](references/agent.md)、[assets/agent/tools.md](assets/agent/tools.md) |
| **搜索与通用工具** | GoogleSearch, TencentSearch, BingSearch, WikipediaSearch, ArxivSearch, BochaSearch, HttpTool, Weather, Calculator, JsonExtractor, JsonConcentrator | Agent 可直接调用，见 [assets/agent/tools.md](assets/agent/tools.md) |
| **SQL / 数据** | SqlManager, SqlCall, MongoDBManager, DBManager | 自然语言转 SQL、表格问答，见 [assets/agent/tools.md](assets/agent/tools.md) |
| **代码与能力** | CodeGenerator, ParameterExtractor, QustionRewrite, code_interpreter, Sandbox | 代码生成与执行，可与 Agent 组合 |
| **评测** | LLMContextRecall, NonLLMContextRecall, ContextRelevance, Faithfulness, ResponseRelevancy | RAG/检索评测，见 [assets/rag/retriever.md](assets/rag/retriever.md) |
| **其他** | Git/GitHub/GitLab/Gitee/GitCode, review, ChineseCorrector, IntentClassifier, StreamCallHelper, WebModule | 代码仓库、审阅、意图分类、流式输出、Web 界面，见 [assets/basic/modules.md](assets/basic/modules.md) |

更多工具用法见 [内置工具](assets/agent/tools.md)。

---

## 流式输出

- **本地模型**: `TrainableModule(..., stream=True)` 或使用 `lazyllm.StreamCallHelper(module)` 对返回值迭代；若模型在 Flow 中，包装最外层 Flow。
- **在线模型**: `OnlineModule(..., stream=True)` 或构造时传入 `stream=True`。  
流式配置（前缀、后缀、颜色等）见 [本地模型与流式输出](assets/basic/local_model_stream.md) 与脚本 [stream_output.py](scripts/stream_output.py)。

---

## 最佳实践

1. **用 Basic 能力构建模块**: 优先 ModuleBase / ActionModule，模型、工具、Flow 均以 Module 形式存在。参考 [references/basic.md](references/basic.md)。
2. **模型统一入口**: 使用 AutoModel 或 OnlineModule / TrainableModule，不直接实例化具体实现类。参考 [Model 使用示例](assets/basic/model.md)。
3. **Prompt 通过 Prompter 注入**: 不硬编码 prompt，保持可配置与复用。参考 [Prompter 使用示例](assets/basic/prompter.md)。
4. **能力通过 AutoRegistry 暴露**: 通过继承 Base 或 Register 装饰器注册，经 `lazyllm.<group>.<key>` 访问。参考 [AutoRegistry 使用示例](assets/basic/registry.md)。
5. **配置与代码解耦**: API Key、模型名、参数等使用 Config 或环境变量。参考 [Config 使用示例](assets/basic/config.md)。
6. **先做 MVP**: 先跑通单模型/单流程，再增加并行、分支、重排等。
7. **复杂逻辑用 Flow**: 多阶段、并行、条件、循环、DAG 用 pipeline / parallel / diverter / loop 等。参考 [references/flow.md](references/flow.md)。
8. **RAG 先保证检索质量**: 先优化切分、Embedding、Retriever、Reranker，再调 Prompt。参考 [references/rag.md](references/rag.md).
9. **Agent 规划与执行分离**: 规划在 Agent，执行在 Tool / Module。参考 [references/agent.md](references/agent.md)。
10. **增量调试**: 先单 Module，再子 Flow，最后完整系统。
11. **优先 Auto 系列**: 如 AutoModel、AutoFinetune、AutoDeploy，必要时再指定具体实现。
12. **第三方依赖走 lazyllm.thirdparty**: 保证懒加载与可选依赖。参考 [Thirdparty 使用示例](assets/basic/thirdparty.md)。

代码示例合集: [基础使用代码合集](scripts/README.md)

---

## 本地文档索引

本 skill 所有链接均为本地引用，可在断网环境下阅读。  
- 基础与模型: [references/basic.md](references/basic.md)、[assets/basic/](assets/basic/)  
- Flow: [references/flow.md](references/flow.md)、[assets/flow/](assets/flow/)  
- Finetune: [references/finetune.md](references/finetune.md)、[assets/finetune/](assets/finetune/)  
- RAG: [references/rag.md](references/rag.md)、[assets/rag/](assets/rag/)  
- Agent 与工具: [references/agent.md](references/agent.md)、[assets/agent/](assets/agent/)  
- 脚本示例: [scripts/README.md](scripts/README.md)  

完整在线文档（需联网）可在项目或官方站点查阅，网站地址：https://docs.lazyllm.ai/zh-cn/latest/
