# LazyLLM 代码模板

本目录包含 LazyLLM 的常用代码模板，可以直接使用或根据需求修改。

## 模板列表

### 基础模板

| 文件 | 功能 | 使用场景 |
|------|------|---------|
| `basic_chatbot.py` | 基础对话机器人 | 简单的对话应用 |
| `multi_turn_chat.py` | 多轮对话 | 需要保存对话历史 |
| `web_interface.py` | Web 界面集成 | 部署为 Web 应用 |

### RAG 模板

| 文件 | 功能 | 使用场景 |
|------|------|---------|
| `basic_rag.py` | 基础 RAG 系统 | 知识库问答 |

### Agent 模板

| 文件 | 功能 | 使用场景 |
|------|------|---------|
| `basic_agent.py` | 基础智能体 | 工具调用和任务执行 |

### Flow 模板

| 文件 | 功能 | 使用场景 |
|------|------|---------|
| `basic_flow.py` | 数据流编排 | 复杂流程和数据处理 |

### 微调模板

| 文件 | 功能 | 使用场景 |
|------|------|---------|
| `basic_finetune.py` | 基础微调 | 模型微调和优化 |

## 使用步骤

### 1. 设置环境变量

```bash
# 以通义千问为例
export LAZYLLM_QWEN_API_KEY=your_api_key_here
```

### 2. 安装依赖（如需要）

```bash
# 基础功能
pip install lazyllm

# 完整功能
lazyllm install full

# 微调功能
lazyllm install llama-factory

# 推理加速
lazyllm install vllm
```

### 3. 修改模板

打开模板文件，修改以下内容：
- 文档路径（RAG 模板）
- 模型名称
- 提示词
- 端口号（Web 模板）

### 4. 运行模板

```bash
python basic_chatbot.py
```

## 快速开始

### 最简单的对话机器人

```bash
# 1. 设置 API Key
export LAZYLLM_QWEN_API_KEY=sk-xxxxx

# 2. 运行
python basic_chatbot.py
```

### 知识库问答系统

```bash
# 1. 设置 API Key
export LAZYLLM_QWEN_API_KEY=sk-xxxxx

# 2. 修改 basic_rag.py 中的 dataset_path
# dataset_path = "/path/to/your/docs"

# 3. 运行
python basic_rag.py
```

### Web 界面

```bash
# 1. 设置 API Key
export LAZYLLM_QWEN_API_KEY=sk-xxxxx

# 2. 运行
python web_interface.py

# 3. 浏览器访问
# http://localhost:23333
```

## 模板定制

### 添加新工具

在 `basic_agent.py` 中添加：

```python
@fc_register('tool')
def my_new_tool(param: str) -> str:
    """新工具的描述"""
    return f"处理结果: {param}"

# 添加到 agent 的工具列表
agent = FunctionCallAgent(llm, tools=['search_tool', 'calculate_tool', 'my_new_tool'])
```

### 修改提示词

```python
# 修改对话提示词
prompt = '你是一个专业领域的专家...'
chat.prompt(lazyllm.ChatPrompter(instruction=prompt))
```

### 添加更多处理步骤

在 `basic_flow.py` 中添加：

```python
with pipeline() as p:
    p.step1 = function1
    p.step2 = function2
    p.step3 = function3
    # 添加更多步骤...
```

## 进阶示例

参考 `references/` 目录下的详细文档：
- [rag.md](../references/rag.md) - RAG 高级功能
- [agent.md](../references/agent.md) - Agent 高级用法
- [flow.md](../references/flow.md) - Flow 编排模式
- [finetune.md](../references/finetune.md) - 微调和加速

## 常见问题

### Q: 如何更换模型？

```python
# 修改模型名称
model = lazyllm.TrainableModule('model-name-here')

# 或使用在线模型
chat = lazyllm.OnlineChatModule(model='qwen-turbo')
```

### Q: 如何调整参数？

```python
# RAG 检索参数
retriever = lazyllm.Retriever(
    doc=documents,
    topk=5,  # 返回更多文档
    similarity_cut_off=0.01  # 降低阈值
)

# 微调参数
model.finetune(
    data=train_data,
    finetune_args={
        'learning_rate': 1e-4,  # 调整学习率
        'num_train_epochs': 5,  # 增加训练轮数
    }
)
```

### Q: 如何调试？

```python
# 打印中间结果
def debug_step(input):
    print(f"中间结果: {input}")
    return input

with pipeline() as p:
    p.step1 = function1
    p.debug = debug_step  # 添加调试步骤
    p.step2 = function2
```

## 相关资源

- LazyLLM 官方文档: https://docs.lazyllm.ai/zh-cn/latest/
- LazyLLM GitHub: https://github.com/LazyAGI/LazyLLM
- 本技能文档: ../SKILL.md
