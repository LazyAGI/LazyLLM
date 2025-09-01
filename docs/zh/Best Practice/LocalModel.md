# TrainableModule 使用指南

TrainableModule 是 LazyLLM 中的核心模块，支持所有类型的模型（包括 LLM、Embedding、Rerank、多模态模型等）的训练、部署和推理。本文档将详细介绍 TrainableModule 的各种使用方式。

## 基础用法

### 创建和使用 TrainableModule

基础创建方式

```python
import lazyllm
model = lazyllm.TrainableModule('qwen2-1.5b')
```

!!! Note

    - 在基础的使用方式下，如果我们没有下载过模型，则会自动去 `huggingface` 或者 `modelscope` 上下载模型。如果我们不希望自动下载模型，则可以为TrainableModule传入参数 `trust_remote_code=False` 。当我们作为客户端连接lazyllm启动的推理服务时，这个做法通常很有效。
    - 模型默认会下载到 `~/.lazyllm/` 目录下。如果我们下午下载到其他目录下，则可以使用环境变量 `LAZYLLM_MODEL_CACHE_DIR=/path/to` 来指定模型的下载路径
    - 由于中国境内的上网策略，因此默认的模型源为 `modelscope` 。如果我们希望更换模型源，则可以使用环境变量 `LAZYLLM_MODEL_SOURCE=huggingface` 来将模型源切换到huggingface。
    - 如果我们本地有下载好的模型权重，则我们可以用绝对路径指定 `model = lazyllm.TrainableModule('/path/to/qwen2-1.5b')`
    - 如果我们将模型下载到统一的位置，则我们可以配置环境变量 `LAZYLLM_MODEL_PATH=/path/to` 来指定模型的根目录

指定微调模型的目录

```python
import lazyllm
model = lazyllm.TrainableModule('qwen2-1.5b', target_path='my_model')
```

我们可以使用 `target_path` 来指定微调模型的位置，在需要微调的场景，模型微调后会保存到 `target_path` 里面，后续推理的时候会从 `target_path` 加载模型进行推理。如果你基于某个模型做了微调，就可以将微调后的模型放到 `target_path` 目录下。


### 基本推理

```python
# 启动模型
model.start()

# 进行推理
response = model("hello")
print(response)
```

LazyLLM的模型依然采用仿函数的模式进行使用，本地模型需要调用 `start()` 进行部署之后，才可以进行调用。

## 微调

### 执行微调

```python
import lazyllm

# 使用自动微调方法
model = lazyllm.TrainableModule('qwen2-1.5b', target_path='/path/to/model')
    .finetune_method(lazyllm.finetune.auto)
    .trainset('/path/to/training/data')
    .mode('finetune')

# 使用特定的微调方法
model = lazyllm.TrainableModule('qwen2-1.5b')
    .finetune_method(lazyllm.finetune.llamafactory, learning_rate=1e-4, num_train_epochs=3)
    .trainset('/path/to/training/data')
    .mode('finetune')

# 执行微调
model.update()
```

微调的细节可以参考 [微调教程](../Tutorial/9.md)

### 支持的微调方法

- `lazyllm.finetune.auto`: 自动选择微调方法
- `lazyllm.finetune.llamafactory`: 使用 LLaMA Factory 进行大模型的微调
- `lazyllm.finetune.collie`: 使用 Collie 进行微调，目前Collie已经停止迭代，因此会在后续的版本移除
- `lazyllm.finetune.flagembedding`: 用于 Embedding 模型的微调

### 微调参数配置

```python
model = lazyllm.TrainableModule('qwen2-1.5b')\
    .finetune_method(lazyllm.finetune.llamafactory,
                     learning_rate=1e-4,      # 学习率
                     num_train_epochs=3,      # 训练轮数
                     per_device_train_batch_size=4,  # 批次大小
                     max_samples=1000,        # 最大样本数
                     val_size=0.1)            # 验证集比例
```

## 流式输出

模型推理的时间都会比较久，通常情况下，一个32B的模型，使用A100的显卡进行推理，面向单个会话，每秒大约能生成30-35个新字符。如果我们想生成一篇2000字的文章，则差不多需要1分钟多的时间。对于用户而言，1分钟的等待时间往往是不可以接受的，通常的解决方案是，变生成变呈现给用户，也就是我们说的流式输出。

### 启用流式输出

```python
# 在创建时启用流式输出
model = lazyllm.TrainableModule('qwen2-1.5b', stream=True)
```

### 流式输出的使用

```python
import lazyllm
model = lazyllm.StreamCallHelper(model)
for msg in model('hello'):
    print(msg)
```

我们借助一个 `StreamCallHelper` 包装我们的模型，即可对模型的调用结果进行迭代，来达到流式输出的目标。当我们的模型在一个flow中的时候，需要包装最外层的flow，而不是模型，例如：

```python
import lazyllm
model = lazyllm.TrainableModule('qwen2-1.5b', stream=True)
ppl = lazyllm.pipeline(model)
ppl = lazyllm.StreamCallHelper(ppl)
for msg in ppl('hello'):
    print(msg)
```

### 流式输出配置

我们可以给流式输出做一些配置，来给流式输出的内容加一些前缀或者后缀，并实现“花花绿绿”的流式输出。具体的配置如下：

```python
# 配置流式输出样式
stream_config = {
    'color': 'green',           # 输出颜色
    'prefix': 'AI: ',          # 前缀
    'prefix_color': 'blue',    # 前缀颜色
    'suffix': 'End\n',            # 后缀
    'suffix_color': 'red'      # 后缀颜色
}

model = lazyllm.TrainableModule('qwen2-1.5b', stream=stream_config)
```

这样，我们的流式输出就会以蓝色的AI开头，流式输出内容的本身是绿色的，再以红色的End结尾。如果我们一个任务中有多个大模型，那么这个能力会非常的好用，我们可以给每个大模型的输出配置不同的前缀和颜色，来呈现给用户。

## Prompt 设置

### 基础 Prompt 设置

```python
# 设置简单的文本 prompt
model = lazyllm.TrainableModule('qwen2-1.5b')
    .prompt("你是一个有用的AI助手，请用简洁的语言回答问题。")

# 设置对话历史
history = [
    ["用户", "你好"],
    ["助手", "你好！有什么可以帮助你的吗？"]
]
model = model.prompt("继续对话", history=history)
```

我们设置的历史对话为 “系统提示词” ，在多用户 / 多会话的环境下，对每个用户 / 会话都生效。我们可以参考 [提示词教程](prompt.md) 了解更多。

### 使用字典格式的 Prompt

```python
# 使用字典格式设置更复杂的 prompt
prompt_config = {
    'system': '你是一个专业的{system_role}',
    'user': '{user_input}',
}

model = lazyllm.TrainableModule('qwen2-1.5b').prompt(prompt_config)
model(dict(system_role='编程助手', user_input='根据用户的需求编程'))
```

### 清除 Prompt

```python
# 清除 prompt，使用空 prompt
model = model.prompt(None)
```

## 输出格式化

### 使用内置格式化器

```python
# 使用 JSON 格式化器
model = lazyllm.TrainableModule('qwen2-1.5b')\
    .formatter(lazyllm.formatter.JsonFormatter('[:][a]'))
```

可以通过 JsonFormatter 对输出中的json进行提取，并获取指定的元素。

### 自定义格式化器

```python
# 使用自定义函数作为格式化器
def my_formatter(text):
    return f"处理后的结果: {text.strip()}"

model = lazyllm.TrainableModule('qwen2-1.5b').formatter(my_formatter)

# 使用链式格式化器
model = model.formatter(lazyllm.formatter.JsonFormatter() | lazyllm.formatter.StrFormatter())
```

## 模型共享

### 基础共享

LazyLLM虽然能方便的部署模型，但在实际使用的时候，如果每个模块都单独部署一个模型，会造成资源的浪费，因此我们引入了模型共享的机制。

```python
# 创建基础模型
base_model = lazyllm.TrainableModule('qwen2-1.5b').start()

# 创建共享实例，使用相同的模型但不同的 prompt
chat_model = base_model.share(prompt="你是一个聊天机器人")
code_model = base_model.share(prompt="你是一个编程助手")
```

### 共享参数配置

```python
# 共享时可以指定不同的配置
shared_model = base_model.share(
    prompt="新的 prompt",                    # 新的 prompt
    format=lazyllm.formatter.JsonFormatter(), # 新的格式化器
    stream={'color': 'blue'}                 # 新的流式配置
)
```

### 共享的优势

- **资源节省**: 多个实例共享同一个模型部署，节省 GPU 内存
- **配置灵活**: 每个共享实例可以有不同的 prompt 和格式化器
- **性能优化**: 避免重复加载模型

## 从 URL 连接

### 连接已有的服务

```python
# 连接到已有的 HTTP 服务
model = lazyllm.TrainableModule().deploy_method(
    lazyllm.deploy.vllm, 
).start()

# 连接到上面部署的模型，假设上面的url是 http://localhost:8000/generate
remote_model = lazyllm.TrainableModule().deploy_method(
    lazyllm.deploy.vllm,
    url='http://localhost:8000/generate/'
)

# 使用远程模型进行推理
response = remote_model("你好")
```

通过这种方式，我们可以实现先启动推理服务，多个不同的用户，再多个不同的进程中使用这些推理服务。

## Embedding 和 Rerank 模型

### Embedding 模型

```python
# 创建 Embedding 模型
embedding_model = lazyllm.TrainableModule('bge-large-zh-v1.5')

# 进行文本嵌入
embeddings = embedding_model("这是一个测试文本")
print(embeddings)  # 返回向量列表

# 微调 Embedding 模型
embedding_model = lazyllm.TrainableModule('bge-large-zh-v1.5')\
    .finetune_method(lazyllm.finetune.flagembedding)\
    .trainset('/path/to/embedding_data')\
    .mode('finetune')
embedding_model.update()
```

对于lazyllm官方模型清单没有的模型，我们需要显式的指定模型的type

```python
# 创建 Embedding 模型
embedding_model = lazyllm.TrainableModule('bge-large-zh-v1.5', type='embed')
```

### Rerank 模型

```python
# 创建 Rerank 模型
rerank_model = lazyllm.TrainableModule('bge-reranker-large')

# 进行重排序
query = "用户查询"
documents = ["文档1", "文档2", "文档3", "文档4"]
top_n = 2

results = rerank_model(query, documents=documents, top_n=top_n)
# 返回 [(index, score), ...] 格式的结果
print(results)
```

对于lazyllm官方模型清单没有的模型，我们需要显式的指定模型的type

```python
# 创建 Embedding 模型
embedding_model = lazyllm.TrainableModule('bge-reranker-large', type='rerank')
```

通常情况下，我们不会单独使用Rerank模型，而是在RAG中进行使用，再RAG中使用Rerank模型可以参考 [RAG最佳实践](rag.md)

## 多模态模型

### 图像生成 (SD)

```python
# 创建 Stable Diffusion 模型
sd_model = lazyllm.TrainableModule('stable-diffusion-3-medium')

# 生成图像
image_prompt = "a beautiful landscape with mountains and lakes"
image_result = sd_model(image_prompt)
# 返回生成的图像路径或数据
```

### 语音转文本 (STT)

```python
# 创建语音识别模型
stt_model = lazyllm.TrainableModule('SenseVoiceSmall')

# 进行语音识别
audio_file = '/path/to/audio.wav'
text_result = stt_model(audio_file)
print(text_result)
```

### 文本转语音 (TTS)

```python
# 创建语音合成模型
tts_model = lazyllm.TrainableModule('ChatTTS')

# 进行语音合成
text = "你好，这是一个测试"
audio_result = tts_model(text)
# 返回生成的音频文件路径
```

### 视觉语言模型 (VLM)

```python
# 创建视觉语言模型
vlm_model = lazyllm.TrainableModule('internvl-chat-2b-v1-5')\
    .deploy_method(lazyllm.deploy.LMDeploy)

# 进行图像问答
image_path = '/path/to/image.jpg'
question = "这张图片中有什么？"
response = vlm_model(encode_query_with_filepaths(question, image_path))
print(response)
```

## OpenAI 格式部署

### 基于 vLLM 启动 OpenAI 服务

```python
# 使用 vLLM 部署 OpenAI 格式的服务
model = lazyllm.TrainableModule('qwen2-1.5b')\
    .deploy_method(lazyllm.deploy.vllm, 
                   openai_api=True,  # 启用 OpenAI API 格式
                   port=8000)

# 启动服务
model.start()

# 服务将在 http://localhost:8000/v1/ 提供 OpenAI 兼容的 API
```

### 连接 OpenAI 格式的服务

```python
# 连接到 OpenAI 格式的服务
openai_model = lazyllm.TrainableModule().deploy_method(
    lazyllm.deploy.vllm,
    url='http://localhost:8000/v1/'
)

# 使用 OpenAI 格式进行推理
response = openai_model("你好")
```

!!! Note

    如果URL以 `v1/` 或 `v1/chat/completions` 结尾，会被认为是openai格式的url；而以 `generate` 结尾，则会被认为是VLLM格式的URL。
