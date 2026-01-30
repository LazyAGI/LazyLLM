# 模型使用

LazyLLM内主要通过以下三种方式获取模型服务: AutoModel, OnlineModule, TrainableModule

## AutoModel使用
LazyLLM的模型使用通过AutoModel进行整合，用于快速创建在线推理模块 OnlineModule 或本地 TrainableModule 的工厂类。会优先采用用户传入的参数，若开启 config 则会根据 auto_model_config_map 中的配置进行覆盖，然后自动判断应当构建在线模块还是本地模块：

    - 当判定为在线模块时，参数会透传给 OnlineModule（自动匹配 OnlineChatModule / OnlineEmbeddingModule / OnlineMultiModalModule）。
    - 当判定为本地模块时，则以 model 与用户参数初始化 TrainableModule，并读取 config map 里的配置参数。

参数：

- model (str) – 指定模型名称。例如 Qwen3-32B。必填。
- config_id (Optional[str]) – 指定配置文件里的id。默认为空。
- source (Optional[str]) – 使用的服务提供方。为在线模块（OnlineModule）指定 qwen / glm / openai 等；若设为 local 则强制创建本地 TrainableModule。
- type (Optional[str]) – 模型类型。若未指定会尝试从 kwargs 中获取或由在线模块自动推断。
- config (Union[str, bool]) – 是否启用 auto_model_config_map 的覆盖逻辑，或者用户指定的 config 文件路径。默认为 True。
- **kwargs – 兼容 model 的同义字段 base_model 和 embed_model_name，不接收其他用户传入的字段。

```python
import lazyllm

llm = lazyllm.AutoModel(
        source="sensenova", 
        model="SenseNova-V6-5-Pro"
    )
response = llm("你好，简单介绍一下你自己。")
print(response)
```

## OnlineModule(在线大模型)

### 基本使用

```python
import lazyllm
# 或按需从 module 子模块导入
# 1.2 构造函数签名
lazyllm.OnlineModule(
    model: Optional[str] = None,   # 模型名称
    source: Optional[str] = None,  # 服务商：qwen / openai / glm / sensenova / kimi / doubao / ppio / siliconflow 等
    *,
    type: Optional[str] = None,    # 类型：llm / vlm / embed / rerank / stt / tts / sd / image_editing 等
    url: Optional[str] = None,     # 自定义 API 基础 URL，覆盖默认官方地址
    **kwargs                       # 透传给具体模块，如 api_key、stream、return_trace 等
)
```

- `type` 的解析顺序：优先使用显式传入的 `type`，否则使用 `kwargs` 中的 `function`，再否则根据 `model` 通过 `get_model_type` 推断；若都没有则默认为 `'llm'`。
- `url`：对 Chat 对应 `base_url`，对 Embedding 对应 `embed_url`，对 MultiModal 对应 `base_url`。

当 `type` 为以下之一时，创建 **OnlineChatModule**（LLM / VLM 等对话模型）：

| type   | 说明     |
|--------|----------|
| `llm`  | 纯文本大模型（默认） |
| `vlm`  | 视觉语言模型       |
| 其他未在下面列出的 | 按 llm 处理 |

当 `type` 属于以下 embedding 相关类型时，创建 **OnlineEmbeddingModule**：

| type               | 说明           |
|--------------------|----------------|
| `embed`            | 文本向量       |
| `cross_modal_embed`| 跨模态向量     |
| `rerank`           | 重排序         |

```python
embed = lazyllm.OnlineModule(type='embed', source='qwen', model='text-embedding-v3')
vec = embed('一段文本')
embed = lazyllm.OnlineModule(type='cross_modal_embed', source='qwen', model='qwen2.5-vl-embedding')
rerank = lazyllm.OnlineModule(type='rerank', source='glm', model='rerank')
```

当 `type` 为以下之一时，创建 **OnlineMultiModalModule**：

| type            | 说明         | 内部 function   |
|-----------------|--------------|-----------------|
| `stt`           | 语音转文字   | `stt`           |
| `tts`           | 文字转语音   | `tts`           |
| `sd`            | 文生图       | `text2image`    |
| `text2image`    | 文生图       | `text2image`    |
| `image_editing` | 图像编辑     | `image_editing` |

```python
# 语音识别 STT
stt = lazyllm.OnlineModule(source='qwen', type='stt', model='qwen-audio-turbo', api_key='xxx')
```

## TrainableModule(本地大模型)

通过TrainableModule接入和部署本地模型，并提供相应的模型服务。

- 在基础的使用方式下，如果没有下载过模型，则会自动去 huggingface 或者 modelscope 上下载模型。如果不希望自动下载模型，则可以为TrainableModule传入参数 trust_remote_code=False 。当客户端连接lazyllm启动的推理服务时，这个做法通常很有效。
- 模型默认会下载到 ~/.lazyllm/ 目录下。如果想要下载到其他目录下，则可以使用环境变量 LAZYLLM_MODEL_CACHE_DIR=/path/to/model 来指定模型的下载路径
- 由于中国境内的上网策略，因此默认的模型源为 modelscope 。如果希望更换模型源，则可以使用环境变量 LAZYLLM_MODEL_SOURCE=huggingface 来将模型源切换到huggingface。
- 如果本地有下载好的模型权重，则可以用绝对路径指定 model = lazyllm.TrainableModule('/path/to/qwen2-1.5b')
- 如果将模型下载到统一的位置，则可以配置环境变量 LAZYLLM_MODEL_PATH=/path/to 来指定模型的根目录

基础使用：

```python
import lazyllm

model = lazyllm.TrainableModule('qwen2-1.5b').start()
response = model('Hello')
print(response)
```

### 模型共享

多个实例可以共享同一个微调模型，节省资源。

#### 基础共享

```python
# 创建基础模型
base_model = lazyllm.TrainableModule('qwen2-1.5b').start()

# 创建共享实例，使用相同的模型但不同的 prompt
chat_model = base_model.share(prompt="你是一个聊天机器人")
code_model = base_model.share(prompt="你是一个编程助手")
```

#### 共享参数配置

```python
# 共享时可以指定不同的配置
shared_model = base_model.share(
    prompt="新的 prompt",                    # 新的 prompt
    format=lazyllm.formatter.JsonFormatter(), # 新的格式化器
    stream={'color': 'blue'}                 # 新的流式配置
)
```

#### 从URL连接

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

#### Embedding模型共享

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

#### Rerank模型共享

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

