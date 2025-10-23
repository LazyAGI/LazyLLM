# 语音小助手

本示例展示了如何使用 LazyLLM 快速构建一个具备语音输入与语音输出能力的智能语音助手（Voice Assistant）。

!!! abstract "通过本节，您将学习如何构建一个语音交互式智能助手，包括以下要点"

    - 如何使用 [Pipeline][lazyllm.flow.Pipeline] 实现多模块的顺序编排与数据流式传递。
    - 如何使用 [TrainableModule][lazyllm.module.TrainableModule] 调用本地语音模型，实现语音与文字的相互转换。
    - 如何使用 [OnlineChatModule][lazyllm.module.OnlineChatModule] 调用在线大模型，生成自然语言回答。
    - 如何使用 [WebModule][lazyllm.tools.webpages.WebModule] 一键部署语音助手，带来“语音问答”式交互体验。
    - 如何设置参数 `audio=True` 进行实时语音输入。

## 环境准备

### 安装依赖

在使用前，请先执行以下命令安装所需库：

```bash
pip install lazyllm
```

### 导入依赖包

```python
from lazyllm import TrainableModule, OnlineChatModule, WebModule, pipeline
```

> ❗ 注意：在 LazyLLM 中，`Pipeline`（大写）与 `pipeline`（小写）是同一个类对象，系统会自动识别并兼容两种写法。

### 环境变量

在流程中会使用到在线大模型，您需要设置 API 密钥（以 sensenova 为例）：

```bash
export LAZYLLM_SENSENOVA_API_KEY=your_api_key
export LAZYLLM_SENSENOVA_SECRET_KEY=your_secret_key
```

**SenseNova API 密钥申请：**

1. 访问 [SenseNova 平台](https://console.sensecore.cn/)
2. 注册并登录您的账户
3. 在主页面 → 点击右上角用户图标 → 创建新的『AccessKey 访问密钥』
4. 复制生成的 API 密钥并设置到环境变量中

> ❗ 注意：其他平台的 API_KEY 申请方式参考[官方文档](docs.lazyllm.ai/)。

### 2. 模型说明

如果需要使用语音识别和语音合成功能，请确保本地或云端环境中已正确配置相关语音模型。本示例中使用的是本地语音模型：

- `sensevoicesmall`：负责语音识别（Speech-to-Text）；
- `ChatTTS-new`：负责语音合成（Text-to-Speech）。

> 💡 您也可以根据需要更换模型名称或提供自定义模型路径。

## 核心代码

以下为完整示例代码，可直接运行：

```python
# 构建语音助手流水线
with pipeline() as ppl:
    # 语音转文字
    ppl.chat2text = TrainableModule('sensevoicesmall')
    # 大模型对话
    ppl.llm = OnlineChatModule()
    # 文字转语音
    ppl.text2chat = TrainableModule('ChatTTS-new')

# 启动网页服务（支持语音输入输出）
WebModule(ppl, port=10236, title='Voice Assistant', audio=True).start().wait()
```

**参数说明**

- `port`：指定网页访问端口（可通过浏览器访问 http://127.0.0.1:10236）。
- `title`：网页顶部标题，方便展示项目主题。
- `audio`：开启语音输入后，Web 界面会自动显示麦克风按钮，实现语音问答交互。

通过 `.start().wait()` 启动并保持服务运行后，终端会显示本地访问地址（如 `http://127.0.0.1:10236`）。打开浏览器即可使用语音对话助手，实现“语音问→语音答”的完整交互。

## 运行效果

下面为示例图：

![alt text](../assets/voice_assistant_demo.png)

- 打开页面后，点击麦克风图标录制语音；
- 程序会自动识别语音 → 调用大模型生成回答 → 语音播报回复；
- 整个过程在浏览器端实时交互，无需额外配置。
