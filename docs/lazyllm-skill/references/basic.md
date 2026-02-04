# LazyLLM的基础能力

LazyLLM的基础能力主要包含以下内容:

| 组件名称 | 组件功能 | 参考文档 |
|---------|---------|---------|
| Module | 模块抽象 | [ModuleBase使用示例](../assets/basic/modulebase.md) [内置Module的使用](../assets/basic/modules.md) |
| Model | 模型调用 | [Model使用示例](../assets/basic/model.md) |
| Prompter | 提示词设计 | [Prompter使用示例](../assets/basic/prompter.md) |
| Thirdparty | 依赖懒加载 | [Thirdparty使用示例](../assets/basic/thirdparty.md) |
| AutoRegistry | 注册机制 | [AutoRegistry使用示例](../assets/basic/registry.md) |
| Config | 配置中心 | [Config使用示例](../assets/basic/config.md) |

## Module模块抽象

在LazyLLM中ModuleBase 是 LazyLLM 的核心基类，定义了所有模块的统一接口和基础能力。它抽象了模块的训练、部署、推理和评测逻辑，并提供了子模块管理、钩子注册、参数传递和递归更新等机制。用户自定义的模块需要继承 ModuleBase，并实现 forward 方法来定义具体的推理逻辑。

ModuleBase的功能特性:

- 统一管理子模块 (submodules)，自动追踪被持有的 ModuleBase 实例。
- 支持 Option 类型的超参数设置，方便网格搜索与自动调参。
- 提供钩子 (hook) 机制，可在调用前后执行自定义逻辑。
- 封装训练 (train)、服务部署 (server)、评测 (eval) 的更新流程。
- 支持 evalset 的加载与自动并行推理评测。

ModuleBase的具体使用方法参考: [ModuleBase使用示例](../assets/basic/modulebase.md)

内置Module功能:

- ActionModule: 用于将函数、模块、flow、Module等可调用的对象包装一个Module。被包装的Module（包括flow中的Module）都会变成该Module的submodule
- TrainableModule: 可训练模块，所有模型（包括LLM、Embedding等）都通过TrainableModule提供服务
- OnlineModule: 在线模型都通过OnlineModule提供服务
- UrlModule: 可以将ServerModule部署得到的Url包装成一个Module，调用 __call__ 时会访问该服务
- ServerModule: ServerModule 类，继承自 UrlModule，封装了将任意可调用对象部署为 API 服务的能力。通过 FastAPI 实现，可以启动一个主服务和多个卫星服务，并支持流式调用、预处理和后处理逻辑。既可以传入本地可调用对象启动服务，也可以通过 URL 直接连接远程服务。
- TrialModule: 参数网格搜索模块，会遍历其所有的submodule，收集所有的可被搜索的参数，遍历这些参数进行微调、部署和评测
- WebModule: WebModule是LazyLLM为开发者提供的基于Web的交互界面。
- DocWebModule: 文档Web界面模块，继承自ModuleBase，提供基于Web的文档管理交互界面。 

内置Module的具体使用参考: [内置Module的使用](../assets/basic/modules.md)

## Model模型使用

LazyLLM内主要通过以下三种方式获取模型服务: AutoModel, OnlineModule, TrainableModule
详细的使用示例参考: [Model使用示例](../assets/basic/model.md)

### AutoModel使用
LazyLLM的模型使用通过AutoModel进行整合，用于快速创建在线推理模块 OnlineModule 或本地 TrainableModule 的工厂类。会优先采用用户传入的参数，若开启 config 则会根据 auto_model_config_map 中的配置进行覆盖，然后自动判断应当构建在线模块还是本地模块

### 在线大模型

用户可以通过配置API_KEY获取对应供应商的模型服务，目前LazyLLM的内置供应商如下:

- Doubao(豆包)
- PPIO(派欧云)
- OpenAI(OpenAI)
- Qwen(千问)
- GLM(智谱)
- Kimi(Kimi)
- SenseNova(商汤)
- SiliconFlow(硅基流动)
- AiPing(清程极智)

OnlineModule: 根据 `type`、`model` 或 `function` 参数，会自动路由到以下三种模块之一：

- **OnlineChatModule**：大模型对话（LLM / VLM）
- **OnlineEmbeddingModule**：向量嵌入与排序（embed / cross_modal_embed / rerank）
- **OnlineMultiModalModule**：多模态能力（STT、TTS、文生图、图像编辑）进而调用在线大模型和嵌入模型。

### 本地大模型

通过TrainableModule接入和部署本地模型，并提供相应的模型服务。

#### 模型共享

多个实例可以共享同一个微调模型，节省资源。

模型使用部分的具体规则和示例参考: [Model使用示例](../assets/basic/model.md)

## Prompter提示词设计

借助内置Prompter来实现提示词注入

### 内置Prompter:

- EmptyPrompter: 空提示生成器，用于直接返回原始输入。该类不会对输入进行任何处理，适用于无需格式化的调试、测试或占位场景。
- Prompter: 用于生成模型输入的Prompt类，支持模板、历史对话拼接与响应抽取。该类支持从字典、模板名称或文件中加载prompt配置，支持历史对话结构拼接（用于Chat类任务）， 可灵活处理有/无history结构的prompt输入，适配非字典类型输入。
- AlpacaPrompter: Alpaca格式的Prompter，支持工具调用，不支持历史对话。
- ChatPrompter: 用于多轮对话的大模型Prompt构造器，支持工具调用、历史对话与自定义指令模版。支持传入 system/user 拆分的指令结构，自动合并为统一模板。支持额外字段注入和打印提示信息。

**各类prompter的详细参数和使用参考文档** : [Prompter使用示例](../assets/basic/prompter.md)

### 结合大模型

```python
# 使用OnlineChatModule
import lazyllm
instruction = '你是一个由LazyLLM开发的知识问答助手，你的任务是根据提供的上下文信息来回答用户的问题。上下文信息是{context}，用户的问题是{input}, 现在请你做出回答。'
prompter = lazyllm.AlpacaPrompter({"user": instruction})
module = lazyllm.OnlineChatModule('openai').prompt(prompter)
module(dict(context='背景', input='输入'))

# 使用TrainableModule
import lazyllm
instruction = '你是一个由LazyLLM开发的知识问答助手，你的任务是根据提供的上下文信息来回答用户的问题。上下文信息是{context}，用户的问题是{input}, 现在请你做出回答。'
prompter = lazyllm.AlpacaPrompter(instruction)
module = lazyllm.TrainableModule('internlm2-chat-7b').prompt(prompter)
module.start()
module(dict(context='背景', input='输入'))
```

prompter详细使用文档参考: [Prompter使用示例](../assets/basic/prompter.md)

## Thirdparty依赖懒加载

`lazyllm.thirdparty` 提供**懒加载**和**可选依赖**管理：在 LazyLLM 内部统一通过 `lazyllm.thirdparty` 访问第三方库，只有在实际用到时才 `import`，未安装时给出明确的 `pip install` 提示，从而让核心包保持轻量，并按需安装功能所需依赖。

在LazyLLM中的使用方式参考: [Thirdparty使用示例](../assets/basic/thirdparty.md)

## AutoRegistry注册机制

### AutoRegistry 的功能特性

- 提供统一的能力注册与发现机制，支持类与函数的自动注册。
- 基于继承与装饰器完成注册，无需手动维护注册表或映射字典。
- 按能力分组（group）组织注册结果，形成稳定的命名空间结构。
- 支持通过注册 key、类名别名以及默认实现访问已注册能力。
- 支持显式指定注册 key 与禁止注册中间抽象类。
- 所有注册能力统一通过 lazyllm.<group>.<key> 方式访问。

### AutoRegistry 的核心规则

- 任何继承自注册 Base 类的实现类，都会在加载时自动注册。
- 中间抽象类必须设置 __lazyllm_registry_disable__ = True。
- 默认注册 key 由类名自动推导，也可通过 __lazyllm_registry_key__ 显式指定。
- 注册体系中禁止出现手动 registry 字典、列表或集中式映射表。
- 函数形式能力必须通过 Register 装饰器接入注册体系。

### AutoRegistry 内置能力分组

- embed: 向量化与Embedding相关能力
- online: 在线模型与推理能力
- tool: 工具与函数能力
- module: Module 体系相关能力
- flow: Flow 及流程编排相关能力
- dataset: 数据集与数据加载相关能力

### AutoRegistry 访问约定

- 通过注册 key 访问
    lazyllm.<group>.<key>(...)
- 通过类名别名访问
    lazyllm.<group>.<ClassName>(...)
- 当某分组只有一个实现或指定了默认实现时，可简写为
    lazyllm.<group>(...)

### AutoRegistry 扩展约定

- 新增能力时，必须通过继承对应 Base 类或使用 Register 装饰器完成注册。
- 禁止在业务代码中直接实例化具体实现类作为主要使用方式。
- 所有能力的对外暴露入口必须经过 AutoRegistry。

### AutoRegistry 使用参考

[AutoRegistry使用示例](../assets/basic/registry.md)

## Config配置中心

### Config 的功能特性

- 提供全局框架级配置管理，用于控制 LazyLLM 行为及运行参数。
- 支持从配置文件、环境变量和代码显式设置中读取配置项。
- 自动构建全局单例 config 对象，供整个 LazyLLM 生态访问。
- 支持动态临时覆盖配置值（临时上下文切换）。
- 提供基础操作 API：读取、写入、列举以及临时覆盖等。

### Config 的核心规范

- 全局唯一的配置对象名为 lazyllm.config。
- 配置值优先级顺序：代码设置（显式） > 环境变量 > 配置文件。
- 环境变量统一使用前缀 LAZYLLM_（可通过参数调整）。
- 不同配置项可通过注册 API 动态定义到 config 中。

具体的使用示例参考: [Config使用示例](../assets/basic/config.md)
