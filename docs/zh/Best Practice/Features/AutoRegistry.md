# 继承即注册规范

## 1. 机制概述

### 1.1 背景与功能

在 LazyLLM 中，在线模型与服务模块（Online Modules）数量众多，类型差异明显，且需要支持持续扩展。
为降低模块接入成本、统一管理方式，并避免显式注册带来的维护复杂度，LazyLLM 引入了**基于继承的自动注册机制**。

该机制通过**类继承关系与命名约定**，在模块定义阶段自动完成注册，无需开发者显式调用注册接口。

### 1.2 “注册”所指内容

当一个模块类满足注册条件时，LazyLLM 会自动完成以下工作：

1. 能力分组（Group / Type）注册

    在 lazyllm 命名空间下创建或扩展对应的能力分组节点，例如：
    - `lazyllm.online.chat`
    - `lazyllm.online.stt`
    - `lazyllm.online.tts`

2. 供应商类（Supplier Class）注册

    将具体实现类挂载到对应分组下，并提供统一的访问入口，例如：
    - `lazyllm.online.chat.doubao`
    - `lazyllm.online.stt.sensenova`

3. 配置项声明（Configuration Keys）

    根据模块能力与供应商信息，自动声明所需的配置项（如 API Key、模型名等），比如：
    - `<supplier>_model_name`
    - `<supplier>_stt_model_name`

## 2. 扩展 LazyLLM 的 Online 类

### 2.1 能力类型与 Base 类选择

LazyLLM 通过 Base 类层级区分模块能力类型。扩展模块时，必须继承与能力类型对应的 Base 类。

典型能力类型包括（示例）：

| 能力类型          | 对应 Base 类                         |
|------------------|-------------------------------------|
| Chat             | OnlineChatModuleBase                |
| Embed            | LazyLLMOnlineEmbedModuleBase        |
| Rerank           | LazyLLMOnlineRerankModuleBase       |
| STT              | LazyLLMOnlineSTTModuleBase          |
| TTS              | LazyLLMOnlineTTSModuleBase          |
| Text-to-Image    | LazyLLMOnlineText2ImageModuleBase   |
| Image Editing    | LazyLLMOnlineImageEditingModuleBase |

> 💡 规则：模块继承的 Base 类，决定其被注册到哪个能力分组。

### 2.2 类命名规范（强约束）

模块类名必须遵循以下命名规则：

```html
<SupplierName><TypeSuffix>
```

其中：

- SupplierName：供应商名称（如 Doubao、SenseNova）
- TypeSuffix：能力类型后缀，与所继承 Base 类一致（如 Chat、STT、TTS）

正确示例：

- DoubaoChat
- SenseNovaSTT
- QwenTextToImage

错误示例（将导致注册失败）：

- 类名未以能力后缀结尾，比如 `DoubaoModule`
- 类名与继承的 Base 类型不一致

## 3. 现有 Online 模块体系结构

### 3.1 Online 模块继承层级

如图所示，LazyLLM Online 模块采用分层继承结构：

- 顶层 Base 类：LazyLLMOnlineBase 类，用来定义 Online 模块的统一命名空间（`lazyllm.online`）

- 能力 Base 类
    - 能力大类，如 LazyLLMOnlineChatModuleBase, OnlineEmbeddingModuleBase 以及 OnlineMultiModalBase 类，用来定义能力大分组。其中 `LazyLLMOnlineChatModuleBase` 类定义 `lazyllm.online.chat` 分组，其他两个类则会跳过分组注册，由能力子类注册具体的能力标签。

    - 能力子类，如 `LazyLLMOnlineRerankModuleBase`, `LazyLLMOnlineTTSModuleBase` 以及 `LazyLLMOnlineText2ImageModuleBase`。用来定义具体的能力分组，比如 `lazyllm.online.rerank`, `lazyllm.online.tts` 等。

- 供应商类：具体服务的实现类，遵循 [2.2 章节](#22-类命名规范强约束) 中的类命名要求。

![auto_registry.png](../assets/auto_registry.png)

### 3.2 注册结果与访问方式

注册完成后，模块可通过以下形式访问：

```python
import lazyllm

doubao_chat_cls = lazyllm.online.chat.doubao(**kwargs)
sensenova_stt_cls = lazyllm.online.stt.sensenova(**kwargs)
```

访问方式对使用者保持稳定，不依赖具体实现类的模块路径。

## 4. 扩展与定制规则

### 4.1 通用配置与能力配置

LazyLLM 区分两类配置项：

1. 供应商级配置

    - 如 `{supplier}_api_key`

    - 与具体能力类型无关，会在初次注册时为所有供应商添加对应的 api key 配置

2. 能力级配置

    - 如 `{supplier}_stt_model_name`

    - 仅在对应能力类型存在，为某种能力的供应商类设置 model name 配置

> 注：配置项的声明由 LazyLLM 在注册过程中自动完成，扩展供应商类时通常无需显式声明配置，除非存在供应商特有的额外需求。

### 4.2 扩展供应商类的基本规则

在大多数情况下，扩展一个新的 Online 供应商类仅需以下三步：

- 步骤 1：选择能力类型并继承对应 Base 类

    根据所实现的能力类型，选择并继承对应的 Online Base 类，例如：

    ```python
    class MyProviderChat(OnlineChatModuleBase):
        ...
    ```

    该继承关系决定该类被注册到：

    ```bash
    lazyllm.online.chat
    ```

- 步骤 2：按照命名规范定义类名

    如[2.2 章节](#22-类命名规范强约束)所述，类名必须满足：

    ```html
    <SupplierName><TypeSuffix>
    ```

    例如：
    - `MyProviderChat`
    - `MyProviderSTT`

    LazyLLM 将基于类名自动解析供应商标识，并生成对应的访问入口：

    ```bash
    lazyllm.online.chat.myprovider(...)
    ```

- 步骤 3：实现供应商自身逻辑

    在类中实现初始化与调用逻辑，例如客户端创建、请求封装等。

    ```python
    class MyProviderChat(OnlineChatModuleBase):
        def __init__(self, api_key: str, base_url: str = "..."):
            ...
    ```

    完成以上步骤后，该供应商类将自动参与注册，无需任何额外操作。

### 4.3 扩展多能力的供应商子类

当同一供应商需要支持多种能力（如 Chat、STT、Embedding）时，推荐通过供应商私有 Base 类组织公共逻辑。

例如：

```python
class _MyProviderBase:
    def __init__(self, api_key: str, base_url: str):
        ...
```

各能力实现类分别继承对应的 Online Base 与供应商 Base：

```python
class MyProviderChat(OnlineChatModuleBase, _MyProviderBase):
    ...

class MyProviderSTT(LazyLLMOnlineSTTModuleBase, _MyProviderBase):
    ...
```

在注册过程中，LazyLLM 会根据供应商与能力类型自动生成常用配置项，包括但不限于：

- `{supplier}_api_key`
- `{supplier}_model_name`
- `{supplier}_{capability}_model_name`

扩展供应商类时，通常不需要显式声明这些配置项，只需在初始化时按约定读取即可。
