# LazyLLM 的注册机制

## 1. 设计背景：为什么 LazyLLM 需要注册机制

LazyLLM 的定位决定了它必须是一个高度可扩展的框架。
随着功能演进，框架内部逐步形成了多类可扩展组件体系，例如：

- 流程编排相关的 Flow 与 Node
- 各类可组合的 Tool
- 不同来源与能力的模型与服务（如 Online Modules）
- 启动与运行相关的 Launcher 组件

这些组件在设计上具有几个显著共性：

第一，**组件数量多且持续增长**。
LazyLLM 并非一个封闭系统，框架本身以及第三方开发者都会不断引入新的实现类。如果每新增一个组件都需要修改集中式注册表，维护成本会迅速上升。

第二，**需要统一、稳定的访问入口**。
从使用者视角看，组件应当通过 lazyllm.xxx.yyy 的形式被访问，而不依赖具体的模块路径或文件结构。这种访问方式必须在实现调整或重构后仍然保持稳定。

第三，**需要降低扩展门槛**。
理想状态下，开发者只需“新增一个类”，即可将新能力接入 LazyLLM 体系，而不必理解或修改复杂的注册流程。

在上述背景下，LazyLLM 引入并系统化了**继承即注册机制**，用于统一解决“组件发现、能力分组与访问入口生成”等问题。

## 2. 设计思路：LazyLLM 注册机制的设计原理

LazyLLM 的注册机制用于统一管理框架内部各类可扩展组件，并为其生成稳定、可预测的访问入口。该机制贯穿 LazyLLM 的多个子系统，是框架整体架构中的一项基础设计。

注册逻辑在组件定义阶段自动执行，通过解析类的继承关系与命名约定，将组件组织到分层的能力结构中，并最终以统一的调用接口对外暴露。

### 2.1 核心设计原则

LazyLLM 的注册机制可以用一句话概括其设计哲学：

> 让“写类 / 写函数”这件事本身，完成注册。

围绕这一目标，LazyLLM 形成了以下几条明确的设计原则。

- 继承关系表达语义，而不是仅用于代码复用

    在 LazyLLM 中，继承某个 Base 类，不只是“复用父类逻辑”，而是明确声明：
    这个类属于哪一类能力体系。

- 避免显式、集中式的注册调用

    框架刻意避免 registry.register(...) 这类显式 API。
    注册应当发生在类定义阶段，而不是由开发者手动触发。

- 支持层级化的能力组织结构

    注册机制天然支持 group.subgroup.xxx 这样的多级能力划分，而不是把所有实现平铺在同一层级。

- 对最终用户暴露统一、可调用的入口

    注册的最终结果不是“类列表”或“映射表”，而是
    ```python
    lazyllm.xxx.yyy(...)
    ```
    这种接近函数调用的使用体验。

在这些原则约束下，LazyLLM 的注册机制并不是某一个工具类，而是一套结构化系统。

### 2.2 三层架构概览

从实现与职责划分的角度看，LazyLLM 的注册系统由三层组成：

1. LazyDict：注册结果的表现层

    负责解决“注册之后，用户如何访问和调用”的问题。

2. LazyLLMRegisterMetaClass：注册逻辑的核心控制层

    负责决定“类在定义时是否被注册、注册到哪里、以什么形式暴露”。

3. Register 装饰器：函数到类的适配层

    负责将“函数形式的能力”统一纳入基于类的注册体系。

这三层各自职责明确，但只有协同工作，才能形成 LazyLLM 中看到的最终效果。

### 2.3 LazyDict：注册结果如何被“使用”

LazyDict 是注册机制中**唯一直接面向使用者的部分**。

从使用体验上看，LazyDict 并不是一个普通的 dict，而是一个经过特殊设计的容器，其目标是让注册结果“像模块、像对象、又像函数”。

LazyDict 重点解决的是以下问题：

- 访问方式统一

    用户不需要知道实现类定义在哪个文件，只需通过
    ```
    lazyllm.<group>.<key>
    ```
    即可访问对应能力。

- 命名宽松、降低记忆成本

    在合理范围内，LazyDict 对大小写、命名形式做了自动匹配，减少使用时的摩擦。

- 函数式调用体验

    在存在默认实现或单一实现的情况下，允许直接
    ```
    lazyllm.<group>(...)
    ```
    而无需显式指定具体 key。

需要强调的是：
> LazyDict 并不决定“注册什么”，它只负责“如何暴露注册结果”。

在系统内部，LazyDict 本质上是某个 Base 类对应的“子类注册表”，例如：

```python
LazyDict(
  impl_a -> ClassA,
  impl_b -> ClassB,
)
```

随后被挂载到 `lazyllm.<group>`，形成对外可见的统一入口。

### 2.4 LazyLLMRegisterMetaClass：类注册的核心机制

LazyLLM 的类注册行为由统一的元类机制控制。
所有参与注册的类，都会在定义阶段经过该元类的处理，从而决定其是否被纳入注册体系、归属到哪个能力分组，以及是否对外暴露访问入口。

这一设计使得注册行为与类的生命周期绑定，而不是依赖显式的注册调用。
开发者只需通过继承关系和命名约定表达语义，具体的注册过程由框架自动完成。

在整体架构中，LazyLLMRegisterMetaClass 的职责是：

- 解析类结构所表达的能力归属信息
- 将可注册的实现类组织到对应的能力分组中
- 为后续的统一访问入口生成必要的注册信息

关于注册判定规则、分组逻辑以及边界行为，将在下一章中展开说明。

### 2.5 Register 装饰器：函数接入注册体系的统一入口

LazyLLM 的注册体系以“类”为核心构建，但在实际使用中，部分能力更适合以函数形式表达。
为统一这两种开发方式，LazyLLM 提供了 Register 装饰器，用于将函数形式的能力纳入同一套注册机制。

Register 装饰器的作用在于：

- 为函数构造一个等价的、可注册的类表示
- 使函数能力能够复用基于元类的注册流程
- 保证函数与类在注册结果和访问方式上的一致性

通过这一适配层，LazyLLM 实现了“类与函数并行接入、统一管理”的注册模型。
函数注册的具体行为与规则，将在后续章节中结合示例进一步说明。

## 3. LazyLLM 注册机制的详细剖析

本章对 LazyLLM 注册机制中的关键规则与行为进行说明，重点描述能力分组的定义方式、注册判定条件、注册 key 的生成规则，以及注册结果的访问与取值约定。这些规则适用于 LazyLLM 中的各类组件，是框架层面对注册行为的统一约束。

后续小节将分别对上述规则进行展开说明。

### 3.1 能力分组的定义规则

LazyLLM 通过 `Base 类定义能力分组（group）`。
能力分组是注册机制中的基础结构，用于组织同一类能力下的所有实现。

分组 Base 类需满足以下命名约定：

```
LazyLLM + <GroupName> + Base
```

样例：

```python
# 定义 group：embed
# 命名形态：LazyLLM + Embed + Base
class LazyLLMEmbedBase(metaclass=LazyLLMRegisterMetaClass):
    pass

# 导入/加载该模块后，框架会产生对应入口（示意）：
# lazyllm.embed  -> LazyDict(...)
```

当注册系统检测到类名符合上述形式时，将其视为一个 分组定义类，并执行以下操作：

- **创建一个用于该分组的注册容器（LazyDict）**  
  例如，定义 `LazyLLMEmbedBase` 时，会创建一个用于存放 Embed 实现类的 `LazyDict` 实例。

- **将该注册容器绑定到 `lazyllm.<group>` 命名空间下**  
  例如，上述 `LazyDict` 会直接绑定为 `lazyllm.embed`，作为该分组的统一访问入口。

- **记录该分组在注册体系中的路径信息（支持层级结构）**  
  例如在多级结构下，可形成 `lazyllm.tool.search` 这类分组路径。

- **将该分组对应的注册容器写入全局注册表中**  
  后续所有继承自 `LazyLLMEmbedBase` 的实现类，都会注册到该 `LazyDict` 中。

通过这种方式，LazyLLM 将“能力分组”的定义与类结构绑定，而不是通过显式配置声明。
分组的创建顺序与模块加载顺序一致，分组一经创建即可被后续实现类复用。

### 3.2 注册判定与 disable 机制

并非所有参与继承结构的类都应当被注册为对外可访问的实现。
LazyLLM 在类定义阶段会对每个类进行注册判定，用于区分：

- **结构性类**：用于组织继承关系或复用公共逻辑
- **实现类**：需要注册到分组中，并对外暴露访问入口

对于需要显式跳过注册的类，LazyLLM 提供了 `disable 机制`。

样例：

```python
# 仅用于复用 HTTP 请求逻辑的中间基类
class _EmbedHTTPMixin(metaclass=LazyLLMRegisterMetaClass):
    __lazyllm_group_disable__ = True

    def _request(self, payload: dict):
        ...
```

当注册系统检测到类中包含 disable 标记时，将执行以下行为：

- **跳过该类的注册流程**  
  例如，上述 _EmbedHTTPMixin 不会出现在 lazyllm.embed 中。

- **不向分组注册容器中写入该类**  
  即不会作为 lazyllm.<group>.<key> 的可访问对象存在。

- **不影响该类被继承或复用**  
  例如，具体的 Embed 实现类仍可继承该 mixin 作为内部实现细节.

disable 机制用于明确区分“内部结构层”和“对外 API 层”，避免中间抽象类污染分组命名空间。

### 3.3 注册 key 的生成与控制

当一个类被判定为可注册实现类后，注册系统会为其生成一个 **group 内的注册 key**，用于构成最终的访问路径：

```
lazyllm.<group>.<key>
```

注册 key 的生成遵循以下规则：

- **默认情况下，由类名推导并规范化**  
  例如，从 OpenAIEmbed 推导出 openai 作为注册 key。

- **同一 group 内，注册 key 需保持唯一**  
  不同实现类不能映射到相同的 key。

- **允许通过类属性显式指定注册 key**  
  用于对外改名或兼容已有接口。

样例：

```python
class OpenAIEmbed(LazyLLMEmbedBase):
    __lazyllm_registry_key__ = "openai"

    def __init__(self, api_key: str, model: str):
        ...
```

在该示例中：

- **openai 作为对外访问 key**  
  对应访问路径为 `lazyllm.embed.openai`。

- **类名与访问 key 解耦**  
  类仍命名为 OpenAIEmbed，但对外 API 使用 openai。

- **不影响继承关系与内部逻辑**  
  注册 key 仅用于访问层，不参与能力判定与继承判断。

通过将“类名”和“访问 key”分离，LazyLLM 支持在不破坏用户调用代码的前提下，调整实现结构或命名方式。

### 3.4 注册结果的访问与取值规则

注册完成后，各分组及其实现类会通过统一命名空间对外暴露。
分组对应的注册容器即 `lazyllm.<group>`，其类型为 `LazyDict。`

访问与取值行为遵循以下约定：

- 分组名称与注册 key 不区分大小写  
  例如，`lazyllm.embed.openai` 与 `lazyllm.Embed.OpenAI` 在语义上等价。

- 实现类的类名可作为访问别名使用  
  `例如，OpenAIEmbed` 可通过类名形式访问同一实现。

- 在存在默认实现或唯一实现时，可直接通过分组调用  
  即省略具体 key，直接调用 `lazyllm.<group>(...)`。

样例：

```python
import lazyllm

# 使用注册 key 访问
embed1 = lazyllm.embed.openai(...)
# 使用类名别名访问
embed2 = lazyllm.embed.OpenAIEmbed(...)
# 使用分组直接调用（存在默认或唯一实现时）
embed3 = lazyllm.embed(...)
```

上述三种访问方式可能指向同一实现对象，具体取值行为由 `LazyDict` 统一处理。
用户无需了解注册系统内部的存储结构，只需遵循统一的访问约定即可。

### 3.5 层级分组与路径解析

LazyLLM 的能力分组支持层级结构。
当分组 Base 类之间存在继承关系时，注册系统会将其映射为多级分组路径，并在访问时按层级逐级解析。

在这种结构下：

- 每一级分组都对应一个注册容器（`LazyDict`）
- 分组路径通过点号形式对外暴露
- 访问过程由外层分组逐级解析至内层分组，最终定位到具体实现

例如，以下访问路径：

```
lazyllm.online.chat.glm(...)
```

对应的结构关系为：

- online：一级分组
- chat：online 下的子分组
- glm：chat 分组下的实现 key

层级分组用于表达能力的语义结构，使注册结果在保持统一访问方式的同时，具备清晰的能力组织层次。

## 4. 注册机制在 Online 模块中的应用

![auto_registry.png](../../../assets/auto_registry.png)

Online 模块是 LazyLLM 中注册机制应用完整、层次较为清晰的一个子系统。
其设计目标并不是提供单一能力的模型封装，而是构建一套**可按能力分组**、**可按供应商扩展**、并能**被统一调度**的在线服务体系。

从整体结构上看，Online 模块由 **三个层次** 组成，这些层次在上图中自上而下展开，并通过继承关系与注册机制连接在一起。

### 4.1 Online 模块的整体结构

#### 4.1.1 注册入口层：LazyLLMOnlineBase

图的最上层是 `LazyLLMOnlineBase`。
该类是所有 Online 模块的**统一入口 Base**，并通过 `LazyLLMRegisterMetaClass` 接入 LazyLLM 的注册系统。

这一层的职责是：

- 将 Online 模块整体纳入 LazyLLM 的注册体系
- 定义 `lazyllm.online` 这一顶层分组
- 在模块加载阶段完成 Online 相关配置项的注册与整合

可以理解为：
只要一个类最终继承自 LazyLLMOnlineBase，它就具备被 LazyLLM 识别为 Online 模块一部分的前提条件。

#### 4.1.2 能力层：按能力类型划分的 Online Base

在 `LazyLLMOnlineBase` 之下，Online 模块首先按能力类型进行划分，对应图中的第二、第三层 Base 类，例如：

- `LazyLLMOnlineChatModuleBase`
- `OnlineEmbeddingModuleBase`
- `LazyLLMOnlineRerankModuleBase`
- `LazyLLMOnlineSTTModuleBase`

这些 Base 类的作用是：

- 定义具体的能力分组（如 `online.chat`、`online.embed`、`online.stt` 等）
- 约束该能力下实现类应当满足的接口与行为
- 为后续的供应商实现提供稳定的继承锚点

在注册机制层面，这一层 **直接对应能力分组（group）** 的形成。

### 4.1.3 供应商实现层：具体 Online 服务实现

在能力 Base 类之下，是具体的供应商实现类，例如图中的：

- `GLMChat`
- `GLMEmbed`
- `GLMRerank`
- `GLMSTT`
- `GLMTextToImage`

这一层的特点是：

- 每个类对应一个真实可用的在线服务实现
- 类名同时承担“实现标识”和“注册 key 来源”的角色
- 通过继承能力 Base，自动注册到对应的能力分组中

例如：

- `GLMChat` → `lazyllm.online.chat.glm`
- `GLMEmbed` → `lazyllm.online.embed.glm`
- `GLMSTT` → `lazyllm.online.stt.glm`

这些映射关系完全由注册机制自动完成，与实现类所在的文件路径无关。

### 4.2 Online 模块的使用与访问方式

在注册机制完成 Online 模块的能力分组与实现类挂载之后，所有 Online 能力都会通过统一的命名空间对外暴露。
用户与上层调度逻辑并不直接依赖具体实现类，而是通过 `lazyllm.online` 提供的访问入口完成实例化与调用。

#### 4.2.1 基于能力与供应商的直接访问

最直接的使用方式，是通过 **能力分组 + 供应商 key** 访问对应的 Online 实现。

例如：

```python
import lazyllm

chat = lazyllm.online.chat.glm(...)
embed = lazyllm.online.embed.glm(...)
stt = lazyllm.online.stt.glm(...)
```

在上述示例中：

- `online` 表示 Online 模块的顶层分组
- `chat / embed / stt` 表示具体能力类型
- `glm` 表示某一供应商在该能力下的实现

这些访问路径均由注册机制自动生成，与实现类所在的模块路径或文件结构无关。

#### 4.2.2 基于类名别名的访问方式

除了使用注册 key，Online 模块还支持通过**实现类名作为别名**进行访问。
这一能力由注册系统在类注册阶段自动生成，用于增强可读性与调试友好性。

例如：

```python
chat = lazyllm.online.chat.GLMChat(...)
embed = lazyllm.online.embed.GLMEmbed(...)
```

类名访问与 key 访问在语义上等价，指向同一实现类。
在实际使用中，通常推荐使用 **key 形式** 作为稳定接口，而类名形式更多用于开发与调试场景。

#### 4.2.3 默认实现与能力级直接调用

在某些能力分组下，如果只存在一个可用实现，或明确指定了默认实现，Online 模块允许**省略供应商 key**，直接通过能力分组进行调用。

示例：

```python
chat = lazyllm.online.chat(...)
```

该调用方式最终会解析到该能力分组下的默认实现。
是否允许这种写法，以及默认实现的选择规则，由分组注册容器（`LazyDict`）统一管理。

### 4.3 Online 模块的配置与扩展规则

Online 模块在注册阶段会自动生成并管理一组通用配置项，用于统一不同供应商与能力类型的初始化与调用行为。本节说明这些配置项的组织方式，以及在扩展 Online 能力或供应商时应遵循的定制规则。

#### 4.3.1 通用配置项的组织方式

Online 模块的配置项按作用范围分为两类：**供应商级配置**与**能力级配置**。
这两类配置均由注册机制在类加载阶段自动声明，并统一接入 LazyLLM 的配置系统。

- **供应商级配置**  
    用于描述与具体能力无关的通用信息，最常见的是认证与访问相关配置，例如：

    ```
    {supplier}_api_key
    {supplier}_base_url
    ```

    该类配置在某个供应商的任意 Online 实现类首次被注册时生成，并在该供应商的所有能力实现中共享。

- **能力级配置**  
    用于描述某一能力下的特定参数，通常与模型名称或能力行为相关，例如：

    ```
    {supplier}_model_name
    {supplier}_{capability}_model_name
    ```

    能力级配置仅在对应能力存在时生成，不会影响同一供应商的其他能力实现。

通过这种分层方式，Online 模块避免了配置项在不同能力之间的相互干扰，同时保持了配置命名的一致性。

#### 4.3.2 实现类中对配置项的使用约定

Online 实现类通常不需要显式声明配置项，而是在初始化阶段按约定读取所需配置值。
推荐的做法是：

- 将配置读取集中在 __init__ 中完成
- 明确区分供应商级配置与能力级配置的使用场景
- 不在实现类中引入与注册机制耦合的配置声明逻辑

示例（示意）：

```python
class MyProviderChat(OnlineChatModuleBase):
    def __init__(self, api_key: str = None, model: str = None, **kw):
        self.api_key = api_key or lazyllm.config.get("myprovider_api_key")
        self.model = model or lazyllm.config.get("myprovider_chat_model_name")
```

通过遵循这一约定，实现类可以在不关心配置来源的情况下，获得一致的配置行为。

#### 4.3.3 Online 能力的扩展方式

Online 模块的扩展以**新增能力实现类**为基本形式。
无论是为某一能力新增供应商，还是为同一供应商扩展多种能力，其核心原则都是：通过继承结构表达能力归属，通过命名与注册机制生成访问入口。

##### 单一能力的实现扩展

在最常见的场景下，供应商只需实现某一种 Online 能力。
此时，只需继承对应能力的 Online Base 类，并在实现类中完成该能力的具体逻辑即可。

示例：

```python
class MyProviderChat(OnlineChatModuleBase):
    def __init__(self, api_key: str, base_url: str = "..."):
        ...
```

该实现类在加载后将自动注册到对应的能力分组下，并生成稳定的访问路径，例如：

```
lazyllm.online.chat.myprovider
```

整个过程无需显式注册或额外配置声明。

---

##### 多能力供应商的组织方式

当同一供应商需要同时支持多种 Online 能力时，推荐通过**供应商私有基类**复用公共逻辑，而不是在能力实现类中重复代码。

推荐结构如下：

```python
class _MyProviderBase:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url

class MyProviderChat(OnlineChatModuleBase, _MyProviderBase):
    ...

class MyProviderSTT(LazyLLMOnlineSTTModuleBase, _MyProviderBase):
    ...
```

在该组织方式下：

- 各能力实现类分别注册到对应的能力分组
- 公共逻辑集中在供应商私有基类中
- 访问路径与配置规则在不同能力间保持一致

---

##### 继承与配置项的自动初始化

在 Online 模块中，继承不仅用于触发实现类的注册，同时也会自动初始化相关配置项。
当实现类被注册时，框架会根据供应商与能力类型，自动生成并接入对应的配置键，例如：

- `{supplier}_api_key`
- `{supplier}_model_name`
- `{supplier}_{capability}_model_name`

实现类通常只需在初始化阶段按约定读取这些配置项，无需显式声明或注册配置。
