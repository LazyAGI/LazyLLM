### 复杂应用的构建

### 基于内部工具构建应用

!!! Note "注意"

    虽然看起来是由模块组合成数据流，然后统一启动，但我们并不是“静态图”，主要表现为：

    - 我们使用了基础的 Python 语法来搭建应用，使用方式非常接近于传统的编程方式。
    - 你仍然可以利用 Python 强大而灵活的特性，在运行的过程中改变已经搭建好的应用的拓扑结构，而后续的执行依据你修改之后的结构。
    - 你可以灵活的向模块的连接处，注入你所希望执行的 hook 函数，这些函数甚至可以是运行时定义的。

下面是基于 LazyLLM 构建的应用示例：

- [聊天机器人](../Cookbook/robot.md)
    - 模块：[TrainableModule][lazyllm.module.TrainableModule]
    - 工具：[WebModule][lazyllm.tools.WebModule]
    - 数据流：无
- [绘画大师](../Cookbook/painting_master.md)
    - 模块：[TrainableModule][lazyllm.module.TrainableModule]
    - 工具：[WebModule][lazyllm.tools.WebModule]
    - 数据流：[Pipeline][lazyllm.flow.Pipeline]
- [多模态机器人](../Cookbook/multimodal_robot.md)
    - 模块：[TrainableModule][lazyllm.module.TrainableModule]
    - 工具：[WebModule][lazyllm.tools.WebModule]
    - 数据流：[Pipeline][lazyllm.flow.Pipeline]、[Switch][lazyllm.flow.Switch]
- [大作家](../Cookbook/great_writer.md)
    - 模块：[TrainableModule][lazyllm.module.TrainableModule]
    - 工具：[WebModule][lazyllm.tools.WebModule]
    - 数据流：[Pipeline][lazyllm.flow.Pipeline]、[Warp][lazyllm.flow.Warp]
- [知识库问答助手](../Cookbook/rag.md)
    - 模块：[TrainableModule][lazyllm.module.TrainableModule]
    - 工具：[WebModule][lazyllm.tools.WebModule]、[Document][lazyllm.tools.Document]、[Retriever][lazyllm.tools.Retriever]、[Reranker][lazyllm.tools.Reranker]
    - 数据流：[Pipeline][lazyllm.flow.Pipeline]、[Parallel][lazyllm.flow.Parallel]

### 结合开源工具构建应用

### 典型场景应用
