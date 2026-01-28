# Flow (数据流编排)

Flow 是 LazyLLM 的核心功能之一，用于构建复杂的数据处理流程，支持线性、分支、并行、循环等多种模式。

## 核心概念

Flow 通过以下组件实现灵活的数据流编排：

| 组件名称 | 组件功能 | 参考文档 |
|---------|---------|---------|
| Pipeline | 顺序执行的线性流程 | [pipeline使用示例](../assets/flow/pipeline.md) |
| Parallel | 并行执行的任务流程 | [parallel使用示例](../assets/flow/parallel.md) |
| Diverter | 并行多分支流程 | [diverter使用示例](../assets/flow/diverter.md) |
| Warp | 并行独立处理流程 | [warp使用示例](../assets/flow/warp.md) |
| IFS | if-else流程 | [ifs使用示例](../assets/flow/ifs.md) |
| Switch | 多路选择流程 | [switch使用示例](../assets/flow/switch.md) |
| Loop | 循环执行流程 | [loop使用示例](../assets/flow/loop.md) |
| Graph | DAG图 | [graph使用示例](../assets/flow/graph.md) |
| Bind | 数据绑定 | [bind使用示例](../assets/flow/bind.md) |

## 流程组合

可以通过组合Flow中的组件实现复杂流程，使用示例：[组合使用示例](../assets/flow/combine.md)

## Pipeline (顺序流程)

主要功能:
一个形成处理阶段管道的顺序执行模型。
Pipeline类是一个处理阶段的线性序列，其中一个阶段的输出成为下一个阶段的输入。它支持在最后一个阶段之后添加后续操作。它是 LazyLLMFlowsBase的子类，提供了一个延迟执行模型，并允许以延迟方式包装和注册函数。

具体使用示例：[pipeline使用示例](../assets/flow/pipeline.py)

## Parallel (并行流程)

主要功能:
Parallel的所有组件共享输入，并将结果合并输出。

具体使用示例：[parallel使用示例](../assets/flow/parallel.md)

## Diverter (条件分支)

主要功能:
一个流分流器，将输入通过不同的模块以并行方式路由。
Diverter类是一种专门的并行处理形式，其中多个输入分别通过一系列模块并行处理。然后将输出聚合并作为元组返回。

具体使用示例：[diverter使用示例](../assets/flow/diverter.md)

## Warp （并行独立）

主要功能:
一个流形变器，将单个模块并行应用于多个输入。（只允许一个函数在warp中）
Warp类设计用于将同一个处理模块应用于一组输入。可以有效地将单个模块“形变”到输入上，使每个输入都并行处理。输出被收集并作为元组返回。需要注意的是，这个类不能用于异步任务，如训练和部署。

具体使用示例：[warp使用示例](../assets/flow/warp.md)

## IFS （if-else）

主要功能:
IFS（If-Else Flow Structure）类设计用于根据给定条件的评估有条件地执行两个提供的路径之一（真路径或假路径）。执行选定路径后，可以应用可选的后续操作，并且如果指定，输入可以与输出一起返回。

具体使用示例：[ifs使用示例](../assets/flow/ifs.md)

## Switch (多路选择)

主要功能:
一个根据条件选择并执行流的控制流机制。
Switch类提供了一种根据表达式的值或条件的真实性选择不同流的方法。

具体使用示例：[switch使用示例](../assets/flow/switch.md)

## Loop (循环流程)

主要功能:
初始化一个循环流结构，该结构将一系列函数重复应用于输入，直到满足停止条件或达到指定的迭代次数。
Loop结构允许定义一个简单的控制流，其中一系列步骤在循环中应用，可以使用可选的停止条件来根据步骤的输出提前退出循环。

具体使用示例：[loop使用示例](../assets/flow/loop.md)

## Graph（DAG图）

主要功能:
一个基于有向无环图（DAG）的复杂流控制结构。
Graph类允许您创建复杂的处理图，其中节点表示处理函数，边表示数据流。它支持拓扑排序来确保正确的执行顺序，并可以处理多输入和多输出的复杂依赖关系。
Graph类特别适用于需要复杂数据流和依赖管理的场景，如机器学习管道、数据处理工作流等。

具体使用示例：[graph使用示例](../assets/flow/graph.md)

## Bind （数据绑定）

主要功能:
通过bind，可以自由的在流程中从上游向下游传递参数。

具体使用示例：[bind使用示例](../assets/flow/bind.md)

## 最佳实践

### 1. 合理使用 bind

- 使用 `bind` 传递上游数据到下游
- 使用 `_0`, `_1` 占位符引用输入
- 使用 `p.input` 引用流程初始输入

### 2. 选择合适的流程类型

- **Pipeline**: 顺序执行的步骤
- **Parallel**: 可同时执行的独立任务
- **Diverter**: 根据条件选择不同处理路径
- **Warp**: 单模块多输入场景
- **IFS**： if-else条件判断任务
- **Switch**: 根据值选择不同处理路径
- **Loop**: 需要重复执行的任务
- **Graph**: 需要使用DAG图的任务
- **Bind**: 需要灵活传递流程参数

### 3. 命名规范

- 为流程节点命名，便于引用
- 使用有意义的变量名
- 保持命名一致性

### 4. 错误处理

```python
from lazyllm import pipeline

def safe_process(input):
    try:
        return input + 1
    except Exception as e:
        return f"Error: {e}"

with pipeline() as ppl:
    ppl.step1 = safe_process
    ppl.step2 = lambda x: x * 2
```

## 使用场景

- RAG 流程编排
- 多 Agent 协作
- 数据预处理管道
- 批量任务处理
- 条件分支逻辑
- 复杂业务流程
