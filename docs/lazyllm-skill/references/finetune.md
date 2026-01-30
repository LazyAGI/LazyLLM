# 微调 (Finetune)

LazyLLM 提供了完整的模型微调解决方案，支持多种微调方法和模型类型，通过 TrainableModule 实现统一的微调接口。

## 核心概念

LazyLLM 的微调功能基于 TrainableModule，支持：

| 组件名称 | 组件功能 | 参考文档 |
|---------|---------|---------|
| 微调框架 | Apacalora, Collie, LlamaFactory, Flagembedding, Dummy, Auto | [微调方法介绍](../assets/finetune/finetune_framework.md) |
| 微调分类 | 大模型微调和embedding模型微调区分 | [不同模型微调实现示例](../assets/finetune/fintune_example.md) |
| 推理框架 | Lightllm, VLLM, LMDeploy, Dummy, Auto | [部署示例](../assets/finetune/deploy_framework.md) |
| 推理执行 | 大模型部署的实际操作 | [部署操作示例](../assets/finetune/deploy_example.md) |

## 支持的微调方法

- `lazyllm.finetune.auto`: 自动选择微调方法
- `lazyllm.finetune.alpacalora`: 使用 Apacalora 进行大模型微调
- `lazyllm.finetune.llamafactory`: 使用 LLaMA Factory 进行大模型的微调
- `lazyllm.finetune.collie`: 使用 Collie 进行微调（已停止迭代，后续版本将移除）
- `lazyllm.finetune.flagembedding`: 使用FlagEmbedding框架，用于 Embedding 和Reranker 模型的微调
- `lazyllm.finetune.dummy`: 创建一个finetune占位符，主要用于测试或演示

### 各类方法参数详细内容参考

[微调方法介绍](../assets/finetune/finetune_framework.md)

### 微调基础使用示例

1.使用autofinetune进微调

```python
import lazyllm

# 使用自动微调方法
model = lazyllm.TrainableModule('qwen2-1.5b', target_path='/path/to/model') \
    .finetune_method(lazyllm.finetune.auto) \
    .trainset('/path/to/training/data') \
    .mode('finetune')

# 执行微调
model.update()
```

2.指定微调方法

```python
import lazyllm

# 使用特定的微调方法
model = lazyllm.TrainableModule('qwen2-1.5b') \
    .finetune_method(lazyllm.finetune.llamafactory, learning_rate=1e-4, num_train_epochs=3) \
    .trainset('/path/to/training/data') \
    .mode('finetune')

# 执行微调
model.update()
```

## 不同模型类型的微调

### 大语言模型微调

大语言模型使用 LLaMA-Factory 进行微调，支持多种 LoRA 方法。

### Embedding 模型微调

Embedding 模型使用 FlagEmbedding 框架进行微调，主要用于优化向量表示。

### Reranker 模型微调

Reranker 模型用于文档重排序，提升检索质量。

### 详细内容参考

[不同模型微调实现示例](../assets/finetune/fintune_example.md)


## 部署模型

部署已有的模型文件以提供对应的模型服务。LazylLM支持Lightllm, VLLM, LMDeploy, Dummy, Auto多种推理方式，详细参数配置参考[部署示例](../assets/finetune/deploy_framework.md)，在了解各种框架参数配置的基础上，参考[部署操作示例](../assets/finetune/deploy_example.md)实现具体的部署。

## 最佳实践

### 1. 数据准备

- 使用高质量、多样化的训练数据
- 确保数据格式符合微调框架要求
- 适当的数据清洗和预处理

### 2. 超参数调优

- **学习率**：通常在 1e-5 到 5e-4 之间
- **Batch size**：根据显存大小调整
- **训练轮数**：避免过拟合，使用验证集监控

### 3. 模型选择

- **小规模微调**：使用 LoRA 等参数高效方法
- **领域适配**：选择合适的预训练模型
- **资源考虑**：根据硬件条件选择模型大小

### 4. 评估策略

- 使用验证集评估微调效果
- 监控训练和验证损失
- 设置早停策略防止过拟合

### 5. 部署优化

- 选择合适的推理引擎（vLLM、LMDeploy等）
- 配置合适的并发参数
- 使用模型共享节省资源

## 常见问题

### Q: 如何选择微调方法？

使用 `lazyllm.finetune.auto` 让 LazyLLM 自动选择，或根据模型类型手动指定：
-大语言模型 → `lazyllm.finetune.llamafactory`
-Embedding模型 → `lazyllm.finetune.flagembedding`

### Q: 微调后模型如何保存和加载？

使用 `target_path` 参数指定保存目录，加载时使用该目录路径。

### Q: 如何处理显存不足？

- 减小 `per_device_train_batch_size`
- 使用梯度累积
- 选择参数高效的微调方法（如LoRA）

### Q: 微调需要多长时间？

取决于模型大小、数据量和硬件条件。通常几小时到几天不等。

## 使用场景

- **领域知识模型**：微调特定领域的对话模型
- **个性化助手**：基于个人数据微调
- **企业应用**：微调符合企业需求的模型
- **RAG优化**：微调嵌入和重排序模型
- **多模态应用**：微调视觉语言模型

## 相关资源

- [微调教程](https://docs.lazyllm.ai/zh-cn/stable/Tutorial/9/)
- [LLaMA-Factory文档](https://github.com/hiyouga/LLaMA-Factory)
- [FlagEmbedding文档](https://github.com/FlagOpen/FlagEmbedding)
- [Flow数据流编排](./flow.md)