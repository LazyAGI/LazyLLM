# 不同模型的微调方法

这部分按照不同模型类型（分为大模型，Embedding 和 Rerank 模型）介绍各类微调方法的使用
每个方法具体的参数配置参考: 
[微调方法介绍](./finetune_framework.md)

## 大模型

以llamafactory微调框架举例

```python
import lazyllm
from lazyllm import finetune, deploy, launchers

model = lazyllm.TrainableModule(model_path)\
    .mode('finetune')\
    .trainset(train_data_path)\
    .finetune_method((finetune.llamafactory, {
        'learning_rate': 1e-4,
        'cutoff_len': 5120,
        'max_samples': 20000,
        'val_size': 0.01,
        'per_device_train_batch_size': 2,
        'num_train_epochs': 2.0,
        'launcher': launchers.sco(ngpus=8)
    }))
model.update()
```

上面代码中，使用LazyLLM的TrainableModule来实现微调
- 模型配置：

    model_path指定了我们要微调的模型，这里我们用Internlm2-Chat-7B，直接指定其所在路径即可；

- 微调配置：

    * mode设置了启动微调模式finetune；
    * trainset 设置了训练用的数据集路径，这里用到的就是我们前面处理好的训练集；
    * finetune_method 设置了用哪个微调框架及其参数，这里传入了一个元组（只能设置两个元素）：
    * 第一个元素指定了使用的微调框架是Llama-Factory：finetune.llamafactory
    * 第二个元素是一个字典，包含了对该微调框架的参数配置；

- 评测配置：
    
    这里通过.evalset来配置了我们之前处理好的评测集；

- 启动任务：

    通过 update 触发任务：模型先进行微调，微调完成后模型会部署起来，部署好后会自动使用评测集全部都过一遍推理以获得结果；

其余微调框架则通过修改finetune_method中的参数进行选择，并在{}中配置需要的参数。不同微调框架支持的参数参考[微调方法介绍](./finetune_framework.md)

## Embedding 和 Rerank 模型

微调Embedding和Rerank模型需要使用flagembedding作为推理框架

```python
embed = lazyllm.TrainableModule(embed_path)\
    .mode('finetune').trainset(train_data_path)\
    .finetune_method((
        lazyllm.finetune.flagembedding,
        {
            'launcher': lazyllm.launchers.remote(nnode=1, nproc=1, ngpus=4),
            'per_device_train_batch_size': 16,
            'num_train_epochs': 2,
        }
    ))
    
docs = Document(kb_path, embed=embed, manager=False)
docs.create_node_group(name='split_sent', transform=lambda s: s.split('\n'))
retriever = lazyllm.Retriever(doc=docs, group_name="split_sent", similarity="cosine", topk=1)
retriever.update()

# 使用微调后的模型
embed = lazyllm.TrainableModule('bge-large-zh-v1.5', 'path/to/sft/bge')
```

这里代码和前面使用LazyLLM的TrainableModule来对LLM进行微调的配置是一致的：

- embed_path: 用于指定微调的模型；
- train_data_path：用于训练的数据集路径；
- lazyllm.finetune.flagembedding: 指定微调的框架；

关键参数：

- ngpus=4: 使用4张GPU进行并行训练
- per_device_batch_size=16: 每GPU批处理大小
- num_train_epochs=2: 训练2个epoch
