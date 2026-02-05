# 各类模型的部署方法

这部分主要介绍如何使用不同的推理方法进行模型部署
每个不同方法的具体参数参考：
[部署操作示例](./deploy_example.md) 

## 模型部署

各类模型均可以统一使用TrainableModule.deploy_method(v, **kw)进行部署:

设置 TrainableModule 的部署方法及其参数

Parameters:

- v (LazyLLMDeployBase) – 部署方法，可选值包括 deploy.auto / deploy.lightllm / deploy.vllm 等
- kw (**dict) – 部署方法所需的参数，对应 v 的参数

此处以dummy部署方法举例，其他部署方法只需要修改deploy_method中的名称，参考[部署操作示例](./deploy_example.md) 中不同推理框架参数配置kw。

```python
import lazyllm
m = lazyllm.module.TrainableModule().deploy_method(deploy.dummy).mode('finetune')
m.evalset([1, 2, 3])
m.update()
print(m.eval_result)
["reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1}"]
```
