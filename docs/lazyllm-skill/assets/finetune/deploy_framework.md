# deploy部署框架介绍

LazyLLM支持Lightllm, VLLM, LMDeploy, Dummy, Auto多种部署方法。
这部分主要介绍每个部署方法的参数和可额外添加的关键字。

具体的使用参考:
[部署操作示例](./deploy_example.md)

## Lightllm

基于 LightLLM 框架提供的推理能力，用于对大语言模型进行推理。

### 参数

- trust_remote_code (bool, default: True ) – 是否信任远程代码，默认为True
- launcher (Launcher, default: remote(ngpus=1) ) – 任务启动器，默认为单GPU远程启动器
- log_path (str, default: None ) – 日志文件路径，默认为None
- **kw – 其他LightLLM服务器配置参数，仅支持更换以下参数:
    
    * tp (int) – 张量并行参数，默认为 1。
    * max_total_token_num (int) – 最大总token数，默认为 64000。
    * eos_id (int) – 结束符ID，默认为 2。
    * port (int) – 服务的端口号，默认为 None。此情况下LazyLLM会自动生成随机端口号。
    * host (str) – 服务的IP地址，默认为 0.0.0.0。
    * nccl_port (int) – NCCL 端口，默认为 None。此情况下LazyLLM会自动生成随机端口号。
    * tokenizer_mode (str) – tokenizer的加载模式，默认为 auto。
    * running_max_req_size (int) – 推理引擎最大的并行请求数， 默认为 256。
    * data_type (str) – 模型权重的数据类型，默认为 float16。
    * max_req_total_len (int) – 请求的最大总长度，默认为 64000。
    * max_req_input_len (int) – 输入的最大长度，默认为 4096。
    * long_truncation_mode (str) – 长文本的截断模式，默认为 head。

```python
from lazyllm import deploy
infer = deploy.lightllm()
```

## Vllm

基于 VLLM 框架提供的推理能力，用于大语言模型的部署与推理。

### 参数

- trust_remote_code (bool, default: True ) – 是否允许加载来自远程服务器的模型代码，默认为 True。
- launcher (launcher, default: remote(ngpus=1) ) – 模型启动器，默认为 launchers.remote(ngpus=1)。
- log_path (str, default: None ) – 日志保存路径，若为 None 则不保存日志。
- openai_api (bool, default: None ) – 是否使用 OpenAI API 接口启动 VLLM 服务，默认为 False。
- kw – 关键字参数，用于更新默认的部署参数。仅支持更换以下参数:
    * tensor-parallel-size (int) – 张量并行大小，默认为 1。
    * dtype (str) – 模型权重和激活值的数据类型，默认为 auto。可选：half、float16、bfloat16、float、float32。
    * kv-cache-dtype (str) – KV 缓存的数据类型，默认为 auto。可选：fp8、fp8_e5m2、fp8_e4m3。
    * device (str) – VLLM 支持的硬件类型，默认为 auto。可选：cuda、neuron、cpu。
    * block-size (int) – token 块大小，默认为 16。
    * port (int | str) – 服务端口号，默认为 auto，即随机分配。
    * host (str) – 服务绑定的 IP 地址，默认为 0.0.0.0。
    * seed (int) – 随机数种子，默认为 0。
    * tokenizer_mode (str) – Tokenizer 加载模式，默认为 auto。
    * max-num-seqs (int) – 推理引擎支持的最大并行请求数，默认为 256。
    * pipeline-parallel-size (int) – 流水线并行大小，默认为 1。
    * max-num-batched-tokens (int) – 最大批处理 token 数，默认为 64000。

```python
from lazyllm import deploy
infer = deploy.vllm()
```

## LMDeploy

基于 LMDeploy 框架，用于启动并管理大语言模型的推理服务。

### 参数

- launcher (Optional[launcher], default: remote(ngpus=1) ) – 服务启动器，默认使用 launchers.remote(ngpus=1)。
- trust_remote_code (bool, default: True ) – 是否信任远程代码，默认为 True。
- log_path (Optional[str], default: None ) – 日志输出路径，默认为 None。
- **kw – 关键字参数，用于更新默认的部署配置。仅支持更换以下参数:

    * tp (int) – 张量并行参数，默认为 1。
    * server-name (str) – 服务监听的 IP 地址，默认为 0.0.0.0。
    * server-port (Optional[int]) – 服务端口号，默认为 None，此时会自动随机分配 30000–40000 区间的端口。
    * max-batch-size (int) – 最大批处理大小，默认为 128。
    * chat-template (Optional[str]) – 对话模板文件路径。若模型不是视觉语言模型且未指定模板，将使用默认模板。
    * eager-mode (bool) – 是否启用 eager 模式，受环境变量 LMDEPLOY_EAGER_MODE 控制，默认为 False。

```python
# Basic use:
from lazyllm import deploy
infer = deploy.LMDeploy()
# MultiModal:
import lazyllm
from lazyllm import deploy, globals
from lazyllm.components.formatter import encode_query_with_filepaths
chat = lazyllm.TrainableModule('InternVL3_5-1B').deploy_method(deploy.LMDeploy)
chat.update_server()
inputs = encode_query_with_filepaths('What is it?', ['path/to/image'])
res = chat(inputs)
```

## DummyDeploy

一个用于测试的模拟部署类，实现了一个简单的流水线风格部署服务，该类主要用于内部测试和示例用途。它接收符合 message_format 格式的输入，根据是否启用 stream 参数，返回字符串或逐步输出的模拟响应。

### 参数

- launcher: 部署器实例，默认值为 launchers.remote(sync=False)。 
- stream (bool): 是否以流式方式输出结果。
- kw: 其他传递给父类的关键字参数。

## AutoDeploy

根据输入的参数自动选择合适的推理框架和参数，以对大语言模型进行推理。具体而言，基于输入的：base_model 的模型参数、max_token_num、launcher 中GPU的类型以及卡数，该类可以自动选择出合适的推理框架（如: Lightllm 或 Vllm）及所需的参数。

### 参数

- base_model (str) – 用于进行微调的基模型，要求是基模型的路径或模型名。用于提供基模型信息。
- source (config[model_source]) – 指定模型的下载源。可通过设置环境变量 LAZYLLM_MODEL_SOURCE 来配置，目前仅支持 huggingface 或 modelscope 。若不设置，lazyllm不会启动自动模型下载。
- trust_remote_code (bool) – 是否允许加载来自远程服务器的模型代码，默认为 True。
- launcher (launcher, default: remote() ) – 微调的启动器，默认为 launchers.remote(ngpus=1)。
- stream (bool) – 是否为流式响应，默认为 False。
- type (str) – 类型参数，默认为 None，及llm类型，另外还支持embed类型。
- max_token_num (int) – 输入微调模型的token最大长度，默认为1024。
- launcher (launcher, default: remote() ) – 微调的启动器，默认为 launchers.remote(ngpus=1)。
- kw – 关键字参数，用于更新默认的训练参数。注意这里能够指定的关键字参数取决于 LazyLLM 推测出的框架，因此建议谨慎设置。

```python
from lazyllm import deploy
deploy.auto('internlm2-chat-7b')
```

## 注意

这部分主要介绍各推理框架的参数，具体的使用方式参考: 
[部署操作示例](./deploy_example.md)
