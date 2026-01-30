# 微调框架介绍

LazyLLM支持Apacalora, Collie, LlamaFactory, Flagembedding, Dummy, Auto多种微调方法。
这部分主要介绍每个微调方法的参数和可额外添加的关键字。

具体的使用参考: [不同模型微调实现示例](./fintune_example.md)

## ApacaloraFinetune

基于 alpaca-lora 项目提供的 LoRA 微调能力，用于对大语言模型进行 LoRA 微调。

### 参数

- base_model (str) – 用于微调的基模型本地路径。
- target_path (str) – 微调后 LoRA 权重保存路径。
- merge_path (Optional[str], default: None ) – 合并 LoRA 权重后的模型保存路径，默认 None。 若未提供，则在 target_path 下创建 "lazyllm_lora" 与 "lazyllm_merge" 目录。
- model_name (Optional[str], default: 'LLM' ) – 模型名称，用于日志前缀，默认 LLM。
- cp_files (Optional[str], default: 'tokeniz*' ) – 从基模型路径复制配置文件到 merge_path，默认 tokeniz*。
- launcher (launcher, default: remote(ngpus=1) ) – 微调启动器，默认 launchers.remote(ngpus=1)。
- kw (dict, default: {} ) – 用于更新默认训练参数的关键字参数，允许更新如下参数：

    * data_path (Optional[str]) – 数据路径，默认 None。
    * batch_size (Optional[int]) – 批大小，默认 64。
    * micro_batch_size (Optional[int]) – 微批大小，默认 4。
    * num_epochs (Optional[int]) – 训练轮数，默认 2。
    * learning_rate (Optional[float]) – 学习率，默认 5.e-4。
    * cutoff_len (Optional[int]) – 截断长度，默认 1030。
    * filter_nums (Optional[int]) – 过滤器数量，默认 1024。
    * val_set_size (Optional[int]) – 验证集大小，默认 200。
    * lora_r (Optional[int]) – LoRA 秩，默认 8。
    * lora_alpha (Optional[int]) – LoRA 融合因子，默认 32。
    * lora_dropout (Optional[float]) – LoRA 丢弃率，默认 0.05。
    * lora_target_modules (Optional[str]) – LoRA 目标模块，默认 [wo,wqkv]。
    * modules_to_save (Optional[str]) – 全量微调模块，默认 [tok_embeddings,output]。
    * deepspeed (Optional[str]) – DeepSpeed 配置路径，默认使用仓库预制 ds.json。
    * prompt_template_name (Optional[str]) – 提示模板名称，默认 alpaca。
    * train_on_inputs (Optional[bool]) – 是否在输入上训练，默认 True。
    * show_prompt (Optional[bool]) – 是否显示提示，默认 False。
    * nccl_port (Optional[int]) – NCCL 端口，默认随机在 19000-20500。

```python
from lazyllm import finetune
trainer = finetune.alpacalora('path/to/base/model', 'path/to/target')
```

## CollieFinetune

基于 Collie 框架提供的 LoRA 微调能力，用于对大语言模型进行 LoRA 微调。

### 参数

- base_model (str) – 用于微调的基模型路径。
- target_path (str) – 微调后 LoRA 权重保存路径。
- merge_path (Optional[str], default: None ) – 合并 LoRA 权重后的模型路径，默认 None。 若未提供，则在 target_path 下创建 "lazyllm_lora" 与 "lazyllm_merge" 目录。
- model_name (Optional[str], default: 'LLM' ) – 模型名称，用于日志前缀，默认 "LLM"。
- cp_files (Optional[str], default: 'tokeniz*' ) – 指定从基模型路径复制到 merge_path 的配置文件，默认 "tokeniz*"。
- launcher (launcher, default: remote(ngpus=1) ) – 微调启动器，默认 launchers.remote(ngpus=1)。
- kw (dict, default: {} ) – 用于更新默认训练参数的关键字参数。仅允许更新如下参数：

    * data_path (Optional[str]) – 数据路径，默认 None。
    * batch_size (Optional[int]) – 批大小，默认 64。
    * micro_batch_size (Optional[int]) – 微批大小，默认 4。
    * num_epochs (Optional[int]) – 训练轮数，默认 3。
    * learning_rate (Optional[float]) – 学习率，默认 5.e-4。
    * dp_size (Optional[int]) – 数据并行参数，默认 8。
    * pp_size (Optional[int]) – 流水线并行参数，默认 1。
    * tp_size (Optional[int]) – 张量并行参数，默认 1。
    * lora_r (Optional[int]) – LoRA 秩，默认 8。
    * lora_alpha (Optional[int]) – LoRA 融合因子，默认 16。
    * lora_dropout (Optional[float]) – LoRA 丢弃率，默认 0.05。
    * lora_target_modules (Optional[str]) – LoRA 目标模块，默认 [wo,wqkv]。
    * modules_to_save (Optional[str]) – 全量微调模块，默认 [tok_embeddings,output]。
    * prompt_template_name (Optional[str]) – 提示模板名称，默认 "alpaca"。

```python
from lazyllm import finetune
trainer = finetune.collie('path/to/base/model', 'path/to/target')
```

## LlamafactoryFinetune

基于 LLaMA-Factory 框架提供的训练能力，用于对大语言模型(或视觉语言模型)进行训练。

### 参数

- base_model – 用于进行训练的基模型路径。支持本地路径，若路径不存在则尝试从配置的模型路径中查找。
- target_path – 训练完成后，模型权重保存的目标路径。
- merge_path (str, default: None ) – 模型合并LoRA权重后的保存路径，默认为None。 如果未指定，将在 target_path 下自动创建两个目录： - "lazyllm_lora"（用于存放LoRA训练权重） - "lazyllm_merge"（用于存放合并后的模型权重）
- config_path (str, default: None ) – 训练配置的 YAML 文件路径，默认None。 如果未指定，则使用默认配置文件 llama_factory/sft.yaml。 配置文件支持覆盖默认训练参数。
- export_config_path (str, default: None ) – LoRA权重合并导出配置的 YAML 文件路径，默认None。 如果未指定，则使用默认配置文件 llama_factory/lora_export.yaml。
- lora_r (int, default: None ) – LoRA的秩（rank），若提供则覆盖配置中的 lora_rank。
- modules_to_save (str, default: None ) – 额外需要保存的模型模块名称列表，格式类似于Python列表字符串，如 "[module1,module2]"。
- lora_target_modules (str, default: None ) – 目标LoRA微调的模块名称列表，格式同上。
- launcher (launcher, default: remote(ngpus=1, sync=True) ) – 微调任务的启动器，默认为单卡同步远程启动器 launchers.remote(ngpus=1, sync=True)。
- **kw – 关键字参数，用于动态覆盖默认训练配置中的参数。仅允许更新如下参数：

    * stage (Literal['pt', 'sft', 'rm', 'ppo', 'dpo', 'kto']) – 默认值是：sft。将在训练中执行的阶段。
    * do_train (bool) – 默认值是：True。是否运行训练。
    * finetuning_type (Literal['lora', 'freeze', 'full']) – 默认值是：lora。要使用的微调方法。
    * lora_target (str) – 默认值是：all。要应用LoRA的目标模块的名称。使用逗号分隔多个模块。使用all指定所有线性模块。
    * template (Optional[str]) – 默认值是：None。用于构建训练和推理提示的模板。
    * cutoff_len (int) – 默认值是：1024。数据集中token化后输入的截止长度。
    * max_samples (Optional[int]) – 默认值是：1000。出于调试目的，截断每个数据集的示例数量。
    * overwrite_cache (bool) – 默认值是：True。覆盖缓存的训练和评估集。
    * preprocessing_num_workers (Optional[int]) – 默认值是：16。用于预处理的进程数。
    * dataset_dir (str) – 默认值是：lazyllm_temp_dir。包含数据集的文件夹的路径。如果没有明确指定，LazyLLM将在当前工作目录的 .temp 文件夹中生成一个 dataset_info.json 文件，供LLaMA-Factory使用。
    * logging_steps (float) – 默认值是：10。每X个更新步骤记录一次日志。应该是整数或范围在 [0,1) 的浮点数。如果小于1，将被解释为总训练步骤的比例。
    * save_steps (float) – 默认值是：500。每X个更新步骤保存一次检查点。应该是整数或范围在 [0,1) 的浮点数。如果小于1，将被解释为总训练步骤的比例。
    * plot_loss (bool) – 默认值是：True。是否保存训练损失曲线。
    * overwrite_output_dir (bool) – 默认值是：True。覆盖输出目录的内容。
    * per_device_train_batch_size (int) – 默认值是：1。每个GPU/TPU/MPS/NPU核心/CPU的训练批次的大小。
    * gradient_accumulation_steps (int) – 默认值是：8。在执行反向传播及参数更新前，要累积的更新步骤数。
    * learning_rate (float) – 默认值是：1e-04。AdamW的初始学习率。
    * num_train_epochs (float) – 默认值是：3.0。要执行的总训练周期数。
    * lr_scheduler_type (Union[SchedulerType, str]) – 默认值是：cosine。要使用的调度器类型。
    * warmup_ratio (float) – 默认值是：0.1。在总步骤的 warmup_ratio 分之一阶段内进行线性预热。
    * fp16 (bool) – 默认值是：True。是否使用fp16（混合）精度，而不是32位。
    * ddp_timeout (Optional[int]) – 默认值是：180000000。覆盖分布式训练的默认超时时间（值应以秒为单位给出）。
    * report_to (Union[NoneType, str, List[str]]) – 默认值是：tensorboard。要将结果和日志报告到的集成列表。
    * val_size (float) – 默认值是：0.1。验证集的大小，应该是整数或范围在[0,1)的浮点数。
    * per_device_eval_batch_size (int) – 默认值是：1。每个GPU/TPU/MPS/NPU核心/CPU的验证集批次大小。
    * eval_strategy (Union[IntervalStrategy, str]) – 默认值是：steps。要使用的验证评估策略。
    * eval_steps (Optional[float]) – 默认值是：500。每X个步骤运行一次验证评估。应该是整数或范围在[0,1)的浮点数。如果小于1，将被解释为总训练步骤的比例。

```python
from lazyllm import finetune
trainer = finetune.llamafactory('internlm2-chat-7b', 'path/to/target')
```

## FlagembeddingFinetune

基于 FlagEmbedding 框架提供的训练能力，用于训练嵌入和重排模型。

### 参数

- base_model (str) – 用于训练的基础模型。必须是基础模型的路径。
- target_path (str) – 训练后模型权重保存的路径。
- launcher (launcher, default: remote(ngpus=1, sync=True) ) – 微调的启动器，默认为 launchers.remote(ngpus=1, sync=True)。
- kw – 用于更新默认训练参数的关键字参数。仅允许更新如下参数：

    * 对于嵌入模型
        * train_group_size (int) – 默认为：8。训练组的大小。用于控制每个训练集中的负样本数量。
        * query_max_len (int) – 默认为：512。经过分词后，段落的最大总输入序列长度。超过此长度的序列将被截断，较短的序列将被填充。
        * passage_max_len (int) – 默认为：512。经过分词后，段落的最大总输入序列长度。超过此长度的序列将被截断，较短的序列将被填充。
        * pad_to_multiple_of (int) – 默认为：8。如果设置，将序列填充为提供值的倍数。
        * query_instruction_for_retrieval (str) – 默认为：Represent this sentence for searching relevant passages:。查询query的指令。
        * query_instruction_format (str) – 默认为：{}{}。查询指令格式。
        * learning_rate (float) – 默认为：1e-5。学习率。
        * num_train_epochs (int) – 默认为：1。要执行的总训练周期数。
        * per_device_train_batch_size (int) – 默认为：2。训练批量大小。
        * gradient_accumulation_steps (int) – 默认为：1。在执行反向/更新传递之前要累积的更新步骤数。
        * dataloader_drop_last (bool) – 默认为：True。如果数据集大小不能被批量大小整除，则丢弃最后一个不完整的批量，即 DataLoader 只返回完整的批量。
        * warmup_ratio (float) – 默认为：0.1。线性调度器的预热比率。
        * weight_decay (float) – 默认为：0.01。AdamW 中的权重衰减。
        * deepspeed (str) – 默认为：''。DeepSpeed 配置文件的路径，默认使用 LazyLLM 代码仓库中的预置文件：ds_stage0.json。
        * logging_steps (int) – 默认为：1。更新日志的频率。
        * save_steps (int) – 默认为：1000。保存频率。
        * temperature (float) – 默认为：0.02。用于相似度评分的温度。
        * sentence_pooling_method (str) – 默认为：cls。池化方法。可用选项：'cls', 'mean', 'last_token'。
        * normalize_embeddings (bool) – 默认为：True。是否归一化嵌入。
        * kd_loss_type (str) – 默认为：kl_div。知识蒸馏的损失类型。可用选项：'kl_div', 'm3_kd_loss'。
        * overwrite_output_dir (bool) – 默认为：True。用于允许程序覆盖现有的输出目录。
        * fp16 (bool) – 默认为：True。是否使用 fp16（混合）精度而不是 32 位。
        * gradient_checkpointing (bool) – 默认为：True。是否启用梯度检查点。
        * negatives_cross_device (bool) – 默认为：True。是否在设备间共享负样本。
    * 对于重排模型
        * train_group_size (int) – 默认为：8。训练组的大小。用于控制每个训练集中的负样本数量。
        * query_max_len (int) – 默认为：256。经过分词后，段落的最大总输入序列长度。超过此长度的序列将被截断，较短的序列将被填充。
        * passage_max_len (int) – 默认为：256。经过分词后，段落的最大总输入序列长度。超过此长度的序列将被截断，较短的序列将被填充。
        * pad_to_multiple_of (int) – 默认为：8。如果设置，将序列填充为提供值的倍数。
        * learning_rate (float) – 默认为：6e-5。学习率。
        * num_train_epochs (int) – 默认为：1。要执行的总训练周期数。
        * per_device_train_batch_size (int) – 默认为：2。训练批量大小。
        * gradient_accumulation_steps (int) – 默认为：1。在执行反向/更新传递之前要累积的更新步骤数。
        * dataloader_drop_last (bool) – 默认为：True。如果数据集大小不能被批量大小整除，则丢弃最后一个不完整的批量，即 DataLoader 只返回完整的批量。
        * warmup_ratio (float) – 默认为：0.1。线性调度器的预热比率。
        * weight_decay (float) – 默认为：0.01。AdamW 中的权重衰减。
        * deepspeed (str) – 默认为：''。DeepSpeed 配置文件的路径，默认使用 LazyLLM 代码仓库中的预置文件：ds_stage0.json。
        * logging_steps (int) – 默认为：1。更新日志的频率。
        * save_steps (int) – 默认为：1000。保存频率。
        * overwrite_output_dir (bool) – 默认为：True。用于允许程序覆盖现有的输出目录。
        * fp16 (bool) – 默认为：True。是否使用 fp16（混合）精度而不是 32 位。
        * gradient_checkpointing (bool) – 默认为：True。是否启用梯度检查点。

```python
from lazyllm import finetune
finetune.FlagembeddingFinetune('bge-m3', 'path/to/target')
```

## AutoFinetune

可根据输入的参数自动选择合适的微调框架和参数，以对大语言模型进行微调。

具体而言，基于输入的：base_model 的模型参数、ctx_len、batch_size、lora_r、launcher 中GPU的类型以及卡数，该类可以自动选择出合适的微调框架（如: AlpacaloraFinetune 或 CollieFinetune）及所需的参数。

### 参数

- base_model (str) – 用于进行微调的基模型。要求是基模型的路径。
- source (config[model_source]) – 指定模型的下载源。可通过设置环境变量 LAZYLLM_MODEL_SOURCE 来配置，目前仅支持 huggingface 或 modelscope 。若不设置，lazyllm不会启动自动模型下载。
- target_path (str) – 微调后模型保存LoRA权重的路径。
- merge_path (str) – 模型合并LoRA权重后的路径，默认为 None。如果未指定，则会在 target_path 下创建 "lazyllm_lora" 和 "lazyllm_merge" 目录，分别作为 target_path 和 merge_path 。
- ctx_len (int) – 输入微调模型的token最大长度，默认为 1024。
- batch_size (int) – 批处理大小，默认为 32。
- lora_r (int) – LoRA 的秩，默认为 8；该数值决定添加参数的量，数值越小参数量越小。
- launcher (launcher, default: remote() ) – 微调的启动器，默认为 launchers.remote(ngpus=1)。
- kw – 关键字参数，用于更新默认的训练参数。注意这里能够指定的关键字参数取决于 LazyLLM 推测出的框架，因此建议谨慎设置。

```python
from lazyllm import finetune
finetune.auto("internlm2-chat-7b", 'path/to/target')
```

## DummyFinetune

用于占位实现微调逻辑。 此类主要用于演示或测试目的，因为它不执行任何实际的微调操作。

### 参数

base_model – 字符串，指定基础模型的名称，默认为 'base'。
target_path – 字符串，指定微调输出的目标路径，默认为 'target'。
launcher – 启动器实例，用于执行命令。默认为launchers.remote()。
**kw – 其他关键字参数，这些参数会被保存以供后续使用。

```python
from lazyllm.components import DummyFinetune
from lazyllm import launchers
# 创建一个 DummyFinetune 实例
finetuner = DummyFinetune(base_model='example-base', target_path='example-target', launcher=launchers.local(), custom_arg='custom_value')
# 调用 cmd 方法生成占位命令
command = finetuner.cmd('--example-arg', key='value')
print(command)
... echo 'dummy finetune!, and init-args is {'custom_arg': 'custom_value'}'
```

## 注意

这部分主要介绍各微调框架的参数，具体的使用方式参考: 
[不同模型微调实现示例](./fintune_example.md)