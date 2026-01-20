# flake8: noqa E501
from . import utils
import functools
import lazyllm

add_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm.components)
add_english_doc = functools.partial(utils.add_english_doc, module=lazyllm.components)
add_example = functools.partial(utils.add_example, module=lazyllm.components)


add_chinese_doc('register', '''\
LazyLLM提供的Component的注册机制，可以将任意函数注册成LazyLLM的Component。被注册的函数无需显式的import，即可通过注册器提供的分组机制，在任一位置被索引到。

.. function:: register(cls, *, rewrite_func) -> Decorator

函数调用后返回一个装饰器，它会将被装饰的函数包装成一个Component注册到名为cls的组中.

Args:
    cls (str): 函数即将被注册到的组的名字，要求组必须存在，默认的组有 ``finetune`` 、 ``deploy`` ，用户可以调用 ``new_group`` 创建新的组
    rewrite_func (str): 注册后要重写的函数名称，默认为 ``apply`` ，当需要注册一个bash命令时需传入 ``cmd`` 

.. function:: register.cmd(cls) -> Decorator

函数调用后返回一个装饰器，它会将被装饰的函数包装成一个Component注册到名为cls的组中。被包装的函数需要返回一个可执行的bash命令。

Args:
    cls (str): 函数即将被注册到的组的名字，要求组必须存在，默认的组有 ``finetune`` 、 ``deploy`` ，用户可以调用 ``new_group`` 创建新的组

.. function:: register.new_group(group_name) -> None

新建一个ComponentGroup, 新建后的group会自动加入到__builtin__中，无需import即可在任一位置访问到该group.

Args:
    group_name (str): 待创建的group的名字
''')

add_english_doc('register', '''\
LazyLLM provides a registration mechanism for Components, allowing any function to be registered as a Component of LazyLLM. The registered functions can be indexed at any location through the grouping mechanism provided by the registrar, without the need for explicit import.

.. function:: register(cls, *, rewrite_func) -> Decorator

This function call returns a decorator that wraps the decorated function into a Component and registers it into the group named `cls`.

Args:
    cls (str): The name of the group to which the function will be registered. The group must exist. The default groups are `finetune` and `deploy`. Users can create new groups by calling `new_group`.
    rewrite_func (str): The name of the function to be rewritten after registration. The default is `'apply'`. If registering a bash command, pass `'cmd'`.

.. function:: register.cmd(cls) -> Decorator

This function call returns a decorator that wraps the decorated function into a Component and registers it into the group named `cls`. The wrapped function needs to return an executable bash command.

Args:
    cls (str): The name of the group to which the function will be registered. The group must exist. The default groups are `finetune` and `deploy`. Users can create new groups by calling `new_group`.

.. function:: register.new_group(group_name) -> None

Creates a new ComponentGroup. The newly created group will be automatically added to __builtin__ and can be accessed at any location without the need for import.

Args:
    group_name (str): The name of the group to be created.
''')

add_example('register', ['''\
>>> import lazyllm
>>> @lazyllm.component_register('mygroup')
... def myfunc(input):
...     return input
...
>>> lazyllm.mygroup.myfunc()(1)
1
''', '''\
>>> import lazyllm
>>> @lazyllm.component_register.cmd('mygroup')
... def mycmdfunc(input):
...     return f'echo {input}'
...
>>> lazyllm.mygroup.mycmdfunc()(1)
PID: 2024-06-01 00:00:00 lazyllm INFO: (lazyllm.launcher) Command: echo 1
PID: 2024-06-01 00:00:00 lazyllm INFO: (lazyllm.launcher) PID: 1
''', '''\
>>> import lazyllm
>>> lazyllm.component_register.new_group('mygroup')
>>> lazyllm.mygroup
{}
'''])

# ============= Finetune
add_chinese_doc('finetune.LazyLLMFinetuneBase', """\
LazyLLM微调基础组件类，继承自ComponentBase。

提供大语言模型微调的基础功能，支持远程启动器配置和模型路径管理。

Args:
    base_model (str): 基础模型路径或标识
    target_path (str): 微调后模型输出路径
    launcher (Launcher, optional): 任务启动器，默认为远程启动器
""")

add_english_doc('finetune.LazyLLMFinetuneBase', """\
LazyLLM fine-tuning base component class, inherits from ComponentBase.

Provides base functionality for large language model fine-tuning, supports remote launcher configuration and model path management.

Args:
    base_model (str): Base model path or identifier
    target_path (str): Fine-tuned model output path
    launcher (Launcher, optional): Task launcher, defaults to remote launcher
""")

# Finetune-AlpacaloraFinetune
add_chinese_doc('finetune.AlpacaloraFinetune', '''\
此类是 ``LazyLLMFinetuneBase`` 的子类，基于 [alpaca-lora](https://github.com/tloen/alpaca-lora) 项目提供的 LoRA 微调能力，用于对大语言模型进行 LoRA 微调。

Args:
    base_model (str): 用于微调的基模型本地路径。
    target_path (str): 微调后 LoRA 权重保存路径。
    merge_path (Optional[str]): 合并 LoRA 权重后的模型保存路径，默认 ``None``。
        若未提供，则在 ``target_path`` 下创建 "lazyllm_lora" 与 "lazyllm_merge" 目录。
    model_name (Optional[str]): 模型名称，用于日志前缀，默认 ``LLM``。
    cp_files (Optional[str]): 从基模型路径复制配置文件到 ``merge_path``，默认 ``tokeniz*``。
    launcher (lazyllm.launcher): 微调启动器，默认 ``launchers.remote(ngpus=1)``。
    kw (dict): 用于更新默认训练参数的关键字参数，允许更新如下参数：

Keyword Args:
    data_path (Optional[str]): 数据路径，默认 ``None``。
    batch_size (Optional[int]): 批大小，默认 64。
    micro_batch_size (Optional[int]): 微批大小，默认 4。
    num_epochs (Optional[int]): 训练轮数，默认 2。
    learning_rate (Optional[float]): 学习率，默认 5.e-4。
    cutoff_len (Optional[int]): 截断长度，默认 1030。
    filter_nums (Optional[int]): 过滤器数量，默认 1024。
    val_set_size (Optional[int]): 验证集大小，默认 200。
    lora_r (Optional[int]): LoRA 秩，默认 8。
    lora_alpha (Optional[int]): LoRA 融合因子，默认 32。
    lora_dropout (Optional[float]): LoRA 丢弃率，默认 0.05。
    lora_target_modules (Optional[str]): LoRA 目标模块，默认 ``[wo,wqkv]``。
    modules_to_save (Optional[str]): 全量微调模块，默认 ``[tok_embeddings,output]``。
    deepspeed (Optional[str]): DeepSpeed 配置路径，默认使用仓库预制 ds.json。
    prompt_template_name (Optional[str]): 提示模板名称，默认 ``alpaca``。
    train_on_inputs (Optional[bool]): 是否在输入上训练，默认 ``True``。
    show_prompt (Optional[bool]): 是否显示提示，默认 ``False``。
    nccl_port (Optional[int]): NCCL 端口，默认随机在 19000-20500。
''')

add_english_doc('finetune.AlpacaloraFinetune', '''\
This class is a subclass of ``LazyLLMFinetuneBase``, based on the LoRA fine-tuning capabilities provided by the [alpaca-lora](https://github.com/tloen/alpaca-lora) project, used for LoRA fine-tuning of large language models.

Args:
    base_model (str): Path to the base model for fine-tuning.
    target_path (str): Path to save LoRA weights of the fine-tuned model.
    merge_path (Optional[str]): Path to save merged LoRA weights, default ``None``.
        If not provided, "lazyllm_lora" and "lazyllm_merge" directories are created under ``target_path``.
    model_name (Optional[str]): Model name used as log prefix, default "LLM".
    cp_files (Optional[str]): Configuration files copied from base model path to ``merge_path``, default ``tokeniz*``.
    launcher (lazyllm.launcher): Launcher for fine-tuning, default ``launchers.remote(ngpus=1)``.
    kw (dict): Keyword arguments to update default training parameters:

Keyword Args:
    data_path (Optional[str]): Path to dataset, default ``None``.
    batch_size (Optional[int]): Batch size, default 64.
    micro_batch_size (Optional[int]): Micro-batch size, default 4.
    num_epochs (Optional[int]): Number of training epochs, default 2.
    learning_rate (Optional[float]): Learning rate, default 5.e-4.
    cutoff_len (Optional[int]): Cutoff length, default 1030.
    filter_nums (Optional[int]): Number of filters, default 1024.
    val_set_size (Optional[int]): Validation set size, default 200.
    lora_r (Optional[int]): LoRA rank, default 8.
    lora_alpha (Optional[int]): LoRA fusion factor, default 32.
    lora_dropout (Optional[float]): LoRA dropout rate, default 0.05.
    lora_target_modules (Optional[str]): LoRA target modules, default ``[wo,wqkv]``.
    modules_to_save (Optional[str]): Modules for full fine-tuning, default ``[tok_embeddings,output]``.
    deepspeed (Optional[str]): Path to DeepSpeed config, default uses repository pre-made ds.json.
    prompt_template_name (Optional[str]): Name of prompt template, default "alpaca".
    train_on_inputs (Optional[bool]): Whether to train on inputs, default ``True``.
    show_prompt (Optional[bool]): Whether to show the prompt, default ``False``.
    nccl_port (Optional[int]): NCCL port, default random between 19000-20500.
''')

add_example('finetune.AlpacaloraFinetune', '''\
>>> from lazyllm import finetune
>>> trainer = finetune.alpacalora('path/to/base/model', 'path/to/target')
''')

add_chinese_doc('finetune.AlpacaloraFinetune.cmd', """\
生成用于执行Alpaca-LoRA微调和模型合并的shell命令序列。

Args:
    trainset (str): 训练数据集路径，支持相对data_path配置的路径或绝对路径
    valset (str, optional): 验证数据集路径，未指定时将从训练集中自动划分

**Returns:**\n
- str or list: 当不需要合并模型时返回单个命令字符串，需要合并时返回包含微调命令、合并命令和文件拷贝命令的列表
""")

add_english_doc('finetune.AlpacaloraFinetune.cmd', """\
Generate shell command sequence for Alpaca-LoRA fine-tuning and model merging.

Args:
    trainset (str): Training dataset path, supports both relative path (to configured data_path) and absolute path
    valset (str, optional): Validation dataset path, will auto-split from trainset if not specified

**Returns:**\n
- str or list: Returns a single command string when no merging needed, otherwise returns a list containing:
                 [fine-tune command, merge command, file copy command]
""")

add_example('finetune.AlpacaloraFinetune.cmd', """\
>>> from lazyllm import finetune
>>> trainer = finetune.alpacalora('path/to/base/model', 'path/to/target')
>>> cmd = trainer.cmd("my_dataset.json")
""")

# Finetune-CollieFinetune
add_chinese_doc('finetune.CollieFinetune', '''\
此类是 ``LazyLLMFinetuneBase`` 的子类，基于 [Collie](https://github.com/OpenLMLab/collie) 框架提供的 LoRA 微调能力，用于对大语言模型进行 LoRA 微调。

Args:
    base_model (str): 用于微调的基模型路径。
    target_path (str): 微调后 LoRA 权重保存路径。
    merge_path (Optional[str]): 合并 LoRA 权重后的模型路径，默认 ``None``。
        若未提供，则在 ``target_path`` 下创建 "lazyllm_lora" 与 "lazyllm_merge" 目录。
    model_name (Optional[str]): 模型名称，用于日志前缀，默认 "LLM"。
    cp_files (Optional[str]): 指定从基模型路径复制到 ``merge_path`` 的配置文件，默认 "tokeniz*"。
    launcher (lazyllm.launcher): 微调启动器，默认 ``launchers.remote(ngpus=1)``。
    kw (dict): 用于更新默认训练参数的关键字参数。仅允许更新如下参数：

Keyword Args:
    data_path (Optional[str]): 数据路径，默认 ``None``。
    batch_size (Optional[int]): 批大小，默认 64。
    micro_batch_size (Optional[int]): 微批大小，默认 4。
    num_epochs (Optional[int]): 训练轮数，默认 3。
    learning_rate (Optional[float]): 学习率，默认 5.e-4。
    dp_size (Optional[int]): 数据并行参数，默认 8。
    pp_size (Optional[int]): 流水线并行参数，默认 1。
    tp_size (Optional[int]): 张量并行参数，默认 1。
    lora_r (Optional[int]): LoRA 秩，默认 8。
    lora_alpha (Optional[int]): LoRA 融合因子，默认 16。
    lora_dropout (Optional[float]): LoRA 丢弃率，默认 0.05。
    lora_target_modules (Optional[str]): LoRA 目标模块，默认 ``[wo,wqkv]``。
    modules_to_save (Optional[str]): 全量微调模块，默认 ``[tok_embeddings,output]``。
    prompt_template_name (Optional[str]): 提示模板名称，默认 "alpaca"。
''')

add_english_doc('finetune.CollieFinetune', '''\
This class is a subclass of ``LazyLLMFinetuneBase``, based on the LoRA fine-tuning capabilities provided by the [Collie](https://github.com/OpenLMLab/collie) framework, used for LoRA fine-tuning of large language models.

Args:
    base_model (str): Path to the base model for fine-tuning.
    target_path (str): Path to save LoRA weights of the fine-tuned model.
    merge_path (Optional[str]): Path to save merged LoRA weights, default ``None``.
        If not provided, "lazyllm_lora" and "lazyllm_merge" directories are created under ``target_path``.
    model_name (Optional[str]): Model name used as log prefix, default "LLM".
    cp_files (Optional[str]): Configuration files copied from base model path to ``merge_path``, default "tokeniz*".
    launcher (lazyllm.launcher): Launcher for fine-tuning, default ``launchers.remote(ngpus=1)``.
    kw (dict): Keyword arguments to update default training parameters:

Keyword Args:
    data_path (Optional[str]): Path to dataset, default ``None``.
    batch_size (Optional[int]): Batch size, default 64.
    micro_batch_size (Optional[int]): Micro-batch size, default 4.
    num_epochs (Optional[int]): Number of training epochs, default 3.
    learning_rate (Optional[float]): Learning rate, default 5.e-4.
    dp_size (Optional[int]): Data parallelism parameter, default 8.
    pp_size (Optional[int]): Pipeline parallelism parameter, default 1.
    tp_size (Optional[int]): Tensor parallelism parameter, default 1.
    lora_r (Optional[int]): LoRA rank, default 8.
    lora_alpha (Optional[int]): LoRA fusion factor, default 16.
    lora_dropout (Optional[float]): LoRA dropout rate, default 0.05.
    lora_target_modules (Optional[str]): LoRA target modules, default ``[wo,wqkv]``.
    modules_to_save (Optional[str]): Modules for full fine-tuning, default ``[tok_embeddings,output]``.
    prompt_template_name (Optional[str]): Name of prompt template, default "alpaca".
''')

add_example('finetune.CollieFinetune', '''\
>>> from lazyllm import finetune
>>> trainer = finetune.collie('path/to/base/model', 'path/to/target')
''')

# Finetune-LlamafactoryFinetune
add_chinese_doc('finetune.LlamafactoryFinetune', '''\
此类是 ``LazyLLMFinetuneBase`` 的子类，基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 框架提供的训练能力，用于对大语言模型(或视觉语言模型)进行训练。

Args:
    base_model: 用于进行训练的基模型路径。支持本地路径，若路径不存在则尝试从配置的模型路径中查找。
    target_path: 训练完成后，模型权重保存的目标路径。
    merge_path (str, optional): 模型合并LoRA权重后的保存路径，默认为None。
        如果未指定，将在 ``target_path`` 下自动创建两个目录：
        - "lazyllm_lora"（用于存放LoRA训练权重）
        - "lazyllm_merge"（用于存放合并后的模型权重）
    config_path (str, optional): 训练配置的 YAML 文件路径，默认None。
        如果未指定，则使用默认配置文件 ``llama_factory/sft.yaml``。
        配置文件支持覆盖默认训练参数。
    export_config_path (str, optional): LoRA权重合并导出配置的 YAML 文件路径，默认None。
        如果未指定，则使用默认配置文件 ``llama_factory/lora_export.yaml``。
    lora_r (int, optional): LoRA的秩（rank），若提供则覆盖配置中的 ``lora_rank``。
    modules_to_save (str, optional): 额外需要保存的模型模块名称列表，格式类似于Python列表字符串，如 "[module1,module2]"。
    lora_target_modules (str, optional): 目标LoRA微调的模块名称列表，格式同上。
    launcher (lazyllm.launcher, optional): 微调任务的启动器，默认为单卡同步远程启动器 ``launchers.remote(ngpus=1, sync=True)``。
    **kw: 关键字参数，用于动态覆盖默认训练配置中的参数。

此类的关键字参数及其默认值如下：

Keyword Args:
    stage (typing.Literal['pt', 'sft', 'rm', 'ppo', 'dpo', 'kto']): 默认值是：``sft``。将在训练中执行的阶段。
    do_train (bool): 默认值是：``True``。是否运行训练。
    finetuning_type (typing.Literal['lora', 'freeze', 'full']): 默认值是：``lora``。要使用的微调方法。
    lora_target (str): 默认值是：``all``。要应用LoRA的目标模块的名称。使用逗号分隔多个模块。使用`all`指定所有线性模块。
    template (typing.Optional[str]): 默认值是：``None``。用于构建训练和推理提示的模板。
    cutoff_len (int): 默认值是：``1024``。数据集中token化后输入的截止长度。
    max_samples (typing.Optional[int]): 默认值是：``1000``。出于调试目的，截断每个数据集的示例数量。
    overwrite_cache (bool): 默认值是：``True``。覆盖缓存的训练和评估集。
    preprocessing_num_workers (typing.Optional[int]): 默认值是：``16``。用于预处理的进程数。
    dataset_dir (str): 默认值是：``lazyllm_temp_dir``。包含数据集的文件夹的路径。如果没有明确指定，LazyLLM将在当前工作目录的 ``.temp`` 文件夹中生成一个 ``dataset_info.json`` 文件，供LLaMA-Factory使用。
    logging_steps (float): 默认值是：``10``。每X个更新步骤记录一次日志。应该是整数或范围在 ``[0,1)`` 的浮点数。如果小于1，将被解释为总训练步骤的比例。
    save_steps (float): 默认值是：``500``。每X个更新步骤保存一次检查点。应该是整数或范围在 ``[0,1)`` 的浮点数。如果小于1，将被解释为总训练步骤的比例。
    plot_loss (bool): 默认值是：``True``。是否保存训练损失曲线。
    overwrite_output_dir (bool): 默认值是：``True``。覆盖输出目录的内容。
    per_device_train_batch_size (int): 默认值是：``1``。每个GPU/TPU/MPS/NPU核心/CPU的训练批次的大小。
    gradient_accumulation_steps (int): 默认值是：``8``。在执行反向传播及参数更新前，要累积的更新步骤数。
    learning_rate (float): 默认值是：``1e-04``。AdamW的初始学习率。
    num_train_epochs (float): 默认值是：``3.0``。要执行的总训练周期数。
    lr_scheduler_type (typing.Union[transformers.trainer_utils.SchedulerType, str]): 默认值是：``cosine``。要使用的调度器类型。
    warmup_ratio (float): 默认值是：``0.1``。在总步骤的 ``warmup_ratio`` 分之一阶段内进行线性预热。
    fp16 (bool): 默认值是：``True``。是否使用fp16（混合）精度，而不是32位。
    ddp_timeout (typing.Optional[int]): 默认值是：``180000000``。覆盖分布式训练的默认超时时间（值应以秒为单位给出）。
    report_to (typing.Union[NoneType, str, typing.List[str]]): 默认值是：``tensorboard``。要将结果和日志报告到的集成列表。
    val_size (float): 默认值是：``0.1``。验证集的大小，应该是整数或范围在`[0,1)`的浮点数。
    per_device_eval_batch_size (int): 默认值是：``1``。每个GPU/TPU/MPS/NPU核心/CPU的验证集批次大小。
    eval_strategy (typing.Union[transformers.trainer_utils.IntervalStrategy, str]): 默认值是：``steps``。要使用的验证评估策略。
    eval_steps (typing.Optional[float]): 默认值是：``500``。每X个步骤运行一次验证评估。应该是整数或范围在`[0,1)`的浮点数。如果小于1，将被解释为总训练步骤的比例。
''')

add_english_doc('finetune.LlamafactoryFinetune', '''\
This class is a subclass of ``LazyLLMFinetuneBase``, based on the training capabilities provided by the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework, used for training large language models(or visual language models).

Args:
    base_model: Path to the base model used for training. Supports local paths; if the path does not exist, it will attempt to locate it from the configured model directory.
    target_path: Target directory to save model weights after training is completed.
    merge_path (str, optional): Path to save the model after merging LoRA weights. Defaults to None.
        If not specified, two directories will be automatically created under ``target_path``:
        - "lazyllm_lora" (for storing LoRA fine-tuned weights)
        - "lazyllm_merge" (for storing the merged model weights)
    config_path (str, optional): Path to the YAML file containing training configuration. Defaults to None.
        If not specified, the default config file ``llama_factory/sft.yaml`` will be used.
        This file can override default training parameters.
    export_config_path (str, optional): Path to the YAML file for LoRA weight export/merging configuration. Defaults to None.
        If not specified, the default config file ``llama_factory/lora_export.yaml`` will be used.
    lora_r (int, optional): Rank of the LoRA adaptation. If provided, overrides the ``lora_rank`` value in the configuration.
    modules_to_save (str, optional): List of additional module names to be saved. Should be provided as a string in Python list format, e.g., "[module1, module2]".
    lora_target_modules (str, optional): List of module names to apply LoRA fine-tuning to. Format is the same as above.
    launcher (lazyllm.launcher, optional): Launcher for the fine-tuning task. Defaults to a single-GPU, synchronous remote launcher: ``launchers.remote(ngpus=1, sync=True)``.
    **kw: Additional keyword arguments used to dynamically override default parameters in the training configuration.

Keyword Args:
    stage (typing.Literal['pt', 'sft', 'rm', 'ppo', 'dpo', 'kto']): Default is: ``sft``. Which stage will be performed in training.
    do_train (bool): Default is: ``True``. Whether to run training.
    finetuning_type (typing.Literal['lora', 'freeze', 'full']): Default is: ``lora``. Which fine-tuning method to use.
    lora_target (str): Default is: ``all``. Name(s) of target modules to apply LoRA. Use commas to separate multiple modules. Use `all` to specify all the linear modules.
    template (typing.Optional[str]): Default is: ``None``. Which template to use for constructing prompts in training and inference.
    cutoff_len (int): Default is: ``1024``. The cutoff length of the tokenized inputs in the dataset.
    max_samples (typing.Optional[int]): Default is: ``1000``. For debugging purposes, truncate the number of examples for each dataset.
    overwrite_cache (bool): Default is: ``True``. Overwrite the cached training and evaluation sets.
    preprocessing_num_workers (typing.Optional[int]): Default is: ``16``. The number of processes to use for the pre-processing.
    dataset_dir (str): Default is: ``lazyllm_temp_dir``. Path to the folder containing the datasets. If not explicitly specified, LazyLLM will generate a ``dataset_info.json`` file in the ``.temp`` folder in the current working directory for use by LLaMA-Factory.
    logging_steps (float): Default is: ``10``. Log every X updates steps. Should be an integer or a float in range ``[0,1)``. If smaller than 1, will be interpreted as ratio of total training steps.
    save_steps (float): Default is: ``500``. Save checkpoint every X updates steps. Should be an integer or a float in range ``[0,1)``. If smaller than 1, will be interpreted as ratio of total training steps.
    plot_loss (bool): Default is: ``True``. Whether or not to save the training loss curves.
    overwrite_output_dir (bool): Default is: ``True``. Overwrite the content of the output directory.
    per_device_train_batch_size (int): Default is: ``1``. Batch size per GPU/TPU/MPS/NPU core/CPU for training.
    gradient_accumulation_steps (int): Default is: ``8``. Number of updates steps to accumulate before performing a backward/update pass.
    learning_rate (float): Default is: ``1e-04``. The initial learning rate for AdamW.
    num_train_epochs (float): Default is: ``3.0``. Total number of training epochs to perform.
    lr_scheduler_type (typing.Union[transformers.trainer_utils.SchedulerType, str]): Default is: ``cosine``. The scheduler type to use.
    warmup_ratio (float): Default is: ``0.1``. Linear warmup over warmup_ratio fraction of total steps.
    fp16 (bool): Default is: ``True``. Whether to use fp16 (mixed) precision instead of 32-bit.
    ddp_timeout (typing.Optional[int]): Default is: ``180000000``. Overrides the default timeout for distributed training (value should be given in seconds).
    report_to (typing.Union[NoneType, str, typing.List[str]]): Default is: ``tensorboard``. The list of integrations to report the results and logs to.
    val_size (float): Default is: ``0.1``. Size of the development set, should be an integer or a float in range `[0,1)`.
    per_device_eval_batch_size (int): Default is: ``1``. Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation.
    eval_strategy (typing.Union[transformers.trainer_utils.IntervalStrategy, str]): Default is: ``steps``. The evaluation strategy to use.
    eval_steps (typing.Optional[float]): Default is: ``500``. Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps.
''')

add_example('finetune.LlamafactoryFinetune', '''\
>>> from lazyllm import finetune
>>> trainer = finetune.llamafactory('internlm2-chat-7b', 'path/to/target')
<lazyllm.llm.finetune type=LlamafactoryFinetune>
''')
add_chinese_doc('finetune.LlamafactoryFinetune.cmd', """\
生成LLaMA-Factory微调命令序列，包括训练和模型合并命令。

Args:
    trainset (str): 训练数据集路径(支持相对lazyllm.config['data_path']的路径)
    valset (str, optional): 验证数据集路径(当前实现中未直接使用)

**Returns:**\n
- str: 完整的shell命令字符串，包含:
    - 训练命令(自动配置参数)
    - 日志重定向(保存到目标路径)
    - 可选的模型合并命令(当配置LoRA时)

注意事项:
    - 自动生成带时间戳的训练日志文件
    - 临时文件会在使用后自动清理
    - 支持多种数据格式(alpaca/sharegpt等)
    - 多模态数据(图像/视频/音频)会自动检测处理
""")

add_english_doc('finetune.LlamafactoryFinetune.cmd', """\
Generate LLaMA-Factory fine-tuning command sequence, including training and model merge commands.

Args:
    trainset (str): Training dataset path (supports relative path to lazyllm.config['data_path'])
    valset (str, optional): Validation dataset path (not directly used in current implementation)

**Returns:**\n
- str: Complete shell command string containing:
    - Training command (with auto-configured parameters)
    - Log redirection (saved to target path)
    - Optional model merge command (when LoRA is configured)

Notes:
    - Automatically generates timestamped training log files
    - Temporary files are automatically cleaned up after use
    - Supports multiple data formats (alpaca/sharegpt etc.)
    - Multimodal data (images/videos/audios) is automatically detected and handled
""")

# Finetune-FlagembeddingFinetune
add_chinese_doc('finetune.FlagembeddingFinetune', '''\
该类是 ``LazyLLMFinetuneBase`` 的子类，基于 [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) 框架提供的训练能力，用于训练嵌入和重排模型。

Args:
    base_model (str): 用于训练的基础模型。必须是基础模型的路径。
    target_path (str): 训练后模型权重保存的路径。
    launcher (lazyllm.launcher): 微调的启动器，默认为 ``launchers.remote(ngpus=1, sync=True)``。
    kw: 用于更新默认训练参数的关键字参数。

该类嵌入模型的关键字参数及其默认值如下：

Keyword Args:
    train_group_size (int): 默认为：``8``。训练组的大小。用于控制每个训练集中的负样本数量。
    query_max_len (int): 默认为：``512``。经过分词后，段落的最大总输入序列长度。超过此长度的序列将被截断，较短的序列将被填充。
    passage_max_len (int): 默认为：``512``。经过分词后，段落的最大总输入序列长度。超过此长度的序列将被截断，较短的序列将被填充。
    pad_to_multiple_of (int): 默认为：``8``。如果设置，将序列填充为提供值的倍数。
    query_instruction_for_retrieval (str): 默认为：``Represent this sentence for searching relevant passages: ``。查询query的指令。
    query_instruction_format (str): 默认为：``{}{}``。查询指令格式。
    learning_rate (float): 默认为：``1e-5``。学习率。
    num_train_epochs (int): 默认为：``1``。要执行的总训练周期数。
    per_device_train_batch_size (int): 默认为：``2``。训练批量大小。
    gradient_accumulation_steps (int): 默认为：``1``。在执行反向/更新传递之前要累积的更新步骤数。
    dataloader_drop_last (bool): 默认为：``True``。如果数据集大小不能被批量大小整除，则丢弃最后一个不完整的批量，即 DataLoader 只返回完整的批量。
    warmup_ratio (float): 默认为：``0.1``。线性调度器的预热比率。
    weight_decay (float): 默认为：``0.01``。AdamW 中的权重衰减。
    deepspeed (str): 默认为：````。DeepSpeed 配置文件的路径，默认使用 LazyLLM 代码仓库中的预置文件：``ds_stage0.json``。
    logging_steps (int): 默认为：``1``。更新日志的频率。
    save_steps (int): 默认为：``1000``。保存频率。
    temperature (float): 默认为：``0.02``。用于相似度评分的温度。
    sentence_pooling_method (str): 默认为：``cls``。池化方法。可用选项：'cls', 'mean', 'last_token'。
    normalize_embeddings (bool): 默认为：``True``。是否归一化嵌入。
    kd_loss_type (str): 默认为：``kl_div``。知识蒸馏的损失类型。可用选项：'kl_div', 'm3_kd_loss'。
    overwrite_output_dir (bool): 默认为：``True``。用于允许程序覆盖现有的输出目录。
    fp16 (bool): 默认为：``True``。是否使用 fp16（混合）精度而不是 32 位。
    gradient_checkpointing (bool): 默认为：``True``。是否启用梯度检查点。
    negatives_cross_device (bool): 默认为：``True``。是否在设备间共享负样本。

该类重排模型的关键字参数及其默认值如下：

Keyword Args:
    train_group_size (int): 默认为：``8``。训练组的大小。用于控制每个训练集中的负样本数量。
    query_max_len (int): 默认为：``256``。经过分词后，段落的最大总输入序列长度。超过此长度的序列将被截断，较短的序列将被填充。
    passage_max_len (int): 默认为：``256``。经过分词后，段落的最大总输入序列长度。超过此长度的序列将被截断，较短的序列将被填充。
    pad_to_multiple_of (int): 默认为：``8``。如果设置，将序列填充为提供值的倍数。
    learning_rate (float): 默认为：``6e-5``。学习率。
    num_train_epochs (int): 默认为：``1``。要执行的总训练周期数。
    per_device_train_batch_size (int): 默认为：``2``。训练批量大小。
    gradient_accumulation_steps (int): 默认为：``1``。在执行反向/更新传递之前要累积的更新步骤数。
    dataloader_drop_last (bool): 默认为：``True``。如果数据集大小不能被批量大小整除，则丢弃最后一个不完整的批量，即 DataLoader 只返回完整的批量。
    warmup_ratio (float): 默认为：``0.1``。线性调度器的预热比率。
    weight_decay (float): 默认为：``0.01``。AdamW 中的权重衰减。
    deepspeed (str): 默认为：````。DeepSpeed 配置文件的路径，默认使用 LazyLLM 代码仓库中的预置文件：``ds_stage0.json``。
    logging_steps (int): 默认为：``1``。更新日志的频率。
    save_steps (int): 默认为：``1000``。保存频率。
    overwrite_output_dir (bool): 默认为：``True``。用于允许程序覆盖现有的输出目录。
    fp16 (bool): 默认为：``True``。是否使用 fp16（混合）精度而不是 32 位。
    gradient_checkpointing (bool): 默认为：``True``。是否启用梯度检查点。
''')

add_english_doc('finetune.FlagembeddingFinetune', '''\
This class is a subclass of ``LazyLLMFinetuneBase``, based on the training capabilities provided by the [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) framework, used for training embedding and reranker models.

Args:
    base_model (str): The base model used for training. It is required to be the path of the base model.
    target_path (str): The path where the trained model weights are saved.
    launcher (lazyllm.launcher): The launcher for fine-tuning, default is ``launchers.remote(ngpus=1, sync=True)``.
    kw: Keyword arguments used to update the default training parameters.

The keyword arguments and their default values for this class of embedding model are as follows:

Keyword Args:
    train_group_size (int): Default is: ``8``. The size of train group. It is used to control the number of negative samples in each training set.
    query_max_len (int): Default is: ``512``. The maximum total input sequence length after tokenization for passage. Sequences longer than this will be truncated, sequences shorter will be padded.
    passage_max_len (int): Default is: ``512``. The maximum total input sequence length after tokenization for passage. Sequences longer than this will be truncated, sequences shorter will be padded.
    pad_to_multiple_of (int): Default is: ``8``. If set will pad the sequence to be a multiple of the provided value.
    query_instruction_for_retrieval (str): Default is: ``Represent this sentence for searching relevant passages: ``. Instruction for query.
    query_instruction_format (str): Default is: ``{}{}``. Format for query instruction.
    learning_rate (float): Default is: ``1e-5``. Learning rate.
    num_train_epochs (int): Default is: ``1``. Total number of training epochs to perform.
    per_device_train_batch_size (int): Default is: ``2``. Train batch size
    gradient_accumulation_steps (int): Default is: ``1``. Number of updates steps to accumulate before performing a backward/update pass.
    dataloader_drop_last (bool): Default is: ``True``. When it='True', the last incomplete batch is dropped if the dataset size is not divisible by the batch size, meaning DataLoader only returns complete batches.
    warmup_ratio (float): Default is: ``0.1``. Warmup ratio for linear scheduler.
    weight_decay (float): Default is: ``0.01``. Weight decay in AdamW.
    deepspeed (str): Default is: ````. The path of the DeepSpeed configuration file, default to use the pre-made configuration file in the LazyLLM code repository: ``ds_stage0.json``.
    logging_steps (int): Default is: ``1``. Logging frequency according to logging strategy.
    save_steps (int): Default is: ``1000``. Saving frequency.
    temperature (float): Default is: ``0.02``. Temperature used for similarity score
    sentence_pooling_method (str): Default is: ``cls``. The pooling method. Available options: 'cls', 'mean', 'last_token'.
    normalize_embeddings (bool): Default is: ``True``. Whether to normalize the embeddings.
    kd_loss_type (str): Default is: ``kl_div``. The loss type for knowledge distillation. Available options:'kl_div', 'm3_kd_loss'.
    overwrite_output_dir (bool): Default is: ``True``. It is used to allow the program to overwrite an existing output directory.
    fp16 (bool): Default is: ``True``.  Whether to use fp16 (mixed) precision instead of 32-bit.
    gradient_checkpointing (bool): Default is: ``True``. Whether enable gradient checkpointing.
    negatives_cross_device (bool): Default is: ``True``. Whether share negatives across devices.

The keyword arguments and their default values for this class of reranker model are as follows:

Keyword Args:
    train_group_size (int): Default is: ``8``. The size of train group. It is used to control the number of negative samples in each training set.
    query_max_len (int): Default is: ``256``. The maximum total input sequence length after tokenization for passage. Sequences longer than this will be truncated, sequences shorter will be padded.
    passage_max_len (int): Default is: ``256``. The maximum total input sequence length after tokenization for passage. Sequences longer than this will be truncated, sequences shorter will be padded.
    pad_to_multiple_of (int): Default is: ``8``. If set will pad the sequence to be a multiple of the provided value.
    learning_rate (float): Default is: ``6e-5``. Learning rate.
    num_train_epochs (int): Default is: ``1``. Total number of training epochs to perform.
    per_device_train_batch_size (int): Default is: ``2``. Train batch size
    gradient_accumulation_steps (int): Default is: ``1``. Number of updates steps to accumulate before performing a backward/update pass.
    dataloader_drop_last (bool): Default is: ``True``. When it='True', the last incomplete batch is dropped if the dataset size is not divisible by the batch size, meaning DataLoader only returns complete batches.
    warmup_ratio (float): Default is: ``0.1``. Warmup ratio for linear scheduler.
    weight_decay (float): Default is: ``0.01``. Weight decay in AdamW.
    deepspeed (str): Default is: ````. The path of the DeepSpeed configuration file, default to use the pre-made configuration file in the LazyLLM code repository: ``ds_stage0.json``.
    logging_steps (int): Default is: ``1``. Logging frequency according to logging strategy.
    save_steps (int): Default is: ``1000``. Saving frequency.
    overwrite_output_dir (bool): Default is: ``True``. It is used to allow the program to overwrite an existing output directory.
    fp16 (bool): Default is: ``True``.  Whether to use fp16 (mixed) precision instead of 32-bit.
    gradient_checkpointing (bool): Default is: ``True``. Whether enable gradient checkpointing.
''')

add_example('finetune.FlagembeddingFinetune', '''\
>>> from lazyllm import finetune
>>> finetune.FlagembeddingFinetune('bge-m3', 'path/to/target')
<lazyllm.llm.finetune type=FlagembeddingFinetune>
''')

# Finetune-Auto
add_chinese_doc('auto.AutoFinetune', '''\
此类是 ``LazyLLMFinetuneBase`` 的子类，可根据输入的参数自动选择合适的微调框架和参数，以对大语言模型进行微调。

具体而言，基于输入的：``base_model`` 的模型参数、``ctx_len``、``batch_size``、``lora_r``、``launcher`` 中GPU的类型以及卡数，该类可以自动选择出合适的微调框架（如: ``AlpacaloraFinetune`` 或 ``CollieFinetune``）及所需的参数。

Args:
    base_model (str): 用于进行微调的基模型。要求是基模型的路径。
    source (lazyllm.config['model_source']): 指定模型的下载源。可通过设置环境变量 ``LAZYLLM_MODEL_SOURCE`` 来配置，目前仅支持 ``huggingface`` 或 ``modelscope`` 。若不设置，lazyllm不会启动自动模型下载。
    target_path (str): 微调后模型保存LoRA权重的路径。
    merge_path (str): 模型合并LoRA权重后的路径，默认为 ``None``。如果未指定，则会在 ``target_path`` 下创建 "lazyllm_lora" 和 "lazyllm_merge" 目录，分别作为 ``target_path`` 和  ``merge_path`` 。
    ctx_len (int): 输入微调模型的token最大长度，默认为 ``1024``。
    batch_size (int): 批处理大小，默认为 ``32``。
    lora_r (int): LoRA 的秩，默认为 ``8``；该数值决定添加参数的量，数值越小参数量越小。
    launcher (lazyllm.launcher): 微调的启动器，默认为 ``launchers.remote(ngpus=1)``。
    kw: 关键字参数，用于更新默认的训练参数。注意这里能够指定的关键字参数取决于 LazyLLM 推测出的框架，因此建议谨慎设置。
''')

add_english_doc('auto.AutoFinetune', '''\
This class is a subclass of ``LazyLLMFinetuneBase`` and can automatically select the appropriate fine-tuning framework and parameters based on the input arguments to fine-tune large language models.

Specifically, based on the input model parameters of ``base_model``, ``ctx_len``, ``batch_size``, ``lora_r``, the type and number of GPUs in ``launcher``, this class can automatically select the appropriate fine-tuning framework (such as: ``AlpacaloraFinetune`` or ``CollieFinetune``) and the required parameters.

Args:
    base_model (str): The base model used for fine-tuning. It is required to be the path of the base model.
    source (lazyllm.config['model_source']): Specifies the model download source. This can be configured by setting the environment variable ``LAZYLLM_MODEL_SOURCE``.
    target_path (str): The path where the LoRA weights of the fine-tuned model are saved.
    merge_path (str): The path where the model merges the LoRA weights, default to ``None``. If not specified, "lazyllm_lora" and "lazyllm_merge" directories will be created under ``target_path`` as ``target_path`` and ``merge_path`` respectively.
    ctx_len (int): The maximum token length for input to the fine-tuned model, default to ``1024``.
    batch_size (int): Batch size, default to ``32``.
    lora_r (int): LoRA rank, default to ``8``; this value determines the amount of parameters added, the smaller the value, the fewer the parameters.
    launcher (lazyllm.launcher): The launcher for fine-tuning, default to ``launchers.remote(ngpus=1)``.
    kw: Keyword arguments, used to update the default training parameters. Note that additional keyword arguments cannot be arbitrarily specified, as they depend on the framework inferred by LazyLLM, so it is recommended to set them with caution.
''')

add_example('auto.AutoFinetune', '''\
>>> from lazyllm import finetune
>>> finetune.auto("internlm2-chat-7b", 'path/to/target')
<lazyllm.llm.finetune type=AlpacaloraFinetune>
''')


# Finetune-EasyR1Finetune
add_chinese_doc('finetune.EasyR1Finetune', '''\
此类是 ``LazyLLMFinetuneBase`` 的子类，基于 [EasyR1](https://github.com/hiyouga/EasyR1) 框架提供的强化学习的能力，用于对大语言模型进行强化学习训练。

Args:
    base_model: 用于进行训练的基模型路径。支持本地路径，若路径不存在则尝试从配置的模型路径中查找。
    target_path: 训练完成后，模型权重保存的目标路径。
    merge_path (str, optional): 同 ``target_path``，不需要配置。
    launcher (lazyllm.launcher, optional): 训练任务的启动器，默认为``None``, 使用单卡同步远程启动器 ``launchers.remote(ngpus=1, sync=True)``。
    **kw: 关键字参数，用于动态覆盖默认训练配置中的参数。

此类的关键字参数及其默认值如下：

Keyword Args:
    data.max_prompt_length (int): 默认值是：``2048``。用于限定输入 prompt 的最大 token 长度，超出会根据实现截断或丢弃多余 tokens。
    data.max_response_length (int): 默认值是：``2048``。模型生成时允许的最大响应长度（token 数）。
    data.rollout_batch_size (int): 默认值是：``128``。rollout（策略采样/生成）时的 batch 大小，增大可提高吞吐但占用更多显存。
    data.val_batch_size (int): 默认值是：``1024``。验证/评估阶段的 batch 大小，通常可设大以提高评估效率，但受显存限制。
    data.format_prompt (typing.Optional[typing.Union[str, callable]]): 默认值是：``None``。用于将原始样本格式化为模型输入 prompt 的模板或函数。为 ``None`` 时使用框架/数据集默认格式化逻辑。
    worker.actor.global_batch_size (int): 默认值是：``128``。actor（用于生成与训练的组件）的一次更新对应的全局 batch 大小。
    worker.actor.micro_batch_size_per_device_for_update (int): 默认值是：``4``。训练更新（反向传播）阶段每个设备上的微批大小，用于计算梯度累积步数。
    worker.actor.micro_batch_size_per_device_for_experience (int): 默认值是：``16``。生成经验（rollout）阶段每个设备上的微批大小，通常大于更新阶段以提高采样吞吐。
    worker.rollout.gpu_memory_utilization (float): 默认值是：``0.6``。控制 rollout 阶段每张 GPU 可用显存比例，框架会据此估算可用 batch/并行度以避免 OOM。
    worker.rollout.tensor_parallel_size (int): 默认值是：``1``。rollout/采样阶段使用的 tensor 并行度大小，>1 时启用张量并行以分摊显存与计算。
    worker.reward.reward_function (typing.Optional[typing.Union[str, callable]]): 默认值是：``None``。用于计算 reward 的函数或可识别标识。函数原型通常为 func(samples) -> rewards。自定义 reward 函数须高效且可序列化（或可在子进程/远程环境中调用），以免成为训练瓶颈。
    trainer.total_epochs (int): 默认值是：``2``。训练的总轮次。对于强化学习微调场景通常不需很大轮次，可通过 rollout 次数与 batch 调整训练强度。
    trainer.n_gpus_per_node (int): 默认值是：``1``。每个节点上用于训练的 GPU 数量。
    trainer.save_freq (int): 默认值是：``5``。以 epoch 为单位的 checkpoint 保存频率。设置为 0 或负值的行为由实现决定（可能只在结束时保存）。
    trainer.save_checkpoint_path (typing.Optional[str]): 默认值是：``None``。指定 checkpoint 的保存路径，若 ``None`` 则使用 ``target_path`` 或框架默认路径。
    trainer.save_model_only (bool): 默认值是：``False``。是否仅保存模型权重而不保存优化器/调度器等训练状态。
''')

add_english_doc('finetune.EasyR1Finetune', '''\
This class is a subclass of ``LazyLLMFinetuneBase`` and provides reinforcement learning training capabilities for large language models using the EasyR1 framework (https://github.com/hiyouga/EasyR1).

Args:
    base_model (str): Path to the base model used for training. A local path is supported; if the path does not exist it will try to locate the model from configured model locations.
    target_path (str): Destination path where the trained model weights will be saved.
    merge_path (str, optional): Same purpose as ``target_path``; not required in most cases.
    launcher (lazyllm.launcher, optional): Launcher used to start the training job. Defaults to ``None``, in which case a single-GPU synchronous remote launcher ``launchers.remote(ngpus=1, sync=True)`` is used.
    **kw: Additional keyword arguments used to dynamically override parameters in the default training configuration.

The following keyword arguments and their default values are supported:

Keyword Args:
    data.max_prompt_length (int): Default: ``2048``. Maximum number of tokens allowed for the input prompt. Inputs longer than this will be truncated or otherwise handled according to the implementation.
    data.max_response_length (int): Default: ``2048``. Maximum number of tokens the model is allowed to generate as a response.
    data.rollout_batch_size (int): Default: ``128``. Batch size used during rollouts (policy sampling/generation). Increasing this can improve throughput but will use more GPU memory.
    data.val_batch_size (int): Default: ``1024``. Batch size used during validation/evaluation. This can typically be set large to speed up evaluation, subject to memory limits.
    data.format_prompt (typing.Optional[typing.Union[str, callable]]): Default: ``None``. Template string or callable used to format raw examples into model prompts. If ``None``, the framework or dataset default formatting is used.
    worker.actor.global_batch_size (int): Default: ``128``. Global batch size per update for actors (components responsible for generation and interaction with the environment).
    worker.actor.micro_batch_size_per_device_for_update (int): Default: ``4``. Micro-batch size per device for the optimization/update phase; used to determine gradient accumulation steps.
    worker.actor.micro_batch_size_per_device_for_experience (int): Default: ``16``. Micro-batch size per device during experience generation (rollout); typically larger than the update micro-batch to increase sampling throughput.
    worker.rollout.gpu_memory_utilization (float): Default: ``0.6``. Fraction of each GPU's memory available for rollout. The framework uses this to estimate safe batch sizes/parallelism to avoid OOM.
    worker.rollout.tensor_parallel_size (int): Default: ``1``. Tensor parallelism degree during rollout/sampling. Values >1 enable tensor parallelism to share memory and computation across devices.
    worker.reward.reward_function (typing.Optional[typing.Union[str, callable]]): Default: ``None``. A function or identifiable name used to compute rewards. Typical signature is ``func(samples) -> rewards``. Custom reward functions should be efficient and serializable (or callable in worker/subprocess/remote environments) to avoid becoming a training bottleneck.
    trainer.total_epochs (int): Default: ``2``. Total number of training epochs. In RL fine-tuning scenarios this usually does not need to be large; training intensity is commonly adjusted via number of rollouts and batch sizes.
    trainer.n_gpus_per_node (int): Default: ``1``. Number of GPUs used per node for training.
    trainer.save_freq (int): Default: ``5``. Checkpoint save frequency measured in epochs. Behavior for 0 or negative values is implementation-defined (for example, only saving at the end).
    trainer.save_checkpoint_path (typing.Optional[str]): Default: ``None``. Path to save checkpoints. If ``None``, ``target_path`` or the framework default path is used.
    trainer.save_model_only (bool): Default: ``False``. Whether to save only model weights (``True``) instead of saving optimizer/scheduler and other training state.
''')

add_example('finetune.EasyR1Finetune', '''\
>>> from lazyllm import finetune
>>> finetune.easyr1('qwen2-0.5b-instruct', 'path/to/target')
<lazyllm.llm.finetune type=EasyR1Finetune>
''')

add_chinese_doc('finetune.EasyR1Finetune.cmd', """\
生成EasyR1训练命令序列。

Args:
    trainset (str): 训练数据集路径(支持相对lazyllm.config['data_path']的路径)
    valset (str, optional): 验证数据集路径

**Returns:**\n
- str: 完整的shell命令字符串，包含:
    - 训练命令(自动配置参数)
    - 日志重定向(保存到目标路径)
""")

add_english_doc('finetune.EasyR1Finetune.cmd', """\
Generate EasyR1 fine-tuning command sequence.
Args:
    trainset (str): Training dataset path (supports relative path to lazyllm.config['data_path'])
    valset (str, optional): Validation dataset path

**Returns:**\n
- str: Complete shell command string containing:
    - Training command (with auto-configured parameters)
    - Log redirection (saved to target path)
""")

# ============= Deploy

add_chinese_doc('LazyLLMDeployBase', '''\
此类是 ``ComponentBase`` 的一个子类，提供了LazyLLM部署的基础功能。它支持多种媒体类型的编码转换，并提供了结果提取和流式处理的配置选项。

Args:
    launcher (LauncherBase): 用于部署的启动器实例，默认为远程启动器(``launchers.remote()``)。

注意事项: 
    - 继承此类时需要实现具体的部署逻辑
    - 可以通过重写extract_result方法来自定义结果提取逻辑
''')

add_english_doc('LazyLLMDeployBase', '''\
This class is a subclass of ``ComponentBase`` that provides basic functionality for LazyLLM deployment. It supports encoding conversion for various media types and provides configuration options for result extraction and streaming processing.

Args:
    launcher (LauncherBase): Launcher instance for deployment, defaults to remote launcher (``launchers.remote()``).

Notes: 
    - Need to implement specific deployment logic when inheriting this class
    - Can customize result extraction logic by overriding the extract_result method
''')

add_example('LazyLLMDeployBase', '''\
>>> import lazyllm
>>> from lazyllm.components.deploy.base import LazyLLMDeployBase
>>> class MyDeployer(LazyLLMDeployBase):
...     def __call__(self, inputs):
...         return processed_result
        def extract_result(output, inputs):
...         return output.json()['result']
>>> deployer = MyDeployer()
>>> result = deployer.extract_result(raw_output, input_data)
''')

add_chinese_doc('LazyLLMDeployBase.extract_result', """\
从模型输出中提取最终结果，默认实现直接返回原始输出，子类可重写此方法实现自定义结果提取逻辑。

Args:
    output: 模型原始输出
    inputs: 原始输入数据，可用于结果后处理

**Returns:**\n
- 处理后的最终结果
""")

add_english_doc('LazyLLMDeployBase.extract_result', """\
Extract final result from model output. The default implementation returns raw output directly, subclasses can override this method to implement custom result extraction logic.

Args:
    output: Raw model output
    inputs: Original input data, can be used for post-processing

**Returns:**\n
- Processed final result
""")


# Deploy-AbstractEmbedding
add_chinese_doc('deploy.embed.AbstractEmbedding', '''\
抽象嵌入基类，为所有嵌入模型提供统一的接口和基础功能。此类定义了嵌入模型的标准接口，包括模型加载、调用和序列化等功能。

Args:
    base_embed (str): 嵌入模型的基础路径或标识符，用于指定要加载的嵌入模型。
    source (str, optional): 模型来源，默认为 ``None``。如果未指定，将使用 LazyLLM 配置中的默认模型来源。
    init (bool): 是否在初始化时立即加载模型，默认为 ``False``。如果为 ``True``，将在对象创建时立即调用 ``load_embed()`` 方法。
''')

add_english_doc('deploy.embed.AbstractEmbedding', '''\
Abstract embedding base class that provides unified interface and basic functionality for all embedding models. This class defines the standard interface for embedding models, including model loading, calling, and serialization capabilities.

Args:
    base_embed (str): The base path or identifier of the embedding model, used to specify which embedding model to load.
    source (str, optional): Model source, default to ``None``. If not specified, will use the default model source from LazyLLM configuration.
    init (bool): Whether to load the model immediately during initialization, default to ``False``. If ``True``, will call the ``load_embed()`` method immediately when the object is created.
''')

add_chinese_doc('deploy.embed.AbstractEmbedding.load_embed', '''\
加载嵌入模型的抽象方法。此方法由子类实现，用于执行具体的模型加载逻辑。
''')

add_english_doc('deploy.embed.AbstractEmbedding.load_embed', '''\
Abstract method for loading embedding models. This method is implemented by subclasses to perform specific model loading logic.

**Note**: This method is currently under development.
''')

add_chinese_doc('deploy.embed.HuggingFaceEmbedding', '''\
HuggingFace嵌入模型管理类，用于管理和注册不同的嵌入模型实现。

属性：
    _model_id_mapping (dict): 模型ID到具体实现类的映射字典。

Args:
    base_embed (str): 基础嵌入模型的路径或名称。
    source (Optional[str]): 模型来源，默认为None。
''')

add_english_doc('deploy.embed.HuggingFaceEmbedding', '''\
HuggingFace embedding model management class for managing and registering different embedding model implementations.

Attributes:
    _model_id_mapping (dict): Mapping dictionary from model IDs to implementation classes.

Args:
    base_embed (str): Path or name of the base embedding model.
    source (Optional[str]): Model source, defaults to None.
''')

add_chinese_doc('deploy.embed.HuggingFaceEmbedding.get_emb_cls', '''\
获取模型对应的嵌入实现类。

Args:
    model_name (str): 模型名称或路径。

**Returns:**\n
- type: 返回对应的嵌入模型实现类，如果未找到则返回默认实现LazyHuggingFaceDefaultEmbedding。
''')

add_english_doc('deploy.embed.HuggingFaceEmbedding.get_emb_cls', '''\
Get the embedding implementation class for a model.

Args:
    model_name (str): Model name or path.

**Returns:**\n
- type: Returns corresponding embedding model implementation class, defaults to LazyHuggingFaceDefaultEmbedding if not found.
''')

add_chinese_doc('deploy.embed.HuggingFaceEmbedding.register', '''\
注册模型ID到特定实现类的装饰器。

Args:
    model_ids (List[str]): 要注册的模型ID列表。

**Returns:**\n
- Callable: 返回装饰器函数。
''')

add_english_doc('deploy.embed.HuggingFaceEmbedding.register', '''\
Decorator for registering model IDs to specific implementation classes.

Args:
    model_ids (List[str]): List of model IDs to register.

**Returns:**\n
- Callable: Returns decorator function.
''')

add_chinese_doc('deploy.embed.HuggingFaceEmbedding.load_embed', '''\
加载嵌入模型。

该方法会调用内部嵌入实现类的load_embed方法来加载模型。
''')

add_english_doc('deploy.embed.HuggingFaceEmbedding.load_embed', '''\
Load the embedding model.

This method calls the load_embed method of the internal embedding implementation class to load the model.
''')

# Deploy-Lightllm
add_chinese_doc('deploy.Lightllm', '''\
此类是 ``LazyLLMDeployBase`` 的子类，基于 [LightLLM](https://github.com/ModelTC/lightllm) 框架提供的推理能力，用于对大语言模型进行推理。

Args:
    trust_remote_code (bool, optional): 是否信任远程代码，默认为True
    launcher (Launcher, optional): 任务启动器，默认为单GPU远程启动器
    log_path (str, optional): 日志文件路径，默认为None
    **kw: 其他LightLLM服务器配置参数
此类的关键字参数及其默认值如下：

Keyword Args: 
    tp (int): 张量并行参数，默认为 ``1``。
    max_total_token_num (int): 最大总token数，默认为 ``64000``。
    eos_id (int): 结束符ID，默认为 ``2``。
    port (int): 服务的端口号，默认为 ``None``。此情况下LazyLLM会自动生成随机端口号。
    host (str): 服务的IP地址，默认为 ``0.0.0.0``。
    nccl_port (int): NCCL 端口，默认为 ``None``。此情况下LazyLLM会自动生成随机端口号。
    tokenizer_mode (str): tokenizer的加载模式，默认为 ``auto``。
    running_max_req_size (int): 推理引擎最大的并行请求数， 默认为 ``256``。
    data_type (str): 模型权重的数据类型，默认为 ``float16``。
    max_req_total_len (int): 请求的最大总长度，默认为 ``64000``。
    max_req_input_len (int): 输入的最大长度，默认为 ``4096``。
    long_truncation_mode (str): 长文本的截断模式，默认为 ``head``。
''')

add_english_doc('deploy.Lightllm', '''\
This class is a subclass of ``LazyLLMDeployBase``, based on the inference capabilities provided by the [LightLLM](https://github.com/ModelTC/lightllm) framework, used for inference with large language models.

Args:
    trust_remote_code (bool, optional): Whether to trust remote code, defaults to True
    launcher (Launcher, optional): Task launcher, defaults to single GPU remote launcher
    log_path (str, optional): Log file path, defaults to None
    **kw: Other LightLLM server configuration parameters
The keyword arguments and their default values for this class are as follows:

Keyword Args: 
    tp (int): Tensor parallelism parameter, default is ``1``.
    max_total_token_num (int): Maximum total token number, default is ``64000``.
    eos_id (int): End-of-sentence ID, default is ``2``.
    port (int): Service port number, default is ``None``, in which case LazyLLM will automatically generate a random port number.
    host (str): Service IP address, default is ``0.0.0.0``.
    nccl_port (int): NCCL port, default is ``None``, in which case LazyLLM will automatically generate a random port number.
    tokenizer_mode (str): Tokenizer loading mode, default is ``auto``.
    running_max_req_size (int): Maximum number of parallel requests for the inference engine, default is ``256``.
    data_type (str): Data type for model weights, default is ``float16``.
    max_req_total_len (int): Maximum total length for requests, default is ``64000``.
    max_req_input_len (int): Maximum input length, default is ``4096``.
    long_truncation_mode (str): Truncation mode for long texts, default is ``head``.
''')

add_example('deploy.Lightllm', '''\
>>> from lazyllm import deploy
>>> infer = deploy.lightllm()
''')

add_chinese_doc('deploy.Lightllm.cmd', '''\
该方法用于生成启动LightLLM服务的命令。

Args:
    finetuned_model (str): 微调后的模型路径。
    base_model (str): 基础模型路径，当finetuned_model无效时使用。

**Returns:**\n
- LazyLLMCMD: 一个包含启动命令的LazyLLMCMD对象。
''')

add_english_doc('deploy.Lightllm.cmd', '''\
This method generates the command to start the LightLLM service.

Args:
    finetuned_model (str): Path to the fine-tuned model.
    base_model (str): Path to the base model, used when finetuned_model is invalid.

**Returns:**\n
- LazyLLMCMD: A LazyLLMCMD object containing the startup command.
''')

add_chinese_doc('deploy.Lightllm.geturl', '''\
获取LightLLM服务的URL地址。

Args:
    job (optional): 任务对象，默认为None，此时使用self.job。

**Returns:**\n
- str: 服务的URL地址，格式为"http://{ip}:{port}/generate"。
''')

add_english_doc('deploy.Lightllm.geturl', '''\
Get the URL address of the LightLLM service.

Args:
    job (optional): Job object, defaults to None, in which case self.job is used.

**Returns:**\n
- str: The service URL address in the format "http://{ip}:{port}/generate".
''')

add_chinese_doc('deploy.Lightllm.extract_result', '''\
从服务响应中提取生成的文本结果。

Args:
    x (str): 服务返回的响应文本。
    inputs (str): 输入文本。

**Returns:**\n
- str: 提取出的生成文本。

异常:
    Exception: 当解析JSON响应失败时抛出异常。
''')

add_english_doc('deploy.Lightllm.extract_result', '''\
Extract generated text from the service response.

Args:
    x (str): Response text from the service.
    inputs (str): Input text.

**Returns:**\n
- str: The extracted generated text.

Raises:
    Exception: When JSON response parsing fails.
''')

# Deploy-Vllm
add_chinese_doc('deploy.Vllm', '''\
此类是 ``LazyLLMDeployBase`` 的子类，基于 [VLLM](https://github.com/vllm-project/vllm) 框架提供的推理能力，用于大语言模型的部署与推理。

Args:
    trust_remote_code (bool): 是否允许加载来自远程服务器的模型代码，默认为 ``True``。
    launcher (lazyllm.launcher): 模型启动器，默认为 ``launchers.remote(ngpus=1)``。
    log_path (str): 日志保存路径，若为 ``None`` 则不保存日志。
    openai_api (bool): 是否使用 OpenAI API 接口启动 VLLM 服务，默认为 ``False``。
    kw: 关键字参数，用于更新默认的部署参数。除支持的关键字参数外，不允许传入额外参数。

此类支持的关键字参数及其默认值如下：

Keyword Args: 
    tensor-parallel-size (int): 张量并行大小，默认为 ``1``。
    dtype (str): 模型权重和激活值的数据类型，默认为 ``auto``。可选：``half``、``float16``、``bfloat16``、``float``、``float32``。
    kv-cache-dtype (str): KV 缓存的数据类型，默认为 ``auto``。可选：``fp8``、``fp8_e5m2``、``fp8_e4m3``。
    device (str): VLLM 支持的硬件类型，默认为 ``auto``。可选：``cuda``、``neuron``、``cpu``。
    block-size (int): token 块大小，默认为 ``16``。
    port (int | str): 服务端口号，默认为 ``auto``，即随机分配。
    host (str): 服务绑定的 IP 地址，默认为 ``0.0.0.0``。
    seed (int): 随机数种子，默认为 ``0``。
    tokenizer_mode (str): Tokenizer 加载模式，默认为 ``auto``。
    max-num-seqs (int): 推理引擎支持的最大并行请求数，默认为 ``256``。
    pipeline-parallel-size (int): 流水线并行大小，默认为 ``1``。
    max-num-batched-tokens (int): 最大批处理 token 数，默认为 ``64000``。
''')

add_english_doc('deploy.Vllm', '''\
This class is a subclass of ``LazyLLMDeployBase``, leveraging the [VLLM](https://github.com/vllm-project/vllm) framework to deploy and run inference on large language models.

Args:
    trust_remote_code (bool): Whether to allow loading of model code from remote sources. Default is ``True``.
    launcher (lazyllm.launcher): The launcher used to start the model. Default is ``launchers.remote(ngpus=1)``.
    log_path (str): Path to store logs. If ``None``, logs will not be saved.
    openai_api (bool): Whether to start VLLM with OpenAI-compatible API. Default is ``False``.
    kw: Keyword arguments used to override default deployment parameters. No extra arguments beyond the supported ones are allowed.

The supported keyword arguments and their default values are as follows:

Keyword Args: 
    tensor-parallel-size (int): Tensor parallelism size. Default is ``1``.
    dtype (str): Data type for model weights and activations. Default is ``auto``. Options include: ``half``, ``float16``, ``bfloat16``, ``float``, ``float32``.
    kv-cache-dtype (str): Data type for KV cache. Default is ``auto``. Options include: ``fp8``, ``fp8_e5m2``, ``fp8_e4m3``.
    device (str): Backend device type supported by VLLM. Default is ``auto``. Options include: ``cuda``, ``neuron``, ``cpu``.
    block-size (int): Token block size. Default is ``16``.
    port (int | str): Service port number. Default is ``auto`` (random assignment).
    host (str): Service binding IP address. Default is ``0.0.0.0``.
    seed (int): Random seed. Default is ``0``.
    tokenizer_mode (str): Tokenizer loading mode. Default is ``auto``.
    max-num-seqs (int): Maximum number of concurrent requests supported by the inference engine. Default is ``256``.
    pipeline-parallel-size (int): Pipeline parallelism size. Default is ``1``.
    max-num-batched-tokens (int): Maximum number of batched tokens. Default is ``64000``.
''')

add_example('deploy.Vllm', '''\
>>> from lazyllm import deploy
>>> infer = deploy.vllm()
''')

add_chinese_doc('deploy.Vllm.cmd', '''\
构造用于启动 vLLM 推理服务的命令。

该方法会自动检测模型路径是否有效，并根据当前配置参数动态生成可执行命令，支持多节点部署时自动加入 ray 启动命令。

Args:
    finetuned_model (str): 微调后的模型路径。
    base_model (str): 备用基础模型路径（当 finetuned_model 无效时启用）。
    master_ip (str): 分布式部署中的主节点 IP，仅在多节点时启用。

**Returns:**\n
- LazyLLMCMD: 可执行命令对象，包含启动指令、结果回调函数及健康检查方法。
''')

add_english_doc('deploy.Vllm.cmd', '''\
Build the command to launch the vLLM inference service.

This method validates the model path and constructs an executable command string based on current configuration. In distributed mode, it will also prepend the ray cluster start command.

Args:
    finetuned_model (str): Path to the fine-tuned model.
    base_model (str): Fallback base model path if finetuned_model is invalid.
    master_ip (str): IP address of the master node in a distributed setup.

**Returns:**\n
- LazyLLMCMD: The command object with shell instruction, return value handler, and health checker.
''')

add_chinese_doc('deploy.Vllm.geturl', '''\
获取 vLLM 服务的推理地址。

根据运行模式（Display 模式或实际部署）返回相应的 URL，用于访问模型的生成接口。

Args:
    job (Job, optional): 部署任务对象。默认取当前模块绑定的 job。

**Returns:**\n
- str: 推理服务的 HTTP 地址。
''')

add_english_doc('deploy.Vllm.geturl', '''\
Get the inference service URL for the vLLM deployment.

Depending on the execution mode (Display or actual deployment), this method returns the appropriate URL for accessing the model's generate endpoint.

Args:
    job (Job, optional): Deployment job object. Defaults to the module's associated job.

**Returns:**\n
- str: The HTTP URL for inference service.
''')

add_chinese_doc('deploy.Vllm.extract_result', '''\
从 VLLM 接口返回的 JSON 字符串中提取推理结果。

Args:
    x (str): VLLM 服务返回的原始 JSON 字符串。
    inputs (Any): 输入参数（此处未使用，保留接口一致性）。

**Returns:**\n
- str: 模型生成的文本结果。
''')

add_english_doc('deploy.Vllm.extract_result', '''\
Extracts the inference result from the JSON string returned by the VLLM service.

Args:
    x (str): Raw JSON string returned from the VLLM service.
    inputs (Any): Input arguments (not used here, kept for interface consistency).

**Returns:**\n
- str: The generated text result from the model.
''')

# Deploy-EmbeddingDeploy
add_chinese_doc('deploy.EmbeddingDeploy', '''\
此类是 ``LazyLLMDeployBase`` 的子类，用于部署文本嵌入（Embedding）服务。支持稠密向量（dense）和稀疏向量（sparse）两种嵌入方式，可使用 HuggingFace 模型或 FlagEmbedding 模型。

Args:
    launcher (Optional[lazyllm.launcher]): 启动器实例，默认为 ``None``。
    model_type (Optional[str]): 模型类型，默认为 ``'embed'``。
    log_path (Optional[str]): 日志文件路径，默认为 ``None``。
    embed_type (Optional[str]): 嵌入类型，可选 ``'dense'`` 或 ``'sparse'``，默认为 ``'dense'``。
    trust_remote_code (bool): 是否信任远程代码，默认为 ``True``。
    port (Optional[int]): 服务端口号，默认为 ``None``，此情况下 LazyLLM 会自动生成随机端口号。

Call Arguments:
    finetuned_model (Optional[str]): 微调后的模型路径或名称。\\n
    base_model (Optional[str]): 基础模型路径或名称，当 finetuned_model 无效时会使用此模型。\\n

Message Format:
    输入格式为包含 text（文本）和 images（图像列表）的字典。\\n
    - text (str): 需要编码的文本内容 \\n
    - images (Union[str, List[str]]): 需要编码的图像列表，可选 \\n
''')

add_english_doc('deploy.EmbeddingDeploy', '''\
This class is a subclass of ``LazyLLMDeployBase``, designed for deploying text embedding services. It supports both dense and sparse embedding methods, compatible with HuggingFace models and FlagEmbedding models.

Args:
    launcher (Optional[lazyllm.launcher]): The launcher instance, defaults to ``None``.
    model_type (Optional[str]): Model type, defaults to ``'embed'``.
    log_path (Optional[str]): Path for log file, defaults to ``None``.
    embed_type (Optional[str]): Embedding type, either ``'dense'`` or ``'sparse'``, defaults to ``'dense'``.
    trust_remote_code (bool): Whether to trust remote code, defaults to ``True``.
    port (Optional[int]): Service port number, defaults to ``None``, in which case LazyLLM will generate a random port.

Call Arguments:
    finetuned_model (Optional[str]): Path or name of the fine-tuned model. \n
    base_model (Optional[str]): Path or name of the base model, used when finetuned_model is invalid. \n

Message Format:
    Input format is a dictionary containing text and images list.\n
    - text (str): Text content to be encoded\n
    - images (Union[str, List[str]]): List of images to be encoded (optional)\n
''')

add_example('deploy.EmbeddingDeploy', '''\
>>> from lazyllm import deploy
>>> embed_service = deploy.EmbeddingDeploy(embed_type='dense')
>>> embed_service('path/to/model')
''')

# Deploy-RerankDeploy
add_chinese_doc('deploy.embed.RerankDeploy', '''\
此类是 ``EmbeddingDeploy`` 的子类，用于部署重排序（Rerank）服务。支持使用HuggingFace模型进行文本重排序。

Args:
    launcher (lazyllm.launcher): 启动器，默认为 ``None``。
    model_type (str): 模型类型，默认为 ``'embed'``。
    log_path (str): 日志文件路径，默认为 ``None``。
    trust_remote_code (bool): 是否信任远程代码，默认为 ``True``。
    port (int): 服务端口号，默认为 ``None``，此情况下LazyLLM会自动生成随机端口号。

Call Arguments:
    finetuned_model: 微调后的模型路径或模型名称。\n
    base_model: 基础模型路径或模型名称，当finetuned_model无效时会使用此模型。\n

Message Format:
    输入格式为包含query（查询文本）、documents（候选文档列表）和top_n（返回的文档数量）的字典。\n
    - query: 查询文本\n
    - documents: 候选文档列表\n
    - top_n: 返回的文档数量，默认为1\n
''')

add_english_doc('deploy.embed.RerankDeploy', '''\
This class is a subclass of ``EmbeddingDeploy``, designed for deploying reranking services. It supports text reranking using HuggingFace models.

Args:
    launcher (lazyllm.launcher): The launcher instance, defaults to ``None``.
    model_type (str): Model type, defaults to ``'embed'``.
    log_path (str): Path for log file, defaults to ``None``.
    trust_remote_code (bool): Whether to trust remote code, defaults to ``True``.
    port (int): Service port number, defaults to ``None``, in which case LazyLLM will generate a random port.

Call Arguments:
    finetuned_model: Path or name of the fine-tuned model. \n
    base_model: Path or name of the base model, used when finetuned_model is invalid.\n

Message Format:
    Input format is a dictionary containing query (query text), documents (list of candidate documents), and top_n (number of documents to return).\\n
    - query: Query text \n
    - documents: List of candidate documents \n
    - top_n: Number of documents to return, defaults to 1 \n
''')

add_example('deploy.embed.RerankDeploy', '''\
>>> from lazyllm import deploy
>>> rerank_service = deploy.embed.RerankDeploy()
>>> rerank_service('path/to/model')
>>> input_data = {
...     "query": "What is machine learning?",
...     "documents": [
...         "Machine learning is a branch of AI.",
...         "Machine learning uses data to improve.",
...         "Deep learning is a subset of machine learning."
...     ],
...     "top_n": 2
... }
>>> result = rerank_service(input_data)
''')

# Deploy-embed
add_chinese_doc('deploy.embed.LazyHuggingFaceRerank', '''\
基于 HuggingFace CrossEncoder 的重排序（Rerank）封装类。  
用于根据查询与候选文档的相关性分数，对文档进行排序。  
支持在初始化时下载并加载指定的重排序模型，并可选择延迟加载以提升启动性能。

Args:
    base_rerank (str): 重排序模型名称或本地路径。支持 HuggingFace Hub 模型标识符或本地路径。
    source (Optional[str]): 模型来源，支持 `huggingface` 和 `modelscope`，默认为全局配置项 `model_source`。
    init (bool): 是否在实例化时立即加载模型。若为 `False`，将在首次调用时延迟加载。
''')

add_english_doc('deploy.embed.LazyHuggingFaceRerank', '''\
Wrapper class for HuggingFace CrossEncoder-based reranking.  
Ranks candidate documents by relevance score with respect to a given query.  
Supports downloading and loading a specified rerank model at initialization, with optional lazy loading for faster startup.

Args:
    base_rerank (str): Name or local path of the rerank model. Supports HuggingFace Hub identifiers or local paths.
    source (Optional[str]): Source of the model, supports `huggingface` and `modelscope`. Defaults to global config `model_source`.
    init (bool): Whether to load the model immediately upon instantiation. If `False`, the model will be loaded lazily on first call.
''')

add_chinese_doc('deploy.embed.LazyHuggingFaceRerank.load_reranker', '''\
加载重排序模型。  

该方法会基于 `self.base_rerank` 初始化一个 `sentence_transformers.CrossEncoder` 实例，  
并赋值给类属性 `self.reranker`，用于后续的重排序任务。  
''')

add_english_doc('deploy.embed.LazyHuggingFaceRerank.load_reranker', '''\
Load the rerank model.  

This method initializes a `sentence_transformers.CrossEncoder` instance using `self.base_rerank`  
and assigns it to the class attribute `self.reranker` for subsequent reranking tasks.  
''')

add_chinese_doc('deploy.embed.LazyHuggingFaceRerank.rebuild', '''\
重建 `LazyHuggingFaceRerank` 实例的类方法。  
主要用于序列化（pickle/cloudpickle）时的反序列化过程，根据提供的参数重新实例化对象。

Args:
    base_rerank (str): 模型名称或路径。
    init (bool): 是否在重建时立即加载模型。

**Returns:**\n
- LazyHuggingFaceRerank: 重新构建的类实例。
''')

add_english_doc('deploy.embed.LazyHuggingFaceRerank.rebuild', '''\
Class method to rebuild a `LazyHuggingFaceRerank` instance.  
Used primarily for deserialization during pickle/cloudpickle operations,  
reinstantiating the object with the provided parameters.

Args:
    base_rerank (str): Model name or path.
    init (bool): Whether to load the model immediately upon rebuilding.

**Returns:**\n
- LazyHuggingFaceRerank: The rebuilt class instance.
''')

add_chinese_doc('deploy.embed.LazyFlagEmbedding', '''\
支持懒加载的 FlagEmbedding 嵌入模块封装。

该类包装了 FlagEmbedding 的加载和调用逻辑，提供对稀疏和稠密嵌入的支持，并通过 lazyllm.once_flag() 机制实现懒加载。适用于嵌入模型的本地/远程下载、初始化与编码流程的封装，便于与 LazyLLM 系统集成。

Args:
    base_embed (str): 嵌入模型名称或路径。
    sparse (bool): 是否使用稀疏嵌入模式，默认为 False。
    source (str, optional): 模型下载源，若未提供则使用 lazyllm 全局配置。
    init (bool): 是否在初始化时立即加载模型，默认为 False。
''')

add_english_doc('deploy.embed.LazyFlagEmbedding', '''\
A lazily loaded wrapper for the FlagEmbedding module.

This class encapsulates loading and usage of FlagEmbedding, with support for both sparse and dense embeddings. It leverages the lazyllm.once_flag() mechanism to initialize only once on demand, and integrates with LazyLLM's model downloading utilities.

Args:
    base_embed (str): The model name or path to be used as the embedding backend.
    sparse (bool): Whether to enable sparse embedding output. Defaults to False.
    source (str, optional): Source URL or identifier for model downloading. Defaults to global config.
    init (bool): Whether to initialize the model immediately upon construction. Defaults to False.
''')

add_chinese_doc('deploy.embed.LazyFlagEmbedding.load_embed', '''\
加载嵌入模型并初始化到设备上。

该方法根据系统是否支持 CUDA 自动选择运行设备（GPU 或 CPU），并从本地或远程加载预训练的 FlagEmbedding 模型。
''')

add_english_doc('deploy.embed.LazyFlagEmbedding.load_embed', '''\
Load the embedding model onto the appropriate device.

This method selects the available device (GPU or CPU) and initializes the pretrained FlagEmbedding model from the provided path or model hub.
''')

add_chinese_doc('deploy.embed.LazyFlagEmbedding.rebuild', '''\
重建 LazyFlagEmbedding 实例的方法。

该类方法用于在序列化或跨进程传递时，重新构造带有初始化配置的 LazyFlagEmbedding 实例。

Args:
    base_embed (str): 嵌入模型的路径或模型名称。
    sparse (bool): 是否启用稀疏嵌入。
    init (bool): 是否在构造时立即加载模型。

**Returns:**\n
- LazyFlagEmbedding: 一个新的 LazyFlagEmbedding 实例。
''')

add_english_doc('deploy.embed.LazyFlagEmbedding.rebuild', '''\
Rebuild a LazyFlagEmbedding instance.

This class method reconstructs an instance of LazyFlagEmbedding, typically used during deserialization or multiprocessing scenarios.

Args:
    base_embed (str): The path or name of the embedding model.
    sparse (bool): Whether to enable sparse embedding mode.
    init (bool): Whether to load the model immediately during instantiation.

**Returns:**\n
- LazyFlagEmbedding: A newly constructed LazyFlagEmbedding instance.
''')

# Deploy-Mindie
add_chinese_doc('deploy.Mindie', '''\
此类是 ``LazyLLMDeployBase`` 的一个子类, 用于部署和管理MindIE大模型推理服务。它封装了MindIE服务的配置生成、进程启动和API交互的全流程。

Args:
    trust_remote_code (bool): 是否信任远程代码(如HuggingFace模型)。默认为 ``True``。
    launcher: 任务启动器实例，默认为 ``launchers.remote()``。
    log_path (str): 日志保存路径，若为 ``None`` 则不保存日志。
    **kw: 其他配置参数

Keyword Args: 
            npuDeviceIds: NPU设备ID列表(如 ``[[0,1]]`` 表示使用2张卡)
            worldSize: 模型并行数量
            port: 服务端口（设为 ``'auto'`` 时自动分配30000-40000的随机端口)
            maxSeqLen: 最大序列长度
            maxInputTokenLen: 单次输入最大token数
            maxPrefillTokens: 预填充token上限
            config: 自定义配置文件

Notes
                : 
   必须预先设置环境变量 ``LAZYLLM_MINDIE_HOME`` 指向MindIE安装目录, 若未指定 ``finetuned_model`` 或路径无效，会自动回退到 ``base_model``
''')

add_english_doc('deploy.Mindie', '''\
This class is a subclass of ``LazyLLMDeployBase``, designed for deploying and managing the MindIE large language model inference service. It encapsulates the full workflow including configuration generation, process launching, and API interaction for the MindIE service.

Args:
    trust_remote_code (bool): Whether to trust remote code (e.g., from HuggingFace models). Default is ``True``.
    launcher: Instance of the task launcher. Default is ``launchers.remote()``.
    log_path (str): Path to save logs. If ``None``, logs will not be saved.
    **kw: Other configuration parameters.

Keyword Args: 
            npuDeviceIds: List of NPU device IDs (e.g., ``[[0,1]]`` indicates using 2 devices)
            worldSize: Model parallelism size
            port: Service port (set to ``'auto'`` for auto-assignment between 30000–40000)
            maxSeqLen: Maximum sequence length
            maxInputTokenLen: Maximum number of tokens per input
            maxPrefillTokens: Maximum number of prefill tokens
            config: Custom configuration file

Notes:
    You must set the environment variable ``LAZYLLM_MINDIE_HOME`` to point to the MindIE installation directory. 
    If ``finetuned_model`` is not specified or the path is invalid, it will automatically fall back to ``base_model``.
''')


add_example('deploy.Mindie', '''\
>>> import lazyllm
>>> from lazyllm.components.deploy import Mindie            
>>> deployer = Mindie(
...     port=30000,
...     launcher=lazyllm.launchers.remote(),
...     max_seq_len=32000,
...     log_path="/path/to/logs"
... )
>>> cmd = deployer.cmd(
...     finetuned_model="/path/to/finetuned_model",
...     base_model="/path/to/base_model")
>>> print("Service URL:", cmd.geturl())

''')

add_english_doc('deploy.Mindie.load_config', '''\
Loads and parses the MindIE configuration file.

Args:
    config_path (str): Path to the JSON configuration file

**Returns:**\n
- dict: Parsed configuration dictionary

Notes:
    - Handles both default and custom configuration files
    - Uses JSON format for configuration
    - Creates backup of original config before modification
''')

add_chinese_doc('deploy.Mindie.load_config', '''\
加载并解析MindIE配置文件。

Args:
    config_path (str): JSON配置文件的路径

**Returns:**\n
- dict: 解析后的配置字典

注意事项:
    - 处理默认和自定义配置文件
    - 使用JSON格式配置
    - 修改前会创建原始配置的备份
''')

add_english_doc('deploy.Mindie.save_config', '''\
Saves the current configuration to file.

Notes:
    - Automatically creates backup of existing config
    - Writes to the standard MindIE config location
    - Uses JSON format with proper indentation
    - Called automatically during deployment
''')

add_chinese_doc('deploy.Mindie.save_config', '''\
保存当前配置到文件。

注意事项:
    - 自动创建现有配置的备份
    - 写入到标准MindIE配置位置
    - 使用带缩进的JSON格式
    - 部署时自动调用
''')

add_english_doc('deploy.Mindie.update_config', '''\
Updates the configuration dictionary with current settings.

Notes:
    - Handles multiple configuration sections:
        - Model deployment parameters
        - Server settings
        - Scheduling parameters
''')

add_chinese_doc('deploy.Mindie.update_config', '''\
使用当前设置更新配置字典。

注意事项:
    - 处理多个配置部分:
        - 模型部署参数
        - 服务器设置
        - 调度参数
''')

add_english_doc('deploy.Mindie.cmd', '''\
Generates the command to start the MindIE service.

Args:
    finetuned_model (str): Path to the fine-tuned model
    base_model (str): Path to the base model (fallback if finetuned_model is invalid)
    master_ip (str): Master node IP address (currently unused)

**Returns:**\n
- LazyLLMCMD: Command object for starting the service

Notes:
    - Automatically handles model path validation
    - Updates configuration before service start
    - Supports random port allocation when configured
''')

add_chinese_doc('deploy.Mindie.cmd', '''\
生成启动MindIE服务的命令。

Args:
    finetuned_model (str): 微调模型路径
    base_model (str): 基础模型路径(当微调模型无效时作为后备)
    master_ip (str): 主节点IP地址(当前未使用)

**Returns:**\n
- LazyLLMCMD: 启动服务的命令对象

注意事项:
    - 自动处理模型路径验证
    - 启动服务前更新配置
    - 支持配置随机端口分配
''')

add_english_doc('deploy.Mindie.geturl', '''\
Gets the service URL after deployment.

Args:
    job: Job object (optional, defaults to self.job)

**Returns:**\n
- str: The generate endpoint URL

Notes:
    - Returns different formats based on display mode
    - Includes port number from configuration
''')

add_chinese_doc('deploy.Mindie.geturl', '''\
获取部署后的服务URL。

Args:
    job: 任务对象(可选，默认为self.job)

**Returns:**\n
- str: generate接口的URL

注意事项:
    - 根据显示模式返回不同格式
    - 包含配置中的端口号
''')

add_english_doc('deploy.Mindie.extract_result', '''\
Extracts the generated text from the API response.

Args:
    x: Raw API response
    inputs: Original inputs (unused)

**Returns:**\n
- str: The generated text

Notes:
    - Parses JSON response
    - Returns first text entry from response
''')

add_chinese_doc('deploy.Mindie.extract_result', '''\
从API响应中提取生成的文本。

Args:
    x: 原始API响应
    inputs: 原始输入(未使用)

**Returns:**\n
- str: 生成的文本

注意事项:
    - 解析JSON响应
    - 返回响应中的第一个文本条目
''')

# Deploy-LMDeploy
add_chinese_doc('deploy.LMDeploy', '''\
``LMDeploy`` 类，继承自 ``LazyLLMDeployBase``，基于 [LMDeploy](https://github.com/InternLM/lmdeploy) 框架，  
用于启动并管理大语言模型的推理服务。

Args:
    launcher (Optional[lazyllm.launcher]): 服务启动器，默认使用 ``launchers.remote(ngpus=1)``。  
    trust_remote_code (bool): 是否信任远程代码，默认为 ``True``。  
    log_path (Optional[str]): 日志输出路径，默认为 ``None``。  
    **kw: 关键字参数，用于更新默认的部署配置。除下列参数外，不允许传入额外参数。  

Keyword Args:
    tp (int): 张量并行参数，默认为 ``1``。  
    server-name (str): 服务监听的 IP 地址，默认为 ``0.0.0.0``。  
    server-port (Optional[int]): 服务端口号，默认为 ``None``，此时会自动随机分配 30000–40000 区间的端口。  
    max-batch-size (int): 最大批处理大小，默认为 ``128``。  
    chat-template (Optional[str]): 对话模板文件路径。若模型不是视觉语言模型且未指定模板，将使用默认模板。  
    eager-mode (bool): 是否启用 eager 模式，受环境变量 ``LMDEPLOY_EAGER_MODE`` 控制，默认为 ``False``。  
''')

add_english_doc('deploy.LMDeploy', '''\
The ``LMDeploy`` class, a subclass of ``LazyLLMDeployBase``,  
leverages [LMDeploy](https://github.com/InternLM/lmdeploy) to launch and manage large language model inference services.

Args:
    launcher (Optional[lazyllm.launcher]): The service launcher, defaults to ``launchers.remote(ngpus=1)``.  
    trust_remote_code (bool): Whether to trust remote code, defaults to ``True``.  
    log_path (Optional[str]): Path to store logs, defaults to ``None``.  
    **kw: Keyword arguments used to update the default deployment configuration. No extra arguments beyond those listed below are allowed.  

Keyword Args:
    tp (int): Tensor parallelism factor, defaults to ``1``.  
    server-name (str): The IP address on which the service listens, defaults to ``0.0.0.0``.  
    server-port (Optional[int]): Port number for the service. Defaults to ``None``; in this case, a random port between 30000–40000 will be assigned.  
    max-batch-size (int): Maximum batch size, defaults to ``128``.  
    chat-template (Optional[str]): Path to the chat template file. If the model is not a vision-language model and no template is specified, a default template will be used.  
    eager-mode (bool): Whether to enable eager mode, controlled by the environment variable ``LMDEPLOY_EAGER_MODE``, defaults to ``False``.  
''')

add_example('deploy.LMDeploy', '''\
>>> # Basic use:
>>> from lazyllm import deploy
>>> infer = deploy.LMDeploy()
>>>
>>> # MultiModal:
>>> import lazyllm
>>> from lazyllm import deploy, globals
>>> from lazyllm.components.formatter import encode_query_with_filepaths
>>> chat = lazyllm.TrainableModule('InternVL3_5-1B').deploy_method(deploy.LMDeploy)
>>> chat.update_server()
>>> inputs = encode_query_with_filepaths('What is it?', ['path/to/image'])
>>> res = chat(inputs)
''')

add_chinese_doc('deploy.LMDeploy.cmd', '''\
该方法用于生成启动LMDeploy服务的命令。

Args:
    finetuned_model (str): 微调后的模型路径。
    base_model (str): 基础模型路径，当finetuned_model无效时使用。

**Returns:**\n
- LazyLLMCMD: 一个包含启动命令的LazyLLMCMD对象。
''')

add_english_doc('deploy.LMDeploy.cmd', '''\
This method generates the command to start the LMDeploy service.

Args:
    finetuned_model (str): Path to the fine-tuned model.
    base_model (str): Path to the base model, used when finetuned_model is invalid.

**Returns:**\n
- LazyLLMCMD: A LazyLLMCMD object containing the startup command.
''')

add_chinese_doc('deploy.LMDeploy.geturl', '''\
获取LMDeploy服务的URL地址。

Args:
    job (optional): 任务对象，默认为None，此时使用self.job。

**Returns:**\n
- str: 服务的URL地址，格式为"http://{ip}:{port}/v1/chat/interactive"。
''')

add_english_doc('deploy.LMDeploy.geturl', '''\
Get the URL address of the LMDeploy service.

Args:
    job (optional): Job object, defaults to None, in which case self.job is used.

**Returns:**\n
- str: The service URL address in the format "http://{ip}:{port}/v1/chat/interactive".
''')

add_chinese_doc('deploy.LMDeploy.extract_result', '''\
解析模型推理结果，从返回的 JSON 字符串中提取文本输出。

Args:
    x (str): 模型返回的 JSON 格式字符串。  
    inputs (dict): 原始输入数据（此参数未被直接使用，保留作接口兼容）。  

**Returns:**\n
- str: 从响应中解析得到的文本结果。  
''')

add_english_doc('deploy.LMDeploy.extract_result', '''\
Parses the model inference result and extracts the text output from a JSON response string.

Args:
    x (str): JSON-formatted string returned by the model.  
    inputs (dict): The original input data (not directly used, reserved for interface compatibility).  

**Returns:**\n
- str: The text result extracted from the response.  
''')
add_chinese_doc('deploy.text_to_speech.utils.TTSBase', """\
TTS（文本转语音）服务的基类。

提供文本转语音服务的部署基础框架，支持模型加载和RelayServer部署。

Args:
    launcher (LazyLLMLaunchersBase, optional): 任务启动器
    log_path (str, optional): 日志文件路径
    port (int, optional): 服务端口号
""")

add_english_doc('deploy.text_to_speech.utils.TTSBase', """\
Base class for TTS (Text-to-Speech) services.

Provides the deployment framework for text-to-speech services, supporting model loading and RelayServer deployment.

Args:
    launcher (LazyLLMLaunchersBase, optional): Task launcher
    log_path (str, optional): Log file path
    port (int, optional): Service port number
""")

# Deploy-Infinity
add_chinese_doc('deploy.Infinity', '''\
此类是 ``LazyLLMDeployBase`` 的子类，基于 [Infinity](https://github.com/michaelfeil/infinity) 框架提供的高性能文本嵌入、重排序和CLIP等能力。

Args:
    launcher (lazyllm.launcher): Infinity 的启动器，默认为 ``launchers.remote(ngpus=1)``。
    kw: 关键字参数，用于更新默认的训练参数。请注意，除了以下列出的关键字参数外，这里不能传入额外的关键字参数。

此类的关键字参数及其默认值如下：

Keyword Args: 
    launcher (Launcher, optional): 启动器配置，默认为remote(ngpus=1)。
    model_type (str, optional): 模型类型，默认为'embed'。
    log_path (str, optional): 日志文件路径，默认为None。
    **kw: 额外的配置参数，包括host、port、batch-size等。

''')

add_english_doc('deploy.Infinity', '''\
This class is a subclass of ``LazyLLMDeployBase``, providing high-performance text-embeddings, reranking, and CLIP capabilities based on the [Infinity](https://github.com/michaelfeil/infinity) framework.

Args:
    launcher (lazyllm.launcher): The launcher for Infinity, defaulting to ``launchers.remote(ngpus=1)``.
    kw: Keyword arguments for updating default training parameters. Note that no additional keyword arguments can be passed here except those listed below.

The keyword arguments and their default values for this class are as follows:

Keyword Args: 
    keys_name_handle (Dict): Key name mapping dictionary.
    message_format (Dict): Default message format template.
    default_headers (Dict): Default HTTP request headers.
    target_name (str): API target endpoint name.
''')

add_example('deploy.Infinity', '''\
>>> import lazyllm
>>> from lazyllm import deploy
>>> deploy.Infinity()
<lazyllm.llm.deploy type=Infinity>
''')

add_chinese_doc('deploy.Infinity.extract_result', '''\
从Infinity API响应中提取结果数据。
解析Infinity服务的JSON响应，根据返回的对象类型提取嵌入向量或重排序结果。

Args:
    x (str): API返回的JSON字符串响应。
    inputs (Dict): 原始输入数据，用于确定返回结果的格式。
''')

add_english_doc('deploy.Infinity.extract_result', '''\
Extract result data from Infinity API response.
Parses JSON response from Infinity service and extracts embedding vectors or reranking results based on the returned object type.

Args:
    x (str): JSON string response returned by API.
    inputs (Dict): Original input data used to determine the return format.
''')

add_chinese_doc('deploy.Infinity.geturl', '''\
获取Infinity服务的URL地址。根据部署模式和作业状态，返回对应的API访问URL地址。

Args:
    job (Optional[Any]): 作业对象，如果为None则使用当前实例的job属性。
''')

add_english_doc('deploy.Infinity.geturl', '''\
Get the URL address of the Infinity service. Returns the corresponding API access URL address based on deployment mode and job status.

Args:
    job (Optional[Any]): Job object, if None uses the current instance's job property.
''')


add_chinese_doc('deploy.relay.base.FastapiApp.get', """\
注册GET请求路由装饰器。

Args:
    path (str): 路由路径
    **kw: 其他FastAPI路由参数

**Returns:**\n
- 装饰器函数
""")

add_english_doc('deploy.relay.base.FastapiApp.get', """\
Register GET request route decorator.

Args:
    path (str): Route path
    **kw: Other FastAPI route parameters

**Returns:**\n
- Decorator function
""")

add_chinese_doc('deploy.relay.base.FastapiApp.post', """\
注册POST请求路由装饰器。

Args:
    path (str): 路由路径
    **kw: 其他FastAPI路由参数

**Returns:**\n
- 装饰器函数
""")

add_english_doc('deploy.relay.base.FastapiApp.post', """\
Register POST request route decorator.

Args:
    path (str): Route path
    **kw: Other FastAPI route parameters

**Returns:**\n
- Decorator function
""")

add_chinese_doc('deploy.relay.base.FastapiApp.list', """\
注册LIST请求路由装饰器。

Args:
    path (str): 路由路径
    **kw: 其他FastAPI路由参数

**Returns:**\n
- 装饰器函数
""")

add_english_doc('deploy.relay.base.FastapiApp.list', """\
Register LIST request route decorator.

Args:
    path (str): Route path
    **kw: Other FastAPI route parameters

**Returns:**\n
- Decorator function
""")

add_chinese_doc('deploy.relay.base.FastapiApp.delete', """\
注册DELETE请求路由装饰器。

Args:
    path (str): 路由路径
    **kw: 其他FastAPI路由参数

**Returns:**\n
- 装饰器函数
""")

add_english_doc('deploy.relay.base.FastapiApp.delete', """\
Register DELETE request route decorator.

Args:
    path (str): Route path
    **kw: Other FastAPI route parameters

**Returns:**\n
- Decorator function
""")

add_chinese_doc('deploy.relay.base.FastapiApp.update', """\
更新并清空路由服务注册表。
""")

add_english_doc('deploy.relay.base.FastapiApp.update', """\
Update and clear route service registry.
""")

# RelayServer class documentation
add_chinese_doc('deploy.relay.base.RelayServer', '''\
RelayServer类是一个用于部署FastAPI服务的基类，它可以将一个函数转换为HTTP服务。这个类支持设置前处理函数、后处理函数，
并可以自动分配端口号。它主要用于将模型推理功能转换为HTTP服务，便于分布式部署和调用。

Args:
    port (int, optional): 服务监听端口号。如果为None则自动分配30000-40000之间的随机端口
    func (callable): 主要的模型推理函数，接收请求数据并返回推理结果
    pre_func (callable, optional): 预处理函数，在调用主函数前执行，用于数据清洗和转换
    post_func (callable, optional): 后处理函数，在调用主函数后执行，用于结果格式化和增强
    pythonpath (str, optional): 额外的Python模块搜索路径，用于解决依赖导入问题
    log_path (str, optional): 服务日志文件的存储目录路径
    cls (str, optional): 服务类别的标识符，用于日志目录命名
    launcher (Launcher): 任务启动器实例，控制服务的部署方式，默认为远程异步启动器
''')

add_english_doc('deploy.relay.base.RelayServer', '''\
RelayServer is a base class for deploying FastAPI services that converts a function into an HTTP service. It supports 
setting pre-processing and post-processing functions, and can automatically allocate port numbers. It's mainly used 
to convert model inference functionality into HTTP services for distributed deployment and invocation.

Args:
    port (int, optional): Service listening port number. If None, automatically assigns a random port between 30000-40000
    func (callable): Main model inference function that receives request data and returns inference results
    pre_func (callable, optional): Pre-processing function, executed before calling the main function for data cleaning and transformation
    post_func (callable, optional): Post-processing function, executed after calling the main function for result formatting and enhancement
    pythonpath (str, optional): Additional Python module search path for resolving dependency import issues
    log_path (str, optional): Storage directory path for service log files
    cls (str, optional): Service category identifier for log directory naming
    launcher (Launcher): Task launcher instance that controls service deployment method, defaults to remote asynchronous launche
''')

add_example('deploy.relay.base.RelayServer', '''\
>>> from lazyllm.components.deploy.relay.base import RelayServer
>>> def my_function(text):
...     return f"Processed: {text}"
>>> server = RelayServer(port=35000, func=my_function)
>>> server.start()  # This will start the server
>>> print(server.geturl())  # Get the service URL
http://localhost:35000/generate
''')

# cmd method documentation
add_chinese_doc('deploy.relay.base.RelayServer.cmd', '''\
cmd方法用于生成启动服务器的命令。它会将当前的函数和配置转换为一个可执行的命令字符串。

Args:
    func: 可选，要部署的新函数。如果不提供，则使用初始化时的函数。

**Returns:**\n
- 返回一个LazyLLMCMD对象，包含服务器启动命令和相关配置。
''')

add_english_doc('deploy.relay.base.RelayServer.cmd', '''\
The cmd method generates the command to start the server. It converts the current function and configuration into 
an executable command string.

Args:
    func: Optional, new function to deploy. If not provided, uses the function from initialization.

**Returns:**\n
- Returns a LazyLLMCMD object containing the server start command and related configuration.
''')

add_example('deploy.relay.base.RelayServer.cmd', '''\
>>> server = RelayServer(port=35000)
>>> def new_function(text):
...     return f"New process: {text}"
>>> cmd_obj = server.cmd(new_function)
>>> print(cmd_obj)  # Will show the command that would be executed
''')

# geturl method documentation
add_chinese_doc('deploy.relay.base.RelayServer.geturl', '''\
geturl方法用于获取服务的访问URL。该URL可用于向服务发送HTTP请求。

Args:
    job: 可选，指定的任务对象。如果为None，则使用当前实例的任务。

**Returns:**\n
- 返回服务的完整URL地址，格式为 http://<ip>:<port>/generate
''')

add_english_doc('deploy.relay.base.RelayServer.geturl', '''\
The geturl method returns the access URL for the service. This URL can be used to send HTTP requests to the service.

Args:
    job: Optional, specified job object. If None, uses the current instance's job.

**Returns:**\n
- Returns the complete URL of the service in the format http://<ip>:<port>/generate
''')

add_example('deploy.relay.base.RelayServer.geturl', '''\
>>> server = RelayServer(port=35000)
>>> server.start()
>>> url = server.geturl()
>>> print(url)  # Shows the service endpoint URL
http://localhost:35000/generate
>>> # You can now use this URL to make HTTP requests to your service
''')

add_chinese_doc('deploy.base.DummyDeploy', '''\
DummyDeploy(launcher=launchers.remote(sync=False), *, stream=False, **kw)

一个用于测试的模拟部署类，继承自 `LazyLLMDeployBase` 和 `flows.Pipeline`，实现了一个简单的流水线风格部署服务，
支持流式输出（可选）。

该类主要用于内部测试和示例用途。它接收符合 `message_format` 格式的输入，根据是否启用 `stream` 参数，返回
字符串或逐步输出的模拟响应。

Args：
    launcher: 部署器实例，默认值为 `launchers.remote(sync=False)`。
    stream (bool): 是否以流式方式输出结果。
    kw: 其他传递给父类的关键字参数。

Call Arguments:
    keys_name_handle (dict): 输入字段名的映射。 \n
    message_format (dict): 默认请求模板，包括输入内容与生成参数。 \n
''')

add_english_doc('deploy.base.DummyDeploy', '''\
DummyDeploy(launcher=launchers.remote(sync=False), *, stream=False, **kw)

A mock deployment class for testing purposes. It extends both `LazyLLMDeployBase` and `flows.Pipeline`,
simulating a simple pipeline-style deployable service with optional streaming support.

This class is primarily intended for internal testing and demonstration. It receives inputs in the format defined
by `message_format`, and returns a dummy response or a streaming response depending on the `stream` flag.

Args:
    launcher: Deployment launcher instance, defaulting to `launchers.remote(sync=False)`.
    stream (bool): Whether to simulate streaming output.
    kw: Additional keyword arguments passed to the superclass.

Call Arguments:
    keys_name_handle (dict): Mapping of input keys for request formatting. \n
    message_format (dict): Default request template including input and generation parameters. \n
''')

# Deploy-Auto
add_chinese_doc('auto.AutoDeploy', '''\
此类是 ``LazyLLMDeployBase`` 的子类，可根据输入的参数自动选择合适的推理框架和参数，以对大语言模型进行推理。

具体而言，基于输入的：``base_model`` 的模型参数、``max_token_num``、``launcher`` 中GPU的类型以及卡数，该类可以自动选择出合适的推理框架（如: ``Lightllm`` 或 ``Vllm``）及所需的参数。

Args:
    base_model (str): 用于进行微调的基模型，要求是基模型的路径或模型名。用于提供基模型信息。
    source (lazyllm.config['model_source']): 指定模型的下载源。可通过设置环境变量 ``LAZYLLM_MODEL_SOURCE`` 来配置，目前仅支持 ``huggingface`` 或 ``modelscope`` 。若不设置，lazyllm不会启动自动模型下载。
    trust_remote_code (bool): 是否允许加载来自远程服务器的模型代码，默认为 ``True``。
    launcher (lazyllm.launcher): 微调的启动器，默认为 ``launchers.remote(ngpus=1)``。
    stream (bool): 是否为流式响应，默认为 ``False``。
    type (str): 类型参数，默认为 ``None``，及``llm``类型，另外还支持``embed``类型。
    max_token_num (int): 输入微调模型的token最大长度，默认为``1024``。
    launcher (lazyllm.launcher): 微调的启动器，默认为 ``launchers.remote(ngpus=1)``。
    kw: 关键字参数，用于更新默认的训练参数。注意这里能够指定的关键字参数取决于 LazyLLM 推测出的框架，因此建议谨慎设置。
''')

add_english_doc('auto.AutoDeploy', '''\
This class is a subclass of ``LazyLLMDeployBase`` that automatically selects the appropriate inference framework and parameters based on the input arguments for inference with large language models.

Specifically, based on the input ``base_model`` parameters, ``max_token_num``, the type and number of GPUs in ``launcher``, this class can automatically select the appropriate inference framework (such as ``Lightllm`` or ``Vllm``) and the required parameters.

Args:
    base_model (str): The base model for fine-tuning, which is required to be the name or the path to the base model. Used to provide base model information.
    source (lazyllm.config['model_source']): Specifies the model download source. This can be configured by setting the environment variable ``LAZYLLM_MODEL_SOURCE``.
    trust_remote_code (bool): Whether to allow loading of model code from remote servers, default is ``True``.
    launcher (lazyllm.launcher): The launcher for fine-tuning, default is ``launchers.remote(ngpus=1)``.
    stream (bool): Whether the response is streaming, default is ``False``.
    type (str): Type parameter, default is ``None``, which corresponds to the ``llm`` type. Additionally, the ``embed`` type is also supported.
    max_token_num (int): The maximum token length for the input fine-tuning model, default is ``1024``.
    launcher (lazyllm.launcher): The launcher for fine-tuning, default is ``launchers.remote(ngpus=1)``.
    kw: Keyword arguments used to update default training parameters. Note that whether additional keyword arguments can be specified depends on the framework inferred by LazyLLM, so it is recommended to set them carefully.
''')

add_example('auto.AutoDeploy', '''\
>>> from lazyllm import deploy
>>> deploy.auto('internlm2-chat-7b')
<lazyllm.llm.deploy type=Lightllm> 
''')

add_chinese_doc('auto.AutoDeploy.get_deployer', '''\
根据模型类型获取对应的部署器类。

自动检测模型类型并返回最适合的部署器类、启动器和配置参数。

Args:
    base_model (str): 基础模型名称或路径。
    source (Optional[str], optional): 模型来源。
    trust_remote_code (bool, optional): 是否信任远程代码。
    launcher (Optional[LazyLLMLaunchersBase], optional): 启动器实例。
    type (Optional[str], optional): 模型类型。
    log_path (Optional[str], optional): 日志文件路径。
    **kw: 其他配置参数。
''')

add_english_doc('auto.AutoDeploy.get_deployer', '''\
Get corresponding deployer class based on model type.

Automatically detects model type and returns the most suitable deployer class, 
launcher and configuration parameters.

Args:
    base_model (str): Base model name or path.
    source (Optional[str], optional): Model source.
    trust_remote_code (bool, optional): Whether to trust remote code.
    launcher (Optional[LazyLLMLaunchersBase], optional): Launcher instance.
    type (Optional[str], optional): Model type.
    log_path (Optional[str], optional): Log file path.
    **kw: Other configuration parameters.

**Returns:**\n
- Tuple: Returns (deployer class, launcher, configuration parameters dict) triple.
''')

# ModelManager
add_chinese_doc('ModelManager', '''\
ModelManager 是 LazyLLM 提供的模型管理与下载工具类，支持本地搜索和 Huggingface/Modelscope 下载。  

Args:
    model_source (Optional[str]): 模型下载源，仅支持 ``huggingface`` 或 ``modelscope``。
        未提供时使用 LAZYLLM_MODEL_SOURCE，若未设置则默认 ``modelscope``。
    token (Optional[str]): 下载私有模型的访问令牌。未提供时使用 LAZYLLM_MODEL_SOURCE_TOKEN。
    model_path (Optional[str]): 冒号分隔的本地绝对路径列表，用于下载前搜索模型。未提供时使用 LAZYLLM_MODEL_PATH。
    cache_dir (Optional[str]): 本地缓存目录，用于存放下载的模型。未提供时使用 LAZYLLM_MODEL_CACHE_DIR，默认 ``~/.lazyllm/model``。

Static Methods:
    get_model_type(model: str) -> str
        返回指定模型类型，如 ``llm``、``chat``，未识别返回 ``llm``。
    get_model_prompt_keys(model: str) -> dict
        返回模型的 prompt key 映射字典。
    validate_model_path(model_path: str) -> bool
        检查目录下是否存在有效模型文件（扩展名: ``.pt``, ``.bin``, ``.safetensors``）。

Instance Methods:
    download(model: Optional[str] = '', call_back: Optional[Callable] = None) -> str | bool
        下载指定模型。流程：
        1. 在 model_path 列出的本地目录搜索；
        2. 未找到则在 cache_dir 下搜索；
        3. 仍未找到则从 model_source 下载并存放 cache_dir。

        Args:
            model (Optional[str]): 目标模型名称，可使用简略名称或下载源完整名称。
            call_back (Optional[Callable]): 下载进度回调函数，可选。
''')

add_english_doc('ModelManager', '''\
ModelManager is a utility class in LazyLLM for managing and downloading models, supporting local search and Huggingface/Modelscope downloads.  

Args:
    model_source (Optional[str]): Model download source, only ``huggingface`` or ``modelscope`` supported.
        Defaults to LAZYLLM_MODEL_SOURCE, and ``modelscope`` if unset.
    token (Optional[str]): Access token for private models. Defaults to LAZYLLM_MODEL_SOURCE_TOKEN.
    model_path (Optional[str]): Colon-separated list of local absolute paths to search before download. Defaults to LAZYLLM_MODEL_PATH.
    cache_dir (Optional[str]): Directory for downloaded models. Defaults to LAZYLLM_MODEL_CACHE_DIR, or ``~/.lazyllm/model``.

Static Methods:
    get_model_type(model: str) -> str
        Returns model type, e.g., ``llm`` or ``chat``; returns ``llm`` if unrecognized.
    get_model_prompt_keys(model: str) -> dict
        Returns the prompt key mapping dictionary for the model.
    validate_model_path(model_path: str) -> bool
        Checks if directory contains valid model files (extensions: ``.pt``, ``.bin``, ``.safetensors``).

Instance Methods:
    download(model: Optional[str] = '', call_back: Optional[Callable] = None) -> str | bool
        Downloads the specified model. Process:
        1. Search in local directories listed in model_path;
        2. If not found, search in cache_dir;
        3. If still not found, download from model_source to cache_dir.

        Args:
            model (Optional[str]): Target model name, can be abbreviated or full name from source.
            call_back (Optional[Callable]): Optional callback function for download progress.
''')

add_example('ModelManager', '''\
>>> from lazyllm.components import ModelManager
>>> downloader = ModelManager(model_source='modelscope')
>>> downloader.download('chatglm3-6b')
''')

add_chinese_doc('ModelManager.get_model_type', '''\
根据模型名称获取模型类型（如 LLM、VLM 等）。

Args:
    model (str): 模型名称或路径，必须为非空字符串。

**Returns:**\n
- str: 模型类型，如果无法匹配则返回 ``llm``。
''')

add_english_doc('ModelManager.get_model_type', '''\
Retrieve the type of a model (e.g., LLM, VLM) based on its name.

Args:
    model (str): Model name or path, must be a non-empty string.

**Returns:**\n
- str: Model type, returns ``llm`` if no match is found.
''')

add_chinese_doc('ModelManager.get_model_prompt_keys', '''\
获取指定模型的 prompt key 映射字典，用于推理时构建输入。  

Args:
    model (str): 模型名称或路径。

**Returns:**\n
- dict: 模型对应的 prompt key 映射，如果不存在则返回空字典
''')

add_english_doc('ModelManager.get_model_prompt_keys', '''\
Get the prompt key mapping dictionary for the specified model, used for constructing inputs during inference.  

Args:
    model (str): Model name or path.

**Returns:**\n
- dict: The prompt key mapping for the model, or an empty dictionary if none exists
''')

add_chinese_doc('ModelManager.validate_model_path', '''\
检查指定路径下是否存在有效的模型文件（.pt, .bin, .safetensors）。  

Args:
    model_path (str): 模型目录路径。

**Returns:**\n
- bool: 如果目录中存在模型文件返回 True，否则返回 False
''')

add_english_doc('ModelManager.validate_model_path', '''\
Check whether the specified path contains valid model files (.pt, .bin, .safetensors).  

Args:
    model_path (str): Path to the model directory.

**Returns:**\n
- bool: True if model files exist in the directory, False otherwise
''')

add_chinese_doc('ModelManager.download', '''\
下载指定名称的模型，如果本地已有则直接返回本地路径；  
支持 Huggingface 和 Modelscope 平台的自动下载，并会在缓存目录创建符号链接以便统一管理。  

Args:
    model (str, optional): 模型名称或路径，默认为空字符串，表示不下载。
    call_back (Optional[Callable], optional): 下载进度回调函数，接受当前下载状态等参数。  

**Returns:**\n
- str | bool: 模型在本地的完整路径，如果下载失败返回 False
''')

add_english_doc('ModelManager.download', '''\
Download the specified model by name. If it already exists locally, return the local path.  
Supports automatic download from Huggingface and Modelscope, creating a symbolic link in the cache directory for unified management.  

Args:
    model (str, optional): Model name or path, defaults to empty string which means no download.
    call_back (Optional[Callable], optional): Callback function for download progress, receives current download status.  

**Returns:**\n
- str | bool: Full local path to the model, or False if the download fails
''')

add_chinese_doc('LLMType', '''\
LLMType 枚举类

该枚举用于表示不同类型的大模型（如 LLM、VLM、TTS 等）。
特点：
- 成员值为字符串（继承自 str）。
- 支持大小写不敏感的构造与比较：
    - 构造时，既可以用成员名称，也可以用成员值，大小写不敏感。
    - 比较时，可以直接与字符串进行比较，大小写不敏感。
    - 成员名称索引（如 LLMType['xxx']）同样大小写不敏感。

可用类型：
    - LLM
    - VLM
    - SD
    - TTS
    - STT
    - EMBED
    - REANK
    - CROSS_MODAL_EMBED
    - OCR
''')

add_english_doc('LLMType', '''\
LLMType Enum Class

This enum represents different types of large models (e.g., LLM, VLM, TTS).
Features:
- Members are strings (inherits from str).
- Case-insensitive construction and comparison:
    - When constructing, both member names and values are accepted, case-insensitive.
    - When comparing, it can be directly compared with strings, case-insensitive.
    - Member name lookup (LLMType['xxx']) is also case-insensitive.

Available types:
    - LLM
    - VLM
    - SD
    - TTS
    - STT
    - EMBED
    - REANK
    - CROSS_MODAL_EMBED
    - OCR
''')

add_example('LLMType', '''\
>>> LLMType("llm")
<LLMType.LLM: 'LLM'>

>>> LLMType("llm") == LLMType.LLM
True

>>> LLMType("LLM") == LLMType.LLM
True

>>> LLMType.LLM == "llm"
True

>>> LLMType.LLM == "LLM"
True

>>> LLMType("CROSS_modal_embed")
<LLMType.CROSS_MODAL_EMBED: 'CROSS_MODAL_EMBED'>
''')

# ============= Formatter

# FormatterBase
add_chinese_doc('formatter.LazyLLMFormatterBase', '''\
此类是格式化器的基类，格式化器是模型输出结果的格式化器，用户可以自定义格式化器，也可以使用LazyLLM提供的格式化器。
''')

add_english_doc('formatter.LazyLLMFormatterBase', '''\
This class is the base class of the formatter. The formatter is the formatter of the model output result. Users can customize the formatter or use the formatter provided by LazyLLM.
''')

add_example('formatter.LazyLLMFormatterBase', '''\
>>> from lazyllm.components.formatter import LazyLLMFormatterBase
>>> class MyFormatter(LazyLLMFormatterBase):
...     def __init__(self, formatter: str = None):
...         self._formatter = formatter
...         if self._formatter:
...             self._parse_formatter()
...         else:
...             self._slices = None
...     def _parse_formatter(self):
...         slice_str = self._formatter.strip()[1:-1]
...         slices = []
...         parts = slice_str.split(":")
...         start = int(parts[0]) if parts[0] else None
...         end = int(parts[1]) if len(parts) > 1 and parts[1] else None
...         step = int(parts[2]) if len(parts) > 2 and parts[2] else None
...         slices.append(slice(start, end, step))
...         self._slices = slices
...     def _load(self, data):
...         return [int(x) for x in data.strip('[]').split(',')]
...     def _parse_py_data_by_formatter(self, data):
...         if self._slices is not None:
...             result = []
...             for s in self._slices:
...                 if isinstance(s, slice):
...                     result.extend(data[s])
...                 else:
...                     result.append(data[int(s)])
...             return result
...         else:
...             return data
...
>>> fmt = MyFormatter("[1:3]")
>>> res = fmt.format("[1,2,3,4,5]")
>>> print(res)
[2, 3]
''')

add_chinese_doc('formatter.LazyLLMFormatterBase.format', """\
格式化输入消息。

Args:
    msg: 输入消息，可以是字符串或其他格式

**Returns:**\n
- 格式化后的数据，具体类型由子类实现决定
""")

add_english_doc('formatter.LazyLLMFormatterBase.format', """\
Format input message.

Args:
    msg: Input message, can be string or other format

**Returns:**\n
- Formatted data, specific type determined by subclass implementation
""")

# JsonLikeFormatter
add_chinese_doc('formatter.formatterbase.JsonLikeFormatter', '''\
该类用于以类 JSON 的格式提取嵌套结构（如 dict、list、tuple）中的子字段内容。

其功能通过格式化字符串 `formatter` 来控制，格式类似于数组/字典的索引切片表达式。例如：

- `[0]` 表示取第 0 个元素
- `[0][{key}]` 表示取第 0 个元素并获取其中 key 字段
- `[0,1][{a,b}]` 表示同时提取第 0 和第 1 个对象的 a 和 b 字段
- `[::2]` 表示步长为 2 的切片
- `*[0][{x}]` 表示以包装格式返回处理后的数据（用于进一步结构化）

Args:
    formatter (str, optional): 控制提取规则的格式字符串。若为 None，则返回原始数据。
''')

add_english_doc('formatter.formatterbase.JsonLikeFormatter', '''\
This class is used to extract subfields from nested structures (like dicts, lists, tuples) using a JSON-like indexing syntax.

The behavior is driven by a formatter string similar to Python-style slicing and dictionary access:

- `[0]` fetches the first item
- `[0][{key}]` accesses the `key` field in the first item
- `[0,1][{a,b}]` fetches the `a` and `b` fields from the first and second items
- `[::2]` does slicing with a step of 2
- `*[0][{x}]` means return a wrapped/structured result

Args:
    formatter (str, optional): A format string that controls how to slice and extract the structure. If None, the input will be returned directly.
''')

add_example('formatter.formatterbase.JsonLikeFormatter', '''\
>>> from lazyllm.components.formatter.formatterbase import JsonLikeFormatter
>>> formatter = JsonLikeFormatter("[{a,b}]")
''')

add_chinese_doc('formatter.formatterbase.PipelineFormatter', """\
流水线格式化器，用于将数据处理流水线封装为格式化器。

该类将Pipeline实例包装为格式化器，支持通过管道操作符组合多个格式化器。

Args:
    formatter (Pipeline): 要封装的流水线实例
""")

add_english_doc('formatter.formatterbase.PipelineFormatter', """\
Pipeline formatter for encapsulating data processing pipelines as formatters.

This class wraps Pipeline instances as formatters and supports combining multiple formatters through pipe operators.

Args:
    formatter (Pipeline): Pipeline instance to encapsulate
""")

# PythonFormatter
add_chinese_doc('formatter.formatterbase.PythonFormatter', '''\
预留格式化器类，用于支持 Python 风格的数据提取语法，待开发。

当前继承自 JsonLikeFormatter，无额外功能。
''')

add_english_doc('formatter.formatterbase.PythonFormatter', '''\
Reserved formatter class for supporting Python-style data extraction syntax. To be developed.

Currently inherits from JsonLikeFormatter with no additional behavior.
''')

# FileFormatter
add_chinese_doc('formatter.FileFormatter', '''\
用于处理带文档上下文的查询字符串格式转换的格式化器。

支持三种模式：

- "decode"：将结构化查询字符串解码为包含 query 和 files 的字典。
- "encode"：将包含 query 和 files 的字典编码为结构化查询字符串。
- "merge"：将多个结构化查询字符串合并为一个整体查询。

Args:
    formatter (str): 指定操作模式，可为 "decode"、"encode" 或 "merge"（默认为 "decode"）。
''')

add_english_doc('formatter.FileFormatter', '''\
A formatter that transforms query strings with document context between structured formats.

Supports three modes:

- "decode": Decodes structured query strings into dictionaries with `query` and `files`.
- "encode": Encodes a dictionary with `query` and `files` into a structured query string.
- "merge": Merges multiple structured query strings into one.

Args:
    formatter (str): The operation mode. Must be one of "decode", "encode", or "merge". Defaults to "decode".
''')

add_example('formatter.FileFormatter', '''\
>>> from lazyllm.components.formatter import FileFormatter

>>> # Decode mode
>>> fmt = FileFormatter('decode')
''')

# YamlFormatter
add_chinese_doc('formatter.YamlFormatter', '''\
用于从 YAML 格式的字符串中提取结构化信息的格式化器。

继承自 JsonLikeFormatter，通过内部方法将字符串解析为 Python 对象后使用类 JSON 的方式提取字段。

适合用于处理包含嵌套结构的 YAML 文本，并结合格式化表达式获取目标数据。

''')

add_english_doc('formatter.YamlFormatter', '''\
A formatter for extracting structured information from YAML-formatted strings.

Inherits from JsonLikeFormatter. Uses the internal method to parse YAML strings into Python objects, and then applies JSON-like formatting rules to extract desired fields.

Suitable for handling nested YAML content with formatter-based field selection.
''')

add_example('formatter.YamlFormatter', '''\
>>> from lazyllm.components.formatter import YamlFormatter
>>> formatter = YamlFormatter("{name,age}")
>>> msg = \\"\\"\\" 
... name: Alice
... age: 30
... city: London
... \\"\\"\\"
>>> formatter(msg)
{'name': 'Alice', 'age': 30}
''')

# JsonFormatter
add_chinese_doc('JsonFormatter', '''\
此类是JSON格式化器，即用户希望模型输出的内容格式为JSON，还可以通过索引方式对输出内容中的某个字段进行选择。
''')

add_english_doc('JsonFormatter', '''\
This class is a JSON formatter, that is, the user wants the model to output content is JSON format, and can also select a field in the output content by indexing.
''')

add_example('JsonFormatter', """\
>>> import lazyllm
>>> from lazyllm.components import JsonFormatter
>>> toc_prompt='''
... You are now an intelligent assistant. Your task is to understand the user's input and convert the outline into a list of nested dictionaries. Each dictionary contains a `title` and a `describe`, where the `title` should clearly indicate the level using Markdown format, and the `describe` is a description and writing guide for that section.
... 
... Please generate the corresponding list of nested dictionaries based on the following user input:
... 
... Example output:
... [
...     {
...         "title": "# Level 1 Title",
...         "describe": "Please provide a detailed description of the content under this title, offering background information and core viewpoints."
...     },
...     {
...         "title": "## Level 2 Title",
...         "describe": "Please provide a detailed description of the content under this title, giving specific details and examples to support the viewpoints of the Level 1 title."
...     },
...     {
...         "title": "### Level 3 Title",
...         "describe": "Please provide a detailed description of the content under this title, deeply analyzing and providing more details and data support."
...     }
... ]
... User input is as follows:
... '''
>>> query = "Please help me write an article about the application of artificial intelligence in the medical field."
>>> m = lazyllm.TrainableModule("internlm2-chat-20b").prompt(toc_prompt).start()
>>> ret = m(query, max_new_tokens=2048)
>>> print(f"ret: {ret!r}")  # the model output without specifying a formatter
'Based on your user input, here is the corresponding list of nested dictionaries:\n[\n    {\n        "title": "# Application of Artificial Intelligence in the Medical Field",\n        "describe": "Please provide a detailed description of the application of artificial intelligence in the medical field, including its benefits, challenges, and future prospects."\n    },\n    {\n        "title": "## AI in Medical Diagnosis",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in medical diagnosis, including specific examples of AI-based diagnostic tools and their impact on patient outcomes."\n    },\n    {\n        "title": "### AI in Medical Imaging",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in medical imaging, including the advantages of AI-based image analysis and its applications in various medical specialties."\n    },\n    {\n        "title": "### AI in Drug Discovery and Development",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in drug discovery and development, including the role of AI in identifying potential drug candidates and streamlining the drug development process."\n    },\n    {\n        "title": "## AI in Medical Research",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in medical research, including its applications in genomics, epidemiology, and clinical trials."\n    },\n    {\n        "title": "### AI in Genomics and Precision Medicine",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in genomics and precision medicine, including the role of AI in analyzing large-scale genomic data and tailoring treatments to individual patients."\n    },\n    {\n        "title": "### AI in Epidemiology and Public Health",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in epidemiology and public health, including its applications in disease surveillance, outbreak prediction, and resource allocation."\n    },\n    {\n        "title": "### AI in Clinical Trials",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in clinical trials, including its role in patient recruitment, trial design, and data analysis."\n    },\n    {\n        "title": "## AI in Medical Practice",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in medical practice, including its applications in patient monitoring, personalized medicine, and telemedicine."\n    },\n    {\n        "title": "### AI in Patient Monitoring",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in patient monitoring, including its role in real-time monitoring of vital signs and early detection of health issues."\n    },\n    {\n        "title": "### AI in Personalized Medicine",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in personalized medicine, including its role in analyzing patient data to tailor treatments and predict outcomes."\n    },\n    {\n        "title": "### AI in Telemedicine",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in telemedicine, including its applications in remote consultations, virtual diagnoses, and digital health records."\n    },\n    {\n        "title": "## AI in Medical Ethics and Policy",\n        "describe": "Please provide a detailed description of the ethical and policy considerations surrounding the use of artificial intelligence in the medical field, including issues related to data privacy, bias, and accountability."\n    }\n]'
>>> m = lazyllm.TrainableModule("internlm2-chat-20b").formatter(JsonFormatter("[:][title]")).prompt(toc_prompt).start()
>>> ret = m(query, max_new_tokens=2048)
>>> print(f"ret: {ret}")  # the model output of the specified formaater
['# Application of Artificial Intelligence in the Medical Field', '## AI in Medical Diagnosis', '### AI in Medical Imaging', '### AI in Drug Discovery and Development', '## AI in Medical Research', '### AI in Genomics and Precision Medicine', '### AI in Epidemiology and Public Health', '### AI in Clinical Trials', '## AI in Medical Practice', '### AI in Patient Monitoring', '### AI in Personalized Medicine', '### AI in Telemedicine', '## AI in Medical Ethics and Policy']
""")

# EmptyFormatter
add_chinese_doc('EmptyFormatter', '''\
此类是空的格式化器，即用户希望对模型的输出不做格式化，用户可以对模型指定该格式化器，也可以不指定(模型默认的格式化器就是空格式化器)
''')

add_english_doc('EmptyFormatter', '''\
This type is the system default formatter. When the user does not specify anything or does not want to format the model output, this type is selected. The model output will be in the same format.
''')

add_example('EmptyFormatter', """\
>>> import lazyllm
>>> from lazyllm.components import EmptyFormatter
>>> toc_prompt='''
... You are now an intelligent assistant. Your task is to understand the user's input and convert the outline into a list of nested dictionaries. Each dictionary contains a `title` and a `describe`, where the `title` should clearly indicate the level using Markdown format, and the `describe` is a description and writing guide for that section.
... 
... Please generate the corresponding list of nested dictionaries based on the following user input:
... 
... Example output:
... [
...     {
...         "title": "# Level 1 Title",
...         "describe": "Please provide a detailed description of the content under this title, offering background information and core viewpoints."
...     },
...     {
...         "title": "## Level 2 Title",
...         "describe": "Please provide a detailed description of the content under this title, giving specific details and examples to support the viewpoints of the Level 1 title."
...     },
...     {
...         "title": "### Level 3 Title",
...         "describe": "Please provide a detailed description of the content under this title, deeply analyzing and providing more details and data support."
...     }
... ]
... User input is as follows:
... '''
>>> query = "Please help me write an article about the application of artificial intelligence in the medical field."
>>> m = lazyllm.TrainableModule("internlm2-chat-20b").prompt(toc_prompt).start()  # the model output without specifying a formatter
>>> ret = m(query, max_new_tokens=2048)
>>> print(f"ret: {ret!r}")
'Based on your user input, here is the corresponding list of nested dictionaries:\n[\n    {\n        "title": "# Application of Artificial Intelligence in the Medical Field",\n        "describe": "Please provide a detailed description of the application of artificial intelligence in the medical field, including its benefits, challenges, and future prospects."\n    },\n    {\n        "title": "## AI in Medical Diagnosis",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in medical diagnosis, including specific examples of AI-based diagnostic tools and their impact on patient outcomes."\n    },\n    {\n        "title": "### AI in Medical Imaging",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in medical imaging, including the advantages of AI-based image analysis and its applications in various medical specialties."\n    },\n    {\n        "title": "### AI in Drug Discovery and Development",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in drug discovery and development, including the role of AI in identifying potential drug candidates and streamlining the drug development process."\n    },\n    {\n        "title": "## AI in Medical Research",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in medical research, including its applications in genomics, epidemiology, and clinical trials."\n    },\n    {\n        "title": "### AI in Genomics and Precision Medicine",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in genomics and precision medicine, including the role of AI in analyzing large-scale genomic data and tailoring treatments to individual patients."\n    },\n    {\n        "title": "### AI in Epidemiology and Public Health",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in epidemiology and public health, including its applications in disease surveillance, outbreak prediction, and resource allocation."\n    },\n    {\n        "title": "### AI in Clinical Trials",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in clinical trials, including its role in patient recruitment, trial design, and data analysis."\n    },\n    {\n        "title": "## AI in Medical Practice",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in medical practice, including its applications in patient monitoring, personalized medicine, and telemedicine."\n    },\n    {\n        "title": "### AI in Patient Monitoring",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in patient monitoring, including its role in real-time monitoring of vital signs and early detection of health issues."\n    },\n    {\n        "title": "### AI in Personalized Medicine",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in personalized medicine, including its role in analyzing patient data to tailor treatments and predict outcomes."\n    },\n    {\n        "title": "### AI in Telemedicine",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in telemedicine, including its applications in remote consultations, virtual diagnoses, and digital health records."\n    },\n    {\n        "title": "## AI in Medical Ethics and Policy",\n        "describe": "Please provide a detailed description of the ethical and policy considerations surrounding the use of artificial intelligence in the medical field, including issues related to data privacy, bias, and accountability."\n    }\n]'
>>> m = lazyllm.TrainableModule("internlm2-chat-20b").formatter(EmptyFormatter()).prompt(toc_prompt).start()  # the model output of the specified formatter
>>> ret = m(query, max_new_tokens=2048)
>>> print(f"ret: {ret!r}")
'Based on your user input, here is the corresponding list of nested dictionaries:\n[\n    {\n        "title": "# Application of Artificial Intelligence in the Medical Field",\n        "describe": "Please provide a detailed description of the application of artificial intelligence in the medical field, including its benefits, challenges, and future prospects."\n    },\n    {\n        "title": "## AI in Medical Diagnosis",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in medical diagnosis, including specific examples of AI-based diagnostic tools and their impact on patient outcomes."\n    },\n    {\n        "title": "### AI in Medical Imaging",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in medical imaging, including the advantages of AI-based image analysis and its applications in various medical specialties."\n    },\n    {\n        "title": "### AI in Drug Discovery and Development",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in drug discovery and development, including the role of AI in identifying potential drug candidates and streamlining the drug development process."\n    },\n    {\n        "title": "## AI in Medical Research",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in medical research, including its applications in genomics, epidemiology, and clinical trials."\n    },\n    {\n        "title": "### AI in Genomics and Precision Medicine",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in genomics and precision medicine, including the role of AI in analyzing large-scale genomic data and tailoring treatments to individual patients."\n    },\n    {\n        "title": "### AI in Epidemiology and Public Health",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in epidemiology and public health, including its applications in disease surveillance, outbreak prediction, and resource allocation."\n    },\n    {\n        "title": "### AI in Clinical Trials",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in clinical trials, including its role in patient recruitment, trial design, and data analysis."\n    },\n    {\n        "title": "## AI in Medical Practice",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in medical practice, including its applications in patient monitoring, personalized medicine, and telemedicine."\n    },\n    {\n        "title": "### AI in Patient Monitoring",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in patient monitoring, including its role in real-time monitoring of vital signs and early detection of health issues."\n    },\n    {\n        "title": "### AI in Personalized Medicine",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in personalized medicine, including its role in analyzing patient data to tailor treatments and predict outcomes."\n    },\n    {\n        "title": "### AI in Telemedicine",\n        "describe": "Please provide a detailed description of how artificial intelligence is used in telemedicine, including its applications in remote consultations, virtual diagnoses, and digital health records."\n    },\n    {\n        "title": "## AI in Medical Ethics and Policy",\n        "describe": "Please provide a detailed description of the ethical and policy considerations surrounding the use of artificial intelligence in the medical field, including issues related to data privacy, bias, and accountability."\n    }\n]'
""")

# FunctionCallFormatter
add_chinese_doc('formatter.formatterbase.FunctionCallFormatter', '''\
函数调用格式化器，用于处理包含函数调用信息的消息字典。

该格式化器专门用于处理函数调用场景下的模型输出，只提取字典中的 'role'、'content' 和 'tool_calls' 字段，过滤掉其他不需要的字段。

主要用于 FunctionCall 等工具调用相关的功能模块。

Args:
    无参数，直接实例化使用。

注意:
    - 输入必须是字典类型，否则会抛出断言错误
    - 只保留字典中存在的 'role'、'content'、'tool_calls' 字段
''')

add_english_doc('formatter.formatterbase.FunctionCallFormatter', '''\
Function call formatter for processing message dictionaries containing function call information.

This formatter is specifically designed for handling model outputs in function calling scenarios. It extracts only the 'role', 'content', and 'tool_calls' fields from the input dictionary, filtering out other unnecessary fields.

Primarily used in function calling-related modules such as FunctionCall.

Args:
    No parameters, instantiate directly.

Note:
    - Input must be a dictionary type, otherwise an assertion error will be raised
    - Only preserves 'role', 'content', and 'tool_calls' fields if they exist in the dictionary
''')

add_example('formatter.formatterbase.FunctionCallFormatter', '''\
>>> from lazyllm.components.formatter.formatterbase import FunctionCallFormatter
>>> formatter = FunctionCallFormatter()
>>> 
>>> # 处理包含函数调用的消息
>>> msg = {
...     'role': 'assistant',
...     'content': 'I will call a function to get the weather.',
...     'tool_calls': [
...         {
...             'id': 'call_123',
...             'type': 'function',
...             'function': {
...                 'name': 'get_weather',
...                 'arguments': '{"location": "Beijing"}'
...             }
...         }
...     ],
...     'other_field': 'will be filtered'
... }
>>> result = formatter.format(msg)
>>> print(result)
{'role': 'assistant', 'content': 'I will call a function to get the weather.', 'tool_calls': [{'id': 'call_123', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"location": "Beijing"}'}}]}
>>> 
>>> # 处理只包含部分字段的消息
>>> msg2 = {
...     'role': 'assistant',
...     'content': 'Hello, how can I help you?'
... }
>>> result2 = formatter.format(msg2)
>>> print(result2)
{'role': 'assistant', 'content': 'Hello, how can I help you?'}
''')

# encode_query_with_filepaths
add_chinese_doc('formatter.encode_query_with_filepaths', '''\
将查询文本和文件路径编码为带有文档上下文的结构化字符串格式。

当指定文件路径时，该函数会将查询内容与文件路径打包成 JSON 格式，并在前缀 ``__lazyllm_docs__`` 的基础上编码返回。否则仅返回原始查询文本。

Args:
    query (str): 用户查询字符串，默认为空字符串。
    files (str or List[str]): 与查询相关的文档路径，可为单个字符串或字符串列表。

**Returns:**\n
- str: 编码后的结构化查询字符串，或原始查询。

Raises:
    AssertionError: 如果 `files` 不是字符串或字符串列表，或列表中元素类型错误。
''')

add_english_doc('formatter.encode_query_with_filepaths', '''\
Encodes a query string together with associated file paths into a structured string format with context.

If file paths are provided, the query and file list will be wrapped into a JSON object prefixed with ``__lazyllm_docs__``. Otherwise, it returns the original query string.

Args:
    query (str): The user query string. Defaults to an empty string.
    files (str or List[str]): File path(s) associated with the query. Can be a single string or a list of strings.

**Returns:**\n
- str: A structured encoded query string or the raw query.

Raises:
    AssertionError: If `files` is not a string or list of strings.
''')

add_example('formatter.encode_query_with_filepaths', '''\
>>> from lazyllm.components.formatter import encode_query_with_filepaths

>>> # Encode a query along with associated documentation files
>>> encode_query_with_filepaths("Generate questions based on the document", files=["a.md"])
'<lazyllm-query>{"query": "Generate questions based on the document", "files": ["a.md"]}'
''')

# decode_query_with_filepaths
add_chinese_doc('formatter.decode_query_with_filepaths', '''\
将结构化查询字符串解析为包含原始查询和文件路径的字典格式。

当输入字符串以特殊前缀 ``__lazyllm_docs__`` 开头时，函数会尝试从中提取 JSON 格式的查询信息；否则将原样返回字符串内容。

Args:
    query_files (str): 编码后的查询字符串，可能包含文档路径和查询内容。

**Returns:**\n
- Union[dict, str]: 若为结构化格式则返回包含 'query' 和 'files' 的字典，否则返回原始查询字符串。

Raises:
    AssertionError: 如果输入参数不是字符串类型。
    ValueError: 如果字符串为结构化格式但解析 JSON 失败。
''')

add_english_doc('formatter.decode_query_with_filepaths', '''\
Decodes a structured query string into a dictionary containing the original query and file paths.

If the input string starts with the special prefix ``__lazyllm_docs__``, it attempts to parse the JSON content; otherwise, it returns the raw query string as-is.

Args:
    query_files (str): The encoded query string that may include both query and file paths.

**Returns:**\n
- Union[dict, str]: A dictionary containing 'query' and 'files' if structured, otherwise the original query string.

Raises:
    AssertionError: If the input is not a string.
    ValueError: If the string is prefixed but JSON decoding fails.
''')

add_example('formatter.decode_query_with_filepaths', '''\
>>> from lazyllm.components.formatter import decode_query_with_filepaths

>>> # Decode a structured query with files
>>> decode_query_with_filepaths('<lazyllm-query>{"query": "Summarize the content", "files": ["doc.md"]}')
{'query': 'Summarize the content', 'files': ['doc.md']}

>>> # Decode a plain string without files
>>> decode_query_with_filepaths("This is just a simple question")
'This is just a simple question'
''')

# lazyllm_merge_query
add_chinese_doc('formatter.lazyllm_merge_query', '''\
将多个查询字符串（可能包含文档路径）合并为一个统一的结构化查询字符串。

每个输入参数可以是普通查询字符串或由 ``encode_query_with_filepaths`` 编码后的结构化字符串。函数会自动解码、拼接查询文本，并合并所有涉及的文档路径，最终重新编码为统一的查询格式。

Args:
    *args (str): 多个查询字符串。每个字符串可以是普通文本或已编码的带文件路径的结构化查询。

**Returns:**\n
- str: 合并后的结构化查询字符串，包含统一的查询内容与文件路径。
''')

add_english_doc('formatter.lazyllm_merge_query', '''\
Merges multiple query strings (potentially with associated file paths) into a single structured query string.

Each argument can be a plain query string or a structured query created by ``encode_query_with_filepaths``. The function decodes each input, concatenates all query texts, and merges the associated file paths. The final result is re-encoded into a single query string with unified context.

Args:
    *args (str): Multiple query strings. Each can be either plain text or an encoded structured query with files.

**Returns:**\n
- str: A single structured query string containing the merged query and file paths.
''')

add_example('formatter.lazyllm_merge_query', '''\
>>> from lazyllm.components.formatter import encode_query_with_filepaths, lazyllm_merge_query

>>> # Merge two structured queries with English content and associated files
>>> q1 = encode_query_with_filepaths("Please summarize document one", files=["doc1.md"])
>>> q2 = encode_query_with_filepaths("Add details from document two", files=["doc2.md"])
>>> lazyllm_merge_query(q1, q2)
'<lazyllm-query>{"query": "Please summarize document oneAdd details from document two", "files": ["doc1.md", "doc2.md"]}'

>>> # Merge plain English text queries without documents
>>> lazyllm_merge_query("What is AI?", "Explain deep learning.")
'What is AI?Explain deep learning.'
''')

# ============= Prompter

# Prompter
add_chinese_doc('Prompter', '''\
用于生成模型输入的Prompt类，支持模板、历史对话拼接与响应抽取。

该类支持从字典、模板名称或文件中加载prompt配置，支持历史对话结构拼接（用于Chat类任务），
可灵活处理有/无history结构的prompt输入，适配非字典类型输入。

Args:
    prompt (Optional[str]): 模板Prompt字符串，支持格式化字段。
    response_split (Optional[str]): 对模型响应进行切分的分隔符，仅用于抽取模型回答。
    chat_prompt (Optional[str]): 多轮对话使用的Prompt模板，必须包含history字段。
    history_symbol (str): 表示历史对话字段的名称，默认为'llm_chat_history'。
    eoa (Optional[str]): 对话中 assistant/user 分隔符。
    eoh (Optional[str]): 多轮history中 user-assistant 分隔符。
    show (bool): 是否打印最终生成的Prompt，默认False。
''')

add_english_doc('Prompter', '''\
Prompt generator class for LLM input formatting. Supports template-based prompting, history injection, and response extraction.

This class allows prompts to be defined via string templates, loaded from dicts, files, or predefined names.
It supports history-aware formatting for multi-turn conversations and adapts to both mapping and string input types.

Args:
    prompt (Optional[str]): Prompt template string with format placeholders.
    response_split (Optional[str]): Optional delimiter to split model response and extract useful output.
    chat_prompt (Optional[str]): Chat template string, must contain a history placeholder.
    history_symbol (str): Name of the placeholder for historical messages, default is 'llm_chat_history'.
    eoa (Optional[str]): Delimiter between assistant/user in history items.
    eoh (Optional[str]): Delimiter between user-assistant pairs.
    show (bool): Whether to print the final prompt when generating. Default is False.
''')

add_example('Prompter', '''\
>>> from lazyllm import Prompter

>>> p = Prompter(prompt="Answer the following: {question}")
>>> p.generate_prompt("What is AI?")
'Answer the following: What is AI?'

>>> p.generate_prompt({"question": "Define machine learning"})
'Answer the following: Define machine learning'

>>> p = Prompter(
...     prompt="Instruction: {instruction}",
...     chat_prompt="Instruction: {instruction}\\\\nHistory:\\\\n{llm_chat_history}",
...     history_symbol="llm_chat_history",
...     eoa="</s>",
...     eoh="|"
... )
>>> p.generate_prompt(
...     input={"instruction": "Translate this."},
...     history=[["hello", "你好"], ["how are you", "你好吗"]]
... )
'Instruction: Translate this.\\\\nHistory:\\\\nhello|你好</s>how are you|你好吗'

>>> prompt_conf = {
...     "prompt": "Task: {task}",
...     "response_split": "---"
... }
>>> p = Prompter.from_dict(prompt_conf)
>>> p.generate_prompt("Summarize this article.")
'Task: Summarize this article.'

>>> full_output = "Task: Summarize this article.---This is the summary."
>>> p.get_response(full_output)
'This is the summary.'
''')

# Prompter.from_dict
add_chinese_doc('Prompter.from_dict', '''\
通过字典配置初始化一个 Prompter 实例。

Args:
    prompt (Dict): 包含 prompt 相关字段的配置字典，需包含 `prompt` 键，其他为可选。
    show (bool): 是否显示生成的 prompt，默认为 False。

**Returns:**\n
- Prompter: 返回一个初始化的 Prompter 实例。
''')

add_english_doc('Prompter.from_dict', '''\
Initializes a Prompter instance from a prompt configuration dictionary.

Args:
    prompt (Dict): A dictionary containing prompt-related configuration. Must include 'prompt' key.
    show (bool): Whether to display the generated prompt. Defaults to False.

**Returns:**\n
- Prompter: An initialized Prompter instance.
''')

# Prompter.from_template
add_chinese_doc('Prompter.from_template', '''\
根据模板名称加载 prompt 配置并初始化 Prompter 实例。

Args:
    template_name (str): 模板名称，必须在 `templates` 中存在。
    show (bool): 是否显示生成的 prompt，默认为 False。

**Returns:**\n
- Prompter: 返回一个初始化的 Prompter 实例。
''')

add_english_doc('Prompter.from_template', '''\
Loads prompt configuration from a template name and initializes a Prompter instance.

Args:
    template_name (str): Name of the template. Must exist in the `templates` dictionary.
    show (bool): Whether to display the generated prompt. Defaults to False.

**Returns:**\n
- Prompter: An initialized Prompter instance.
''')

# Prompter.from_file
add_chinese_doc('Prompter.from_file', '''\
从 JSON 文件中读取配置并初始化 Prompter 实例。

Args:
    fname (str): JSON 配置文件路径。
    show (bool): 是否显示生成的 prompt，默认为 False。

Returns:
    Prompter: 返回一个初始化的 Prompter 实例。
''')

add_english_doc('Prompter.from_file', '''\
Loads prompt configuration from a JSON file and initializes a Prompter instance.

Args:
    fname (str): Path to the JSON configuration file.
    show (bool): Whether to display the generated prompt. Defaults to False.

Returns:
    Prompter: An initialized Prompter instance.
''')

# Prompter.empty
add_chinese_doc('Prompter.empty', '''\
创建一个空的 Prompter 实例。

Returns:
    Prompter: 返回一个无 prompt 配置的 Prompter 实例。
''')

add_english_doc('Prompter.empty', '''\
Creates an empty Prompter instance.

Returns:
    Prompter: A Prompter instance without any prompt configuration.
''')

# Prompter.generate_prompt
add_chinese_doc('Prompter.generate_prompt', '''\
根据输入和可选的历史记录生成最终 Prompt。

Args:
    input (Union[str, Dict]): 用户输入。可以是字符串或包含多字段的字典。
    history (Optional[List[List[str]]]): 多轮对话历史，例如 [['u1', 'a1'], ['u2', 'a2']]。
    tools (Optional[Any]): 目前未支持工具调用，此字段必须为 None。
    label (Optional[str]): 附加在 prompt 末尾的标签，通常用于训练。
    show (bool): 是否显示生成的 prompt，默认 False。

Returns:
    str: 格式化后的 prompt 字符串。
''')

add_english_doc('Prompter.generate_prompt', '''\
Generates a formatted prompt string based on input and optional conversation history.

Args:
    input (Union[str, Dict]): User input. Can be a single string or a dictionary with multiple fields.
    history (Optional[List[List[str]]]): Multi-turn dialogue history, e.g., [['u1', 'a1'], ['u2', 'a2']].
    tools (Optional[Any]): Not supported. Must be None.
    label (Optional[str]): Optional label to append to the prompt, commonly used for training.
    show (bool): Whether to print the generated prompt. Defaults to False.

Returns:
    str: The final formatted prompt string.
''')

# Prompter.get_response
add_chinese_doc('Prompter.get_response', '''\
从 LLM 返回结果中提取模型的回答内容。

Args:
    response (str): 模型完整响应文本。
    input (Optional[str]): 如果模型输出以输入开头，将会自动去除输入部分。

Returns:
    str: 提取后的模型响应内容。
''')

add_english_doc('Prompter.get_response', '''\
Extracts the actual model answer from the full response returned by an LLM.

Args:
    response (str): The full raw output from the model.
    input (Optional[str]): If the response starts with the input, that part will be removed.

Returns:
    str: The cleaned model response.
''')

add_chinese_doc('prompter.PrompterBase', '''\
Prompter的基类，自定义的Prompter需要继承此基类，并通过基类提供的 ``_init_prompt`` 函数来设置Prompt模板和Instruction的模板，以及截取结果所使用的字符串。可以查看 :[prompt](/Best%20Practice/prompt) 进一步了解Prompt的设计思想和使用方式。

Prompt模板和Instruction模板都用 ``{}`` 表示要填充的字段，其中Prompt可包含的字段有 ``system``, ``history``, ``tools``, ``user`` 等，而instruction_template可包含的字段为 ``instruction`` 和 ``extra_keys`` 。
``instruction`` 由应用的开发者传入， ``instruction`` 中也可以带有 ``{}`` 用于让定义可填充的字段，方便用户填入额外的信息。如果 ``instruction`` 字段为字符串，则认为是系统instruction；如果是字典，则它包含的key只能是 ``user`` 和 ``system`` 两种选择。 ``user`` 表示用户输入的instruction，在prompt中放在用户输入前面， ``system`` 表示系统instruction，在prompt中凡在system prompt后面。
''')

add_english_doc('prompter.PrompterBase', '''\
The base class of Prompter. A custom Prompter needs to inherit from this base class and set the Prompt template and the Instruction template using the `_init_prompt` function provided by the base class, as well as the string used to capture results. Refer to  [prompt](/Best%20Practice/prompt) for further understanding of the design philosophy and usage of Prompts.

Both the Prompt template and the Instruction template use ``{}`` to indicate the fields to be filled in. The fields that can be included in the Prompt are `system`, `history`, `tools`, `user` etc., while the fields that can be included in the instruction_template are `instruction` and `extra_keys`. If the ``instruction`` field is a string, it is considered as a system instruction; if it is a dictionary, it can only contain the keys ``user`` and ``system``. ``user`` represents the user input instruction, which is placed before the user input in the prompt, and ``system`` represents the system instruction, which is placed after the system prompt in the prompt.
``instruction`` is passed in by the application developer, and the ``instruction`` can also contain ``{}`` to define fillable fields, making it convenient for users to input additional information.
''')

add_example('prompter.PrompterBase', '''\
>>> from lazyllm.components.prompter import PrompterBase
>>> class MyPrompter(PrompterBase):
...     def __init__(self, instruction = None, extra_keys = None, show = False):
...         super(__class__, self).__init__(show)
...         instruction_template = f'{instruction}\\\\n{{extra_keys}}\\\\n'.replace('{extra_keys}', PrompterBase._get_extro_key_template(extra_keys))
...         self._init_prompt("<system>{system}</system>\\\\n</instruction>{instruction}</instruction>{history}\\\\n{input}\\\\n, ## Response::", instruction_template, '## Response::')
... 
>>> p = MyPrompter('ins {instruction}')
>>> p.generate_prompt('hello')
'<system>You are an AI-Agent developed by LazyLLM.</system>\\\\n</instruction>ins hello\\\\n\\\\n</instruction>\\\\n\\\\n, ## Response::'
>>> p.generate_prompt('hello world', return_dict=True)
{'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\\\nins hello world\\\\n\\\\n'}, {'role': 'user', 'content': ''}]}
''')

add_chinese_doc('prompter.PrompterBase.pre_hook', '''\
设置预处理钩子函数，供外部在生成提示词前对输入数据进行自定义处理。

Args:
    func (Optional[Callable]): 一个可调用对象，作为预处理钩子函数，接收并处理输入数据。

**Returns:**\n
- LazyLLMPrompterBase: 返回自身实例，方便链式调用。
''')

add_english_doc('prompter.PrompterBase.pre_hook', '''\
Sets a pre-processing hook function, allowing external custom processing of input data before prompt generation.

Args:
    func (Optional[Callable]): A callable object to be used as the pre-processing hook function, which receives and processes input data.

**Returns:**\n
- LazyLLMPrompterBase: Returns the instance itself to support method chaining.
''')

add_chinese_doc('prompter.PrompterBase.generate_prompt', '''\
根据用户输入，生成对应的Prompt.

Args:
    input (Option[str | Dict]):  Prompter的输入，如果是dict，会填充到instruction的槽位中；如果是str，则会作为输入。
    history (Option[List[List | Dict]]): 历史对话，可以为 ``[[u, s], [u, s]]`` 或 openai的history格式，默认为None。
    tools (Option[List[Dict]]: 可以使用的工具合集，大模型用作FunctionCall时使用，默认为None
    label (Option[str]): 标签，训练或微调时使用，默认为None
    show (bool): 标志是否打印生成的Prompt，默认为False
    return_dict (bool): 标志是否返回dict，一般情况下使用 ``OnlineChatModule`` 时会设置为True。如果返回dict，则仅填充 ``instruction``。默认为False
''')

add_english_doc('prompter.PrompterBase.generate_prompt', '''\

Generate a corresponding Prompt based on user input.

Args:
    input (Option[str | Dict]): The input from the prompter, if it's a dict, it will be filled into the slots of the instruction; if it's a str, it will be used as input.
    history (Option[List[List | Dict]]): Historical conversation, can be ``[[u, s], [u, s]]`` or in openai's history format, defaults to None.
    tools (Option[List[Dict]]): A collection of tools that can be used, used when the large model performs FunctionCall, defaults to None.
    label (Option[str]): Label, used during fine-tuning or training, defaults to None.
    show (bool): Flag indicating whether to print the generated Prompt, defaults to False.
    return_dict (bool): Flag indicating whether to return a dict, generally set to True when using ``OnlineChatModule``. If returning a dict, only the ``instruction`` will be filled. Defaults to False.
''')

add_chinese_doc('prompter.PrompterBase.get_response', '''\
用作对Prompt的截断，只保留有价值的输出

Args:
     output (str): 大模型的输出
     input (Option[[str]): 大模型的输入，若指定此参数，会将输出中包含输入的部分全部截断，默认为None
''')

add_english_doc('prompter.PrompterBase.get_response', '''\
Used to truncate the Prompt, keeping only valuable output.

Args:
        output (str): The output of the large model.
        input (Option[str]): The input of the large model. If this parameter is specified, any part of the output that includes the input will be completely truncated. Defaults to None.
''')

# EmptyPrompter
add_chinese_doc('prompter.EmptyPrompter', '''\
继承自 `LazyLLMPrompterBase` 的空提示生成器，用于直接返回原始输入。

该类不会对输入进行任何处理，适用于无需格式化的调试、测试或占位场景。
''')

add_english_doc('prompter.EmptyPrompter', '''\
An empty prompt generator that inherits from `LazyLLMPrompterBase`, and directly returns the original input.

This class performs no formatting and is useful for debugging, testing, or as a placeholder.
''')

add_example('prompter.EmptyPrompter', '''\
>>> from lazyllm.components.prompter import EmptyPrompter
>>> prompter = EmptyPrompter()
>>> prompter.generate_prompt("Hello LazyLLM")
'Hello LazyLLM'
>>> prompter.generate_prompt({"query": "Tell me a joke"})
{'query': 'Tell me a joke'}
>>> # Even with additional parameters, the input is returned unchanged
>>> prompter.generate_prompt("No-op", history=[["Hi", "Hello"]], tools=[{"name": "search"}], label="debug")
'No-op'
''')

# EmptyPrompter.generate_prompt
add_chinese_doc('prompter.EmptyPrompter.generate_prompt', '''\
直接返回输入的Prompt实现，继承自 `LazyLLMPrompterBase`。

该方法不会对输入做任何格式化操作，适用于调试、测试或占位场景。

Args:
    input (Any): 任意输入，作为Prompt返回。
    history (Option[List[List | Dict]]): 历史对话，可忽略，默认None。
    tools (Option[List[Dict]]): 工具参数，可忽略，默认None。
    label (Option[str]): 标签，可忽略，默认None。
    show (bool): 是否打印返回内容，默认为False。
''')

add_english_doc('prompter.EmptyPrompter.generate_prompt', '''\
A prompt passthrough implementation that inherits from `LazyLLMPrompterBase`.

This method directly returns the input without any formatting. Useful for debugging, testing, or placeholder use.

Args:
    input (Any): The input to be returned directly as the prompt.
    history (Option[List[List | Dict]]): Dialogue history, ignored. Defaults to None.
    tools (Option[List[Dict]]): Tool definitions, ignored. Defaults to None.
    label (Option[str]): Label, ignored. Defaults to None.
    show (bool): Whether to print the returned prompt. Defaults to False.
''')

add_english_doc('prompter.builtinPrompt.LazyLLMPrompterBase', '''\
LazyLLM prompter base class for managing and generating model prompts.

Args:
    show (bool): Whether to display generated prompts, defaults to False.
    tools (Optional[List]): List of available tools, defaults to None.
    history (Optional[List]): Conversation history, defaults to None.

Attributes:
    ISA (str): Instruction separator start token "<!lazyllm-spliter!>".\n
    ISE (str): Instruction separator end token "</!lazyllm-spliter!>".\n

Configuration Items:
    system: System role setting \n
    sos/eos: Session start/end markers \n
    soh/eoh: Human input start/end markers \n
    soa/eoa: AI response start/end markers \n
    soe/eoe: Tool execution result start/end markers \n
    tool_start_token/tool_end_token: Tool call start/end markers \n
    tool_args_token: Tool arguments marker \n
''')

add_chinese_doc('prompter.builtinPrompt.LazyLLMPrompterBase', '''\
LazyLLM提示词基类，用于管理和生成模型提示词。

Args:
    show (bool): 是否显示生成的提示词，默认为False。
    tools (Optional[List]): 可用工具列表，默认为None。
    history (Optional[List]): 对话历史记录，默认为None。

Attributes:
    ISA (str): 指令分隔符起始标记 "<!lazyllm-spliter!>"。\n
    ISE (str): 指令分隔符结束标记 "</!lazyllm-spliter!>"。\n

Configuration Items:
    - system: 系统角色设定\n
    - sos/eos: 会话开始/结束标记\n
    - soh/eoh: 人类输入开始/结束标记\n
    - soa/eoa: AI回复开始/结束标记\n
    - soe/eoe: 工具执行结果开始/结束标记\n
    - tool_start_token/tool_end_token: 工具调用开始/结束标记\n
    - tool_args_token: 工具参数标记\n
''')

add_chinese_doc('AlpacaPrompter', '''\
Alpaca格式的Prompter，支持工具调用，不支持历史对话。

Args:
    instruction (Option[str]): 大模型的任务指令，至少带一个可填充的槽位(如 ``{instruction}``)。或者使用字典指定 ``system`` 和 ``user`` 的指令。
    extra_keys (Option[List]): 额外的字段，用户的输入会填充这些字段。
    show (bool): 标志是否打印生成的Prompt，默认为False
    tools (Option[list]): 大模型可以使用的工具集合，默认为None
''')

add_english_doc('AlpacaPrompter', '''\
Alpaca-style Prompter, supports tool calls, does not support historical dialogue.


Args:
    instruction (Option[str]): Task instructions for the large model, with at least one fillable slot (e.g. ``{instruction}``). Or use a dictionary to specify the ``system`` and ``user`` instructions.
    extra_keys (Option[List]): Additional fields that will be filled with user input.
    show (bool): Flag indicating whether to print the generated Prompt, default is False.
    tools (Option[list]): Tool-set which is provived for LLMs, default is None.
''')

add_example('AlpacaPrompter', '''\
>>> from lazyllm import AlpacaPrompter
>>> p = AlpacaPrompter('hello world {instruction}')
>>> p.generate_prompt('this is my input')
'You are an AI-Agent developed by LazyLLM.\\\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\\\n\\\\n ### Instruction:\\\\nhello world this is my input\\\\n\\\\n\\\\n### Response:\\\\n'
>>> p.generate_prompt('this is my input', return_dict=True)
{'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\\\n\\\\n ### Instruction:\\\\nhello world this is my input\\\\n\\\\n'}, {'role': 'user', 'content': ''}]}
>>>
>>> p = AlpacaPrompter('hello world {instruction}, {input}', extra_keys=['knowledge'])
>>> p.generate_prompt(dict(instruction='hello world', input='my input', knowledge='lazyllm'))
'You are an AI-Agent developed by LazyLLM.\\\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\\\n\\\\n ### Instruction:\\\\nhello world hello world, my input\\\\n\\\\nHere are some extra messages you can referred to:\\\\n\\\\n### knowledge:\\\\nlazyllm\\\\n\\\\n\\\\n### Response:\\\\n'
>>> p.generate_prompt(dict(instruction='hello world', input='my input', knowledge='lazyllm'), return_dict=True)
{'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\\\n\\\\n ### Instruction:\\\\nhello world hello world, my input\\\\n\\\\nHere are some extra messages you can referred to:\\\\n\\\\n### knowledge:\\\\nlazyllm\\\\n\\\\n'}, {'role': 'user', 'content': ''}]}
>>>
>>> p = AlpacaPrompter(dict(system="hello world", user="this is user instruction {input}"))
>>> p.generate_prompt(dict(input="my input"))
'You are an AI-Agent developed by LazyLLM.\\\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\\\n\\\\n ### Instruction:\\\\nhello word\\\\n\\\\n\\\\n\\\\nthis is user instruction my input### Response:\\\\n'
>>> p.generate_prompt(dict(input="my input"), return_dict=True)
{'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\\\n\\\\n ### Instruction:\\\\nhello world'}, {'role': 'user', 'content': 'this is user instruction my input'}]}
''')

add_chinese_doc('ChatPrompter', '''\
用于多轮对话的大模型Prompt构造器，继承自 `LazyLLMPrompterBase`。

支持工具调用、历史对话与自定义指令模版。支持传入 system/user 拆分的指令结构，自动合并为统一模板。支持额外字段注入和打印提示信息。

Args:
    instruction (Option[str | Dict[str, str]]): Prompt模板指令，可为字符串或包含 `system` 和 `user` 的字典。若为字典，将自动拼接并注入特殊标记分隔符。
    extra_keys (Option[List[str]]): 额外的字段列表，用户输入中的内容会被插入对应槽位，用于丰富上下文。
    show (bool): 是否打印生成的Prompt，默认False。
    tools (Option[List]): 可选的工具列表，用于FunctionCall任务，默认None。
    history (Option[List[List[str]]]): 可选的历史对话，用于对话记忆，格式为[[user, assistant], ...]，默认None。
''')

add_english_doc('ChatPrompter', '''\
Prompt constructor for multi-turn dialogue, inherits from `LazyLLMPrompterBase`.

Supports tool calling, conversation history, and customizable instruction templates. Accepts instructions as either plain string or dict with separate `system` and `user` components, automatically merging them into a unified prompt template. Also supports injecting extra user-defined fields.

Args:
    instruction (Option[str | Dict[str, str]]): The prompt instruction template. Can be a string or a dict with `system` and `user` keys. If a dict is given, the components will be merged using special delimiters.
    extra_keys (Option[List[str]]): A list of additional keys that will be filled by user input to enrich the prompt context.
    show (bool): Whether to print the generated prompt. Default is False.
    tools (Option[List]): A list of tools available to the model for function-calling tasks. Default is None.
    history (Option[List[List[str]]]): Dialogue history in the format [[user, assistant], ...]. Used to provide conversational memory. Default is None.
''')

add_example('ChatPrompter', '''\
>>> from lazyllm import ChatPrompter

- Simple instruction string
>>> p = ChatPrompter('hello world')
>>> p.generate_prompt('this is my input')
'You are an AI-Agent developed by LazyLLM.hello world\\\\nthis is my input\\\\n'

>>> p.generate_prompt('this is my input', return_dict=True)
{'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\\\nhello world'}, {'role': 'user', 'content': 'this is my input'}]}

- Using extra_keys
>>> p = ChatPrompter('hello world {instruction}', extra_keys=['knowledge'])
>>> p.generate_prompt({
...     'instruction': 'this is my ins',
...     'input': 'this is my inp',
...     'knowledge': 'LazyLLM-Knowledge'
... })
'You are an AI-Agent developed by LazyLLM.hello world this is my ins\\\\nHere are some extra messages you can referred to:\\\\n\\\\n### knowledge:\\\\nLazyLLM-Knowledge\\\\nthis is my inp\\\\n'

- With conversation history
>>> p.generate_prompt({
...     'instruction': 'this is my ins',
...     'input': 'this is my inp',
...     'knowledge': 'LazyLLM-Knowledge'
... }, history=[['s1', 'e1'], ['s2', 'e2']])
'You are an AI-Agent developed by LazyLLM.hello world this is my ins\\\\nHere are some extra messages you can referred to:\\\\n\\\\n### knowledge:\\\\nLazyLLM-Knowledge\\\\ns1|e1\\\\ns2|e2\\\\nthis is my inp\\\\n'

- Using dict format for system/user instructions
>>> p = ChatPrompter(dict(system="hello world", user="this is user instruction {input}"))
>>> p.generate_prompt({'input': "my input", 'query': "this is user query"})
'You are an AI-Agent developed by LazyLLM.hello world\\\\nthis is user instruction my input this is user query\\\\n'

>>> p.generate_prompt({'input': "my input", 'query': "this is user query"}, return_dict=True)
{'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\\\nhello world'}, {'role': 'user', 'content': 'this is user instruction my input this is user query'}]}
''')

# ============= MultiModal

add_english_doc('StableDiffusionDeploy', '''\
Stable Diffusion Model Deployment Class. This class is used to deploy the stable diffusion model to a specified server for network invocation.

Args:
    launcher (Optional[LazyLLMLaunchersBase], optional): Launcher instance. Defaults to ``None``
    log_path (Optional[str], optional): Log file path. Defaults to ``None``
    trust_remote_code (bool, optional): Whether to trust remote code. Defaults to ``True``
    port (Optional[int], optional): Service port number. Defaults to ``None``
''')

add_chinese_doc('StableDiffusionDeploy', '''\
Stable Diffusion 模型部署类。该类用于将SD模型部署到指定服务器上，以便可以通过网络进行调用。

Args:
    launcher (Optional[LazyLLMLaunchersBase], optional): 启动器实例。默认为 ``None``
    log_path (Optional[str], optional): 日志文件路径。默认为 ``None``
    trust_remote_code (bool, optional): 是否信任远程代码。默认为 ``True``
    port (Optional[int], optional): 服务端口号。默认为 ``None``

''')

add_example('StableDiffusionDeploy', ['''\
>>> from lazyllm import launchers, UrlModule
>>> from lazyllm.components import StableDiffusionDeploy
>>> deployer = StableDiffusionDeploy(launchers.remote())
>>> url = deployer(base_model='stable-diffusion-3-medium')
>>> model = UrlModule(url=url)
>>> res = model('a tiny cat.')
>>> print(res)
... <lazyllm-query>{"query": "", "files": ["path/to/sd3/image_xxx.png"]}
'''])

add_english_doc('ChatTTSDeploy', '''\
ChatTTS Model Deployment Class.

Keyword Args: 
    keys_name_handle (dict): A key mapping dictionary used to handle parameter name conversion between 
                            internal and external API interfaces. Defaults to `{'inputs': 'inputs'}`.

    message_format (dict): The request payload structure containing three main sections: \n
        - `inputs` (str): The raw text content to be synthesized into speech. \n
        - `refinetext` (dict): Text refinement and stylization parameters controlling speech expression: \n
            * `prompt` (str): Voice style control tags, e.g., "[oral_2][laugh_0][break_6]" \n
            * `top_P` (float): Nucleus sampling parameter for decoding strategy (default: 0.7) \n
            * `top_K` (int): Top-K sampling parameter (default: 20) \n
            * `temperature` (float): Sampling temperature controlling randomness (default: 0.7) \n
            * `repetition_penalty` (float): Repetition penalty to avoid redundant generation (default: 1.0) \n
            * `max_new_token` (int): Maximum number of tokens to generate (default: 384) \n
            * `min_new_token` (int): Minimum number of tokens to generate (default: 0) \n
            * `show_tqdm` (bool): Whether to display progress bar during generation (default: True) \n
            * `ensure_non_empty` (bool): Ensure non-empty generation result (default: True) \n
        - `infercode` (dict): Inference and encoding parameters affecting audio quality: \n
            * `prompt` (str): Voice speed control tags, e.g., "[speed_5]" \n
            * `spk_emb` (Optional): Speaker embedding vector for specifying voice characteristics (default: None) \n
            * `temperature` (float): Sampling temperature for audio generation (default: 0.3) \n
            * `repetition_penalty` (float): Repetition penalty coefficient (default: 1.05) \n
            * `max_new_token` (int): Maximum number of tokens for audio generation (default: 2048) \n
''')

add_chinese_doc('ChatTTSDeploy', '''\
ChatTTS 模型部署类。

Keyword Args: 
    keys_name_handle (dict): 键名映射字典，用于处理内部和外部API接口之间的参数名称转换。
                            默认为 `{'inputs': 'inputs'}`。

    message_format (dict): 请求负载结构，包含三个主要部分：\n
        - `inputs` (str): 要合成为语音的原始文本内容。\n
        - `refinetext` (dict): 文本细化和风格化参数，控制语音表达：\n
            * `prompt` (str): 语音风格控制标签，例如："[oral_2][laugh_0][break_6]"\n
            * `top_P` (float): 核采样参数，用于解码策略（默认值：0.7）\n
            * `top_K` (int): Top-K 采样参数（默认值：20）\n
            * `temperature` (float): 采样温度，控制随机性（默认值：0.7）\n
            * `repetition_penalty` (float): 重复惩罚，避免冗余生成（默认值：1.0）\n
            * `max_new_token` (int): 最大生成token数（默认值：384）\n
            * `min_new_token` (int): 最小生成token数（默认值：0）\n
            * `show_tqdm` (bool): 是否在生成过程中显示进度条（默认值：True）\n
            * `ensure_non_empty` (bool): 确保生成非空结果（默认值：True）\n
        - `infercode` (dict): 推理和编码参数，影响音频质量：\n
            * `prompt` (str): 语速控制标签，例如："[speed_5]"\n
            * `spk_emb` (可选): 说话人嵌入向量，用于指定音色特征（默认值：None）\n
            * `temperature` (float): 音频生成的采样温度（默认值：0.3）\n
            * `repetition_penalty` (float): 重复惩罚系数（默认值：1.05）\n
            * `max_new_token` (int): 音频生成的最大token数（默认值：2048）\n
''')

add_example('ChatTTSDeploy', ['''\
>>> from lazyllm import launchers, UrlModule
>>> from lazyllm.components import ChatTTSDeploy
>>> deployer = ChatTTSDeploy(launchers.remote())
>>> url = deployer(base_model='ChatTTS')
>>> model = UrlModule(url=url)
>>> res = model('Hello World!')
>>> print(res)
... <lazyllm-query>{"query": "", "files": ["path/to/chattts/sound_xxx.wav"]}
'''])

add_english_doc('BarkDeploy', '''\
Bark Model Deployment Class. This class is used to deploy the Bark model to a specified server for network invocation.

`__init__(self, launcher=None)`
Constructor, initializes the deployment class.

Args:
    launcher (lazyllm.launcher): An instance of the launcher used to start the remote service.

`__call__(self, finetuned_model=None, base_model=None)`
Deploys the model and returns the remote service address.

Args:
    finetuned_model (str): If provided, this model will be used for deployment; if not provided or the path is invalid, `base_model` will be used.
    base_model (str): The default model, which will be used for deployment if `finetuned_model` is invalid.
    Return (str): The URL address of the remote service.

Notes:
    - Input for infer: `str`.  The text corresponding to the audio to be generated.
    - Return of infer: The string encoded from the generated file paths, starting with the encoding flag "<lazyllm-query>", followed by the serialized dictionary. The key `files` in the dictionary stores a list, with elements being the paths of the generated audio files.
    - Supported models: [bark](https://huggingface.co/suno/bark)
''')

add_chinese_doc('BarkDeploy', '''\
Bark 模型部署类。该类用于将Bark模型部署到指定服务器上，以便可以通过网络进行调用。

`__init__(self, launcher=None)`
构造函数，初始化部署类。

Args:
    launcher(lazyllm.launcher): 用于启动远程服务的启动器实例。

`__call__(self, finetuned_model=None, base_model=None)`
部署模型，并返回远程服务地址。

Args: 
    finetuned_model (str): 如果提供，则使用该模型进行部署；如果未提供或路径无效，则使用 `base_model`。
    base_model (str): 默认模型，如果 `finetuned_model` 无效，则使用该模型进行部署。
    返回值 (str): 远程服务的URL地址。

Notes:
    - 推理的输入：字符串。待生成音频的对应文字。
    - 推理的返回值：从生成的文件路径编码的字符串， 编码标志以 "<lazyllm-query>"开头，后面跟序列化后的字典, 字典中 `files`键存放了一个列表，元素是生成的音频文件路径。
    - 支持的模型为：[bark](https://huggingface.co/suno/bark)
''')

add_example('BarkDeploy', ['''\
>>> from lazyllm import launchers, UrlModule
>>> from lazyllm.components import BarkDeploy
>>> deployer = BarkDeploy(launchers.remote())
>>> url = deployer(base_model='bark')
>>> model = UrlModule(url=url)
>>> res = model('Hello World!')
>>> print(res)
... <lazyllm-query>{"query": "", "files": ["path/to/bark/sound_xxx.wav"]}
'''])

add_english_doc('MusicGenDeploy', '''\
MusicGen Model Deployment Class. This class is used to deploy the MusicGen model to a specified server for network invocation.

`__init__(self, launcher=None)`
Constructor, initializes the deployment class.

Args:
    launcher (lazyllm.launcher): An instance of the launcher used to start the remote service.

`__call__(self, finetuned_model=None, base_model=None)`
Deploys the model and returns the remote service address.

Args:
    finetuned_model (str): If provided, this model will be used for deployment; if not provided or the path is invalid, `base_model` will be used.
    base_model (str): The default model, which will be used for deployment if `finetuned_model` is invalid.
    Return (str): The URL address of the remote service.

Notes:
    - Input for infer: `str`.  The text corresponding to the audio to be generated.
    - Return of infer: The string encoded from the generated file paths, starting with the encoding flag "<lazyllm-query>", followed by the serialized dictionary. The key `files` in the dictionary stores a list, with elements being the paths of the generated audio files.
    - Supported models: [musicgen-small](https://huggingface.co/facebook/musicgen-small)
''')

add_chinese_doc('MusicGenDeploy', '''\
MusicGen 模型部署类。该类用于将MusicGen模型部署到指定服务器上，以便可以通过网络进行调用。

`__init__(self, launcher=None)`
构造函数，初始化部署类。

Args:
    launcher(lazyllm.launcher): 用于启动远程服务的启动器实例。

`__call__(self, finetuned_model=None, base_model=None)`
部署模型，并返回远程服务地址。

Args: 
    finetuned_model (str): 如果提供，则使用该模型进行部署；如果未提供或路径无效，则使用 `base_model`。
    base_model (str): 默认模型，如果 `finetuned_model` 无效，则使用该模型进行部署。
    返回值 (str): 远程服务的URL地址。

Notes:
    - 推理的输入：字符串。待生成音频的对应文字。
    - 推理的返回值：从生成的文件路径编码的字符串， 编码标志以 "<lazyllm-query>"开头，后面跟序列化后的字典, 字典中 `files`键存放了一个列表，元素是生成的音频文件路径。
    - 支持的模型为：[musicgen-small](https://huggingface.co/facebook/musicgen-small)
''')

add_example('MusicGenDeploy', ['''\
>>> from lazyllm import launchers, UrlModule
>>> from lazyllm.components import MusicGenDeploy
>>> deployer = MusicGenDeploy(launchers.remote())
>>> url = deployer(base_model='musicgen-small')
>>> model = UrlModule(url=url)
>>> model('Symphony with flute as the main melody')
... <lazyllm-query>{"query": "", "files": ["path/to/musicgen/sound_xxx.wav"]}
'''])

add_english_doc('SenseVoiceDeploy', '''\
SenseVoice Model Deployment Class. This class is used to deploy the SenseVoice model to a specified server for network invocation.

`__init__(self, launcher=None)`
Constructor, initializes the deployment class.

Args:
    launcher (Optional[LazyLLMLaunchersBase]): Launcher instance, defaults to None.
    log_path (Optional[str]): Log file path, defaults to None.
    trust_remote_code (bool): Whether to trust remote code, defaults to True.
    port (Optional[int]): Service port number, defaults to None.

Notes:
    - Input for infer: `str`. The audio path or link.
    - Return of infer: `str`. The recognized content.
    - Supported models: [SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)
''')

add_chinese_doc('SenseVoiceDeploy', '''\
SenseVoice 模型部署类。该类用于将SenseVoice模型部署到指定服务器上，以便可以通过网络进行调用。

`__init__(self, launcher=None)`
构造函数，初始化部署类。

Args:
    launcher (Optional[LazyLLMLaunchersBase]): Launcher instance, defaults to None.
    log_path (Optional[str]): Log file path, defaults to None.
    trust_remote_code (bool): Whether to trust remote code, defaults to True.
    port (Optional[int]): Service port number, defaults to None.

Notes:
    - 推理的输入：字符串。音频路径或者链接。
    - 推理的返回值：字符串。识别出的内容。
    - 支持的模型为：[SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)
''')

add_example('SenseVoiceDeploy', ['''\
>>> import os
>>> import lazyllm
>>> from lazyllm import launchers, UrlModule
>>> from lazyllm.components import SenseVoiceDeploy
>>> deployer = SenseVoiceDeploy(launchers.remote())
>>> url = deployer(base_model='SenseVoiceSmall')
>>> model = UrlModule(url=url)
>>> model('path/to/audio') # support format: .mp3, .wav
... xxxxxxxxxxxxxxxx
'''])

add_chinese_doc('deploy.speech_to_text.sense_voice.SenseVoice', '''\
SenseVoice 类，封装了基于 FunASR 的语音转文本模型加载与调用逻辑。  
支持懒加载、自动模型下载，输入可为字符串路径、URL 或包含音频的字典。  

Args:
    base_path (str): 模型路径或标识符，将通过 ModelManager 下载到本地。  
    source (Optional[str]): 模型来源，若未指定则使用 ``lazyllm.config['model_source']``。  
    init (bool): 是否在初始化时立即加载模型，默认为 ``False``。  

Attributes:
    base_path (str): 下载或解析后的模型路径。  
    model (Optional[funasr.AutoModel]): FunASR 语音识别模型实例，初始化后可用。  
    init_flag: 用于懒加载的标志，确保模型只加载一次。  
''')

add_english_doc('deploy.speech_to_text.sense_voice.SenseVoice', '''\
The SenseVoice class encapsulates FunASR-based speech-to-text model loading and invocation.  
It supports lazy initialization, automatic model downloading, and accepts string paths, URLs, or dicts containing audio.  

Args:
    base_path (str): Model path or identifier, downloaded locally via ModelManager.  
    source (Optional[str]): Model source, defaults to ``lazyllm.config['model_source']`` if not specified.  
    init (bool): Whether to load the model immediately during initialization. Defaults to ``False``.  

Attributes:
    base_path (str): Resolved local path of the downloaded model.  
    model (Optional[funasr.AutoModel]): Instance of the FunASR speech recognition model, available after initialization.  
    init_flag: A flag used for lazy loading, ensuring the model is loaded only once.  
''')

add_english_doc('deploy.speech_to_text.sense_voice.SenseVoice.load_stt', '''\
Initializes and loads the FunASR speech-to-text model. Supports Huawei NPU acceleration if `torch_npu` is available.

Uses `fsmn-vad` for voice activity detection (VAD), supporting long utterances.
Maximum single segment duration is set to 30 seconds.
Default inference device is `cuda:0` (GPU).

The loaded model is assigned to `self.model` for subsequent audio transcription.

Note:
- If the environment has `torch_npu` installed, the method will import it to enable Ascend NPU acceleration.
''')

add_chinese_doc('deploy.speech_to_text.sense_voice.SenseVoice.load_stt', '''\
初始化并加载 FunASR 语音转文本模型，如果存在 `torch_npu` 则支持华为 NPU 加速。

使用 `fsmn-vad` 进行语音活动检测（VAD），支持长语音段。
单段语音最大持续时间为 30 秒。
默认推理设备为 `cuda:0`（GPU）。

加载的模型将保存在 `self.model` 中，用于后续音频转写。

注意：
- 如果当前环境中存在 `torch_npu`，函数会导入以支持昇腾 NPU 加速。
''')

add_chinese_doc('deploy.speech_to_text.sense_voice.SenseVoice.rebuild', '''\
类方法，用于在反序列化过程中重新构建 `SenseVoice` 实例（例如使用 `cloudpickle`）。  

Args:
    base_path (str): 语音识别模型路径。  
    init (bool): 实例化时是否立即初始化并加载模型。

**Returns:**\n
- SenseVoice: 返回一个新的 `SenseVoice` 实例，用于支持序列化/多进程兼容。
''')

add_english_doc('deploy.speech_to_text.sense_voice.SenseVoice.rebuild', '''\
Class method to reconstruct a `SenseVoice` instance during deserialization (e.g., with `cloudpickle`).  

Args:
    base_path (str): Path to the speech-to-text model.  
    init (bool): Whether to initialize and load the model upon instantiation.

**Returns:**\n
- SenseVoice: A new `SenseVoice` instance, used for serialization/multiprocessing compatibility.
''')

add_english_doc('deploy.text_to_speech.TTSDeploy', '''\
TTSDeploy is a factory class for creating instances of different Text-to-Speech (TTS) deployment types based on the specified name.

Args:
    name: A string specifying the type of deployment instance to be created.
    **kwarg: Keyword arguments to be passed to the constructor of the corresponding deployment instance.

Returns:
    If the name argument is 'bark', an instance of [BarkDeploy][lazyllm.components.BarkDeploy] is returned.
    If the name argument is 'ChatTTS', an instance of [ChatTTSDeploy][lazyllm.components.ChatTTSDeploy] is returned.
    If the name argument starts with 'musicgen', an instance of [MusicGenDeploy][lazyllm.components.MusicGenDeploy] is returned.
    If the name argument does not match any of the above cases, a RuntimeError exception is raised, indicating the unsupported model.            
''')

add_chinese_doc('deploy.text_to_speech.TTSDeploy', '''\
TTSDeploy 是一个用于根据指定的名称创建不同类型文本到语音(TTS)部署实例的工厂类。

Args:
    name：字符串，用于指定要创建的部署实例的类型。
    **kwarg：关键字参数，用于传递给对应部署实例的构造函数。

Returns:
    如果 name 参数为 ‘bark’，则返回一个 [BarkDeploy][lazyllm.components.BarkDeploy] 实例。
    如果 name 参数为 ‘ChatTTS’，则返回一个 [ChatTTSDeploy][lazyllm.components.ChatTTSDeploy] 实例。
    如果 name 参数以 ‘musicgen’ 开头，则返回一个 [MusicGenDeploy][lazyllm.components.MusicGenDeploy] 实例。
    如果 name 参数不匹配以上任何情况，抛出 RuntimeError 异常，说明不支持的模型。            
''')

add_example('TTSDeploy', ['''\
>>> from lazyllm import launchers, UrlModule
>>> from lazyllm.components import TTSDeploy
>>> model_name = 'bark'
>>> deployer = TTSDeploy(model_name, launcher=launchers.remote())
>>> url = deployer(base_model=model_name)
>>> model = UrlModule(url=url)
>>> res = model('Hello World!')
>>> print(res)
... <lazyllm-query>{"query": "", "files": ["path/to/chattts/sound_xxx.wav"]}
'''])

add_english_doc('finetune.base.DummyFinetune', '''\
DummyFinetune is a subclass of [LazyLLMFinetuneBase][lazyllm.components.LazyLLMFinetuneBase] that serves as a placeholder implementation for fine-tuning.
The class is primarily used for demonstration or testing purposes, as it does not perform any actual fine-tuning logic.

Args:
    base_model: A string specifying the base model name. Defaults to 'base'.
    target_path: A string specifying the target path for fine-tuning outputs. Defaults to 'target'.
    launcher: A launcher instance for executing commands. Defaults to [launchers.remote()][lazyllm.launchers.remote].
    **kw: Additional keyword arguments that are stored for later use.

Returns:
    A string representing a dummy command. The string includes the initial arguments passed during initialization.
''')

add_chinese_doc('finetune.base.DummyFinetune', '''\
DummyFinetune 是 [LazyLLMFinetuneBase][lazyllm.components.LazyLLMFinetuneBase] 的子类，用于占位实现微调逻辑。
此类主要用于演示或测试目的，因为它不执行任何实际的微调操作。

Args:
    base_model: 字符串，指定基础模型的名称，默认为 'base'。
    target_path: 字符串，指定微调输出的目标路径，默认为 'target'。
    launcher: 启动器实例，用于执行命令。默认为 [launchers.remote()][lazyllm.launchers.remote]。
    **kw: 其他关键字参数，这些参数会被保存以供后续使用。

Returns:
    一个字符串，表示一个占位命令。该字符串包括初始化时传递的参数。
''')

add_example('finetune.base.DummyFinetune', ['''\
>>> from lazyllm.components import DummyFinetune
>>> from lazyllm import launchers
>>> # 创建一个 DummyFinetune 实例
>>> finetuner = DummyFinetune(base_model='example-base', target_path='example-target', launcher=launchers.local(), custom_arg='custom_value')
>>> # 调用 cmd 方法生成占位命令
>>> command = finetuner.cmd('--example-arg', key='value')
>>> print(command)
... echo 'dummy finetune!, and init-args is {'custom_arg': 'custom_value'}'
'''])

add_english_doc('finetune.base.DummyFinetune.cmd', '''\
The `cmd` method generates a dummy command string for fine-tuning. This method is primarily for testing or demonstration purposes.

Args:
    *args: Positional arguments to be included in the command (not used in this implementation).
    **kw: Keyword arguments to be included in the command (not used in this implementation).

Returns:
    A string representing a dummy command. The string includes the initial arguments (`**kw`) passed during the instance initialization, which are stored in `self.kw`.

Example:
    If the class is initialized with `custom_arg='value'`, calling the `cmd` method will return:
    `"echo 'dummy finetune!, and init-args is {'custom_arg': 'value'}'"`
''')

add_chinese_doc('finetune.base.DummyFinetune.cmd', '''\
`cmd` 方法生成一个用于微调的占位命令字符串。此方法主要用于测试或演示目的。

Args:
    *args: 要包含在命令中的位置参数（在本实现中未使用）。
    **kw: 要包含在命令中的关键字参数（在本实现中未使用）。

Returns:
    一个字符串，表示一个占位命令。该字符串包括初始化时传递的关键字参数 (`**kw`)，存储在 `self.kw` 中。

Example:
    如果类初始化时使用 `custom_arg='value'`，调用 `cmd` 方法将返回：
    `"echo 'dummy finetune!, and init-args is {'custom_arg': 'value'}'"`
''')

add_example('finetune.base.DummyFinetune.cmd', ['''\
>>> from lazyllm.components import DummyFinetune
>>> from lazyllm import launchers
>>> # 创建一个 DummyFinetune 实例，并传递初始化参数
>>> finetuner = DummyFinetune(base_model='example-base', target_path='example-target', launcher=launchers.local(), custom_arg='value')
>>> # 调用 cmd 方法生成占位命令
>>> command = finetuner.cmd()
>>> # 打印生成的占位命令
>>> print(command)
... echo 'dummy finetune!, and init-args is {'custom_arg': 'value'}'
'''])

add_english_doc('OCRDeploy', '''\
OCRDeploy is a subclass of [LazyLLMDeployBase][lazyllm.components.LazyLLMDeployBase] that provides deployment for OCR (Optical Character Recognition) models.
This class is designed to deploy OCR models with additional configurations such as logging, trust for remote code, and port customization.

Attributes:

    keys_name_handle: A dictionary mapping input keys to their corresponding handler keys. For example:
        - "inputs": Handles general inputs.
        - "ocr_files": Also mapped to "inputs".
    message_format: A dictionary specifying the expected message format. For example:
        - {"inputs": "/path/to/pdf"} indicates that the model expects a PDF file path as input.
    default_headers: A dictionary specifying default headers for API requests. Defaults to:
        - {"Content-Type": "application/json"}

Args:
    launcher: A launcher instance for deploying the model. Defaults to `None`.
    log_path: A string specifying the path where logs should be saved. Defaults to `None`.
    trust_remote_code: A boolean indicating whether to trust remote code execution. Defaults to `True`.
    port: An integer specifying the port for the deployment server. Defaults to `None`.

Returns:
    OCRDeploy instance, can be started by calling
''')

add_chinese_doc('OCRDeploy', '''\
OCRDeploy 是 [LazyLLMDeployBase][lazyllm.components.LazyLLMDeployBase] 的子类，用于部署 OCR（光学字符识别）模型。
此类支持额外的配置，例如日志记录、远程代码信任以及端口自定义。

属性:

    keys_name_handle: 一个字典，用于将输入键映射到相应的处理键。例如：
        - "inputs": 处理一般输入。
        - "ocr_files": 同样映射到 "inputs"。
    message_format: 一个字典，指定模型期望的消息格式。例如：
        - {"inputs": "/path/to/pdf"} 表示模型需要一个 PDF 文件路径作为输入。
    default_headers: 一个字典，指定 API 请求的默认头部。默认为：
        - {"Content-Type": "application/json"}

Args:
    launcher: 启动器实例，用于部署模型。默认为 `None`。
    log_path: 字符串，指定日志保存的路径。默认为 `None`。
    trust_remote_code: 布尔值，指示是否信任远程代码执行。默认为 `True`。
    port: 整数，指定部署服务器的端口号。默认为 `None`。

Returns:
    OCRDeploy实例，可通过调用方式启动服务
''')

add_example('OCRDeploy', ['''\
>>> from lazyllm.components import OCRDeploy
>>> from lazyllm import launchers
>>> # 创建一个 OCRDeploy 实例
>>> deployer = OCRDeploy(launcher=launchers.local(), log_path='./logs', port=8080)
>>> # 使用微调的 OCR 模型部署服务器
>>> server = deployer(finetuned_model='ocr-model')
>>> # 打印部署服务器信息
>>> print(server)
... <RelayServer instance ready to handle OCR requests>
'''])

# core.py
add_chinese_doc('core.ComponentBase', '''\
组件基类，提供统一的接口与基础实现，便于创建不同类型的组件。  
组件通过指定的 Launcher 来执行任务，支持自定义任务执行逻辑。

Args:
    launcher (LazyLLMLaunchersBase or type, optional): 组件使用的启动器实例或启动器类，默认为空启动器（empty）。
''')

add_english_doc('core.ComponentBase', '''\
Base class for components, providing a unified interface and basic implementation to facilitate creation of various components.  
Components execute tasks via a specified launcher and support custom task execution logic.

Args:
    launcher (LazyLLMLaunchersBase or type, optional): Launcher instance or launcher class used by the component, defaults to empty launcher.
''')

add_example('core.ComponentBase', '''\
>>> from lazyllm.components.core import ComponentBase
>>> class MyComponent(ComponentBase):
...     def apply(self, x):
...         return x * 2
>>> comp = MyComponent()
>>> comp.name = "ExampleComponent"
>>> print(comp.name)
ExampleComponent
>>> result = comp(10)
>>> print(result)
20
>>> print(comp.apply(5))
10
''')

add_chinese_doc('core.ComponentBase.apply', '''\
组件执行的核心方法，需由子类实现。  
定义组件的具体业务逻辑或任务执行步骤。  

**注意:**  
调用组件时，如果子类重写了此方法，则会调用此方法执行任务。  
''')

add_english_doc('core.ComponentBase.apply', '''\
Core execution method of the component, to be implemented by subclasses.  
Defines the specific business logic or task execution steps of the component.

**Note:**  
If this method is overridden by the subclass, it will be called when the component is invoked.
''')

add_chinese_doc('core.ComponentBase.cmd', '''\
生成组件的执行命令，需由子类实现。  
返回的命令可以是字符串、元组或列表，表示具体执行任务的指令。  

**注意:**  
调用组件时，如果未重写 `apply` 方法，将通过此命令生成任务并由启动器执行。  
''')

add_english_doc('core.ComponentBase.cmd', '''\
Generates the execution command of the component, to be implemented by subclasses.  
The returned command can be a string, tuple, or list, representing the instruction to execute the task.

**Note:**  
If the `apply` method is not overridden, this command will be used to create a job for the launcher to run.
''')

add_chinese_doc('deploy.ray.Distributed', """\
分布式部署类，继承自LazyLLMDeployBase。

提供基于Ray框架的分布式模型部署功能，支持多节点集群部署。

Args:
    launcher: 启动器配置，默认为远程启动器(ngpus=1)
    port (int, optional): 服务端口号，默认为随机端口(30000-40000)

Attributes:
    finetuned_model: 微调后的模型路径
    base_model: 基础模型路径
    master_ip: 主节点IP地址

Methods:
    cmd(finetuned_model, base_model, master_ip): 生成部署命令
    geturl(job): 获取部署服务的URL地址
""")

add_english_doc('deploy.ray.Distributed', """\
Distributed deployment class, inherits from LazyLLMDeployBase.

Provides distributed model deployment functionality based on Ray framework, supports multi-node cluster deployment.

Args:
    launcher: Launcher configuration, defaults to remote launcher(ngpus=1)
    port (int, optional): Service port number, defaults to random port(30000-40000)

Attributes:
    finetuned_model: Fine-tuned model path
    base_model: Base model path
    master_ip: Master node IP address

Methods:
    cmd(finetuned_model, base_model, master_ip): Generate deployment command
    geturl(job): Get deployed service URL address
""")

add_chinese_doc('deploy.ray.Distributed.cmd', """\
生成Ray分布式部署命令。

根据是否为主节点生成相应的Ray启动命令，支持头节点和工作节点两种模式。

Args:
    finetuned_model: 微调后的模型路径
    base_model: 基础模型路径
    master_ip: 主节点IP地址，如果为空则作为头节点启动

Returns:
    LazyLLMCMD: 包含部署命令的对象
""")

add_english_doc('deploy.ray.Distributed.cmd', """\
Generate Ray distributed deployment command.

Generate corresponding Ray startup command based on whether it is a master node, supports both head node and worker node modes.

Args:
    finetuned_model: Fine-tuned model path
    base_model: Base model path
    master_ip: Master node IP address, if empty starts as head node

Returns:
    LazyLLMCMD: Object containing deployment command
""")

add_chinese_doc('deploy.ray.Distributed.geturl', """\
获取分布式部署服务的URL地址。

根据部署模式返回相应的服务地址信息，支持显示模式和实际部署模式。

Args:
    job: 任务对象，默认为当前任务

Returns:
    Package: 包含模型路径和服务地址的打包对象
""")

add_english_doc('deploy.ray.Distributed.geturl', """\
Get URL address of distributed deployment service.

Return corresponding service address information based on deployment mode, supports display mode and actual deployment mode.

Args:
    job: Job object, defaults to current job

Returns:
    Package: Packaged object containing model path and service address
""")
