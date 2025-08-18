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
# Finetune-AlpacaloraFinetune
add_chinese_doc('finetune.AlpacaloraFinetune', '''\
此类是 ``LazyLLMFinetuneBase`` 的子类，基于 [alpaca-lora](https://github.com/tloen/alpaca-lora) 项目提供的LoRA微调能力，用于对大语言模型进行LoRA微调。

Args:
    base_model (str): 用于进行微调的基模型的本地绝对路径。
    target_path (str): 微调后模型保存LoRA权重的本地绝对路径。
    merge_path (str): 模型合并LoRA权重后的路径，默认为 ``None`` 。如果未指定，则会在 ``target_path`` 下创建 "lazyllm_lora" 和 "lazyllm_merge" 目录，分别作为 ``target_path`` 和  ``merge_path`` 。
    model_name (str): 模型的名称，用于设置日志名的前缀，默认为 ``LLM``。
    cp_files (str): 指定复制源自基模型路径下的配置文件，会被复制到  ``merge_path`` ，默认为 ``tokeniz*``
    launcher (lazyllm.launcher): 微调的启动器，默认为 ``launchers.remote(ngpus=1)``。
    kw: 关键字参数，用于更新默认的训练参数。请注意，除了以下列出的关键字参数外，这里不能传入额外的关键字参数。

此类的关键字参数及其默认值如下：

Keyword Args: 
    data_path (str): 数据路径，默认为 ``None``；一般在此类对象被调用时候，作为唯一位置参数传入。
    batch_size (int): 批处理大小，默认为 ``64``。
    micro_batch_size (int): 微批处理大小，默认为 ``4``。
    num_epochs (int): 训练轮数，默认为 ``2``。
    learning_rate (float): 学习率，默认为 ``5.e-4``。
    cutoff_len (int): 截断长度，默认为 ``1030``；输入数据token超过该长度就会被截断。
    filter_nums (int): 过滤器数量，默认为 ``1024``；仅保留低于该token长度数值的输入。
    val_set_size (int): 验证集大小，默认为 ``200``。
    lora_r (int): LoRA 的秩，默认为 ``8``；该数值决定添加参数的量，数值越小参数量越小。
    lora_alpha (int): LoRA 的融合因子，默认为 ``32``；该数值决定LoRA参数对基模型参数的影响度，数值越大影响越大。
    lora_dropout (float): LoRA 的丢弃率，默认为 ``0.05``，一般用于防止过拟合。
    lora_target_modules (str): LoRA 的目标模块，默认为 ``[wo,wqkv]``，该默认值为 InternLM2 模型的；该配置项不同模型的不一样。
    modules_to_save (str): 用于全量微调的模块，默认为 ``[tok_embeddings,output]``，该默认值为 InternLM2 模型的；该配置项不同模型的不一样。
    deepspeed (str): DeepSpeed 配置文件的路径，默认使用 LazyLLM 代码仓库中预制的配置文件： ``ds.json``。
    prompt_template_name (str): 提示模板的名称，默认为 ``alpaca``，即默认使用 LazyLLM 提供的提示模板。
    train_on_inputs (bool): 是否在输入上训练，默认为 ``True``。
    show_prompt (bool): 是否显示提示，默认为 ``False``。
    nccl_port (int): NCCL 端口，默认为 ``19081``。

''')

add_english_doc('finetune.AlpacaloraFinetune', '''\
This class is a subclass of ``LazyLLMFinetuneBase``, based on the LoRA fine-tuning capabilities provided by the [alpaca-lora](https://github.com/tloen/alpaca-lora) project, used for LoRA fine-tuning of large language models.

Args:
    base_model (str): The base model used for fine-tuning. It is required to be the path of the base model.
    target_path (str): The path where the LoRA weights of the fine-tuned model are saved.
    merge_path (str): The path where the model merges the LoRA weights, default to `None`. If not specified, "lazyllm_lora" and "lazyllm_merge" directories will be created under ``target_path`` as ``target_path`` and ``merge_path`` respectively.
    model_name (str): The name of the model, used as the prefix for setting the log name, default to "LLM".
    cp_files (str): Specify configuration files to be copied from the base model path, which will be copied to ``merge_path``, default to ``tokeniz*``
    launcher (lazyllm.launcher): The launcher for fine-tuning, default to ``launchers.remote(ngpus=1)``.
    kw: Keyword arguments, used to update the default training parameters. Note that additional keyword arguments cannot be arbitrarily specified.

The keyword arguments and their default values for this class are as follows:

Keyword Args: 
    data_path (str): Data path, default to ``None``; generally passed as the only positional argument when this object is called.
    batch_size (int): Batch size, default to ``64``.
    micro_batch_size (int): Micro-batch size, default to ``4``.
    num_epochs (int): Number of training epochs, default to ``2``.
    learning_rate (float): Learning rate, default to ``5.e-4``.
    cutoff_len (int): Cutoff length, default to ``1030``; input data tokens will be truncated if they exceed this length.
    filter_nums (int): Number of filters, default to ``1024``; only input with token length below this value is preserved.
    val_set_size (int): Validation set size, default to ``200``.
    lora_r (int): LoRA rank, default to ``8``; this value determines the amount of parameters added, the smaller the value, the fewer the parameters.
    lora_alpha (int): LoRA fusion factor, default to ``32``; this value determines the impact of LoRA parameters on the base model parameters, the larger the value, the greater the impact.
    lora_dropout (float): LoRA dropout rate, default to ``0.05``, generally used to prevent overfitting.
    lora_target_modules (str): LoRA target modules, default to ``[wo,wqkv]``, which is the default for InternLM2 model; this configuration item varies for different models.
    modules_to_save (str): Modules for full fine-tuning, default to ``[tok_embeddings,output]``, which is the default for InternLM2 model; this configuration item varies for different models.
    deepspeed (str): The path of the DeepSpeed configuration file, default to use the pre-made configuration file in the LazyLLM code repository: ``ds.json``.
    prompt_template_name (str): The name of the prompt template, default to "alpaca", i.e., use the prompt template provided by LazyLLM by default.
    train_on_inputs (bool): Whether to train on inputs, default to ``True``.
    show_prompt (bool): Whether to show the prompt, default to ``False``.
    nccl_port (int): NCCL port, default to ``19081``.

''')

add_example('finetune.AlpacaloraFinetune', '''\
>>> from lazyllm import finetune
>>> trainer = finetune.alpacalora('path/to/base/model', 'path/to/target')
''')

# Finetune-CollieFinetune
add_chinese_doc('finetune.CollieFinetune', '''\
此类是 ``LazyLLMFinetuneBase`` 的子类，基于 [Collie](https://github.com/OpenLMLab/collie) 框架提供的LoRA微调能力，用于对大语言模型进行LoRA微调。

Args:
    base_model (str): 用于进行微调的基模型。要求是基模型的路径。
    target_path (str): 微调后模型保存LoRA权重的路径。
    merge_path (str): 模型合并LoRA权重后的路径，默认为None。如果未指定，则会在 ``target_path`` 下创建 "lazyllm_lora" 和 "lazyllm_merge" 目录，分别作为 ``target_path`` 和  ``merge_path`` 。
    model_name (str): 模型的名称，用于设置日志名的前缀，默认为 "LLM"。
    cp_files (str): 指定复制源自基模型路径下的配置文件，会被复制到  ``merge_path`` ，默认为 "tokeniz\*"
    launcher (lazyllm.launcher): 微调的启动器，默认为 ``launchers.remote(ngpus=1)``。
    kw: 关键字参数，用于更新默认的训练参数。请注意，除了以下列出的关键字参数外，这里不能传入额外的关键字参数。

此类的关键字参数及其默认值如下：

Keyword Args: 
    data_path (str): 数据路径，默认为 ``None``；一般在此类对象被调用时候，作为唯一位置参数传入。
    batch_size (int): 批处理大小，默认为 ``64``。
    micro_batch_size (int): 微批处理大小，默认为 ``4``。
    num_epochs (int): 训练轮数，默认为 ``2``。
    learning_rate (float): 学习率，默认为 ``5.e-4``。
    dp_size (int): 数据并行参数，默认为 ``8``。
    pp_size (int): 流水线并行参数，默认为 ``1``。
    tp_size (int): 张量并行参数，默认为 ``1``。
    lora_r (int): LoRA 的秩，默认为 ``8``；该数值决定添加参数的量，数值越小参数量越小。
    lora_alpha (int): LoRA 的融合因子，默认为 ``32``；该数值决定LoRA参数对基模型参数的影响度，数值越大影响越大。
    lora_dropout (float): LoRA 的丢弃率，默认为 ``0.05``，一般用于防止过拟合。
    lora_target_modules (str): LoRA 的目标模块，默认为 ``[wo,wqkv]``，该默认值为 InternLM2 模型的；该配置项不同模型的不一样。
    modules_to_save (str): 用于全量微调的模块，默认为 ``[tok_embeddings,output]``，该默认值为 InternLM2 模型的；该配置项不同模型的不一样。
    prompt_template_name (str): 提示模板的名称，默认为 ``alpaca``，即默认使用 LazyLLM 提供的提示模板。

''')

add_english_doc('finetune.CollieFinetune', '''\
This class is a subclass of ``LazyLLMFinetuneBase``, based on the LoRA fine-tuning capabilities provided by the [Collie](https://github.com/OpenLMLab/collie) framework, used for LoRA fine-tuning of large language models.

Args:
    base_model (str): The base model used for fine-tuning. It is required to be the path of the base model.
    target_path (str): The path where the LoRA weights of the fine-tuned model are saved.
    merge_path (str): The path where the model merges the LoRA weights, default to ``None``. If not specified, "lazyllm_lora" and "lazyllm_merge" directories will be created under ``target_path`` as ``target_path`` and ``merge_path`` respectively.
    model_name (str): The name of the model, used as the prefix for setting the log name, default to "LLM".
    cp_files (str): Specify configuration files to be copied from the base model path, which will be copied to ``merge_path``, default to "tokeniz*"
    launcher (lazyllm.launcher): The launcher for fine-tuning, default to ``launchers.remote(ngpus=1)``.
    kw: Keyword arguments, used to update the default training parameters. Note that additional keyword arguments cannot be arbitrarily specified.

The keyword arguments and their default values for this class are as follows:

Keyword Args: 
    data_path (str): Data path, default to ``None``; generally passed as the only positional argument when this object is called.
    batch_size (int): Batch size, default to ``64``.
    micro_batch_size (int): Micro-batch size, default to ``4``.
    num_epochs (int): Number of training epochs, default to ``2``.
    learning_rate (float): Learning rate, default to ``5.e-4``.
    dp_size (int): Data parallelism parameter, default to `` 8``.
    pp_size (int): Pipeline parallelism parameter, default to ``1``.
    tp_size (int): Tensor parallelism parameter, default to ``1``.
    lora_r (int): LoRA rank, default to ``8``; this value determines the amount of parameters added, the smaller the value, the fewer the parameters.
    lora_alpha (int): LoRA fusion factor, default to ``32``; this value determines the impact of LoRA parameters on the base model parameters, the larger the value, the greater the impact.
    lora_dropout (float): LoRA dropout rate, default to ``0.05``, generally used to prevent overfitting.
    lora_target_modules (str): LoRA target modules, default to ``[wo,wqkv]``, which is the default for InternLM2 model; this configuration item varies for different models.
    modules_to_save (str): Modules for full fine-tuning, default to ``[tok_embeddings,output]``, which is the default for InternLM2 model; this configuration item varies for different models.
    prompt_template_name (str): The name of the prompt template, default to ``alpaca``, i.e., use the prompt template provided by LazyLLM by default.

''')

add_example('finetune.CollieFinetune', '''\
>>> from lazyllm import finetune
>>> trainer = finetune.collie('path/to/base/model', 'path/to/target')
''')
add_chinese_doc('finetune.AlpacaloraFinetune.cmd', """\
生成用于执行Alpaca-LoRA微调和模型合并的shell命令序列。

Args:
    trainset (str): 训练数据集路径，支持相对data_path配置的路径或绝对路径
    valset (str, optional): 验证数据集路径，未指定时将从训练集中自动划分

Returns:
    str or list: 当不需要合并模型时返回单个命令字符串，需要合并时返回包含微调命令、合并命令和文件拷贝命令的列表

""")

add_english_doc('finetune.AlpacaloraFinetune.cmd', """\
Generate shell command sequence for Alpaca-LoRA fine-tuning and model merging.

Args:
    trainset (str): Training dataset path, supports both relative path (to configured data_path) and absolute path
    valset (str, optional): Validation dataset path, will auto-split from trainset if not specified

Returns:
    str or list: Returns a single command string when no merging needed, otherwise returns a list containing:
                 [fine-tune command, merge command, file copy command]


""")

add_example('finetune.AlpacaloraFinetune.cmd', """\
>>> from lazyllm import finetune
>>> trainer = finetune.alpacalora('path/to/base/model', 'path/to/target')
>>> cmd = trainer.cmd("my_dataset.json")

""")
# Finetune-LlamafactoryFinetune
add_chinese_doc('finetune.LlamafactoryFinetune', '''\
此类是 ``LazyLLMFinetuneBase`` 的子类，基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 框架提供的训练能力，用于对大语言模型(或视觉语言模型)进行训练。

Args:
    base_model (str): 用于进行训练的基模型路径。支持本地路径，若路径不存在则尝试从配置的模型路径中查找。
    target_path (str): 训练完成后，模型权重保存的目标路径。
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
    base_model (str): Path to the base model used for training. Supports local paths; if the path does not exist, it will attempt to locate it from the configured model directory.
    target_path (str): Target directory to save model weights after training is completed.
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

返回:
    str: 完整的shell命令字符串，包含:
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

Returns:
    str: Complete shell command string containing:
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

Returns:
    处理后的最终结果

""")

add_english_doc('LazyLLMDeployBase.extract_result', """\
Extract final result from model output. The default implementation returns raw output directly, subclasses can override this method to implement custom result extraction logic.

Args:
    output: Raw model output
    inputs: Original input data, can be used for post-processing

Returns:
    Processed final result

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

**注意**: 此方法目前正在开发中。
''')

add_english_doc('deploy.embed.AbstractEmbedding.load_embed', '''\
Abstract method for loading embedding models. This method is implemented by subclasses to perform specific model loading logic.

**Note**: This method is currently under development.
''')

# Deploy-Lightllm
add_chinese_doc('deploy.Lightllm', '''\
此类是 ``LazyLLMDeployBase`` 的子类，基于 [LightLLM](https://github.com/ModelTC/lightllm) 框架提供的推理能力，用于对大语言模型进行推理。

Args:
    trust_remote_code (bool): 是否允许加载来自远程服务器的模型代码，默认为 ``True``。
    launcher (lazyllm.launcher): 微调的启动器，默认为 ``launchers.remote(ngpus=1)``。
    stream (bool): 是否为流式响应，默认为 ``False``。
    kw: 关键字参数，用于更新默认的训练参数。请注意，除了以下列出的关键字参数外，这里不能传入额外的关键字参数。

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
    trust_remote_code (bool): Whether to allow loading of model code from remote servers, default is ``True``.
    launcher (lazyllm.launcher): The launcher for fine-tuning, default is ``launchers.remote(ngpus=1)``.
    stream (bool): Whether the response is streaming, default is ``False``.
    kw: Keyword arguments used to update default training parameters. Note that not any additional keyword arguments can be specified here.

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

参数:
    finetuned_model (str): 微调后的模型路径。
    base_model (str): 基础模型路径，当finetuned_model无效时使用。

返回值:
    LazyLLMCMD: 一个包含启动命令的LazyLLMCMD对象。
''')

add_english_doc('deploy.Lightllm.cmd', '''\
This method generates the command to start the LightLLM service.

Args:
    finetuned_model (str): Path to the fine-tuned model.
    base_model (str): Path to the base model, used when finetuned_model is invalid.

Returns:
    LazyLLMCMD: A LazyLLMCMD object containing the startup command.
''')

add_chinese_doc('deploy.Lightllm.geturl', '''\
获取LightLLM服务的URL地址。

参数:
    job (optional): 任务对象，默认为None，此时使用self.job。

返回值:
    str: 服务的URL地址，格式为"http://{ip}:{port}/generate"。
''')

add_english_doc('deploy.Lightllm.geturl', '''\
Get the URL address of the LightLLM service.

Args:
    job (optional): Job object, defaults to None, in which case self.job is used.

Returns:
    str: The service URL address in the format "http://{ip}:{port}/generate".
''')

add_chinese_doc('deploy.Lightllm.extract_result', '''\
从服务响应中提取生成的文本结果。

参数:
    x (str): 服务返回的响应文本。
    inputs (str): 输入文本。

返回值:
    str: 提取出的生成文本。

异常:
    Exception: 当解析JSON响应失败时抛出异常。
''')

add_english_doc('deploy.Lightllm.extract_result', '''\
Extract generated text from the service response.

Args:
    x (str): Response text from the service.
    inputs (str): Input text.

Returns:
    str: The extracted generated text.

Raises:
    Exception: When JSON response parsing fails.
''')

# Deploy-Vllm
add_chinese_doc('deploy.Vllm', '''\
此类是 ``LazyLLMDeployBase`` 的子类，基于 [VLLM](https://github.com/vllm-project/vllm) 框架提供的推理能力，用于对大语言模型进行推理。

Args:
    trust_remote_code (bool): 是否允许加载来自远程服务器的模型代码，默认为 ``True``。
    launcher (lazyllm.launcher): 微调的启动器，默认为 ``launchers.remote(ngpus=1)``。
    log_path (str): 日志保存路径，若为 ``None`` 则不保存日志。
    openai_api(bool):是否调用openai接口,默认为``None``。
    kw: 关键字参数，用于更新默认的训练参数。请注意，除了以下列出的关键字参数外，这里不能传入额外的关键字参数。

此类的关键字参数及其默认值如下：

Keyword Args: 
    tensor-parallel-size (int): 张量并行参数，默认为 ``1``。
    dtype (str): 模型权重和激活值的数据类型，默认为 ``auto``。另外可选项还有： ``half``, ``float16``, ``bfloat16``, ``float``, ``float32``。
    kv-cache-dtype (str): 看kv缓存的存储类型，默认为 ``auto``。另外可选的还有：``fp8``, ``fp8_e5m2``, ``fp8_e4m3``。
    device (str): VLLM所支持的后端硬件类型，默认为 ``auto``。另外可选的还有：``cuda``, ``neuron``, ``cpu``。
    block-size (int): 设置 token块的大小，默认为 ``16``。
    port (int): 服务的端口号，默认为 ``auto``。
    host (str): 服务的IP地址，默认为 ``0.0.0.0``。
    seed (int): 随机数种子，默认为 ``0``。
    tokenizer_mode (str): tokenizer的加载模式，默认为 ``auto``。
    max-num-seqs (int): 推理引擎最大的并行请求数， 默认为 ``256``。

''')

add_english_doc('deploy.Vllm', '''\
This class is a subclass of ``LazyLLMDeployBase``, based on the inference capabilities provided by the [VLLM](https://github.com/vllm-project/vllm) framework, used for inference with large language models.

Args:
    trust_remote_code (bool): Whether to allow loading of model code from remote servers, default is ``True``.
    launcher (lazyllm.launcher): The launcher for fine-tuning, default is ``launchers.remote(ngpus=1)``.
    log_path (str): Path to save logs. If ``None``, logs will not be saved.
    openai_api (bool): Whether to call the OpenAI API. Default is ``None``.
    kw: Keyword arguments used to update default training parameters. Note that not any additional keyword arguments can be specified here.

The keyword arguments and their default values for this class are as follows:

Keyword Args: 
    tensor-parallel-size (int): Tensor parallelism parameter, default is ``1``.
    dtype (str): Data type for model weights and activations, default is ``auto``. Other options include: ``half``, ``float16``, ``bfloat16``, ``float``, ``float32``.
    kv-cache-dtype (str): Data type for the key-value cache storage, default is ``auto``. Other options include: ``fp8``, ``fp8_e5m2``, ``fp8_e4m3``.
    device (str): Backend hardware type supported by VLLM, default is ``auto``. Other options include: ``cuda``, ``neuron``, ``cpu``.
    block-size (int): Sets the size of the token block, default is ``16``.
    port (int): Service port number, default is ``auto``.
    host (str): Service IP address, default is ``0.0.0.0``.
    seed (int): Random number seed, default is ``0``.
    tokenizer_mode (str): Tokenizer loading mode, default is ``auto``.
    max-num-seqs (int): Maximum number of parallel requests for the inference engine, default is ``256``.

''')

add_example('deploy.Vllm', '''\
>>> from lazyllm import deploy
>>> infer = deploy.vllm()
''')

# Deploy-EmbeddingDeploy
add_chinese_doc('deploy.EmbeddingDeploy', '''\
此类是 ``LazyLLMDeployBase`` 的子类，用于部署文本嵌入（Embedding）服务。支持稠密向量（dense）和稀疏向量（sparse）两种嵌入方式，可以使用HuggingFace模型或FlagEmbedding模型。

Args:
    launcher (lazyllm.launcher): 启动器，默认为 ``None``。
    model_type (str): 模型类型，默认为 ``'embed'``。
    log_path (str): 日志文件路径，默认为 ``None``。
    embed_type (str): 嵌入类型，可选 ``'dense'`` 或 ``'sparse'``，默认为 ``'dense'``。
    trust_remote_code (bool): 是否信任远程代码，默认为 ``True``。
    port (int): 服务端口号，默认为 ``None``，此情况下LazyLLM会自动生成随机端口号。

调用参数:
    finetuned_model: 微调后的模型路径或模型名称。
    base_model: 基础模型路径或模型名称，当finetuned_model无效时会使用此模型。

消息格式:
    输入格式为包含text（文本）和images（图像列表）的字典。
    - text: 需要编码的文本内容
    - images: 需要编码的图像列表（可选）
''')

add_english_doc('deploy.EmbeddingDeploy', '''\
This class is a subclass of ``LazyLLMDeployBase``, designed for deploying text embedding services. It supports both dense and sparse embedding methods, compatible with HuggingFace models and FlagEmbedding models.

Args:
    launcher (lazyllm.launcher): The launcher instance, defaults to ``None``.
    model_type (str): Model type, defaults to ``'embed'``.
    log_path (str): Path for log file, defaults to ``None``.
    embed_type (str): Embedding type, either ``'dense'`` or ``'sparse'``, defaults to ``'dense'``.
    trust_remote_code (bool): Whether to trust remote code, defaults to ``True``.
    port (int): Service port number, defaults to ``None``, in which case LazyLLM will generate a random port.

Call Arguments:
    finetuned_model: Path or name of the fine-tuned model.
    base_model: Path or name of the base model, used when finetuned_model is invalid.

Message Format:
    Input format is a dictionary containing text and images list.
    - text: Text content to be encoded
    - images: List of images to be encoded (optional)
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

调用参数:
    finetuned_model: 微调后的模型路径或模型名称。
    base_model: 基础模型路径或模型名称，当finetuned_model无效时会使用此模型。

消息格式:
    输入格式为包含query（查询文本）、documents（候选文档列表）和top_n（返回的文档数量）的字典。
    - query: 查询文本
    - documents: 候选文档列表
    - top_n: 返回的文档数量，默认为1
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
    finetuned_model: Path or name of the fine-tuned model.
    base_model: Path or name of the base model, used when finetuned_model is invalid.

Message Format:
    Input format is a dictionary containing query (query text), documents (list of candidate documents), and top_n (number of documents to return).
    - query: Query text
    - documents: List of candidate documents
    - top_n: Number of documents to return, defaults to 1
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
该方法会使用 `sentence_transformers.CrossEncoder` 从指定的 `base_rerank` 路径或名称加载模型，  
通常在延迟加载模式下由首次调用实例时自动触发。
''')

add_english_doc('deploy.embed.LazyHuggingFaceRerank.load_reranker', '''\
Load the rerank model.  
Uses `sentence_transformers.CrossEncoder` to load the model from the specified `base_rerank` path or name.  
Typically triggered automatically on first call when lazy loading is enabled.
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

Returns:
    LazyFlagEmbedding: 一个新的 LazyFlagEmbedding 实例。
''')

add_english_doc('deploy.embed.LazyFlagEmbedding.rebuild', '''\
Rebuild a LazyFlagEmbedding instance.

This class method reconstructs an instance of LazyFlagEmbedding, typically used during deserialization or multiprocessing scenarios.

Args:
    base_embed (str): The path or name of the embedding model.
    sparse (bool): Whether to enable sparse embedding mode.
    init (bool): Whether to load the model immediately during instantiation.

Returns:
    LazyFlagEmbedding: A newly constructed LazyFlagEmbedding instance.
''')


add_chinese_doc('deploy.Vllm.cmd', '''\
构造用于启动 vLLM 推理服务的命令。

该方法会自动检测模型路径是否有效，并根据当前配置参数动态生成可执行命令，支持多节点部署时自动加入 ray 启动命令。

Args:
    finetuned_model (str): 微调后的模型路径。
    base_model (str): 备用基础模型路径（当 finetuned_model 无效时启用）。
    master_ip (str): 分布式部署中的主节点 IP，仅在多节点时启用。

Returns:
    LazyLLMCMD: 可执行命令对象，包含启动指令、结果回调函数及健康检查方法。
''')

add_english_doc('deploy.Vllm.cmd', '''\
Build the command to launch the vLLM inference service.

This method validates the model path and constructs an executable command string based on current configuration. In distributed mode, it will also prepend the ray cluster start command.

Args:
    finetuned_model (str): Path to the fine-tuned model.
    base_model (str): Fallback base model path if finetuned_model is invalid.
    master_ip (str): IP address of the master node in a distributed setup.

Returns:
    LazyLLMCMD: The command object with shell instruction, return value handler, and health checker.
''')

add_chinese_doc('deploy.Vllm.geturl', '''\
获取 vLLM 服务的推理地址。

根据运行模式（Display 模式或实际部署）返回相应的 URL，用于访问模型的生成接口。

Args:
    job (Job, optional): 部署任务对象。默认取当前模块绑定的 job。

Returns:
    str: 推理服务的 HTTP 地址。
''')

add_english_doc('deploy.Vllm.geturl', '''\
Get the inference service URL for the vLLM deployment.

Depending on the execution mode (Display or actual deployment), this method returns the appropriate URL for accessing the model's generate endpoint.

Args:
    job (Job, optional): Deployment job object. Defaults to the module's associated job.

Returns:
    str: The HTTP URL for inference service.
''')

add_chinese_doc('deploy.Vllm.extract_result', '''\
从 vLLM 返回结果中提取文本。

该函数从 JSON 格式的返回值中提取模型输出的文本部分。

Args:
    x (str): JSON 格式的原始返回结果字符串。
    inputs (dict): 原始输入数据（用于兼容接口，当前未使用）。

Returns:
    str: 提取出的文本内容。
''')

add_english_doc('deploy.Vllm.extract_result', '''\
Extract the generated text from a vLLM response.

This function parses the returned JSON and extracts the model-generated text content.

Args:
    x (str): Raw JSON string returned from the API.
    inputs (dict): Original input data (unused; kept for compatibility).

Returns:
    str: The generated text extracted from the response.
''')

# Deploy-Mindie
add_chinese_doc('deploy.Mindie', '''\
此类是 ``LazyLLMDeployBase`` 的一个子类, 用于部署和管理MindIE大模型推理服务。它封装了MindIE服务的配置生成、进程启动和API交互的全流程。
Args:
    trust_remote_code (bool): 是否信任远程代码(如HuggingFace模型)。默认为 ``True``。
    launcher: 任务启动器实例，默认为 ``launchers.remote()``。
    log_path (str): 日志保存路径，若为 ``None`` 则不保存日志。
    **kw: 其他配置参数，支持以下关键参数：
        - npuDeviceIds: NPU设备ID列表(如 ``[[0,1]]`` 表示使用2张卡)
        - worldSize: 模型并行数量
        - port: 服务端口（设为 ``'auto'`` 时自动分配30000-40000的随机端口)
        - maxSeqLen: 最大序列长度
        - maxInputTokenLen: 单次输入最大token数
        - maxPrefillTokens: 预填充token上限
        - config: 自定义配置文件
注意事项: 
   必须预先设置环境变量 ``LAZYLLM_MINDIE_HOME`` 指向MindIE安装目录, 若未指定 ``finetuned_model`` 或路径无效，会自动回退到 ``base_model``
''')

add_english_doc('deploy.Mindie', '''\
This class is a subclass of ``LazyLLMDeployBase``, designed for deploying and managing the MindIE large language model inference service. It encapsulates the full workflow including configuration generation, process launching, and API interaction for the MindIE service.
Args:
    trust_remote_code (bool): Whether to trust remote code (e.g., from HuggingFace models). Default is ``True``.
    launcher: Instance of the task launcher. Default is ``launchers.remote()``.
    log_path (str): Path to save logs. If ``None``, logs will not be saved.
    **kw: Other configuration parameters. Supports the following keys:
        - npuDeviceIds: List of NPU device IDs (e.g., ``[[0,1]]`` indicates using 2 devices)
        - worldSize: Model parallelism size
        - port: Service port (set to ``'auto'`` for auto-assignment between 30000–40000)
        - maxSeqLen: Maximum sequence length
        - maxInputTokenLen: Maximum number of tokens per input
        - maxPrefillTokens: Maximum number of prefill tokens
        - config: Custom configuration file
Note:
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

Returns:
    dict: Parsed configuration dictionary

Notes:
    - Handles both default and custom configuration files
    - Uses JSON format for configuration
    - Creates backup of original config before modification
''')

add_chinese_doc('deploy.Mindie.load_config', '''\
加载并解析MindIE配置文件。

Args:
    config_path (str): JSON配置文件的路径

Returns:
    dict: 解析后的配置字典

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

Returns:
    LazyLLMCMD: Command object for starting the service

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

返回:
    LazyLLMCMD: 启动服务的命令对象

注意事项:
    - 自动处理模型路径验证
    - 启动服务前更新配置
    - 支持配置随机端口分配
''')

add_english_doc('deploy.Mindie.geturl', '''\
Gets the service URL after deployment.

Args:
    job: Job object (optional, defaults to self.job)

Returns:
    str: The generate endpoint URL

Notes:
    - Returns different formats based on display mode
    - Includes port number from configuration
''')

add_chinese_doc('deploy.Mindie.geturl', '''\
获取部署后的服务URL。

Args:
    job: 任务对象(可选，默认为self.job)

返回:
    str: generate接口的URL

注意事项:
    - 根据显示模式返回不同格式
    - 包含配置中的端口号
''')

add_english_doc('deploy.Mindie.extract_result', '''\
Extracts the generated text from the API response.

Args:
    x: Raw API response
    inputs: Original inputs (unused)

Returns:
    str: The generated text

Notes:
    - Parses JSON response
    - Returns first text entry from response
''')

add_chinese_doc('deploy.Mindie.extract_result', '''\
从API响应中提取生成的文本。

Args:
    x: 原始API响应
    inputs: 原始输入(未使用)

返回:
    str: 生成的文本

注意事项:
    - 解析JSON响应
    - 返回响应中的第一个文本条目
''')

# Deploy-LMDeploy
add_chinese_doc('deploy.LMDeploy', '''\
此类是 ``LazyLLMDeployBase`` 的子类，基于 [LMDeploy](https://github.com/InternLM/lmdeploy) 框架提供的推理能力，用于对大语言模型进行推理。

Args:
    launcher (lazyllm.launcher): 微调的启动器，默认为 ``launchers.remote(ngpus=1)``。
    stream (bool): 是否为流式响应，默认为 ``False``。
    trust_remote_code (bool): 是否信任远程代码，默认为 ``True``。
    log_path (str): 日志文件路径，默认为 ``None``。
    kw: 关键字参数，用于更新默认的训练参数。请注意，除了以下列出的关键字参数外，这里不能传入额外的关键字参数。

此类的关键字参数及其默认值如下：

Keyword Args: 
    tp (int): 张量并行参数，默认为 ``1``。
    server-name (str): 服务的IP地址，默认为 ``0.0.0.0``。
    server-port (int): 服务的端口号，默认为 ``None``,此情况下LazyLLM会自动生成随机端口号。
    max-batch-size (int): 最大batch数，默认为 ``128``。
    chat-template (str): 对话模板文件路径，默认为 ``None``。如果模型不是视觉语言模型且未指定模板，将使用默认模板。
    eager-mode (bool): 是否启用eager模式，默认由环境变量 ``LMDEPLOY_EAGER_MODE`` 控制，默认为 ``False``。

''')

add_english_doc('deploy.LMDeploy', '''\
This class is a subclass of ``LazyLLMDeployBase``, leveraging the inference capabilities provided by the [LMDeploy](https://github.com/InternLM/lmdeploy) framework for inference on large language models.

Args:
    launcher (lazyllm.launcher): The launcher for fine-tuning, defaults to ``launchers.remote(ngpus=1)``.
    stream (bool): Whether to enable streaming response, defaults to ``False``.
    trust_remote_code (bool): Whether to trust remote code, defaults to ``True``.
    log_path (str): Path for log file, defaults to ``None``.
    kw: Keyword arguments for updating default training parameters. Note that no additional keyword arguments beyond those listed below can be passed.

Keyword Args: 
    tp (int): Tensor parallelism parameter, defaults to ``1``.
    server-name (str): The IP address of the service, defaults to ``0.0.0.0``.
    server-port (int): The port number of the service, defaults to ``None``. In this case, LazyLLM will automatically generate a random port number.
    max-batch-size (int): Maximum batch size, defaults to ``128``.
    chat-template (str): Path to chat template file, defaults to ``None``. If the model is not a vision-language model and no template is specified, a default template will be used.
    eager-mode (bool): Whether to enable eager mode, controlled by environment variable ``LMDEPLOY_EAGER_MODE``, defaults to ``False``.

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
>>> chat = lazyllm.TrainableModule('Mini-InternVL-Chat-2B-V1-5').deploy_method(deploy.LMDeploy)
>>> chat.update_server()
>>> inputs = encode_query_with_filepaths('What is it?', ['path/to/image'])
>>> res = chat(inputs)
''')

add_chinese_doc('deploy.LMDeploy.cmd', '''\
该方法用于生成启动LMDeploy服务的命令。

参数:
    finetuned_model (str): 微调后的模型路径。
    base_model (str): 基础模型路径，当finetuned_model无效时使用。

返回值:
    LazyLLMCMD: 一个包含启动命令的LazyLLMCMD对象。
''')

add_english_doc('deploy.LMDeploy.cmd', '''\
This method generates the command to start the LMDeploy service.

Args:
    finetuned_model (str): Path to the fine-tuned model.
    base_model (str): Path to the base model, used when finetuned_model is invalid.

Returns:
    LazyLLMCMD: A LazyLLMCMD object containing the startup command.
''')

add_chinese_doc('deploy.LMDeploy.geturl', '''\
获取LMDeploy服务的URL地址。

参数:
    job (optional): 任务对象，默认为None，此时使用self.job。

返回值:
    str: 服务的URL地址，格式为"http://{ip}:{port}/v1/chat/interactive"。
''')

add_english_doc('deploy.LMDeploy.geturl', '''\
Get the URL address of the LMDeploy service.

Args:
    job (optional): Job object, defaults to None, in which case self.job is used.

Returns:
    str: The service URL address in the format "http://{ip}:{port}/v1/chat/interactive".
''')

add_chinese_doc('deploy.LMDeploy.extract_result', '''\
从服务响应中提取生成的文本结果。

参数:
    x (str): 服务返回的响应文本。
    inputs (str): 输入文本。

返回值:
    str: 提取出的生成文本。
''')

add_english_doc('deploy.LMDeploy.extract_result', '''\
Extract generated text from the service response.

Args:
    x (str): Response text from the service.
    inputs (str): Input text.

Returns:
    str: The extracted generated text.
''')

# Deploy-Infinity
add_chinese_doc('deploy.Infinity', '''\
此类是 ``LazyLLMDeployBase`` 的子类，基于 [Infinity](https://github.com/michaelfeil/infinity) 框架提供的高性能文本嵌入、重排序和CLIP等能力。

Args:
    launcher (lazyllm.launcher): Infinity 的启动器，默认为 ``launchers.remote(ngpus=1)``。
    kw: 关键字参数，用于更新默认的训练参数。请注意，除了以下列出的关键字参数外，这里不能传入额外的关键字参数。

此类的关键字参数及其默认值如下：

Keyword Args: 
    host (str): 服务的IP地址，默认为 ``0.0.0.0``。
    port (int): 服务的端口号，默认为 ``None``,此情况下LazyLLM会自动生成随机端口号。
    batch-size (int): 最大batch数， 默认为 ``256``。
''')

add_english_doc('deploy.Infinity', '''\
This class is a subclass of ``LazyLLMDeployBase``, providing high-performance text-embeddings, reranking, and CLIP capabilities based on the [Infinity](https://github.com/michaelfeil/infinity) framework.

Args:
    launcher (lazyllm.launcher): The launcher for Infinity, defaulting to ``launchers.remote(ngpus=1)``.
    kw: Keyword arguments for updating default training parameters. Note that no additional keyword arguments can be passed here except those listed below.

The keyword arguments and their default values for this class are as follows:

Keyword Args: 
    host (str): The IP address of the service, defaulting to ``0.0.0.0``.
    port (int): The port number of the service, defaulting to ``None``, in which case LazyLLM will automatically generate a random port number.
    batch-size (int): The maximum batch size, defaulting to ``256``.
''')

add_example('deploy.Infinity', '''\
>>> import lazyllm
>>> from lazyllm import deploy
>>> deploy.Infinity()
<lazyllm.llm.deploy type=Infinity>
''')

# RelayServer class documentation
add_chinese_doc('deploy.relay.base.RelayServer', '''\
RelayServer类是一个用于部署FastAPI服务的基类，它可以将一个函数转换为HTTP服务。这个类支持设置前处理函数、后处理函数，
并可以自动分配端口号。它主要用于将模型推理功能转换为HTTP服务，便于分布式部署和调用。

主要参数：
    port: 服务端口号，如果为None则随机分配30000-40000之间的端口
    func: 要部署的主函数
    pre_func: 请求预处理函数
    post_func: 响应后处理函数
    pythonpath: 额外的Python路径
    log_path: 日志存储路径
    cls: 服务名称
    launcher: 启动器类型，默认为异步远程启动
''')

add_english_doc('deploy.relay.base.RelayServer', '''\
RelayServer is a base class for deploying FastAPI services that converts a function into an HTTP service. It supports 
setting pre-processing and post-processing functions, and can automatically allocate port numbers. It's mainly used 
to convert model inference functionality into HTTP services for distributed deployment and invocation.

Main parameters:
    port: Service port number, randomly assigned between 30000-40000 if None
    func: Main function to be deployed
    pre_func: Request pre-processing function
    post_func: Response post-processing function
    pythonpath: Additional Python path
    log_path: Log storage path
    cls: Service name
    launcher: Launcher type, defaults to asynchronous remote launch
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

参数：
    func: 可选，要部署的新函数。如果不提供，则使用初始化时的函数。

返回值：
    返回一个LazyLLMCMD对象，包含服务器启动命令和相关配置。
''')

add_english_doc('deploy.relay.base.RelayServer.cmd', '''\
The cmd method generates the command to start the server. It converts the current function and configuration into 
an executable command string.

Args:
    func: Optional, new function to deploy. If not provided, uses the function from initialization.

Returns:
    Returns a LazyLLMCMD object containing the server start command and related configuration.
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

参数：
    job: 可选，指定的任务对象。如果为None，则使用当前实例的任务。

返回值：
    返回服务的完整URL地址，格式为 http://<ip>:<port>/generate
''')

add_english_doc('deploy.relay.base.RelayServer.geturl', '''\
The geturl method returns the access URL for the service. This URL can be used to send HTTP requests to the service.

Args:
    job: Optional, specified job object. If None, uses the current instance's job.

Returns:
    Returns the complete URL of the service in the format http://<ip>:<port>/generate
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

属性：
- keys_name_handle (dict): 输入字段名的映射。
- message_format (dict): 默认请求模板，包括输入内容与生成参数。

参数：
- launcher: 部署器实例，默认值为 `launchers.remote(sync=False)`。
- stream (bool): 是否以流式方式输出结果。
- kw: 其他传递给父类的关键字参数。

方法：
- __call__(*args): 启动部署并返回服务地址。
- __repr__(): 返回流水线的字符串表示。
''')

add_english_doc('deploy.base.DummyDeploy', '''\
DummyDeploy(launcher=launchers.remote(sync=False), *, stream=False, **kw)

A mock deployment class for testing purposes. It extends both `LazyLLMDeployBase` and `flows.Pipeline`,
simulating a simple pipeline-style deployable service with optional streaming support.

This class is primarily intended for internal testing and demonstration. It receives inputs in the format defined
by `message_format`, and returns a dummy response or a streaming response depending on the `stream` flag.

Attributes:
- keys_name_handle (dict): Mapping of input keys for request formatting.
- message_format (dict): Default request template including input and generation parameters.

Parameters:
- launcher: Deployment launcher instance, defaulting to `launchers.remote(sync=False)`.
- stream (bool): Whether to simulate streaming output.
- kw: Additional keyword arguments passed to the superclass.

Methods:
- __call__(*args): Starts the deployment and returns the service URL.
- __repr__(): Returns a string representation of the underlying pipeline.
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

add_chinese_doc('ModelManager', '''\
ModelManager是LazyLLM为开发者提供的自动下载模型的工具类。目前支持从一个本地目录列表查找指定模型，以及从huggingface或者modelscope自动下载模型数据至指定目录。
在使用ModelManager之前，需要设置下列环境变量：

- LAZYLLM_MODEL_SOURCE: 模型下载源，可以设置为 ``huggingface`` 或 ``modelscope`` 。
- LAZYLLM_MODEL_SOURCE_TOKEN: ``huggingface`` 或 ``modelscope`` 提供的token，用于下载私有模型。
- LAZYLLM_MODEL_PATH: 冒号 ``:`` 分隔的本地绝对路径列表用于搜索模型。
- LAZYLLM_MODEL_CACHE_DIR: 下载后的模型在本地的存储目录

Keyword Args: 
    model_source (str, 可选): 模型下载源，目前仅支持 ``huggingface`` 或 ``modelscope`` 。如有必要，ModelManager将从此下载源下载模型数据。如果不提供，默认使用
        LAZYLLM_MODEL_SOURCE环境变量中的设置。如未设置LAZYLLM_MODEL_SOURCE，ModelManager将从 ``modelscope`` 下载模型。
    token (str, 可选): ``huggingface`` 或 ``modelscope`` 提供的token。如果token不为空，ModelManager将使用此token下载模型数据。如果不提供，默认使用
        LAZYLLM_MODEL_SOURCE_TOKEN环境变量中的设置。如未设置LAZYLLM_MODEL_SOURCE_TOKEN，ModelManager将不会自动下载私有模型。
    model_path (str, 可选)：冒号(:)分隔的本地绝对路径列表。在实际下载模型数据之前，ModelManager将在此列表包含的目录中尝试寻找目标模型。如果不提供，默认使用
        LAZYLLM_MODEL_PATH环境变量中的设置。如果为空或LAZYLLM_MODEL_PATH未设置，ModelManager将跳过从model_path中寻找模型的步骤。
    cache_dir (str, 可选): 一个本地目录的绝对路径。下载后的模型将存放在此目录下，如果不提供，默认使用LAZYLLM_MODEL_CACHE_DIR环境变量中的设置。如果
        LAZYLLM_MODEL_PATH未设置，默认值为~/.lazyllm/model
ModelManager.download(model) -> str

用于从model_source下载模型。download函数首先在ModelManager类初始化参数model_path列出的目录中搜索目标模型。如果未找到，会在cache_dir下搜索目标模型。如果仍未找到，
则从model_source上下载模型并存放于cache_dir下。

Args:
    model (str): 目标模型名称。download函数使用此名称从model_source上下载模型。为了方便开发者使用，LazyLLM为常用模型建立了简略模型名称到下载源实际模型名称的映射，
        例如 ``Llama-3-8B`` , ``GLM3-6B`` 或 ``Qwen1.5-7B`` 。具体可参考文件 ``lazyllm/module/utils/downloader/model_mapping.py`` 。model可以接受简略模型名或下载源中的模型全名。
''')

add_english_doc('ModelManager', '''\
ModelManager is a utility class provided by LazyLLM for developers to automatically download models.
Currently, it supports search for models from local directories, as well as automatically downloading model from
huggingface or modelscope. Before using ModelManager, the following environment variables need to be set:

- LAZYLLM_MODEL_SOURCE: The source for model downloads, which can be set to ``huggingface`` or ``modelscope`` .
- LAZYLLM_MODEL_SOURCE_TOKEN: The token provided by ``huggingface`` or ``modelscope`` for private model download.
- LAZYLLM_MODEL_PATH: A colon-separated ``:`` list of local absolute paths for model search.
- LAZYLLM_MODEL_CACHE_DIR: Directory for downloaded models.

Keyword Args: 
    model_source (str, optional): The source for model downloads, currently only supports ``huggingface`` or ``modelscope`` .
        If necessary, ModelManager downloads model data from the source. If not provided, LAZYLLM_MODEL_SOURCE
        environment variable would be used, and if LAZYLLM_MODEL_SOURCE is not set, ModelManager will not download
        any model.
    token (str, optional): The token provided by ``huggingface`` or ``modelscope`` . If the token is present, ModelManager uses
        the token to download model. If not provided, LAZYLLM_MODEL_SOURCE_TOKEN environment variable would be used.
        and if LAZYLLM_MODEL_SOURCE_TOKEN is not set, ModelManager will not download private models, only public ones.
    model_path (str, optional): A colon-separated list of absolute paths. Before actually start to download model,
        ModelManager trys to find the target model in the directories in this list. If not provided,
        LAZYLLM_MODEL_PATH environment variable would be used, and LAZYLLM_MODEL_PATH is not set, ModelManager skips
        looking for models from model_path.
    cache_dir (str, optional): An absolute path of a directory to save downloaded models. If not provided,
        LAZYLLM_MODEL_CACHE_DIR environment variable would be used, and if LAZYLLM_MODEL_PATH is not set, the default
        value is ~/.lazyllm/model.

<span style="font-size: 20px;">&ensp;**`ModelManager.download(model) -> str`**</span>

Download models from model_source. The function first searches for the target model in directories listed in the
model_path parameter of ModelManager class. If not found, it searches under cache_dir. If still not found,
it downloads the model from model_source and stores it under cache_dir.

Args:
    model (str): The name of the target model. The function uses this name to download the model from model_source.
    To further simplify use of the function, LazyLLM provides a mapping dict from abbreviated model names to original
    names on the download source for popular models, such as ``Llama-3-8B`` , ``GLM3-6B`` or ``Qwen1.5-7B``. For more details,
    please refer to the file ``lazyllm/module/utils/downloader/model_mapping.py`` . The model argument can be either
    an abbreviated name or one from the download source.
''')

add_example('ModelManager', '''\
>>> from lazyllm.components import ModelManager
>>> downloader = ModelManager(model_source='modelscope')
>>> downloader.download('chatglm3-6b')
''')

# ============= Formatter

# FormatterBase
add_chinese_doc('formatter.FormatterBase', '''\
此类是格式化器的基类，格式化器是模型输出结果的格式化器，用户可以自定义格式化器，也可以使用LazyLLM提供的格式化器。
主要方法：_parse_formatter:解析索引内容。_load:解析str对象，其中包含python对象的部分被解析出来，比如list，dict等对象。_parse_py_data_by_formatter:根据自定义的格式化器和索引对python对象进行格式化。format:对传入的内容进行格式化，如果内容是字符串类型，先将字符串转化为python对象，再进行格式化。如果内容是python对象，直接进行格式化。
''')

add_english_doc('formatter.FormatterBase', '''\
This class is the base class of the formatter. The formatter is the formatter of the model output result. Users can customize the formatter or use the formatter provided by LazyLLM.
Main methods: _parse_formatter: parse the index content. _load: Parse the str object, and the part containing Python objects is parsed out, such as list, dict and other objects. _parse_py_data_by_formatter: format the python object according to the custom formatter and index. format: format the passed content. If the content is a string type, convert the string into a python object first, and then format it. If the content is a python object, format it directly.
''')

add_example('formatter.FormatterBase', '''\
>>> from lazyllm.components.formatter import FormatterBase
>>> class MyFormatter(FormatterBase):
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

# encode_query_with_filepaths
add_chinese_doc('formatter.encode_query_with_filepaths', '''\
将查询文本和文件路径编码为带有文档上下文的结构化字符串格式。

当指定文件路径时，该函数会将查询内容与文件路径打包成 JSON 格式，并在前缀 ``__lazyllm_docs__`` 的基础上编码返回。否则仅返回原始查询文本。

Args:
    query (str): 用户查询字符串，默认为空字符串。
    files (str or List[str]): 与查询相关的文档路径，可为单个字符串或字符串列表。

Returns:
    str: 编码后的结构化查询字符串，或原始查询。

Raises:
    AssertionError: 如果 `files` 不是字符串或字符串列表，或列表中元素类型错误。
''')

add_english_doc('formatter.encode_query_with_filepaths', '''\
Encodes a query string together with associated file paths into a structured string format with context.

If file paths are provided, the query and file list will be wrapped into a JSON object prefixed with ``__lazyllm_docs__``. Otherwise, it returns the original query string.

Args:
    query (str): The user query string. Defaults to an empty string.
    files (str or List[str]): File path(s) associated with the query. Can be a single string or a list of strings.

Returns:
    str: A structured encoded query string or the raw query.

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

Returns:
    Union[dict, str]: 若为结构化格式则返回包含 'query' 和 'files' 的字典，否则返回原始查询字符串。

Raises:
    AssertionError: 如果输入参数不是字符串类型。
    ValueError: 如果字符串为结构化格式但解析 JSON 失败。
''')

add_english_doc('formatter.decode_query_with_filepaths', '''\
Decodes a structured query string into a dictionary containing the original query and file paths.

If the input string starts with the special prefix ``__lazyllm_docs__``, it attempts to parse the JSON content; otherwise, it returns the raw query string as-is.

Args:
    query_files (str): The encoded query string that may include both query and file paths.

Returns:
    Union[dict, str]: A dictionary containing 'query' and 'files' if structured, otherwise the original query string.

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

Returns:
    str: 合并后的结构化查询字符串，包含统一的查询内容与文件路径。
''')

add_english_doc('formatter.lazyllm_merge_query', '''\
Merges multiple query strings (potentially with associated file paths) into a single structured query string.

Each argument can be a plain query string or a structured query created by ``encode_query_with_filepaths``. The function decodes each input, concatenates all query texts, and merges the associated file paths. The final result is re-encoded into a single query string with unified context.

Args:
    *args (str): Multiple query strings. Each can be either plain text or an encoded structured query with files.

Returns:
    str: A single structured query string containing the merged query and file paths.
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

Returns:
    Prompter: 返回一个初始化的 Prompter 实例。
''')

add_english_doc('Prompter.from_dict', '''\
Initializes a Prompter instance from a prompt configuration dictionary.

Args:
    prompt (Dict): A dictionary containing prompt-related configuration. Must include 'prompt' key.
    show (bool): Whether to display the generated prompt. Defaults to False.

Returns:
    Prompter: An initialized Prompter instance.
''')

# Prompter.from_template
add_chinese_doc('Prompter.from_template', '''\
根据模板名称加载 prompt 配置并初始化 Prompter 实例。

Args:
    template_name (str): 模板名称，必须在 `templates` 中存在。
    show (bool): 是否显示生成的 prompt，默认为 False。

Returns:
    Prompter: 返回一个初始化的 Prompter 实例。
''')

add_english_doc('Prompter.from_template', '''\
Loads prompt configuration from a template name and initializes a Prompter instance.

Args:
    template_name (str): Name of the template. Must exist in the `templates` dictionary.
    show (bool): Whether to display the generated prompt. Defaults to False.

Returns:
    Prompter: An initialized Prompter instance.
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
    - Input for infer: `str`. A description of the image to be generated.
    - Return of infer: The string encoded from the generated file paths, starting with the encoding flag "<lazyllm-query>", followed by the serialized dictionary. The key `files` in the dictionary stores a list, with elements being the paths of the generated image files.
    - Supported models: [stable-diffusion-3-medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium)
''')

add_chinese_doc('StableDiffusionDeploy', '''\
Stable Diffusion 模型部署类。该类用于将SD模型部署到指定服务器上，以便可以通过网络进行调用。

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
    - 推理的输入：字符串。待生成图像的描述。
    - 推理的返回值：从生成的文件路径编码的字符串， 编码标志以 "<lazyllm-query>"开头，后面跟序列化后的字典, 字典中 `files`键存放了一个列表，元素是生成的图像文件路径。
    - 支持的模型为：[stable-diffusion-3-medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium)
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
ChatTTS Model Deployment Class. This class is used to deploy the ChatTTS model to a specified server for network invocation.

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
    - Supported models: [ChatTTS](https://huggingface.co/2Noise/ChatTTS)
''')

add_chinese_doc('ChatTTSDeploy', '''\
ChatTTS 模型部署类。该类用于将ChatTTS模型部署到指定服务器上，以便可以通过网络进行调用。

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
    - 支持的模型为：[ChatTTS](https://huggingface.co/2Noise/ChatTTS)
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
    launcher (lazyllm.launcher): An instance of the launcher used to start the remote service.

`__call__(self, finetuned_model=None, base_model=None)`
Deploys the model and returns the remote service address.

Args:
    finetuned_model (str): If provided, this model will be used for deployment; if not provided or the path is invalid, `base_model` will be used.
    base_model (str): The default model, which will be used for deployment if `finetuned_model` is invalid.
    Return (str): The URL address of the remote service.

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
    launcher(lazyllm.launcher): 用于启动远程服务的启动器实例。

`__call__(self, finetuned_model=None, base_model=None)`
部署模型，并返回远程服务地址。

Args: 
    finetuned_model (str): 如果提供，则使用该模型进行部署；如果未提供或路径无效，则使用 `base_model`。
    base_model (str): 默认模型，如果 `finetuned_model` 无效，则使用该模型进行部署。
    返回值 (str): 远程服务的URL地址。
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

add_english_doc('deploy.speech_to_text.sense_voice.SenseVoice', '''\
SenseVoice(base_path, source=None, init=False)

A speech-to-text wrapper using FunASR models for lazy initialization and audio transcription.
This class supports automatic model downloading, safe initialization, and inference from audio paths or URLs.

Parameters:
- base_path (str): Path or model identifier to download the STT model.
- source (str, optional): Model source name; defaults to `lazyllm.config['model_source']`.
- init (bool): Whether to initialize the model immediately on creation.

Attributes:
- base_path (str): Final resolved path of the model after download.
- model: Loaded FunASR model instance.
- init_flag: A lazy flag used to ensure model is only loaded once.

Methods:
- __call__(string: str | dict) -> str:
    Transcribes the input audio file or URL to text. Accepts base64-encoded content, file paths, or URLs.
- load_stt():
    Loads the FunASR speech-to-text model and related VAD (Voice Activity Detection).
- rebuild(base_path, init):
    Rebuilds the class instance (used for serialization).
- __reduce__():
    Supports pickling by ensuring proper lazy-loading on deserialization.
''')

add_chinese_doc('deploy.speech_to_text.sense_voice.SenseVoice', '''\
SenseVoice(base_path, source=None, init=False)

使用 FunASR 模型进行语音转文本的包装类，支持懒加载与自动模型下载。
支持从音频路径、URL 或 base64 编码音频进行转写，适用于延迟初始化和高效部署。

参数：
- base_path (str): 用于下载语音识别模型的路径或模型标识。
- source (str, 可选): 模型来源，默认使用 `lazyllm.config['model_source']`。
- init (bool): 是否在初始化时立即加载模型。

属性：
- base_path (str): 下载后模型的实际路径。
- model: 加载的 FunASR 模型对象。
- init_flag: 用于懒加载的初始化标志，保证模型只加载一次。

方法：
- __call__(string: str | dict) -> str:
    将输入的音频文件或 URL 转换为文本。支持 base64 编码、文件路径或 URL 输入。
- load_stt():
    加载 FunASR 的语音识别模型和语音活动检测（VAD）模型。
- rebuild(base_path, init):
    用于重新构造类实例（常用于序列化）。
- __reduce__():
    实现 pickling 支持，确保在反序列化时正确懒加载。
''')

add_english_doc('deploy.speech_to_text.sense_voice.SenseVoice.load_stt', '''\
load_stt()

Loads the speech-to-text model using FunASR with optional support for Huawei NPU via `torch_npu`.

The method initializes the model with the following characteristics:
- Uses `fsmn-vad` for voice activity detection with long utterance support.
- Sets maximum single segment time to 30 seconds.
- Selects `cuda:0` as the default inference device.

The model is stored in `self.model` and will be used to transcribe audio input.

Note:
If `torch_npu` is available in the environment, the function attempts to load it for potential Huawei Ascend acceleration.
''')

add_chinese_doc('deploy.speech_to_text.sense_voice.SenseVoice.load_stt', '''\
load_stt()

使用 FunASR 加载语音转文本模型，支持华为 NPU（如存在 `torch_npu`）。

此方法将初始化模型，包含以下设置：
- 使用 `fsmn-vad` 进行语音活动检测（VAD），支持最长 30 秒的单段语音。
- 设置推理设备为 `cuda:0`（默认使用 GPU）。
- 将模型实例保存在 `self.model` 中，用于后续音频转写。

注意：
如果当前环境中存在 `torch_npu`，函数将自动导入以支持华为昇腾设备加速。
''')

add_english_doc('deploy.speech_to_text.sense_voice.SenseVoice.rebuild', '''\
rebuild(base_path: str, init: bool) -> SenseVoice

Class method used to reconstruct a `SenseVoice` instance during deserialization (e.g., when using `cloudpickle`).

Parameters:
- base_path (str): Path to the speech-to-text model.
- init (bool): Whether to immediately initialize and load the model upon creation.

Returns:
- A new instance of `SenseVoice` with the specified configuration.

Note:
This method is internally used to support model serialization and multiprocessing compatibility.
''')

add_chinese_doc('deploy.speech_to_text.sense_voice.SenseVoice.rebuild', '''\
rebuild(base_path: str, init: bool) -> SenseVoice

该类方法用于反序列化（如 `cloudpickle`）过程中重新构建 `SenseVoice` 实例。

参数：
- base_path (str)：语音识别模型的路径。
- init (bool)：是否在实例化时立即加载模型。

返回：
- 一个新的 `SenseVoice` 实例。

说明：
该方法主要用于支持对象的序列化与多进程环境下的兼容重建操作。
''')

add_english_doc('TTSDeploy', '''\
TTSDeploy is a factory class for creating instances of different Text-to-Speech (TTS) deployment types based on the specified name.

`__new__(cls, name, **kwarg)`
The constructor dynamically creates and returns the corresponding deployment instance based on the provided name argument.

Args:
    name: A string specifying the type of deployment instance to be created.
    **kwarg: Keyword arguments to be passed to the constructor of the corresponding deployment instance.

Returns:
    If the name argument is 'bark', an instance of [BarkDeploy][lazyllm.components.BarkDeploy] is returned.
    If the name argument is 'ChatTTS', an instance of [ChatTTSDeploy][lazyllm.components.ChatTTSDeploy] is returned.
    If the name argument starts with 'musicgen', an instance of [MusicGenDeploy][lazyllm.components.MusicGenDeploy] is returned.
    If the name argument does not match any of the above cases, a RuntimeError exception is raised, indicating the unsupported model.            
''')

add_chinese_doc('TTSDeploy', '''\
TTSDeploy 是一个用于根据指定的名称创建不同类型文本到语音(TTS)部署实例的工厂类。

`__new__(cls, name, **kwarg)`
构造函数，根据提供的名称参数动态创建并返回相应的部署实例。

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
     `cmd(self, *args, **kw) -> str`
Generates a dummy command string for fine-tuning. This method is for testing purposes only.
Args:
    *args: Positional arguments to be included in the command.
    **kw: Keyword arguments to be included in the command.
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

     `cmd(self, *args, **kw) -> str`
生成一个用于微调的占位命令字符串。此方法仅用于测试目的。
Args:
    *args: 要包含在命令中的位置参数。
    **kw: 要包含在命令中的关键字参数。
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
    finetuned_model: A string specifying the path or name of the fine-tuned OCR model. Defaults to `None`.
    base_model: A string specifying the base model name. If `finetuned_model` is not provided, `base_model` will be used. Defaults to `None`.
Returns:
    An instance of [RelayServer][lazyllm.deploy.RelayServer], which acts as the deployment server for the OCR model.
Example:
    ```python
    deployer = OCRDeploy(launcher=launchers.local(), log_path='./logs', port=8080)
    server = deployer(finetuned_model='ocr-model')
    print(server)  # RelayServer instance ready to handle OCR requests
    ```
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
    finetuned_model: 字符串，指定微调 OCR 模型的路径或名称。默认为 `None`。
    base_model: 字符串，指定基础模型的名称。如果未提供 `finetuned_model`，将使用 `base_model`。默认为 `None`。
Returns:
    [RelayServer][lazyllm.deploy.RelayServer] 的实例，作为 OCR 模型的部署服务器。
Example:
    ```python
    deployer = OCRDeploy(launcher=launchers.local(), log_path='./logs', port=8080)
    server = deployer(finetuned_model='ocr-model')
    print(server)  # RelayServer 实例，准备处理 OCR 请求
    ```
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

# ============= Launcher

add_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm.launcher)
add_english_doc = functools.partial(utils.add_english_doc, module=lazyllm.launcher)
add_example = functools.partial(utils.add_example, module=lazyllm.launcher)

# Launcher-EmptyLauncher
add_chinese_doc('EmptyLauncher', '''\
此类是 ``LazyLLMLaunchersBase`` 的子类，作为一个本地的启动器。

Args:
    subprocess (bool): 是否使用子进程来启动。默认为 `False`。
    sync (bool): 是否同步执行作业。默认为 `True`，否则为异步执行。

''')

add_english_doc('EmptyLauncher', '''\
This class is a subclass of ``LazyLLMLaunchersBase`` and serves as a local launcher.

Args:
    subprocess (bool): Whether to use a subprocess to launch. Default is ``False``.
    sync (bool): Whether to execute jobs synchronously. Default is ``True``, otherwise it executes asynchronously.

''')

add_example('EmptyLauncher', '''\
>>> import lazyllm
>>> launcher = lazyllm.launchers.empty()
''')

# Launcher-SlurmLauncher
add_chinese_doc('SlurmLauncher', '''\
此类是 ``LazyLLMLaunchersBase`` 的子类，作为slurm启动器。

具体而言，它提供了启动和配置 Slurm 作业的方法，包括指定分区、节点数量、进程数量、GPU 数量以及超时时间等参数。

Args:
    partition (str): 要使用的 Slurm 分区。默认为 ``None``，此时将使用 ``lazyllm.config['partition']`` 中的默认分区。该配置可通过设置环境变量来生效，如 ``export LAZYLLM_SLURM_PART=a100`` 。
    nnode  (int): 要使用的节点数量。默认为 ``1``。
    nproc (int): 每个节点要使用的进程数量。默认为 ``1``。
    ngpus: (int): 每个节点要使用的 GPU 数量。默认为 ``None``, 即不使用 GPU。
    timeout (int): 作业的超时时间（以秒为单位）。默认为 ``None``，此时将不设置超时时间。
    sync (bool): 是否同步执行作业。默认为 ``True``，否则为异步执行。

''')

add_english_doc('SlurmLauncher', '''\
This class is a subclass of ``LazyLLMLaunchersBase`` and acts as a Slurm launcher.

Specifically, it provides methods to start and configure Slurm jobs, including specifying parameters such as the partition, number of nodes, number of processes, number of GPUs, and timeout settings.

Args:
    partition (str): The Slurm partition to use. Defaults to ``None``, in which case the default partition in ``lazyllm.config['partition']`` will be used. This configuration can be enabled by setting environment variables, such as ``export LAZYLLM_SLURM_PART=a100``.
    nnode  (int): The number of nodes to use. Defaults to ``1``.
    nproc (int): The number of processes per node. Defaults to ``1``.
    ngpus (int): The number of GPUs per node. Defaults to ``None``, meaning no GPUs will be used.
    timeout (int): The timeout for the job in seconds. Defaults to ``None``, in which case no timeout will be set.
    sync (bool): Whether to execute the job synchronously. Defaults to ``True``, otherwise it will be executed asynchronously.

''')

add_example('SlurmLauncher', '''\
>>> import lazyllm
>>> launcher = lazyllm.launchers.slurm(partition='partition_name', nnode=1, nproc=1, ngpus=1, sync=False)
''')

# SlurmLauncher methods
add_chinese_doc('SlurmLauncher.makejob', '''\
创建并返回一个 SlurmLauncher.Job 对象。

Args:
    cmd: 要执行的命令字符串。

Returns:
    SlurmLauncher.Job: 配置好的 Slurm 作业对象。
''')

add_english_doc('SlurmLauncher.makejob', '''\
Creates and returns a SlurmLauncher.Job object.

Args:
    cmd: The command string to execute.

Returns:
    SlurmLauncher.Job: A configured Slurm job object.
''')

add_chinese_doc('SlurmLauncher.get_idle_nodes', '''\
获取指定分区中当前可用的节点数量，基于可用 GPU 数量。

该方法通过查询 Slurm 队列状态和节点信息，计算每个节点的可用 GPU 数量，并返回一个字典，其中键为节点 IP，值为可用 GPU 数量。

Args:
    partion (str, optional): 要查询的分区名称。默认为 ``None``，此时使用当前启动器的分区。

Returns:
    dict: 以节点 IP 为键、可用 GPU 数量为值的字典。
''')

add_english_doc('SlurmLauncher.get_idle_nodes', '''\
Obtains the current number of available nodes in the specified partition based on the available number of GPUs.

This method queries the Slurm queue status and node information to calculate the number of available GPUs for each node, and returns a dictionary with node IP as the key and the number of available GPUs as the value.

Args:
    partion (str, optional): The partition name to query. Defaults to ``None``, in which case the current launcher's partition will be used.

Returns:
    dict: A dictionary with node IP as the key and the number of available GPUs as the value.
''')

add_chinese_doc('SlurmLauncher.launch', '''\
启动 Slurm 作业并管理其执行。

该方法启动指定的 Slurm 作业，并根据同步设置决定是否等待作业完成。如果设置为同步执行，会持续监控作业状态直到完成，然后停止作业。

Args:
    job: 要启动的 SlurmLauncher.Job 对象。

Returns:
    作业的返回值。

Raises:
    AssertionError: 如果传入的 job 不是 SlurmLauncher.Job 类型。
''')

add_english_doc('SlurmLauncher.launch', '''\
Launches a Slurm job and manages its execution.

This method starts the specified Slurm job and decides whether to wait for job completion based on the sync setting. If set to synchronous execution, it continuously monitors the job status until completion, then stops the job.

Args:
    job: The SlurmLauncher.Job object to launch.

Returns:
    The return value of the job.

Raises:
    AssertionError: If the provided job is not a SlurmLauncher.Job type.
''')

# Launcher-ScoLauncher
add_chinese_doc('ScoLauncher', '''\
此类是 ``LazyLLMLaunchersBase`` 的子类，作为SCO (Sensecore)启动器。

具体而言，它提供了启动和配置 SCO 作业的方法，包括指定分区、工作空间名称、框架类型、节点数量、进程数量、GPU 数量以及是否使用 torchrun 等参数。

Args:
    partition (str): 要使用的分区。默认为 ``None``，此时将使用 ``lazyllm.config['partition']`` 中的默认分区。该配置可通过设置环境变量来生效，如 ``export LAZYLLM_SLURM_PART=a100`` 。
    workspace_name (str): SCO 上的工作空间名称。默认为 ``lazyllm.config['sco.workspace']`` 中的配置。该配置可通过设置环境变量来生效，如 ``export LAZYLLM_SCO_WORKSPACE=myspace`` 。
    framework (str): 要使用的框架类型，例如 ``pt`` 代表 PyTorch。默认为 ``pt``。
    nnode  (int): 要使用的节点数量。默认为 ``1``。
    nproc (int): 每个节点要使用的进程数量。默认为 ``1``。
    ngpus: (int): 每个节点要使用的 GPU 数量。默认为 ``1``, 使用1块 GPU。
    torchrun (bool): 是否使用 ``torchrun`` 启动作业。默认为 ``False``。
    sync (bool): 是否同步执行作业。默认为 ``True``，否则为异步执行。

''')

add_english_doc('ScoLauncher', '''\
This class is a subclass of ``LazyLLMLaunchersBase`` and acts as a SCO launcher.

Specifically, it provides methods to start and configure SCO jobs, including specifying parameters such as the partition, workspace name, framework type, number of nodes, number of processes, number of GPUs, and whether to use torchrun or not.

Args:
    partition (str): The Slurm partition to use. Defaults to ``None``, in which case the default partition in ``lazyllm.config['partition']`` will be used. This configuration can be enabled by setting environment variables, such as ``export LAZYLLM_SLURM_PART=a100``.
    workspace_name (str): The workspace name on SCO. Defaults to the configuration in ``lazyllm.config['sco.workspace']``. This configuration can be enabled by setting environment variables, such as ``export LAZYLLM_SCO_WORKSPACE=myspace``.
    framework (str): The framework type to use, for example, ``pt`` for PyTorch. Defaults to ``pt``.
    nnode  (int): The number of nodes to use. Defaults to ``1``.
    nproc (int): The number of processes per node. Defaults to ``1``.
    ngpus (int): The number of GPUs per node. Defaults to ``1``, using 1 GPU.
    torchrun (bool): Whether to start the job with ``torchrun``. Defaults to ``False``.
    sync (bool): Whether to execute the job synchronously. Defaults to ``True``, otherwise it will be executed asynchronously.

''')

add_example('ScoLauncher', '''\
>>> import lazyllm
>>> launcher = lazyllm.launchers.sco(partition='partition_name', nnode=1, nproc=1, ngpus=1, sync=False)
''')

# Launcher-RemoteLauncher
add_chinese_doc('RemoteLauncher', '''\
此类是 ``LazyLLMLaunchersBase`` 的一个子类，它充当了一个远程启动器的代理。它根据配置文件中的 ``lazyllm.config['launcher']`` 条目动态地创建并返回一个对应的启动器实例(例如：``SlurmLauncher`` 或 ``ScoLauncher``)。

Args:
    *args: 位置参数，将传递给动态创建的启动器构造函数。
    sync (bool): 是否同步执行作业。默认为 ``False``。
    **kwargs: 关键字参数，将传递给动态创建的启动器构造函数。

注意事项: 
    - ``RemoteLauncher`` 不是一个直接的启动器，而是根据配置动态创建一个启动器。 
    - 配置文件中的 ``lazyllm.config['launcher']`` 指定一个存在于 ``lazyllm.launchers`` 模块中的启动器类名。该配置可通过设置环境变量 ``LAZYLLM_DEFAULT_LAUNCHER`` 来设置。如：``export LAZYLLM_DEFAULT_LAUNCHER=sco`` , ``export LAZYLLM_DEFAULT_LAUNCHER=slurm`` 。
''')

add_english_doc('RemoteLauncher', '''\
This class is a subclass of ``LazyLLMLaunchersBase`` and acts as a proxy for a remote launcher. It dynamically creates and returns an instance of the corresponding launcher based on the ``lazyllm.config['launcher']`` entry in the configuration file (for example: ``SlurmLauncher`` or ``ScoLauncher``).

Args:
    *args: Positional arguments that will be passed to the constructor of the dynamically created launcher.
    sync (bool): Whether to execute the job synchronously. Defaults to ``False``.
    **kwargs: Keyword arguments that will be passed to the constructor of the dynamically created launcher.

Notes: 
    - ``RemoteLauncher`` is not a direct launcher but dynamically creates a launcher based on the configuration. 
    - The ``lazyllm.config['launcher']`` in the configuration file specifies a launcher class name present in the ``lazyllm.launchers`` module. This configuration can be set by setting the environment variable ``LAZYLLM_DEAULT_LAUNCHER``. For example: ``export LAZYLLM_DEAULT_LAUNCHER=sco``, ``export LAZYLLM_DEAULT_LAUNCHER=slurm``.

''')

add_example('RemoteLauncher', '''\
>>> import lazyllm
>>> launcher = lazyllm.launchers.remote(ngpus=1)
''')

# core.py
add_chinese_doc('lazyllm.components.core.ComponentBase', '''\
组件基类，提供统一的接口与基础实现，便于创建不同类型的组件。  
组件通过指定的 Launcher 来执行任务，支持自定义任务执行逻辑。

Args:
    launcher (LazyLLMLaunchersBase or type, optional): 组件使用的启动器实例或启动器类，默认为空启动器（empty）。
''')

add_english_doc('lazyllm.components.core.ComponentBase', '''\
Base class for components, providing a unified interface and basic implementation to facilitate creation of various components.  
Components execute tasks via a specified launcher and support custom task execution logic.

Args:
    launcher (LazyLLMLaunchersBase or type, optional): Launcher instance or launcher class used by the component, defaults to empty launcher.
''')

add_example('lazyllm.components.core.ComponentBase', '''\
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

add_chinese_doc('lazyllm.components.core.ComponentBase.apply', '''\
组件执行的核心方法，需由子类实现。  
定义组件的具体业务逻辑或任务执行步骤。  

**注意:**  
调用组件时，如果子类重写了此方法，则会调用此方法执行任务。  
''')

add_english_doc('lazyllm.components.core.ComponentBase.apply', '''\
Core execution method of the component, to be implemented by subclasses.  
Defines the specific business logic or task execution steps of the component.

**Note:**  
If this method is overridden by the subclass, it will be called when the component is invoked.
''')

add_chinese_doc('lazyllm.components.core.ComponentBase.cmd', '''\
生成组件的执行命令，需由子类实现。  
返回的命令可以是字符串、元组或列表，表示具体执行任务的指令。  

**注意:**  
调用组件时，如果未重写 `apply` 方法，将通过此命令生成任务并由启动器执行。  
''')

add_english_doc('lazyllm.components.core.ComponentBase.cmd', '''\
Generates the execution command of the component, to be implemented by subclasses.  
The returned command can be a string, tuple, or list, representing the instruction to execute the task.

**Note:**  
If the `apply` method is not overridden, this command will be used to create a job for the launcher to run.
''')

add_chinese_doc('Job', '''\
通用任务调度执行类。
该类用于封装一个通过启动器（launcher）调度执行的任务，支持命令包装、同步控制、返回值提取、命令固定等功能。
Args:
    cmd (LazyLLMCMD): 要执行的命令对象。
    launcher (Any): 启动器实例，用于实际任务调度执行。
    sync (bool): 是否为同步执行，默认为 True。
''')

add_english_doc('Job', '''\
Generic task scheduling executor.
This class wraps a task that is launched via a launcher, with features like command fixing, output handling, sync control, and return value capturing.
Args:
    cmd (LazyLLMCMD): The command object to be executed.
    launcher (Any): Launcher instance responsible for task dispatching.
    sync (bool): Whether the task should run synchronously. Defaults to True.
''')

add_chinese_doc('Job.get_executable_cmd', '''\
生成最终可执行命令。
如果已缓存固定命令（fixed），则直接返回。否则根据原始命令进行包裹（wrap）并缓存为 `_fixed_cmd`。
Args:
    fixed (bool): 是否使用已固定的命令对象（若已存在）。
Returns:
    LazyLLMCMD: 可直接执行的命令对象。
''')

add_english_doc('Job.get_executable_cmd', '''\
Generate the final executable command.
If a fixed command already exists, return it. Otherwise, wrap the original command and cache it as `_fixed_cmd`.
Args:
    fixed (bool): Whether to use the cached fixed command.
Returns:
    LazyLLMCMD: The executable command object.
''')

add_chinese_doc('Job.start', '''\
对外接口：启动作业，并支持失败时的自动重试。
若作业执行失败，会根据 `restart` 参数控制重试次数。
Args:
    restart (int): 重试次数。默认为 3。
    fixed (bool): 是否使用固定后的命令。用于避免多次构建。
''')

add_english_doc('Job.start', '''\
Public interface to start the job with optional retry on failure.
If the job fails, retries execution based on the `restart` parameter.
Args:
    restart (int): Number of times to retry upon failure. Default is 3.
    fixed (bool): Whether to use the fixed version of the command.
''')

add_chinese_doc('Job.restart', '''\
重新启动作业流程。
该函数会先停止已有进程，等待 2 秒后重新启动作业。
Args:
    fixed (bool): 是否使用固定后的命令。
''')

add_english_doc('Job.restart', '''\
Restart the job by first stopping it and then restarting after a short delay.
Args:
    fixed (bool): Whether to reuse the fixed command object.
''')

add_chinese_doc('Job.wait', '''\
挂起当前线程，等待作业执行完成。当前实现为空方法（子类可重写）。
''')

add_english_doc('Job.wait', '''\
Suspend the current thread until the job finishes.
Empty implementation by default; can be overridden in subclasses.
''')

add_chinese_doc('Job.stop', '''\
停止当前作业。
该方法为接口定义，需子类实现，当前抛出 NotImplementedError。
''')

add_english_doc('Job.stop', '''\
Stop the current job.
This method is an interface placeholder and must be implemented by subclasses.
''')

add_chinese_doc('Job.status', '''\
当前作业状态。
该属性为接口定义，需子类实现，当前抛出 NotImplementedError。
''')

add_english_doc('Job.status', '''\
Current job status.
This property is abstract and must be implemented by subclasses.
''')

add_chinese_doc('K8sLauncher', '''\
K8sLauncher是一个基于Kubernetes的部署启动器，用于在Kubernetes集群中部署和管理服务。

参数:
    kube_config_path (str): Kubernetes配置文件路径。
    resource_config_path (str): 资源配置文件路径。
    image (str): 容器镜像。
    volume_configs (list): 卷配置列表。
    svc_type (str): 服务类型，默认为"LoadBalancer"。
    namespace (str): Kubernetes命名空间，默认为"default"。
    gateway_name (str): 网关名称，默认为"lazyllm-gateway"。
    gateway_class_name (str): 网关类名称，默认为"istio"。
    host (str): HTTP主机名，默认为None。
    path (str): HTTP路径，默认为'/generate'。
    gateway_retry (int): 网关重试次数。
''')

add_english_doc('K8sLauncher', '''\
K8sLauncher is a Kubernetes-based deployment launcher for deploying and managing services in a Kubernetes cluster.

Args:
    kube_config_path (str): Path to the Kubernetes configuration file.
    resource_config_path (str): Path to the resource configuration file.
    image (str): Container image.
    volume_configs (list): List of volume configurations.
    svc_type (str): Service type, defaults to "LoadBalancer".
    namespace (str): Kubernetes namespace, defaults to "default".
    gateway_name (str): Gateway name, defaults to "lazyllm-gateway".
    gateway_class_name (str): Gateway class name, defaults to "istio".
    host (str): HTTP hostname, defaults to None.
    path (str): HTTP path, defaults to '/generate'.
    gateway_retry (int): Number of gateway retries.
''')

add_chinese_doc('K8sLauncher.makejob', '''\
创建一个Kubernetes作业实例。

参数:
    cmd (str): 要执行的命令。

返回值:
    K8sLauncher.Job: 一个新的Kubernetes作业实例。
''')

add_english_doc('K8sLauncher.makejob', '''\
Create a Kubernetes job instance.

Args:
    cmd (str): The command to execute.

Returns:
    K8sLauncher.Job: A new Kubernetes job instance.
''')

add_chinese_doc('K8sLauncher.launch', '''\
启动一个Kubernetes作业或可调用对象。

参数:
    f (K8sLauncher.Job): 要启动的Kubernetes作业实例。
    *args: 位置参数。
    **kw: 关键字参数。

返回值:
    Any: 作业的返回值。

异常:
    RuntimeError: 当提供的不是Deployment对象时抛出。
''')

add_english_doc('K8sLauncher.launch', '''\
Launch a Kubernetes job or callable object.

Args:
    f (K8sLauncher.Job): The Kubernetes job instance to launch.
    *args: Positional arguments.
    **kw: Keyword arguments.

Returns:
    Any: The return value of the job.

Raises:
    RuntimeError: When the provided object is not a Deployment object.
''')
