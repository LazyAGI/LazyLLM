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
    merge_path (str): 模型合并LoRA权重后的路径，默认为 ``None`` 。如果未指定，则会在 ``target_path`` 下创建 "lora" 和 "merge" 目录，分别作为 ``target_path`` 和  ``merge_path`` 。
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
    merge_path (str): The path where the model merges the LoRA weights, default to `None`. If not specified, "lora" and "merge" directories will be created under ``target_path`` as ``target_path`` and ``merge_path`` respectively.
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
    merge_path (str): 模型合并LoRA权重后的路径，默认为None。如果未指定，则会在 ``target_path`` 下创建 "lora" 和 "merge" 目录，分别作为 ``target_path`` 和  ``merge_path`` 。
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
    merge_path (str): The path where the model merges the LoRA weights, default to ``None``. If not specified, "lora" and "merge" directories will be created under ``target_path`` as ``target_path`` and ``merge_path`` respectively.
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

# Finetune-LlamafactoryFinetune
add_chinese_doc('finetune.LlamafactoryFinetune', '''\
此类是 ``LazyLLMFinetuneBase`` 的子类，基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 框架提供的训练能力，用于对大语言模型(或视觉语言模型)进行训练。

Args:
    base_model (str): 用于进行训练的基模型。要求是基模型的路径。
    target_path (str): 训练后模型保存权重的路径。
    merge_path (str): 模型合并LoRA权重后的路径，默认为None。如果未指定，则会在 ``target_path`` 下创建 "lora" 和 "merge" 目录，分别作为 ``target_path`` 和  ``merge_path`` 。
    config_path (str): LLaMA-Factory的训练配置文件（这里要求格式为yaml），默认为None。如果未指定，则会在当前工作路径下，创建一个名为 ``.temp`` 的文件夹，并在其中生成以 ``train_`` 前缀开头，以 ``.yaml`` 结尾的配置文件。
    export_config_path (str): LLaMA-Factory的Lora权重合并的配置文件（这里要求格式为yaml），默认为None。如果未指定，则会在当前工作路径下中的 ``.temp`` 文件夹内生成以 ``merge_`` 前缀开头，以 ``.yaml`` 结尾的配置文件。
    launcher (lazyllm.launcher): 微调的启动器，默认为 ``launchers.remote(ngpus=1, sync=True)``。
    kw: 关键字参数，用于更新默认的训练参数。

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
    base_model (str): The base model used for training. It is required to be the path of the base model.
    target_path (str): The path where the trained model weights are saved.
    merge_path (str): The path where the model is merged with LoRA weights, default is None. If not specified, "lora" and "merge" directories will be created under ``target_path``, to be used as ``target_path`` and ``merge_path`` respectively.
    config_path (str): The LLaMA-Factory training configuration file (yaml format is required), default is None. If not specified, a ``.temp`` folder will be created in the current working directory, and a configuration file starting with ``train_`` and ending with ``.yaml`` will be generated inside it.
    export_config_path (str): The LLaMA-Factory Lora weight merging configuration file (yaml format is required), default is None. If not specified, a configuration file starting with ``merge_`` and ending with ``.yaml`` will be generated inside the ``.temp`` folder in the current working directory.
    launcher (lazyllm.launcher): The launcher for fine-tuning, default is ``launchers.remote(ngpus=1, sync=True)``.
    kw: Keyword arguments used to update the default training parameters.

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

# Finetune-Auto
add_chinese_doc('auto.AutoFinetune', '''\
此类是 ``LazyLLMFinetuneBase`` 的子类，可根据输入的参数自动选择合适的微调框架和参数，以对大语言模型进行微调。

具体而言，基于输入的：``base_model`` 的模型参数、``ctx_len``、``batch_size``、``lora_r``、``launcher`` 中GPU的类型以及卡数，该类可以自动选择出合适的微调框架（如: ``AlpacaloraFinetune`` 或 ``CollieFinetune``）及所需的参数。

Args:
    base_model (str): 用于进行微调的基模型。要求是基模型的路径。
    source (lazyllm.config['model_source']): 指定模型的下载源。可通过设置环境变量 ``LAZYLLM_MODEL_SOURCE`` 来配置，目前仅支持 ``huggingface`` 或 ``modelscope`` 。若不设置，lazyllm不会启动自动模型下载。
    target_path (str): 微调后模型保存LoRA权重的路径。
    merge_path (str): 模型合并LoRA权重后的路径，默认为 ``None``。如果未指定，则会在 ``target_path`` 下创建 "lora" 和 "merge" 目录，分别作为 ``target_path`` 和  ``merge_path`` 。
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
    merge_path (str): The path where the model merges the LoRA weights, default to ``None``. If not specified, "lora" and "merge" directories will be created under ``target_path`` as ``target_path`` and ``merge_path`` respectively.
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

''')

add_example('deploy.Lightllm', '''\
>>> from lazyllm import deploy
>>> infer = deploy.lightllm()
''')

# Deploy-Vllm
add_chinese_doc('deploy.Vllm', '''\
此类是 ``LazyLLMDeployBase`` 的子类，基于 [VLLM](https://github.com/vllm-project/vllm) 框架提供的推理能力，用于对大语言模型进行推理。

Args:
    trust_remote_code (bool): 是否允许加载来自远程服务器的模型代码，默认为 ``True``。
    launcher (lazyllm.launcher): 微调的启动器，默认为 ``launchers.remote(ngpus=1)``。
    stream (bool): 是否为流式响应，默认为 ``False``。
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
    stream (bool): Whether the response is streaming, default is ``False``.
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

# Deploy-LMDeploy
add_chinese_doc('deploy.LMDeploy', '''\
此类是 ``LazyLLMDeployBase`` 的子类，基于 [LMDeploy](https://github.com/InternLM/lmdeploy) 框架提供的推理能力，用于对大语言模型进行推理。

Args:
    launcher (lazyllm.launcher): 微调的启动器，默认为 ``launchers.remote(ngpus=1)``。
    stream (bool): 是否为流式响应，默认为 ``False``。
    kw: 关键字参数，用于更新默认的训练参数。请注意，除了以下列出的关键字参数外，这里不能传入额外的关键字参数。

此类的关键字参数及其默认值如下：

Keyword Args: 
    tp (int): 张量并行参数，默认为 ``1``。
    server-name (str): 服务的IP地址，默认为 ``0.0.0.0``。
    server-port (int): 服务的端口号，默认为 ``None``,此情况下LazyLLM会自动生成随机端口号。
    max-batch-size (int): 最大batch数， 默认为 ``128``。

''')

add_english_doc('deploy.LMDeploy', '''\
    This class is a subclass of ``LazyLLMDeployBase``, leveraging the inference capabilities provided by the [LMDeploy](https://github.com/InternLM/lmdeploy) framework for inference on large language models.

Args:
    launcher (lazyllm.launcher): The launcher for fine-tuning, defaults to ``launchers.remote(ngpus=1)``.
    stream (bool): Whether to enable streaming response, defaults to ``False``.
    kw: Keyword arguments for updating default training parameters. Note that no additional keyword arguments beyond those listed below can be passed.

Keyword Args: 
    tp (int): Tensor parallelism parameter, defaults to ``1``.
    server_name (str): The IP address of the service, defaults to ``0.0.0.0``.
    server_port (int): The port number of the service, defaults to ``None``. In this case, LazyLLM will automatically generate a random port number.
    max_batch_size (int): Maximum batch size, defaults to ``128``.

''')

add_example('deploy.LMDeploy', '''\
>>> # Basic use:
>>> from lazyllm import deploy
>>> infer = deploy.LMDeploy()
>>>
>>> # MultiModal:
>>> import lazyllm
>>> from lazyllm import deploy, globals
>>> chat = lazyllm.TrainableModule('Mini-InternVL-Chat-2B-V1-5').deploy_method(deploy.LMDeploy)
>>> chat.update_server()
>>> globals['global_parameters']["lazyllm-files"] = {'files': 'path/to/image'}
>>> res = chat('What is it?')
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
>>> m = lazyllm.TrainableModule("internlm2-chat-20b").formatter(JsonFormatter("[:, title]")).prompt(toc_prompt).start()
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

# ============= Prompter

add_chinese_doc('prompter.PrompterBase', '''\
Prompter的基类，自定义的Prompter需要继承此基类，并通过基类提供的 ``_init_prompt`` 函数来设置Prompt模板和Instruction的模板，以及截取结果所使用的字符串。可以查看 :[prompt](/Best%20Practice/prompt) 进一步了解Prompt的设计思想和使用方式。

Prompt模板和Instruction模板都用 ``{}`` 表示要填充的字段，其中Prompt可包含的字段有 ``system``, ``history``, ``tools``, ``user`` 等，而instruction_template可包含的字段为 ``instruction`` 和 ``extro_keys`` 。
``instruction`` 由应用的开发者传入， ``instruction`` 中也可以带有 ``{}`` 用于让定义可填充的字段，方便用户填入额外的信息。如果 ``instruction`` 字段为字符串，则认为是系统instruction；如果是字典，则它包含的key只能是 ``user`` 和 ``system`` 两种选择。 ``user`` 表示用户输入的instruction，在prompt中放在用户输入前面， ``system`` 表示系统instruction，在prompt中凡在system prompt后面。
''')

add_english_doc('prompter.PrompterBase', '''\
The base class of Prompter. A custom Prompter needs to inherit from this base class and set the Prompt template and the Instruction template using the `_init_prompt` function provided by the base class, as well as the string used to capture results. Refer to  [prompt](/Best%20Practice/prompt) for further understanding of the design philosophy and usage of Prompts.

Both the Prompt template and the Instruction template use ``{}`` to indicate the fields to be filled in. The fields that can be included in the Prompt are `system`, `history`, `tools`, `user` etc., while the fields that can be included in the instruction_template are `instruction` and `extro_keys`. If the ``instruction`` field is a string, it is considered as a system instruction; if it is a dictionary, it can only contain the keys ``user`` and ``system``. ``user`` represents the user input instruction, which is placed before the user input in the prompt, and ``system`` represents the system instruction, which is placed after the system prompt in the prompt.
``instruction`` is passed in by the application developer, and the ``instruction`` can also contain ``{}`` to define fillable fields, making it convenient for users to input additional information.
''')

add_example('prompter.PrompterBase', '''\
>>> from lazyllm.components.prompter import PrompterBase
>>> class MyPrompter(PrompterBase):
...     def __init__(self, instruction = None, extro_keys = None, show = False):
...         super(__class__, self).__init__(show)
...         instruction_template = f'{instruction}\\\\n{{extro_keys}}\\\\n'.replace('{extro_keys}', PrompterBase._get_extro_key_template(extro_keys))
...         self._init_prompt("<system>{system}</system>\\\\n</instruction>{instruction}</instruction>{history}\\\\n{input}\\\\n, ## Response::", instruction_template, '## Response::')
... 
>>> p = MyPrompter('ins {instruction}')
>>> p.generate_prompt('hello')
'<system>You are an AI-Agent developed by LazyLLM.</system>\\\\n</instruction>ins hello\\\\n\\\\n</instruction>\\\\n\\\\n, ## Response::'
>>> p.generate_prompt('hello world', return_dict=True)
{'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\\\nins hello world\\\\n\\\\n'}, {'role': 'user', 'content': ''}]}
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

add_chinese_doc('AlpacaPrompter', '''\
Alpaca格式的Prompter，支持工具调用，不支持历史对话。

Args:
    instruction (Option[str]): 大模型的任务指令，至少带一个可填充的槽位(如 ``{instruction}``)。或者使用字典指定 ``system`` 和 ``user`` 的指令。
    extro_keys (Option[List]): 额外的字段，用户的输入会填充这些字段。
    show (bool): 标志是否打印生成的Prompt，默认为False
    tools (Option[list]): 大模型可以使用的工具集合，默认为None
''')

add_english_doc('AlpacaPrompter', '''\
Alpaca-style Prompter, supports tool calls, does not support historical dialogue.


Args:
    instruction (Option[str]): Task instructions for the large model, with at least one fillable slot (e.g. ``{instruction}``). Or use a dictionary to specify the ``system`` and ``user`` instructions.
    extro_keys (Option[List]): Additional fields that will be filled with user input.
    show (bool): Flag indicating whether to print the generated Prompt, default is False.
    tools (Option[list]): Tool-set which is provived for LLMs, default is None.

**Examples:**\n
```python
>>> from lazyllm import AlpacaPrompter
>>> p = AlpacaPrompter('hello world {instruction}')
>>> p.generate_prompt('this is my input')
'You are an AI-Agent developed by LazyLLM.\\\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\\\n\\\\n ### Instruction:\\\\nhello world this is my input\\\\n\\\\n\\\\n### Response:\\\\n'
>>> p.generate_prompt('this is my input', return_dict=True)
{'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\\\n\\\\n ### Instruction:\\\\nhello world this is my input\\\\n\\\\n'}, {'role': 'user', 'content': ''}]}
>>>
>>> p = AlpacaPrompter('hello world {instruction}, {input}', extro_keys=['knowledge'])
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
```

<span style="font-size: 20px;">**`generate_prompt(input: str | Dict[str, str] | None = None, history: List[List[str] | Dict[str, Any]] | None = None, tools: List[Dict[str, Any]] | None = None, label: str | None = None, *, show: bool = False, return_dict: bool = False)→ str | Dict `**</span>

Based on the user's input, generate the corresponding Prompt.

Args:
    instruction (Option[str]): Task instructionspecify the ``system`` and ``user`` instructions.\n
    extro_keys (Option[List]): Additional fields that will be filled with user input.\n
    show (bool): Flag indicating whether to print the generated Prompt, default is False.\n
    tools (Option[list]): Tool-set which is provived for LLMs, default is None.\n


<span style="font-size: 20px;">**`get_response(output: str, input: str | None = None)→ str`**</span>

Used to truncate the Prompt, only retaining valuable outputs.

Args:
    output (str) :The output of the large model
    input (Optional[str]) :The input to the large model. If this parameter is specified, the function will truncate any part of the output that contains the input. Default is None.                
''')

# add_example('AlpacaPrompter', '''\
# >>> from lazyllm import AlpacaPrompter
# >>> p = AlpacaPrompter('hello world {instruction}')
# >>> p.generate_prompt('this is my input')
# 'You are an AI-Agent developed by LazyLLM.\\\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\\\n\\\\n ### Instruction:\\\\nhello world this is my input\\\\n\\\\n\\\\n### Response:\\\\n'
# >>> p.generate_prompt('this is my input', return_dict=True)
# {'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\\\n\\\\n ### Instruction:\\\\nhello world this is my input\\\\n\\\\n'}, {'role': 'user', 'content': ''}]}
# >>>
# >>> p = AlpacaPrompter('hello world {instruction}, {input}', extro_keys=['knowledge'])
# >>> p.generate_prompt(dict(instruction='hello world', input='my input', knowledge='lazyllm'))
# 'You are an AI-Agent developed by LazyLLM.\\\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\\\n\\\\n ### Instruction:\\\\nhello world hello world, my input\\\\n\\\\nHere are some extra messages you can referred to:\\\\n\\\\n### knowledge:\\\\nlazyllm\\\\n\\\\n\\\\n### Response:\\\\n'
# >>> p.generate_prompt(dict(instruction='hello world', input='my input', knowledge='lazyllm'), return_dict=True)
# {'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\\\n\\\\n ### Instruction:\\\\nhello world hello world, my input\\\\n\\\\nHere are some extra messages you can referred to:\\\\n\\\\n### knowledge:\\\\nlazyllm\\\\n\\\\n'}, {'role': 'user', 'content': ''}]}
# >>>
# >>> p = AlpacaPrompter(dict(system="hello world", user="this is user instruction {input}"))
# >>> p.generate_prompt(dict(input="my input"))
# 'You are an AI-Agent developed by LazyLLM.\\\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\\\n\\\\n ### Instruction:\\\\nhello word\\\\n\\\\n\\\\n\\\\nthis is user instruction my input### Response:\\\\n'
# >>> p.generate_prompt(dict(input="my input"), return_dict=True)
# {'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\\\n\\\\n ### Instruction:\\\\nhello world'}, {'role': 'user', 'content': 'this is user instruction my input'}]}

# ''')

add_chinese_doc('ChatPrompter', '''\
多轮对话的Prompt，支持工具调用和历史对话

Args:
    instruction (Option[str]): 大模型的任务指令，可以带0到多个待填充的槽位,用 ``{}`` 表示。针对用户instruction可以通过字典传递，字段为 ``user`` 和 ``system`` 。
    extro_keys (Option[List]): 额外的字段，用户的输入会填充这些字段。
    show (bool): 标志是否打印生成的Prompt，默认为False
''')

add_english_doc('ChatPrompter', '''\
chat prompt, supports tool calls and historical dialogue.

Args:
    instruction (Option[str]): Task instructions for the large model, with 0 to multiple fillable slot, represented by ``{}``. For user instructions, you can pass a dictionary with fields ``user`` and ``system``.
    extro_keys (Option[List]): Additional fields that will be filled with user input.
    show (bool): Flag indicating whether to print the generated Prompt, default is False.

**Examples:**\n
```python
>>> from lazyllm import ChatPrompter
>>> p = ChatPrompter('hello world')
>>> p.generate_prompt('this is my input')
'<|start_system|>You are an AI-Agent developed by LazyLLM.hello world\\\\n\\\\n<|end_system|>\\\\n\\\\n\\\\n<|Human|>:\\\\nthis is my input\\\\n<|Assistant|>:\\\\n'
>>> p.generate_prompt('this is my input', return_dict=True)
{'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\\\nhello world\\\\n\\\\n'}, {'role': 'user', 'content': 'this is my input'}]}
>>>
>>> p = ChatPrompter('hello world {instruction}', extro_keys=['knowledge'])
>>> p.generate_prompt(dict(instruction='this is my ins', input='this is my inp', knowledge='LazyLLM-Knowledge'))
'<|start_system|>You are an AI-Agent developed by LazyLLM.hello world this is my ins\\\\nHere are some extra messages you can referred to:\\\\n\\\\n### knowledge:\\\\nLazyLLM-Knowledge\\\\n\\\\n\\\\n<|end_system|>\\\\n\\\\n\\\\n<|Human|>:\\\\nthis is my inp\\\\n<|Assistant|>:\\\\n'
>>> p.generate_prompt(dict(instruction='this is my ins', input='this is my inp', knowledge='LazyLLM-Knowledge'), return_dict=True)
{'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\\\nhello world this is my ins\\\\nHere are some extra messages you can referred to:\\\\n\\\\n### knowledge:\\\\nLazyLLM-Knowledge\\\\n\\\\n\\\\n'}, {'role': 'user', 'content': 'this is my inp'}]}
>>> p.generate_prompt(dict(instruction='this is my ins', input='this is my inp', knowledge='LazyLLM-Knowledge'), history=[['s1', 'e1'], ['s2', 'e2']])
'<|start_system|>You are an AI-Agent developed by LazyLLM.hello world this is my ins\\\\nHere are some extra messages you can referred to:\\\\n\\\\n### knowledge:\\\\nLazyLLM-Knowledge\\\\n\\\\n\\\\n<|end_system|>\\\\n\\\\n<|Human|>:s1<|Assistant|>:e1<|Human|>:s2<|Assistant|>:e2\\\\n<|Human|>:\\\\nthis is my inp\\\\n<|Assistant|>:\\\\n'
>>>
>>> p = ChatPrompter(dict(system="hello world", user="this is user instruction {input} "))
>>> p.generate_prompt(dict(input="my input", query="this is user query"))
'<|start_system|>You are an AI-Agent developed by LazyLLM.hello world\\\\n\\\\n<|end_system|>\\\\n\\\\n\\\\n<|Human|>:\\\\nthis is user instruction my input this is user query\\\\n<|Assistant|>:\\\\n'
>>> p.generate_prompt(dict(input="my input", query="this is user query"), return_dict=True)
{'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\\\nhello world\\\\n\\\\n'}, {'role': 'user', 'content': 'this is user instruction my input this is user query'}]}
```
<span style="font-size: 20px;">**`generate_prompt(input: str | Dict[str, str] | None = None, history: List[List[str] | Dict[str, Any]] | None = None, tools: List[Dict[str, Any]] | None = None, label: str | None = None, *, show: bool = False, return_dict: bool = False)→ str | Dict `**</span>

Based on the user's input, generate the corresponding Prompt.

Args:
    instruction (Option[str]): Task instructionspecify the ``system`` and ``user`` instructions.\n
    extro_keys (Option[List]): Additional fields that will be filled with user input.\n
    show (bool): Flag indicating whether to print the generated Prompt, default is False.\n
    tools (Option[list]): Tool-set which is provived for LLMs, default is None.\n


<span style="font-size: 20px;">**`get_response(output: str, input: str | None = None)→ str`**</span>

Used to truncate the Prompt, only retaining valuable outputs.

Args:
    output (str) :The output of the large model
    input (Optional[str]) :The input to the large model. If this parameter is specified, the function will truncate any part of the output that contains the input. Default is None.                

''')

# add_example('ChatPrompter', '''\
# >>> from lazyllm import ChatPrompter
# >>> p = ChatPrompter('hello world')
# >>> p.generate_prompt('this is my input')
# '<|start_system|>You are an AI-Agent developed by LazyLLM.hello world\\\\n\\\\n<|end_system|>\\\\n\\\\n\\\\n<|Human|>:\\\\nthis is my input\\\\n<|Assistant|>:\\\\n'
# >>> p.generate_prompt('this is my input', return_dict=True)
# {'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\\\nhello world\\\\n\\\\n'}, {'role': 'user', 'content': 'this is my input'}]}
# >>>
# >>> p = ChatPrompter('hello world {instruction}', extro_keys=['knowledge'])
# >>> p.generate_prompt(dict(instruction='this is my ins', input='this is my inp', knowledge='LazyLLM-Knowledge'))
# '<|start_system|>You are an AI-Agent developed by LazyLLM.hello world this is my ins\\\\nHere are some extra messages you can referred to:\\\\n\\\\n### knowledge:\\\\nLazyLLM-Knowledge\\\\n\\\\n\\\\n<|end_system|>\\\\n\\\\n\\\\n<|Human|>:\\\\nthis is my inp\\\\n<|Assistant|>:\\\\n'
# >>> p.generate_prompt(dict(instruction='this is my ins', input='this is my inp', knowledge='LazyLLM-Knowledge'), return_dict=True)
# {'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\\\nhello world this is my ins\\\\nHere are some extra messages you can referred to:\\\\n\\\\n### knowledge:\\\\nLazyLLM-Knowledge\\\\n\\\\n\\\\n'}, {'role': 'user', 'content': 'this is my inp'}]}
# >>> p.generate_prompt(dict(instruction='this is my ins', input='this is my inp', knowledge='LazyLLM-Knowledge'), history=[['s1', 'e1'], ['s2', 'e2']])
# '<|start_system|>You are an AI-Agent developed by LazyLLM.hello world this is my ins\\\\nHere are some extra messages you can referred to:\\\\n\\\\n### knowledge:\\\\nLazyLLM-Knowledge\\\\n\\\\n\\\\n<|end_system|>\\\\n\\\\n<|Human|>:s1<|Assistant|>:e1<|Human|>:s2<|Assistant|>:e2\\\\n<|Human|>:\\\\nthis is my inp\\\\n<|Assistant|>:\\\\n'
# >>>
# >>> p = ChatPrompter(dict(system="hello world", user="this is user instruction {input} "))
# >>> p.generate_prompt(dict(input="my input", query="this is user query"))
# '<|start_system|>You are an AI-Agent developed by LazyLLM.hello world\\\\n\\\\n<|end_system|>\\\\n\\\\n\\\\n<|Human|>:\\\\nthis is user instruction my input this is user query\\\\n<|Assistant|>:\\\\n'
# >>> p.generate_prompt(dict(input="my input", query="this is user query"), return_dict=True)
# {'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\\\nhello world\\\\n\\\\n'}, {'role': 'user', 'content': 'this is user instruction my input this is user query'}]}
# ''')

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
    - Return of infer: A `str` that is the serialized form of a dictionary, with the keyword “images_base64”, corresponding to a list whose elements are images encoded in base64.
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
    - 推理的返回值：一个字典被序列化后的字符串， 关键字是"images_base64", 对应值为一个列表，其中的元素是被base64编码的图像。
    - 支持的模型为：[stable-diffusion-3-medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium)
''')

add_example('StableDiffusionDeploy', ['''\
>>> from lazyllm import launchers, UrlModule
>>> from lazyllm.components import StableDiffusionDeploy
>>> deployer = StableDiffusionDeploy(launchers.remote())
>>> url = deployer(base_model='stable-diffusion-3-medium')
>>> model = UrlModule(url=url)
>>> res = model('a tiny cat.')
>>> print(len(res), res[:50])
... 1384335 {"images_base64": ["iVBORw0KGgoAAAANSUhEUgAABAAAAA
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
    - Return of infer: A `str` that is the serialized form of a dictionary, with the keyword “sounds”, corresponding to a list where the first element is the sampling rate, and the second element is a list of audio data.
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
    - 推理的返回值：一个字典被序列化后的字符串， 关键字是"sounds", 对应值为一个列表，其中第一个元素是采样率，第二个元素是一个音频数据对应的列表。
    - 支持的模型为：[ChatTTS](https://huggingface.co/2Noise/ChatTTS)
''')

add_example('ChatTTSDeploy', ['''\
>>> from lazyllm import launchers, UrlModule
>>> from lazyllm.components import ChatTTSDeploy
>>> deployer = ChatTTSDeploy(launchers.remote())
>>> url = deployer(base_model='ChatTTS')
>>> model = UrlModule(url=url)
>>> res = model('Hello World!')
>>> print(len(res), res[:50])
... {"sounds": [24000, [-0.006741484627127647, 0.00795
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
    - Return of infer: A `str` that is the serialized form of a dictionary, with the keyword “sounds”, corresponding to a list where the first element is the sampling rate, and the second element is a list of audio data.
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
    - 推理的返回值：一个字典被序列化后的字符串， 关键字是"sounds", 对应值为一个列表，其中第一个元素是采样率，第二个元素是一个音频数据对应的列表。
    - 支持的模型为：[bark](https://huggingface.co/suno/bark)
''')

add_example('BarkDeploy', ['''\
>>> from lazyllm import launchers, UrlModule
>>> from lazyllm.components import BarkDeploy
>>> deployer = BarkDeploy(launchers.remote())
>>> url = deployer(base_model='bark')
>>> model = UrlModule(url=url)
>>> res = model('Hello World!')
>>> print(len(res), res[:50])
... 373843 {"sounds": [24000, [-66.9375, -62.0625, -60.5, -60
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
    - Return of infer: A `str` that is the serialized form of a dictionary, with the keyword “sounds”, corresponding to a list where the first element is the sampling rate, and the second element is a list of audio data.
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
    - 推理的返回值：一个字典被序列化后的字符串， 关键字是"sounds", 对应值为一个列表，其中第一个元素是采样率，第二个元素是一个音频数据对应的列表。
    - 支持的模型为：[musicgen-small](https://huggingface.co/facebook/musicgen-small)
''')

add_example('MusicGenDeploy', ['''\
>>> from lazyllm import launchers, UrlModule
>>> from lazyllm.components import MusicGenDeploy
>>> deployer = MusicGenDeploy(launchers.remote())
>>> url = deployer(base_model='musicgen-small')
>>> model = UrlModule(url=url)
>>> model('Symphony with flute as the main melody')
4981065 {"sounds": [32000, [-931, -1206, -1170, -1078, -10, ...
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
>>> print(len(res), res[:50])
... 387355 {"sounds": [24000, [-111.9375, -106.4375, -106.875
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
