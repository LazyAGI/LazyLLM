# flake8: noqa E501
from . import utils
import functools
import lazyllm


add_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm.components)
add_english_doc = functools.partial(utils.add_english_doc, module=lazyllm.components)
add_example = functools.partial(utils.add_example, module=lazyllm.components)

# Prompter
add_chinese_doc('Prompter', '''\
这是Prompter的文档
''')
add_english_doc('Prompter', '''\
This is doc for Prompter
''')
add_example('Prompter', '''\
def test_prompter():
    pass
''')

add_english_doc('Prompter.is_empty', '''\
This is doc for Prompter.is_empty
''')

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
此类是 ``LazyLLMFinetuneBase`` 的子类，基于 `alpaca-lora <https://github.com/tloen/alpaca-lora>`_ 项目提供的LoRA微调能力，用于对大语言模型进行LoRA微调。

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
This class is a subclass of ``LazyLLMFinetuneBase``, based on the LoRA fine-tuning capabilities provided by the `alpaca-lora <https://github.com/tloen/alpaca-lora>`_ project, used for LoRA fine-tuning of large language models.

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
此类是 ``LazyLLMFinetuneBase`` 的子类，基于 `Collie <https://github.com/OpenLMLab/collie>`_ 框架提供的LoRA微调能力，用于对大语言模型进行LoRA微调。

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
This class is a subclass of ``LazyLLMFinetuneBase``, based on the LoRA fine-tuning capabilities provided by the `Collie <https://github.com/OpenLMLab/collie>`_ framework, used for LoRA fine-tuning of large language models.

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
>>> finetune.auto("Llama-7b", 'path/to/target')
<lazyllm.llm.finetune type=CollieFinetune>
''')

# ============= Deploy
# Deploy-Lightllm
add_chinese_doc('deploy.Lightllm', '''\
此类是 ``LazyLLMDeployBase`` 的子类，基于 `LightLLM <https://github.com/ModelTC/lightllm>`_ 框架提供的推理能力，用于对大语言模型进行推理。

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
    host (int): 服务的IP地址，默认为 ``0.0.0.0``。
    nccl_port (int): NCCL 端口，默认为 ``None``。此情况下LazyLLM会自动生成随机端口号。
    tokenizer_mode (str): tokenizer的加载模式，默认为 ``auto``。
    running_max_req_size (int): 推理引擎最大的并行请求数， 默认为 ``256``。

''')

add_english_doc('deploy.Lightllm', '''\
This class is a subclass of ``LazyLLMDeployBase``, based on the inference capabilities provided by the `LightLLM <https://github.com/ModelTC/lightllm>`_ framework, used for inference with large language models.

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
此类是 ``LazyLLMDeployBase`` 的子类，基于 `VLLM <https://github.com/vllm-project/vllm>`_ 框架提供的推理能力，用于对大语言模型进行推理。

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
    host (int): 服务的IP地址，默认为 ``0.0.0.0``。
    seed (int): 随机数种子，默认为 ``0``。
    tokenizer_mode (str): tokenizer的加载模式，默认为 ``auto``。
    max-num-seqs (int): 推理引擎最大的并行请求数， 默认为 ``256``。

''')

add_english_doc('deploy.Vllm', '''\
This class is a subclass of ``LazyLLMDeployBase``, based on the inference capabilities provided by the `VLLM <https://github.com/vllm-project/vllm>`_ framework, used for inference with large language models.

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
>>> deploy.auto('Llama-7b')
<lazyllm.llm.deploy type=Vllm>    
''')

add_chinese_doc('ModelDownloader', '''\
ModelDownloader是LazyLLM为开发者提供的自动下载模型的工具类。目前支持从一个本地目录列表查找指定模型，以及从huggingface或者modelscope自动下载模型数据至指定目录。
在使用ModelDownloader之前，需要设置下列环境变量：

    - LAZYLLM_MODEL_SOURCE: 模型下载源，可以设置为 ``huggingface`` 或 ``modelscope`` 。
    - LAZYLLM_MODEL_SOURCE_TOKEN: ``huggingface`` 或 ``modelscope`` 提供的token，用于下载私有模型。
    - LAZYLLM_MODEL_PATH: 冒号 ``:`` 分隔的本地绝对路径列表用于搜索模型。
    - LAZYLLM_MODEL_CACHE_DIR: 下载后的模型在本地的存储目录
    
Keyword Args: 
    model_source (str, 可选): 模型下载源，目前仅支持 ``huggingface`` 或 ``modelscope`` 。如有必要，ModelDownloader将从此下载源下载模型数据。如果不提供，默认使用
        LAZYLLM_MODEL_SOURCE环境变量中的设置。如未设置LAZYLLM_MODEL_SOURCE，ModelDownloader将从 ``modelscope`` 下载模型。
    token (str, 可选): ``huggingface`` 或 ``modelscope`` 提供的token。如果token不为空，ModelDownloader将使用此token下载模型数据。如果不提供，默认使用
        LAZYLLM_MODEL_SOURCE_TOKEN环境变量中的设置。如未设置LAZYLLM_MODEL_SOURCE_TOKEN，ModelDownloader将不会自动下载私有模型。
    model_path (str, 可选)：冒号(:)分隔的本地绝对路径列表。在实际下载模型数据之前，ModelDownloader将在此列表包含的目录中尝试寻找目标模型。如果不提供，默认使用
        LAZYLLM_MODEL_PATH环境变量中的设置。如果为空或LAZYLLM_MODEL_PATH未设置，ModelDownloader将跳过从model_path中寻找模型的步骤。
    cache_dir (str, 可选): 一个本地目录的绝对路径。下载后的模型将存放在此目录下，如果不提供，默认使用LAZYLLM_MODEL_CACHE_DIR环境变量中的设置。如果
        LAZYLLM_MODEL_PATH未设置，默认值为~/.lazyllm/model
        
.. function:: ModelDownloader.download(model) -> str

用于从model_source下载模型。download函数首先在ModelDownloader类初始化参数model_path列出的目录中搜索目标模型。如果未找到，会在cache_dir下搜索目标模型。如果仍未找到，
则从model_source上下载模型并存放于cache_dir下。

Args:
    model (str): 目标模型名称。download函数使用此名称从model_source上下载模型。为了方便开发者使用，LazyLLM为常用模型建立了简略模型名称到下载源实际模型名称的映射，
        例如 ``Llama-3-8B`` , ``GLM3-6B`` 或 ``Qwen1.5-7B`` 。具体可参考文件 ``lazyllm/module/utils/downloader/model_mapping.py`` 。model可以接受简略模型名或下载源中的模型全名。
''')

add_english_doc('ModelDownloader', '''\
ModelDownloader is a utility class provided by LazyLLM for developers to automatically download models.
Currently, it supports search for models from local directories, as well as automatically downloading model from
huggingface or modelscope. Before using ModelDownloader, the following environment variables need to be set:

    - LAZYLLM_MODEL_SOURCE: The source for model downloads, which can be set to ``huggingface`` or ``modelscope`` .
    - LAZYLLM_MODEL_SOURCE_TOKEN: The token provided by ``huggingface`` or ``modelscope`` for private model download.
    - LAZYLLM_MODEL_PATH: A colon-separated ``:`` list of local absolute paths for model search.
    - LAZYLLM_MODEL_CACHE_DIR: Directory for downloaded models.

Keyword Args: 
    model_source (str, optional): The source for model downloads, currently only supports ``huggingface`` or ``modelscope`` .
        If necessary, ModelDownloader downloads model data from the source. If not provided, LAZYLLM_MODEL_SOURCE
        environment variable would be used, and if LAZYLLM_MODEL_SOURCE is not set, ModelDownloader will not download
        any model.
    token (str, optional): The token provided by ``huggingface`` or ``modelscope`` . If the token is present, ModelDownloader uses
        the token to download model. If not provided, LAZYLLM_MODEL_SOURCE_TOKEN environment variable would be used.
        and if LAZYLLM_MODEL_SOURCE_TOKEN is not set, ModelDownloader will not download private models, only public ones.
    model_path (str, optional): A colon-separated list of absolute paths. Before actually start to download model,
        ModelDownloader trys to find the target model in the directories in this list. If not provided,
        LAZYLLM_MODEL_PATH environment variable would be used, and LAZYLLM_MODEL_PATH is not set, ModelDownloader skips
        looking for models from model_path.
    cache_dir (str, optional): An absolute path of a directory to save downloaded models. If not provided,
        LAZYLLM_MODEL_CACHE_DIR environment variable would be used, and if LAZYLLM_MODEL_PATH is not set, the default
        value is ~/.lazyllm/model.
        
.. function:: ModelDownloader.download(model) -> str

Download models from model_source. The function first searches for the target model in directories listed in the
model_path parameter of ModelDownloader class. If not found, it searches under cache_dir. If still not found,
it downloads the model from model_source and stores it under cache_dir.

Args:
    model (str): The name of the target model. The function uses this name to download the model from model_source.
    To further simplify use of the function, LazyLLM provides a mapping dict from abbreviated model names to original
    names on the download source for popular models, such as ``Llama-3-8B`` , ``GLM3-6B`` or ``Qwen1.5-7B``. For more details,
    please refer to the file ``lazyllm/module/utils/downloader/model_mapping.py`` . The model argument can be either
    an abbreviated name or one from the download source.
''')

add_example('ModelDownloader', '''\
>>> from lazyllm.components import ModelDownloader
>>> downloader = ModelDownloader(model_source='modelscope')
>>> downloader.download('GLM3-6B')
''')


# ============= Prompter

add_chinese_doc('prompter.PrompterBase', '''\
Prompter的基类，自定义的Prompter需要继承此基类，并通过基类提供的 ``_init_prompt`` 函数来设置Prompt模板和Instruction的模板，以及截取结果所使用的字符串。可以查看 :doc:`/best_practice/prompt` 进一步了解Prompt的设计思想和使用方式。

Prompt模板和Instruction模板都用 ``{}`` 表示要填充的字段，其中Prompt可包含的字段有 ``system``, ``history``, ``tools``等，而instruction_template可包含的字段为 ``instruction`` 和 ``extro_keys`` 。
``instruction`` 由应用的开发者传入， ``instruction`` 中也可以带有 ``{}`` 用于让定义可填充的字段，方便用户填入额外的信息。
''')

add_english_doc('prompter.PrompterBase', '''\
The base class of Prompter. A custom Prompter needs to inherit from this base class and set the Prompt template and the Instruction template using the `_init_prompt` function provided by the base class, as well as the string used to capture results. Refer to :doc:`/best_practice/prompt.rst` for further understanding of the design philosophy and usage of Prompts.

Both the Prompt template and the Instruction template use ``{}`` to indicate the fields to be filled in. The fields that can be included in the Prompt are `system`, `history`, `tools`, etc., while the fields that can be included in the instruction_template are `instruction` and `extro_keys`.
``instruction`` is passed in by the application developer, and the ``instruction`` can also contain ``{}`` to define fillable fields, making it convenient for users to input additional information.
''')

add_example('prompter.PrompterBase', '''\
>>> from lazyllm.components.prompter import PrompterBase
>>> class MyPrompter(PrompterBase):
...     def __init__(self, instruction = None, extro_keys = None, show = False):
...         super(__class__, self).__init__(show)
...         instruction_template = f'{instruction}\\n{{extro_keys}}\\n'.replace('{extro_keys}', PrompterBase._get_extro_key_template(extro_keys))
...         self._init_prompt("<system>{system}</system>\\n</instruction>{instruction}</instruction>{history}\\n{input}\\n, ## Response::", instruction_template, '## Response::')
... 
>>> p = MyPrompter('ins {instruction}')
>>> p.generate_prompt('hello')
'<system>You are an AI-Agent developed by LazyLLM.</system>\\n</instruction>ins hello\\n\\n</instruction>\\n\\n, ## Response::'
>>> p.generate_prompt('hello world', return_dict=True)
{'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\nins hello world\\n\\n'}, {'role': 'user', 'content': ''}]}
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
    instruction (Option[str]): 大模型的任务指令，至少带一个可填充的槽位(如 ``{instruction}``)。
    extro_keys (Option[List]): 额外的字段，用户的输入会填充这些字段。
    show (bool): 标志是否打印生成的Prompt，默认为False
    tools (Option[list]): 大模型可以使用的工具集合，默认为None
''')

add_english_doc('AlpacaPrompter', '''\
Alpaca-style Prompter, supports tool calls, does not support historical dialogue.

Sure! Here is the translation, keeping the original format:

Args:
    instruction (Option[str]): Task instructions for the large model, with at least one fillable slot (e.g. ``{instruction}``).
    extro_keys (Option[List]): Additional fields that will be filled with user input.
    show (bool): Flag indicating whether to print the generated Prompt, default is False.
    tools (Option[list]): Tool-set which is provived for LLMs, default is None.
''')

add_example('AlpacaPrompter', '''\
>>> from lazyllm import AlpacaPrompter
>>> p = AlpacaPrompter('hello world {instruction}')
>>> p.generate_prompt('this is my input')
'You are an AI-Agent developed by LazyLLM.\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\n\\n ### Instruction:\\nhello world this is my input\\n\\n\\n### Response:\\n'
>>> p.generate_prompt('this is my input', return_dict=True)
{'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\n\\n ### Instruction:\\nhello world this is my input\\n\\n'}, {'role': 'user', 'content': ''}]}
>>>
>>> p = AlpacaPrompter('hello world {instruction}, {input}', extro_keys=['knowledge'])
>>> p.generate_prompt(dict(instruction='hello world', input='my input', knowledge='lazyllm'))
'You are an AI-Agent developed by LazyLLM.\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\n\\n ### Instruction:\\nhello world hello world, my input\\n\\nHere are some extra messages you can referred to:\\n\\n### knowledge:\\nlazyllm\\n\\n\\n### Response:\\n'
>>> p.generate_prompt(dict(instruction='hello world', input='my input', knowledge='lazyllm'), return_dict=True)
{'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\n\\n ### Instruction:\\nhello world hello world, my input\\n\\nHere are some extra messages you can referred to:\\n\\n### knowledge:\\nlazyllm\\n\\n'}, {'role': 'user', 'content': ''}]}
''')

add_chinese_doc('ChatPrompter', '''\
多轮对话的Prompt，支持工具调用和历史对话

Args:
    instruction (Option[str]): 大模型的任务指令，可以带0到多个待填充的槽位,用 ``{}`` 表示。
    extro_keys (Option[List]): 额外的字段，用户的输入会填充这些字段。
    show (bool): 标志是否打印生成的Prompt，默认为False
''')

add_english_doc('ChatPrompter', '''\
chat prompt, supports tool calls and historical dialogue.

Args:
    instruction (Option[str]): Task instructions for the large model, with 0 to multiple fillable slot, represented by ``{}``.
    extro_keys (Option[List]): Additional fields that will be filled with user input.
    show (bool): Flag indicating whether to print the generated Prompt, default is False.
''')

add_example('ChatPrompter', '''\
>>> from lazyllm import ChatPrompter
>>> p = ChatPrompter('hello world')
>>> p.generate_prompt('this is my input')
'<|start_system|>You are an AI-Agent developed by LazyLLM.hello world\\n\\n<|end_system|>\\n\\n\\n<|Human|>:\\nthis is my input\\n<|Assistant|>:\\n'
>>> p.generate_prompt('this is my input', return_dict=True)
{'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\nhello world\\n\\n'}, {'role': 'user', 'content': 'this is my input'}]}
>>>
>>> p = ChatPrompter('hello world {instruction}', extro_keys=['knowledge'])
>>> p.generate_prompt(dict(instruction='this is my ins', input='this is my inp', knowledge='LazyLLM-Knowledge'))
'<|start_system|>You are an AI-Agent developed by LazyLLM.hello world this is my ins\\nHere are some extra messages you can referred to:\\n\\n### knowledge:\\nLazyLLM-Knowledge\\n\\n\\n<|end_system|>\\n\\n\\n<|Human|>:\\nthis is my inp\\n<|Assistant|>:\\n'
>>> p.generate_prompt(dict(instruction='this is my ins', input='this is my inp', knowledge='LazyLLM-Knowledge'), return_dict=True)
{'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\nhello world this is my ins\\nHere are some extra messages you can referred to:\\n\\n### knowledge:\\nLazyLLM-Knowledge\\n\\n\\n'}, {'role': 'user', 'content': 'this is my inp'}]}
>>> p.generate_prompt(dict(instruction='this is my ins', input='this is my inp', knowledge='LazyLLM-Knowledge'), history=[['s1', 'e1'], ['s2', 'e2']])
'<|start_system|>You are an AI-Agent developed by LazyLLM.hello world this is my ins\\nHere are some extra messages you can referred to:\\n\\n### knowledge:\\nLazyLLM-Knowledge\\n\\n\\n<|end_system|>\\n\\n<|Human|>:s1<|Assistant|>:e1<|Human|>:s2<|Assistant|>:e2\\n<|Human|>:\\nthis is my inp\\n<|Assistant|>:\\n'
''')

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
