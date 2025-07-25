import os
import copy
import random

import lazyllm
from lazyllm import launchers, ArgsDict, thirdparty
from .base import LazyLLMFinetuneBase


class AlpacaloraFinetune(LazyLLMFinetuneBase):
    """此类是 ``LazyLLMFinetuneBase`` 的子类，基于 [alpaca-lora](https://github.com/tloen/alpaca-lora) 项目提供的LoRA微调能力，用于对大语言模型进行LoRA微调。

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



Examples:
    >>> from lazyllm import finetune
    >>> trainer = finetune.alpacalora('path/to/base/model', 'path/to/target')
    """
    defatult_kw = ArgsDict({
        'data_path': None,
        'batch_size': 64,
        'micro_batch_size': 4,
        'num_epochs': 2,
        'learning_rate': 5.e-4,
        'cutoff_len': 1030,
        'filter_nums': 1024,
        'val_set_size': 200,
        'lora_r': 8,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'lora_target_modules': '[wo,wqkv]',
        'modules_to_save': '[tok_embeddings,output]',
        'deepspeed': '',
        'prompt_template_name': 'alpaca',
        'train_on_inputs': True,
        'show_prompt': False,
        'nccl_port': 19081,
    })
    auto_map = {'micro_batch_size': 'micro_batch_size'}

    def __init__(self,
                 base_model,
                 target_path,
                 merge_path=None,
                 model_name='LLM',
                 cp_files='tokeniz*',
                 launcher=launchers.remote(ngpus=1),
                 **kw
                 ):
        if not merge_path:
            save_path = os.path.join(lazyllm.config['train_target_root'], target_path)
            target_path, merge_path = os.path.join(save_path, "lazyllm_lora"), os.path.join(save_path, "lazyllm_merge")
            os.makedirs(target_path, exist_ok=True)
            os.makedirs(merge_path, exist_ok=True)
        super().__init__(
            base_model,
            target_path,
            launcher=launcher,
        )
        self.folder_path = os.path.dirname(os.path.abspath(__file__))
        deepspeed_config_path = os.path.join(self.folder_path, 'alpaca-lora', 'ds.json')
        self.kw = copy.deepcopy(self.defatult_kw)
        self.kw['deepspeed'] = deepspeed_config_path
        self.kw['nccl_port'] = random.randint(19000, 20500)
        self.kw.check_and_update(kw)
        self.merge_path = merge_path
        self.cp_files = cp_files
        self.model_name = model_name

    def cmd(self, trainset, valset=None) -> str:
        thirdparty.check_packages(['datasets', 'deepspeed', 'fire', 'numpy', 'peft', 'torch', 'transformers'])
        if not os.path.exists(trainset):
            defatult_path = os.path.join(lazyllm.config['data_path'], trainset)
            if os.path.exists(defatult_path):
                trainset = defatult_path
        if not self.kw['data_path']:
            self.kw['data_path'] = trainset

        run_file_path = os.path.join(self.folder_path, 'alpaca-lora', 'finetune.py')
        cmd = (f'python {run_file_path} '
               f'--base_model={self.base_model} '
               f'--output_dir={self.target_path} '
            )
        cmd += self.kw.parse_kwargs()
        cmd += f' 2>&1 | tee {os.path.join(self.target_path, self.model_name)}_$(date +"%Y-%m-%d_%H-%M-%S").log'

        if self.merge_path:
            run_file_path = os.path.join(self.folder_path, 'alpaca-lora', 'utils', 'merge_weights.py')

            cmd = [cmd,
                   f'python {run_file_path} '
                   f'--base={self.base_model} '
                   f'--adapter={self.target_path} '
                   f'--save_path={self.merge_path} ',
                   f' cp {os.path.join(self.base_model, self.cp_files)} {self.merge_path} '
                ]

        # cmd = 'realpath .'
        return cmd
