import os
import copy

import lazyllm
from lazyllm import launchers, ArgsDict, thirdparty
from .base import LazyLLMFinetuneBase


class CollieFinetune(LazyLLMFinetuneBase):
    """This class is a subclass of ``LazyLLMFinetuneBase``, based on the LoRA fine-tuning capabilities provided by the [Collie](https://github.com/OpenLMLab/collie) framework, used for LoRA fine-tuning of large language models.

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



Examples:
    >>> from lazyllm import finetune
    >>> trainer = finetune.collie('path/to/base/model', 'path/to/target')
    """
    defatult_kw = ArgsDict({
        'data_path': None,
        'batch_size': 64,
        'micro_batch_size': 4,
        'num_epochs': 3,
        'learning_rate': 5.e-4,
        'dp_size': 8,
        'pp_size': 1,
        'tp_size': 1,
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.05,
        'lora_target_modules': '[wo,wqkv]',
        'modules_to_save': '[tok_embeddings,output]',
        'prompt_template_name': 'alpaca',
    })
    auto_map = {
        'ddp': 'dp_size',
        'micro_batch_size': 'micro_batch_size',
        'tp': 'tp_size',
    }

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
        self.kw = copy.deepcopy(self.defatult_kw)
        self.kw.check_and_update(kw)
        self.merge_path = merge_path
        self.cp_files = cp_files
        self.model_name = model_name

    def cmd(self, trainset, valset=None) -> str:
        thirdparty.check_packages(['numpy', 'peft', 'torch', 'transformers'])
        if not os.path.exists(trainset):
            defatult_path = os.path.join(lazyllm.config['data_path'], trainset)
            if os.path.exists(defatult_path):
                trainset = defatult_path
        if not self.kw['data_path']:
            self.kw['data_path'] = trainset

        run_file_path = os.path.join(self.folder_path, 'collie', 'finetune.py')
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
                   f' cp {os.path.join(self.base_model,self.cp_files)} {self.merge_path} '
                ]

        return cmd
