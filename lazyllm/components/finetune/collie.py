import os
import copy

import lazyllm
from lazyllm import launchers, ArgsDict, thirdparty
from .base import LazyLLMFinetuneBase


class CollieFinetune(LazyLLMFinetuneBase):
    """This class is a subclass of ``LazyLLMFinetuneBase``, based on the LoRA fine-tuning capabilities provided by the [Collie](https://github.com/OpenLMLab/collie) framework, used for LoRA fine-tuning of large language models.

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
                 launcher=launchers.remote(ngpus=1),  # noqa B008
                 **kw
                 ):
        if not merge_path:
            save_path = os.path.join(lazyllm.config['train_target_root'], target_path)
            target_path, merge_path = os.path.join(save_path, 'lazyllm_lora'), os.path.join(save_path, 'lazyllm_merge')
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
                   f' cp {os.path.join(self.base_model, self.cp_files)} {self.merge_path} '
                ]

        return cmd
