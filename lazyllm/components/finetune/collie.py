from .base import LazyLLMFinetuneBase
from lazyllm import launchers, ArgsDict, thirdparty
import os
import copy


class CollieFinetune(LazyLLMFinetuneBase):
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
            target_path, merge_path = os.path.join(target_path, "lora"), os.path.join(target_path, "merge")
            os.system(f'mkdir -p {target_path} {merge_path}')
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
        if not self.kw['data_path']:
            self.kw['data_path'] = trainset

        run_file_path = os.path.join(self.folder_path, 'collie/finetune.py')
        cmd = (f'python {run_file_path} '
               f'--base_model={self.base_model} '
               f'--output_dir={self.target_path} '
            )
        cmd += self.kw.parse_kwargs()
        cmd += f' 2>&1 | tee {self.target_path}/{self.model_name}_$(date +"%Y-%m-%d_%H-%M-%S").log'

        if self.merge_path:
            run_file_path = os.path.join(self.folder_path, 'alpaca-lora/utils/merge_weights.py')

            cmd = [cmd,
                   f'python {run_file_path} '
                   f'--base={self.base_model} '
                   f'--adapter={self.target_path} '
                   f'--save_path={self.merge_path} ',
                   f' cp {self.base_model}/{self.cp_files} {self.merge_path} '
                ]

        return cmd
