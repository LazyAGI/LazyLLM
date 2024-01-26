from .base import LazyLLMFinetuneBase
from lazyllm import launchers
import os


class AlpacaloraFinetune(LazyLLMFinetuneBase):
    def __init__(self,
                 base_model,
                 target_path,
                 merge_path=None,
                 model_name='LLM',
                 cp_files='tokeniz*',
                 launcher=launchers.slurm(),
                 **kw
                 ):
        super().__init__(
            base_model,
            target_path,
            launcher=launcher,
        )
        self.updata_defaults_dict(kw)
        self.merge_path = merge_path
        self.cp_files = cp_files
        self.model_name = model_name

    def updata_defaults_dict(self, kw):
        self.defaults_dict={
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
            'lora_target_modules': '[query_key_value,dense,dense_4h_to_h,dense_h_to_4h]',
            'modules_to_save': '[word_embeddings, output_layer]',
            'deepspeed': 'ds.json',
            'prompt_with_background': True,
            'train_on_inputs': True,
        }
        assert set(kw.keys()).issubset(set(self.defaults_dict.keys()))
        self.kw = self.defaults_dict.copy()
        self.kw.update(kw)

    def cmd(self, trainset, valset=None) -> str:
        if not self.kw['data_path']:
            self.kw['data_path']=trainset

        folder_path = os.path.dirname(os.path.abspath(__file__))
        run_file_path = os.path.join(folder_path, 'alpaca-lora/finetune.py')
        cmd = (
                f'python {run_file_path} '
                f'--base_model={self.base_model} '
                f'--output_dir={self.target_path} '
            )
        cmd += ' '.join(f'--{k}={v}' if type(v) is not str else f'--{k}=\"{v}\"' for k, v in self.kw.items())
        cmd += f' 2>&1 | tee {self.target_path}/{self.model_name}_$(date +"%Y-%m-%d_%H-%M-%S").log'

        if self.merge_path:
            run_file_path = os.path.join(folder_path, 'alpaca-lora/utils/merge_weights.py')

            cmd = [ cmd,
                    f'python {run_file_path} '
                    f'--base={self.base_model} '
                    f'--adapter={self.target_path} '
                    f'--save_path={self.merge_path} ',
                    f' cp {self.base_model}/{self.cp_files} {self.merge_path} '
                ]

        # cmd = 'realpath .'
        return cmd

