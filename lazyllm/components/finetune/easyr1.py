import os
import copy
import json
import random
import subprocess
from datetime import datetime
from subprocess import CalledProcessError

import lazyllm
from lazyllm import launchers, ArgsDict, thirdparty, LOG
from .base import LazyLLMFinetuneBase


class EasyR1Finetune(LazyLLMFinetuneBase):
    defatult_kw = ArgsDict({
        'data.max_prompt_length': 2048,
        'data.max_response_length': 2048,
        'data.rollout_batch_size': 128,
        'data.val_batch_size': 1024,
        'data.format_prompt': None,
        'worker.actor.global_batch_size': 128,
        'worker.actor.micro_batch_size_per_device_for_update': 4,
        'worker.actor.micro_batch_size_per_device_for_experience': 16,
        'worker.rollout.gpu_memory_utilization': 0.6,
        'worker.rollout.tensor_parallel_size': 1,
        'worker.reward.reward_function': None,
        'trainer.total_epochs': 2,
        'trainer.n_gpus_per_node': 1,
        'trainer.save_freq': 5,
        'trainer.save_checkpoint_path': None,
        'trainer.save_model_only': False,
    }, with_line=False)

    def __init__(self,
                 base_model,
                 target_path,
                 merge_path=None,
                 launcher=None,
                 **kw
                 ):
        if not merge_path:
            merge_path = target_path
        os.makedirs(target_path, exist_ok=True)
        os.makedirs(merge_path, exist_ok=True)
        if launcher is None:
            launcher = launchers.remote(ngpus=1, sync=True)
        super().__init__(
            base_model,
            target_path,
            launcher=launcher,
        )
        self._folder_path = os.path.dirname(os.path.abspath(__file__))
        self.kw = copy.deepcopy(self.defatult_kw)
        self.kw.check_and_update(kw)

    def cmd(self, trainset, valset=None) -> str:
        thirdparty.check_packages(['verl', 'trl'])
        if not os.path.exists(trainset):
            defatult_path = os.path.join(lazyllm.config['data_path'], trainset)
            if os.path.exists(defatult_path):
                trainset = defatult_path
            else:
                raise FileNotFoundError(f'Trainset {trainset} does not exist, please check your path.')
        if not os.path.exists(valset):
            defatult_path = os.path.join(lazyllm.config['data_path'], valset)
            if os.path.exists(defatult_path):
                valset = defatult_path
            else:
                raise FileNotFoundError(f'Valset {valset} does not exist, please check your path.')

        formatted_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        random_value = random.randint(1000, 9999)
        self.log_file_path = f'{self.target_path}/train_log_{formatted_date}_{random_value}.log'

        self.kw['data.train_files'] = trainset
        self.kw['data.val_files'] = valset
        self.kw['worker.actor.model.model_path'] = self.base_model
        self.kw['trainer.n_gpus_per_node'] = self.launcher.ngpus
        if not self.kw['trainer.save_checkpoint_path']:
            self.kw['trainer.save_checkpoint_path'] = self.target_path
        if not self.kw['worker.reward.reward_function']:
            self.kw['worker.reward.reward_function'] = (f'{self._folder_path}/easy_r1/'
                                                        'reward_function/math.py:compute_score')
        if not self.kw['data.format_prompt']:
            self.kw['data.format_prompt'] = f'{self._folder_path}/easy_r1/format_prompt/math.jinja'

        cmd = f'python -m verl.trainer.main config={self._folder_path}/easy_r1/config.yaml '
        cmd += self.kw.parse_kwargs()
        cmd += f' 2>&1 | tee {self.log_file_path}'

        return cmd

    def __call__(self, *args, **kw):
        save_path = super().__call__(*args, **kw)
        ckpt_tracker_file = os.path.join(save_path, 'checkpoint_tracker.json')
        if not os.path.exists(ckpt_tracker_file):
            not_found_msg = 'Training failed, checkpoint_tracker.json not found.'
            LOG.error(not_found_msg)
            return not_found_msg
        with open(ckpt_tracker_file, 'r') as f:
            json_data = json.load(f)
        actor_path = json_data.get('last_actor_path', None)
        if not actor_path or not os.path.exists(actor_path):
            not_found_msg = 'Training failed, last_actor_path not found in checkpoint_tracker.json.'
            LOG.error(not_found_msg)
            return not_found_msg

        self._merge_ckpt(actor_path)
        huggingface_path = os.path.join(actor_path, 'huggingface')
        return huggingface_path

    def _merge_ckpt(self, path):
        try:
            script_path = f'{self._folder_path}/easy_r1/model_merger.py'
            subprocess.run(
                ['python', script_path, '--local_dir', path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except CalledProcessError as e:
            LOG.error(f'Command failed with return code {e.returncode}: {e.stderr}')
            return False
        except Exception as e:
            LOG.error(f'Error merging checkpoints: {e}')
            return False
        return True
