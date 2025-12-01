import os
import yaml
import json
import uuid
import tempfile
import random
from datetime import datetime

import lazyllm
from lazyllm import launchers, ArgsDict, thirdparty
from .base import LazyLLMFinetuneBase
from .llama_factory.model_mapping import match_longest_prefix, llamafactory_mapping_dict


class LlamafactoryFinetune(LazyLLMFinetuneBase):
    auto_map = {
        'gradient_step': 'gradient_accumulation_steps',
        'micro_batch_size': 'per_device_train_batch_size',
    }

    def __init__(self,
                 base_model,
                 target_path,
                 merge_path=None,
                 config_path=None,
                 export_config_path=None,
                 lora_r=None,
                 modules_to_save=None,
                 lora_target_modules=None,
                 launcher=launchers.remote(ngpus=1, sync=True),  # noqa B008
                 **kw
                 ):
        if not os.path.exists(base_model):
            defatult_path = os.path.join(lazyllm.config['model_path'], base_model)
            if os.path.exists(defatult_path):
                base_model = defatult_path
        if not merge_path:
            if target_path.endswith('lazyllm_lora') or '/lazyllm_lora/' in target_path or target_path.endswith('lazyllm_lora/'):
                merge_path = target_path.replace('lazyllm_lora', 'lazyllm_merge')
                os.system(f'mkdir -p {target_path} {merge_path}')
            else:
                save_path = os.path.join(lazyllm.config['train_target_root'], target_path)
                target_path, merge_path = os.path.join(save_path, 'lazyllm_lora'), os.path.join(save_path, 'lazyllm_merge')
                os.system(f'mkdir -p {target_path} {merge_path}')
        super().__init__(
            base_model,
            target_path,
            launcher=launcher,
        )
        self.merge_path = merge_path
        self.temp_yaml_file = None
        self.temp_export_yaml_file = None
        self.config_path = config_path
        self.export_config_path = export_config_path
        self.config_folder_path = os.path.dirname(os.path.abspath(__file__))

        default_config_path = os.path.join(self.config_folder_path, 'llama_factory', 'sft.yaml')
        self.template_dict = ArgsDict(self._load_yaml(default_config_path))

        if self.config_path:
            self.template_dict.update(self._load_yaml(self.config_path))

        if lora_r:
            self.template_dict['lora_rank'] = lora_r
        if modules_to_save:
            self.template_dict['additional_target'] = modules_to_save.strip('[]')
        if lora_target_modules:
            self.template_dict['lora_target'] = lora_target_modules.strip('[]')
        self.template_dict['model_name_or_path'] = base_model
        self.template_dict['output_dir'] = target_path
        self.template_dict['template'] = self._get_template_name(base_model)
        
        # Filter kw to only include keys that exist in template_dict
        # This ensures check_and_update won't fail due to unexpected keys
        # Keys not in template_dict will be silently ignored (they may be used elsewhere)
        filtered_kw = {k: v for k, v in kw.items() if k in self.template_dict}
        
        # Update template_dict with filtered parameters
        self.template_dict.check_and_update(filtered_kw)
        
        # Log key parameters for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"[llamafactory] filtered_kw: {filtered_kw}")

        default_export_config_path = os.path.join(self.config_folder_path, 'llama_factory', 'lora_export.yaml')
        self.export_dict = ArgsDict(self._load_yaml(default_export_config_path))

        if self.export_config_path:
            self.export_dict.update(self._load_yaml(self.export_config_path))

        self.export_dict['model_name_or_path'] = base_model
        self.export_dict['adapter_name_or_path'] = target_path
        self.export_dict['export_dir'] = merge_path
        self.export_dict['template'] = self.template_dict['template']

        self.temp_folder = os.path.join(lazyllm.config['temp_dir'], 'llamafactory_config', str(uuid.uuid4())[:10])
        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder)
        self.log_file_path = None

    def _get_template_name(self, base_model):
        base_name = os.path.basename(base_model).lower()
        key_value = match_longest_prefix(base_name)
        if key_value:
            return key_value
        else:
            raise RuntimeError(f'Cannot find prfix of base_model({base_model}) '
                               f'in DEFAULT_TEMPLATE of LLaMA_Factory: {llamafactory_mapping_dict}')

    def _load_yaml(self, config_path):
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        return config_dict

    def _build_temp_yaml(self, updated_template_str, prefix='train_'):
        fd, temp_yaml_file = tempfile.mkstemp(prefix=prefix, suffix='.yaml', dir=self.temp_folder)
        with os.fdopen(fd, 'w') as temp_file:
            temp_file.write(updated_template_str)
        return temp_yaml_file

    def _build_temp_dataset_info(self, datapaths):
        if isinstance(datapaths, str):
            datapaths = [datapaths]
        elif isinstance(datapaths, list) and all(isinstance(item, str) for item in datapaths):
            pass
        else:
            raise TypeError(f'datapaths({datapaths}) should be str or list of str.')
        temp_dataset_dict = dict()
        for datapath in datapaths:
            datapath = os.path.join(lazyllm.config['data_path'], datapath)
            assert os.path.isfile(datapath)
            file_name, _ = os.path.splitext(os.path.basename(datapath))
            temp_dataset_dict[file_name] = {'file_name': datapath}
            formatting = 'alpaca'
            try:
                with open(datapath, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                if 'messages' in data[0]:
                    formatting = 'sharegpt'
                media_types = []
                for media in ['images', 'videos', 'audios']:
                    if media in data[0]:
                        media_types.append(media)
                if media_types:
                    columns = {item: item for item in media_types}
                    columns.update({'messages': 'messages'})
                    temp_dataset_dict[file_name].update({
                        'tags': {
                            'role_tag': 'role',
                            'content_tag': 'content',
                            'user_tag': 'user',
                            'assistant_tag': 'assistant'
                        },
                        'columns': columns
                    })
            except Exception:
                pass
            temp_dataset_dict[file_name].update({'formatting': formatting})
        self.temp_dataset_info_path = os.path.join(self.temp_folder, 'dataset_info.json')
        with open(self.temp_dataset_info_path, 'w') as json_file:
            json.dump(temp_dataset_dict, json_file, indent=4)
        return self.temp_dataset_info_path, ','.join(temp_dataset_dict.keys())

    def _rm_temp_yaml(self):
        if self.temp_yaml_file:
            if os.path.exists(self.temp_yaml_file):
                os.remove(self.temp_yaml_file)
            self.temp_yaml_file = None

    def cmd(self, trainset, valset=None) -> str:
        thirdparty.check_packages(['datasets', 'deepspeed', 'numpy', 'peft', 'torch', 'transformers', 'trl'])
        # train config update
        if 'dataset_dir' in self.template_dict and self.template_dict['dataset_dir'] == 'lazyllm_temp_dir':
            _, datasets = self._build_temp_dataset_info(trainset)
            self.template_dict['dataset_dir'] = self.temp_folder
        else:
            datasets = trainset
        self.template_dict['dataset'] = datasets

        # save config update
        if self.template_dict['finetuning_type'] == 'lora':
            # For LoRA/QLoRA: use llamafactory-cli export to merge adapter with base model
            updated_export_str = yaml.dump(dict(self.export_dict), default_flow_style=False)
            self.temp_export_yaml_file = self._build_temp_yaml(updated_export_str, prefix='merge_')
        elif self.template_dict['finetuning_type'] == 'full':
            # For Full finetuning: model is already complete in target_path (lazyllm_lora)
            # We need to copy it to merge_path (lazyllm_merge) to maintain consistency
            # This ensures _get_all_finetuned_models and export_model work correctly
            pass  # Will handle in cmds below

        updated_template_str = yaml.dump(dict(self.template_dict), default_flow_style=False)
        self.temp_yaml_file = self._build_temp_yaml(updated_template_str)

        formatted_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        random_value = random.randint(1000, 9999)
        self.log_file_path = f'{self.target_path}/train_log_{formatted_date}_{random_value}.log'

        # Use bash instead of sh to support pipefail
        # This ensures export only runs if training succeeds
        cmds = f'export DISABLE_VERSION_CHECK=1 && bash -c "set -o pipefail && llamafactory-cli train {self.temp_yaml_file} 2>&1 | tee {self.log_file_path}"'
        if self.temp_export_yaml_file:
            # For LoRA/QLoRA: merge adapter with base model
            # Only run export if training succeeds (exit code 0)
            # With pipefail, tee will preserve the exit code from llamafactory-cli
            cmds += f' && llamafactory-cli export {self.temp_export_yaml_file}'
        elif self.template_dict['finetuning_type'] == 'full':
            # For Full finetuning: copy model from lazymm_lora to lazymm_merge
            # This maintains consistency with LoRA/QLoRA workflow
            # Only copy if training succeeds (exit code 0)
            # Only copy model files, exclude training process information (checkpoints, logs, etc.)
            # to save storage space since lazymm_merge is only used for exporting to models directory
            exclude_patterns = [
                '--exclude=checkpoint-*',  # Exclude checkpoint directories
                '--exclude=train_log_*.log',  # Exclude training logs
                '--exclude=trainer_state.json',  # Exclude trainer state
                '--exclude=trainer_log.jsonl',  # Exclude trainer log
                '--exclude=train_results.json',  # Exclude training results
                '--exclude=all_results.json',  # Exclude all results
                '--exclude=eval_results.json',  # Exclude evaluation results
                '--exclude=training_loss.png',  # Exclude training loss plot
                '--exclude=runs/',  # Exclude tensorboard logs
                '--exclude=training_args.bin',  # Exclude training arguments
            ]
            exclude_str = ' '.join(exclude_patterns)
            cmds += f' && rsync -a {exclude_str} {self.target_path}/ {self.merge_path}/ 2>/dev/null || true'
        return cmds
