import os
import yaml
import json
import tempfile

import lazyllm
from lazyllm import launchers, ArgsDict, thirdparty, CaseInsensitiveDict
from .base import LazyLLMFinetuneBase

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
                 launcher=launchers.remote(ngpus=1, sync=True),
                 **kw
                 ):
        if not os.path.exists(base_model):
            defatult_path = os.path.join(lazyllm.config['model_path'], base_model)
            if os.path.exists(defatult_path):
                base_model = defatult_path
        if not merge_path:
            save_path = os.path.join(os.getcwd(), target_path)
            target_path, merge_path = os.path.join(save_path, "lora"), os.path.join(save_path, "merge")
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

        default_config_path = os.path.join(self.config_folder_path, 'llamafactory/sft.yaml')
        self.template_dict = ArgsDict(self.load_yaml(default_config_path))

        if self.config_path:
            self.template_dict.update(self.load_yaml(self.config_path))

        if lora_r:
            self.template_dict['lora_rank'] = lora_r
        if modules_to_save:
            self.template_dict['additional_target'] = modules_to_save.strip('[]')
        if lora_target_modules:
            self.template_dict['lora_target'] = lora_target_modules.strip('[]')
        self.template_dict['model_name_or_path'] = base_model
        self.template_dict['output_dir'] = target_path
        self.template_dict['template'] = self.get_template_name(base_model)
        self.template_dict.check_and_update(kw)

        default_export_config_path = os.path.join(self.config_folder_path, 'llamafactory/lora_export.yaml')
        self.export_dict = ArgsDict(self.load_yaml(default_export_config_path))

        if self.export_config_path:
            self.export_dict.update(self.load_yaml(self.export_config_path))

        self.export_dict['model_name_or_path'] = base_model
        self.export_dict['adapter_name_or_path'] = target_path
        self.export_dict['export_dir'] = merge_path
        self.export_dict['template'] = self.template_dict['template']

        self.temp_folder = os.path.join(os.getcwd(), '.temp')
        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder)

    def get_template_name(self, base_model):
        try:
            from llamafactory.extras.constants import DEFAULT_TEMPLATE
        except Exception:
            # llamfactory need a gpu, 1st import raise error, so import 2nd.
            from llamafactory.extras.constants import DEFAULT_TEMPLATE
        teplate_dict = CaseInsensitiveDict(DEFAULT_TEMPLATE)
        key = os.path.basename(base_model).split('-')[0]
        if key in teplate_dict:
            return teplate_dict[key]
        else:
            raise RuntimeError(f'Cannot find prfix({key}) of base_model({base_model}) '
                               f'in DEFAULT_TEMPLATE of LLaMA_Factory: {DEFAULT_TEMPLATE}')

    def load_yaml(self, config_path):
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        return config_dict

    def build_temp_yaml(self, updated_template_str, prefix='train_'):
        fd, temp_yaml_file = tempfile.mkstemp(prefix=prefix, suffix='.yaml', dir=self.temp_folder)
        with os.fdopen(fd, 'w') as temp_file:
            temp_file.write(updated_template_str)
        return temp_yaml_file

    def build_temp_dataset_info(self, datapaths):
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
        self.temp_dataset_info_path = os.path.join(self.temp_folder, 'dataset_info.json')
        with open(self.temp_dataset_info_path, 'w') as json_file:
            json.dump(temp_dataset_dict, json_file, indent=4)
        return self.temp_dataset_info_path, ','.join(temp_dataset_dict.keys())

    def rm_temp_yaml(self):
        if self.temp_yaml_file:
            if os.path.exists(self.temp_yaml_file):
                os.remove(self.temp_yaml_file)
            self.temp_yaml_file = None

    def cmd(self, trainset, valset=None) -> str:
        thirdparty.check_packages(['datasets', 'deepspeed', 'numpy', 'peft', 'torch', 'transformers', 'trl'])
        # train config update
        if 'dataset_dir' in self.template_dict and self.template_dict['dataset_dir'] == 'lazyllm_temp_dir':
            _, datasets = self.build_temp_dataset_info(trainset)
            self.template_dict['dataset_dir'] = self.temp_folder
        else:
            datasets = trainset
        self.template_dict['dataset'] = datasets

        # save config update
        if self.template_dict['finetuning_type'] == 'lora':
            updated_export_str = yaml.dump(dict(self.export_dict), default_flow_style=False)
            self.temp_export_yaml_file = self.build_temp_yaml(updated_export_str, prefix='merge_')

        updated_template_str = yaml.dump(dict(self.template_dict), default_flow_style=False)
        self.temp_yaml_file = self.build_temp_yaml(updated_template_str)

        cmds = f'llamafactory-cli train {self.temp_yaml_file}'
        if self.temp_export_yaml_file:
            cmds += f' && llamafactory-cli export {self.temp_export_yaml_file}'
        return cmds
