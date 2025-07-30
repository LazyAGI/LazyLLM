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
    """This class is a subclass of ``LazyLLMFinetuneBase``, based on the training capabilities provided by the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework, used for training large language models(or visual language models).

Args:
    base_model (str): The base model used for training. It is required to be the path of the base model.
    target_path (str): The path where the trained model weights are saved.
    merge_path (str): The path where the model is merged with LoRA weights, default is None. If not specified, "lazyllm_lora" and "lazyllm_merge" directories will be created under ``target_path``, to be used as ``target_path`` and ``merge_path`` respectively.
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



Examples:
    >>> from lazyllm import finetune
    >>> trainer = finetune.llamafactory('internlm2-chat-7b', 'path/to/target')
    <lazyllm.llm.finetune type=LlamafactoryFinetune>
    """
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
            save_path = os.path.join(lazyllm.config['train_target_root'], target_path)
            target_path, merge_path = os.path.join(save_path, "lazyllm_lora"), os.path.join(save_path, "lazyllm_merge")
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

        default_export_config_path = os.path.join(self.config_folder_path, 'llama_factory', 'lora_export.yaml')
        self.export_dict = ArgsDict(self.load_yaml(default_export_config_path))

        if self.export_config_path:
            self.export_dict.update(self.load_yaml(self.export_config_path))

        self.export_dict['model_name_or_path'] = base_model
        self.export_dict['adapter_name_or_path'] = target_path
        self.export_dict['export_dir'] = merge_path
        self.export_dict['template'] = self.template_dict['template']

        self.temp_folder = os.path.join(lazyllm.config['temp_dir'], 'llamafactory_config', str(uuid.uuid4())[:10])
        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder)
        self.log_file_path = None

    def get_template_name(self, base_model):
        base_name = os.path.basename(base_model).lower()
        key_value = match_longest_prefix(base_name)
        if key_value:
            return key_value
        else:
            raise RuntimeError(f'Cannot find prfix of base_model({base_model}) '
                               f'in DEFAULT_TEMPLATE of LLaMA_Factory: {llamafactory_mapping_dict}')

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
                    columns.update({"messages": "messages"})
                    temp_dataset_dict[file_name].update({
                        "tags": {
                            "role_tag": "role",
                            "content_tag": "content",
                            "user_tag": "user",
                            "assistant_tag": "assistant"
                        },
                        "columns": columns
                    })
            except Exception:
                pass
            temp_dataset_dict[file_name].update({'formatting': formatting})
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

        formatted_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        random_value = random.randint(1000, 9999)
        self.log_file_path = f'{self.target_path}/train_log_{formatted_date}_{random_value}.log'

        cmds = f'export DISABLE_VERSION_CHECK=1 && llamafactory-cli train {self.temp_yaml_file}'
        cmds += f' 2>&1 | tee {self.log_file_path}'
        if self.temp_export_yaml_file:
            cmds += f' && llamafactory-cli export {self.temp_export_yaml_file}'
        return cmds
