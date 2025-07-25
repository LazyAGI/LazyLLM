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
    """此类是 ``LazyLLMFinetuneBase`` 的子类，基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 框架提供的训练能力，用于对大语言模型(或视觉语言模型)进行训练。

Args:
    base_model (str): 用于进行训练的基模型。要求是基模型的路径。
    target_path (str): 训练后模型保存权重的路径。
    merge_path (str): 模型合并LoRA权重后的路径，默认为None。如果未指定，则会在 ``target_path`` 下创建 "lazyllm_lora" 和 "lazyllm_merge" 目录，分别作为 ``target_path`` 和  ``merge_path`` 。
    config_path (str): LLaMA-Factory的训练配置文件（这里要求格式为yaml），默认为None。如果未指定，则会在当前工作路径下，创建一个名为 ``.temp`` 的文件夹，并在其中生成以 ``train_`` 前缀开头，以 ``.yaml`` 结尾的配置文件。
    export_config_path (str): LLaMA-Factory的Lora权重合并的配置文件（这里要求格式为yaml），默认为None。如果未指定，则会在当前工作路径下中的 ``.temp`` 文件夹内生成以 ``merge_`` 前缀开头，以 ``.yaml`` 结尾的配置文件。
    launcher (lazyllm.launcher): 微调的启动器，默认为 ``launchers.remote(ngpus=1, sync=True)``。
    kw: 关键字参数，用于更新默认的训练参数。

此类的关键字参数及其默认值如下：

Keyword Args:
    stage (typing.Literal['pt', 'sft', 'rm', 'ppo', 'dpo', 'kto']): 默认值是：``sft``。将在训练中执行的阶段。
    do_train (bool): 默认值是：``True``。是否运行训练。
    finetuning_type (typing.Literal['lora', 'freeze', 'full']): 默认值是：``lora``。要使用的微调方法。
    lora_target (str): 默认值是：``all``。要应用LoRA的目标模块的名称。使用逗号分隔多个模块。使用`all`指定所有线性模块。
    template (typing.Optional[str]): 默认值是：``None``。用于构建训练和推理提示的模板。
    cutoff_len (int): 默认值是：``1024``。数据集中token化后输入的截止长度。
    max_samples (typing.Optional[int]): 默认值是：``1000``。出于调试目的，截断每个数据集的示例数量。
    overwrite_cache (bool): 默认值是：``True``。覆盖缓存的训练和评估集。
    preprocessing_num_workers (typing.Optional[int]): 默认值是：``16``。用于预处理的进程数。
    dataset_dir (str): 默认值是：``lazyllm_temp_dir``。包含数据集的文件夹的路径。如果没有明确指定，LazyLLM将在当前工作目录的 ``.temp`` 文件夹中生成一个 ``dataset_info.json`` 文件，供LLaMA-Factory使用。
    logging_steps (float): 默认值是：``10``。每X个更新步骤记录一次日志。应该是整数或范围在 ``[0,1)`` 的浮点数。如果小于1，将被解释为总训练步骤的比例。
    save_steps (float): 默认值是：``500``。每X个更新步骤保存一次检查点。应该是整数或范围在 ``[0,1)`` 的浮点数。如果小于1，将被解释为总训练步骤的比例。
    plot_loss (bool): 默认值是：``True``。是否保存训练损失曲线。
    overwrite_output_dir (bool): 默认值是：``True``。覆盖输出目录的内容。
    per_device_train_batch_size (int): 默认值是：``1``。每个GPU/TPU/MPS/NPU核心/CPU的训练批次的大小。
    gradient_accumulation_steps (int): 默认值是：``8``。在执行反向传播及参数更新前，要累积的更新步骤数。
    learning_rate (float): 默认值是：``1e-04``。AdamW的初始学习率。
    num_train_epochs (float): 默认值是：``3.0``。要执行的总训练周期数。
    lr_scheduler_type (typing.Union[transformers.trainer_utils.SchedulerType, str]): 默认值是：``cosine``。要使用的调度器类型。
    warmup_ratio (float): 默认值是：``0.1``。在总步骤的 ``warmup_ratio`` 分之一阶段内进行线性预热。
    fp16 (bool): 默认值是：``True``。是否使用fp16（混合）精度，而不是32位。
    ddp_timeout (typing.Optional[int]): 默认值是：``180000000``。覆盖分布式训练的默认超时时间（值应以秒为单位给出）。
    report_to (typing.Union[NoneType, str, typing.List[str]]): 默认值是：``tensorboard``。要将结果和日志报告到的集成列表。
    val_size (float): 默认值是：``0.1``。验证集的大小，应该是整数或范围在`[0,1)`的浮点数。
    per_device_eval_batch_size (int): 默认值是：``1``。每个GPU/TPU/MPS/NPU核心/CPU的验证集批次大小。
    eval_strategy (typing.Union[transformers.trainer_utils.IntervalStrategy, str]): 默认值是：``steps``。要使用的验证评估策略。
    eval_steps (typing.Optional[float]): 默认值是：``500``。每X个步骤运行一次验证评估。应该是整数或范围在`[0,1)`的浮点数。如果小于1，将被解释为总训练步骤的比例。



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
