import os
import copy
import random
from datetime import datetime

import lazyllm
from lazyllm import launchers, ArgsDict, thirdparty
from .base import LazyLLMFinetuneBase
from ..utils.downloader import ModelManager


class FlagembeddingFinetune(LazyLLMFinetuneBase):
    """该类是 ``LazyLLMFinetuneBase`` 的子类，基于 [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) 框架提供的训练能力，用于训练嵌入和重排模型。

Args:
    base_model (str): 用于训练的基础模型。必须是基础模型的路径。
    target_path (str): 训练后模型权重保存的路径。
    launcher (lazyllm.launcher): 微调的启动器，默认为 ``launchers.remote(ngpus=1, sync=True)``。
    kw: 用于更新默认训练参数的关键字参数。

该类嵌入模型的关键字参数及其默认值如下：

Keyword Args:
    train_group_size (int): 默认为：``8``。训练组的大小。用于控制每个训练集中的负样本数量。
    query_max_len (int): 默认为：``512``。经过分词后，段落的最大总输入序列长度。超过此长度的序列将被截断，较短的序列将被填充。
    passage_max_len (int): 默认为：``512``。经过分词后，段落的最大总输入序列长度。超过此长度的序列将被截断，较短的序列将被填充。
    pad_to_multiple_of (int): 默认为：``8``。如果设置，将序列填充为提供值的倍数。
    query_instruction_for_retrieval (str): 默认为：``Represent this sentence for searching relevant passages: ``。查询query的指令。
    query_instruction_format (str): 默认为：``{}{}``。查询指令格式。
    learning_rate (float): 默认为：``1e-5``。学习率。
    num_train_epochs (int): 默认为：``1``。要执行的总训练周期数。
    per_device_train_batch_size (int): 默认为：``2``。训练批量大小。
    gradient_accumulation_steps (int): 默认为：``1``。在执行反向/更新传递之前要累积的更新步骤数。
    dataloader_drop_last (bool): 默认为：``True``。如果数据集大小不能被批量大小整除，则丢弃最后一个不完整的批量，即 DataLoader 只返回完整的批量。
    warmup_ratio (float): 默认为：``0.1``。线性调度器的预热比率。
    weight_decay (float): 默认为：``0.01``。AdamW 中的权重衰减。
    deepspeed (str): 默认为：````。DeepSpeed 配置文件的路径，默认使用 LazyLLM 代码仓库中的预置文件：``ds_stage0.json``。
    logging_steps (int): 默认为：``1``。更新日志的频率。
    save_steps (int): 默认为：``1000``。保存频率。
    temperature (float): 默认为：``0.02``。用于相似度评分的温度。
    sentence_pooling_method (str): 默认为：``cls``。池化方法。可用选项：'cls', 'mean', 'last_token'。
    normalize_embeddings (bool): 默认为：``True``。是否归一化嵌入。
    kd_loss_type (str): 默认为：``kl_div``。知识蒸馏的损失类型。可用选项：'kl_div', 'm3_kd_loss'。
    overwrite_output_dir (bool): 默认为：``True``。用于允许程序覆盖现有的输出目录。
    fp16 (bool): 默认为：``True``。是否使用 fp16（混合）精度而不是 32 位。
    gradient_checkpointing (bool): 默认为：``True``。是否启用梯度检查点。
    negatives_cross_device (bool): 默认为：``True``。是否在设备间共享负样本。

该类重排模型的关键字参数及其默认值如下：

Keyword Args:
    train_group_size (int): 默认为：``8``。训练组的大小。用于控制每个训练集中的负样本数量。
    query_max_len (int): 默认为：``256``。经过分词后，段落的最大总输入序列长度。超过此长度的序列将被截断，较短的序列将被填充。
    passage_max_len (int): 默认为：``256``。经过分词后，段落的最大总输入序列长度。超过此长度的序列将被截断，较短的序列将被填充。
    pad_to_multiple_of (int): 默认为：``8``。如果设置，将序列填充为提供值的倍数。
    learning_rate (float): 默认为：``6e-5``。学习率。
    num_train_epochs (int): 默认为：``1``。要执行的总训练周期数。
    per_device_train_batch_size (int): 默认为：``2``。训练批量大小。
    gradient_accumulation_steps (int): 默认为：``1``。在执行反向/更新传递之前要累积的更新步骤数。
    dataloader_drop_last (bool): 默认为：``True``。如果数据集大小不能被批量大小整除，则丢弃最后一个不完整的批量，即 DataLoader 只返回完整的批量。
    warmup_ratio (float): 默认为：``0.1``。线性调度器的预热比率。
    weight_decay (float): 默认为：``0.01``。AdamW 中的权重衰减。
    deepspeed (str): 默认为：````。DeepSpeed 配置文件的路径，默认使用 LazyLLM 代码仓库中的预置文件：``ds_stage0.json``。
    logging_steps (int): 默认为：``1``。更新日志的频率。
    save_steps (int): 默认为：``1000``。保存频率。
    overwrite_output_dir (bool): 默认为：``True``。用于允许程序覆盖现有的输出目录。
    fp16 (bool): 默认为：``True``。是否使用 fp16（混合）精度而不是 32 位。
    gradient_checkpointing (bool): 默认为：``True``。是否启用梯度检查点。



Examples:
    >>> from lazyllm import finetune
    >>> finetune.FlagembeddingFinetune('bge-m3', 'path/to/target')
    <lazyllm.llm.finetune type=FlagembeddingFinetune>
    """
    defatult_embed_kw = ArgsDict({
        'train_group_size': 8,
        'query_max_len': 512,
        'passage_max_len': 512,
        'pad_to_multiple_of': 8,
        'query_instruction_for_retrieval': 'Represent this sentence for searching relevant passages: ',
        'query_instruction_format': '{}{}',
        'learning_rate': 1e-5,
        'num_train_epochs': 1,
        'per_device_train_batch_size': 2,
        'gradient_accumulation_steps': 1,
        'dataloader_drop_last': True,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'deepspeed': '',
        'logging_steps': 1,
        'save_steps': 1000,
        'temperature': 0.02,
        'sentence_pooling_method': 'cls',
        'normalize_embeddings': True,
        'kd_loss_type': 'kl_div',
        'overwrite_output_dir': True,
        'fp16': True,
        'gradient_checkpointing': True,
        'negatives_cross_device': True
    })
    defatult_rerank_kw = ArgsDict({
        'train_group_size': 8,
        'query_max_len': 256,
        'passage_max_len': 256,
        'pad_to_multiple_of': 8,
        'learning_rate': 6e-5,
        'num_train_epochs': 1,
        'per_device_train_batch_size': 2,
        'gradient_accumulation_steps': 1,
        'dataloader_drop_last': True,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'deepspeed': '',
        'logging_steps': 1,
        'save_steps': 1000,
        'overwrite_output_dir': True,
        'fp16': True,
        'gradient_checkpointing': True
    })
    store_true_embed_kw = {'overwrite_output_dir', 'fp16', 'gradient_checkpointing', 'negatives_cross_device'}
    store_true_rerank_kw = {'overwrite_output_dir', 'fp16', 'gradient_checkpointing'}

    def __init__(
        self,
        base_model,
        target_path,
        launcher=launchers.remote(ngpus=1, sync=True),
        **kw
    ):
        model_type = ModelManager.get_model_type(base_model.split('/')[-1])
        if model_type not in ('embed', 'reranker'):
            raise RuntimeError(f'Not supported {model_type} type to finetune.')
        if not os.path.exists(base_model):
            defatult_path = os.path.join(lazyllm.config['model_path'], base_model)
            if os.path.exists(defatult_path):
                base_model = defatult_path
        save_path = os.path.join(lazyllm.config['train_target_root'], target_path)
        target_path = os.path.join(save_path, model_type)
        os.system(f'mkdir -p {target_path}')
        super().__init__(
            base_model,
            target_path,
            launcher=launcher,
        )
        if model_type == 'reranker':
            self.kw = copy.deepcopy(self.defatult_rerank_kw)
            self.store_true_kw = copy.deepcopy(self.store_true_rerank_kw)
            self.module_run_path = 'FlagEmbedding.finetune.reranker.encoder_only.base'
        else:
            self.kw = copy.deepcopy(self.defatult_embed_kw)
            self.store_true_kw = copy.deepcopy(self.store_true_embed_kw)
            self.module_run_path = 'FlagEmbedding.finetune.embedder.encoder_only.base'
        self.kw.check_and_update(kw)
        if not self.kw['deepspeed']:
            folder_path = os.path.dirname(os.path.abspath(__file__))
            deepspeed_config_path = os.path.join(folder_path, 'flag_embedding', 'ds_stage0.json')
            self.kw['deepspeed'] = deepspeed_config_path
        self.nproc_per_node = launcher.ngpus

    def cmd(self, trainset, valset=None) -> str:
        thirdparty.check_packages(['flagembedding'])
        self.kw['train_data'] = trainset

        formatted_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file_path = f'{self.target_path}/train_log_{formatted_date}_{random.randint(1000, 9999)}.log'
        cache_path = os.path.join(os.path.expanduser('~'), '.lazyllm', 'fintune', 'embeding')
        cache_model_path = os.path.join(cache_path, "model")
        cache_data_path = os.path.join(cache_path, "data")
        os.system(f'mkdir -p {cache_model_path} {cache_data_path}')

        cmds = (f'export WANDB_MODE=disabled && torchrun --nproc_per_node {self.nproc_per_node} '
                f'-m {self.module_run_path} '
                f'--model_name_or_path {self.base_model} '
                f'--output_dir {self.target_path} '
                f'--cache_dir {cache_model_path} '
                f'--cache_path {cache_data_path} '
            )
        for key in self.store_true_kw:
            cmds += f'--{key} ' if self.kw.pop(key) else ''
        cmds += self.kw.parse_kwargs()
        cmds += f' 2>&1 | tee {self.log_file_path}'
        return cmds
