import os
import copy
import random
from datetime import datetime

import lazyllm
from lazyllm import launchers, ArgsDict, thirdparty
from .base import LazyLLMFinetuneBase
from ..utils.downloader import ModelManager


class FlagembeddingFinetune(LazyLLMFinetuneBase):
    """This class is a subclass of ``LazyLLMFinetuneBase``, based on the training capabilities provided by the [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) framework, used for training embedding and reranker models.

Args:
    base_model (str): The base model used for training. It is required to be the path of the base model.
    target_path (str): The path where the trained model weights are saved.
    launcher (lazyllm.launcher): The launcher for fine-tuning, default is ``launchers.remote(ngpus=1, sync=True)``.
    kw: Keyword arguments used to update the default training parameters.

The keyword arguments and their default values for this class of embedding model are as follows:

Keyword Args:
    train_group_size (int): Default is: ``8``. The size of train group. It is used to control the number of negative samples in each training set.
    query_max_len (int): Default is: ``512``. The maximum total input sequence length after tokenization for passage. Sequences longer than this will be truncated, sequences shorter will be padded.
    passage_max_len (int): Default is: ``512``. The maximum total input sequence length after tokenization for passage. Sequences longer than this will be truncated, sequences shorter will be padded.
    pad_to_multiple_of (int): Default is: ``8``. If set will pad the sequence to be a multiple of the provided value.
    query_instruction_for_retrieval (str): Default is: ``Represent this sentence for searching relevant passages: ``. Instruction for query.
    query_instruction_format (str): Default is: ``{}{}``. Format for query instruction.
    learning_rate (float): Default is: ``1e-5``. Learning rate.
    num_train_epochs (int): Default is: ``1``. Total number of training epochs to perform.
    per_device_train_batch_size (int): Default is: ``2``. Train batch size
    gradient_accumulation_steps (int): Default is: ``1``. Number of updates steps to accumulate before performing a backward/update pass.
    dataloader_drop_last (bool): Default is: ``True``. When it='True', the last incomplete batch is dropped if the dataset size is not divisible by the batch size, meaning DataLoader only returns complete batches.
    warmup_ratio (float): Default is: ``0.1``. Warmup ratio for linear scheduler.
    weight_decay (float): Default is: ``0.01``. Weight decay in AdamW.
    deepspeed (str): Default is: ````. The path of the DeepSpeed configuration file, default to use the pre-made configuration file in the LazyLLM code repository: ``ds_stage0.json``.
    logging_steps (int): Default is: ``1``. Logging frequency according to logging strategy.
    save_steps (int): Default is: ``1000``. Saving frequency.
    temperature (float): Default is: ``0.02``. Temperature used for similarity score
    sentence_pooling_method (str): Default is: ``cls``. The pooling method. Available options: 'cls', 'mean', 'last_token'.
    normalize_embeddings (bool): Default is: ``True``. Whether to normalize the embeddings.
    kd_loss_type (str): Default is: ``kl_div``. The loss type for knowledge distillation. Available options:'kl_div', 'm3_kd_loss'.
    overwrite_output_dir (bool): Default is: ``True``. It is used to allow the program to overwrite an existing output directory.
    fp16 (bool): Default is: ``True``.  Whether to use fp16 (mixed) precision instead of 32-bit.
    gradient_checkpointing (bool): Default is: ``True``. Whether enable gradient checkpointing.
    negatives_cross_device (bool): Default is: ``True``. Whether share negatives across devices.

The keyword arguments and their default values for this class of reranker model are as follows:

Keyword Args:
    train_group_size (int): Default is: ``8``. The size of train group. It is used to control the number of negative samples in each training set.
    query_max_len (int): Default is: ``256``. The maximum total input sequence length after tokenization for passage. Sequences longer than this will be truncated, sequences shorter will be padded.
    passage_max_len (int): Default is: ``256``. The maximum total input sequence length after tokenization for passage. Sequences longer than this will be truncated, sequences shorter will be padded.
    pad_to_multiple_of (int): Default is: ``8``. If set will pad the sequence to be a multiple of the provided value.
    learning_rate (float): Default is: ``6e-5``. Learning rate.
    num_train_epochs (int): Default is: ``1``. Total number of training epochs to perform.
    per_device_train_batch_size (int): Default is: ``2``. Train batch size
    gradient_accumulation_steps (int): Default is: ``1``. Number of updates steps to accumulate before performing a backward/update pass.
    dataloader_drop_last (bool): Default is: ``True``. When it='True', the last incomplete batch is dropped if the dataset size is not divisible by the batch size, meaning DataLoader only returns complete batches.
    warmup_ratio (float): Default is: ``0.1``. Warmup ratio for linear scheduler.
    weight_decay (float): Default is: ``0.01``. Weight decay in AdamW.
    deepspeed (str): Default is: ````. The path of the DeepSpeed configuration file, default to use the pre-made configuration file in the LazyLLM code repository: ``ds_stage0.json``.
    logging_steps (int): Default is: ``1``. Logging frequency according to logging strategy.
    save_steps (int): Default is: ``1000``. Saving frequency.
    overwrite_output_dir (bool): Default is: ``True``. It is used to allow the program to overwrite an existing output directory.
    fp16 (bool): Default is: ``True``.  Whether to use fp16 (mixed) precision instead of 32-bit.
    gradient_checkpointing (bool): Default is: ``True``. Whether enable gradient checkpointing.



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
