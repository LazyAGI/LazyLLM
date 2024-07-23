# noqa: E121
import lazyllm
from typing import Any, Optional, List, Callable
from dataclasses import dataclass
from functools import partial

@dataclass
class NodeArgs(object):
    type: type
    default: Any = None
    options: Optional[List] = None
    getattr_f: Optional[Callable] = None


all_nodes = dict(

    LocalModel=dict(
        module=lazyllm.TrainableModule,
        init_arguments=dict(
            base_model=NodeArgs(str),
            target_path=NodeArgs(str),
            stream=NodeArgs(bool, True),
            return_trace=NodeArgs(bool, False)),
        builder_argument=dict(
            trainset=NodeArgs(str),
            finetune_method=NodeArgs(str, getattr_f=partial(getattr, lazyllm.finetune)),
            deploy_method=NodeArgs(str, 'vllm', getattr_f=partial(getattr, lazyllm.deploy))),
        other_arguments=dict(
            finetune_method=dict(
                batch_size=NodeArgs(int, 16),
                micro_batch_size=NodeArgs(int, 2),
                num_epochs=NodeArgs(int, 3),
                learning_rate=NodeArgs(float, 5e-4),
                lora_r=NodeArgs(int, 8),
                lora_alpha=NodeArgs(int, 32),
                lora_dropout=NodeArgs(float, 0.05)))
    ),
)
