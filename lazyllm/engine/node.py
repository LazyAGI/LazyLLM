# noqa: E121
import lazyllm
from typing import Any, Optional, List, Callable
from dataclasses import dataclass
from functools import partial

from lazyllm.tools.http_request.http_request import HttpRequest

@dataclass
class NodeArgs(object):
    type: type
    default: Any = None
    options: Optional[List] = None
    getattr_f: Optional[Callable] = None


all_nodes = dict()

all_nodes['LocalLLM'] = dict(
    module=lazyllm.TrainableModule,
    init_arguments=dict(
        base_model=NodeArgs(str),
        target_path=NodeArgs(str),
        stream=NodeArgs(bool, True),
        return_trace=NodeArgs(bool, False)),
    builder_argument=dict(
        trainset=NodeArgs(str),
        prompt=NodeArgs(str),
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
)

all_nodes['OnlineLLM'] = dict(
    module=lazyllm.OnlineChatModule,
    init_arguments=dict(
        source=NodeArgs(str),
        base_model=NodeArgs(str),
        stream=NodeArgs(bool, True),
        return_trace=NodeArgs(bool, False)),
    builder_argument=dict(
        prompt=NodeArgs(str)),
)

all_nodes['VQA'] = all_nodes['SD'] = all_nodes['TTS'] = all_nodes['STT'] = dict(
    module=lazyllm.OnlineChatModule,
    init_arguments=dict(base_model=NodeArgs(str))
)

all_nodes['HttpTool'] = dict(
    module=HttpRequest,
    init_arguments=dict(
        method=NodeArgs(str),
        url=NodeArgs(str),
        api_key=NodeArgs(str, ''),
        headers=NodeArgs(dict, {}),
        params=NodeArgs(dict, {}),
        body=NodeArgs(str, ''),
    ),
    builder_argument=dict(),
    other_arguments=dict()
)

all_nodes['Formatter'] = dict(
    module=partial(getattr, lazyllm.formatter),
    init_arguments=dict(
        formatter=NodeArgs(str),
    )
)
