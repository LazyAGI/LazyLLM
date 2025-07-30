import csv
import os
from .rule import Configurations
from typing import Dict, List, Union, Any, TypeVar
from dataclasses import dataclass, fields
from .protocol import FINETUNE_RULE_SET, DEPLOY_RULE_SET


@dataclass(frozen=True)
class TrainingConfiguration:
    """TrainingConfiguration(framework: str, tp: int, zero: bool, gradient_step: int, sp: int, ddp: int, micro_batch_size: int, tgs: int)"""
    framework: str
    tp: int
    zero: bool
    gradient_step: int
    sp: int
    ddp: int
    micro_batch_size: int
    tgs: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        tp = data["TP"]
        gpu_num = data["GPU_NUM"]
        batch_size = data["BATCH_SIZE"]
        gradient_step = data["GRADIENT_STEP"]

        ddp = gpu_num // tp
        micro_batch_size = batch_size * tp // gpu_num // gradient_step
        assert ddp > 0, f"(gpu num {gpu_num} / tp {tp}  must be greater than 0"
        assert micro_batch_size > 0, (
            f"(batch size {batch_size} * tp {tp} / gpu number {gpu_num} / "
            f"gradient step {gradient_step}) must be greater than 0")

        data.update(SP=1, MICRO_BATCH_SIZE=micro_batch_size, DDP=ddp)

        keys = set(x.name.upper() for x in fields(cls))
        data = {key.lower(): value for key, value in data.items() if key in keys}
        return cls(**data)


@dataclass(frozen=True)
class DeployConfiguration:
    """DeployConfiguration(framework: str, tp: int, tgs: int)"""
    framework: str
    tp: int
    tgs: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        keys = set(x.name.upper() for x in fields(cls))
        data = {key.lower(): value for key, value in data.items() if key in keys}
        return cls(**data)


OutputConfiguration = TypeVar('OutputConfiguration', bound=Union[TrainingConfiguration, DeployConfiguration])


class AutoConfig(object):
    def __init__(self, finetune_file, deploy_file):
        with open(finetune_file) as file:
            reader = csv.reader(file)
            self._finetune = Configurations(FINETUNE_RULE_SET).parse_header(next(reader)).parse_values(reader)
        with open(deploy_file) as file:
            reader = csv.reader(file)
            self._deploy = Configurations(DEPLOY_RULE_SET).parse_header(next(reader)).parse_values(reader)

    def _query(self, *, clazz: type[OutputConfiguration], **kw) -> List[OutputConfiguration]:
        cf = self._finetune if clazz == TrainingConfiguration else self._deploy
        configurations = [clazz.from_dict(arguments) for arguments in cf.lookup({k.upper(): v for k, v in kw.items()})]
        configurations.sort(key=lambda x: x.tgs, reverse=True)
        return configurations

    def query_finetune(self, gpu_type: str, gpu_num: int, model_name: str,
                       ctx_len: int, batch_size: int, lora_r: int):
        return self._query(clazz=TrainingConfiguration, gpu_type=gpu_type, gpu_num=gpu_num, model_name=model_name,
                           ctx_len=ctx_len, batch_size=batch_size, lora_r=lora_r)

    def query_deploy(self, gpu_type: str, gpu_num: int, model_name: str, max_token_num):
        return self._query(clazz=DeployConfiguration, gpu_type=gpu_type, gpu_num=gpu_num,
                           model_name=model_name, max_token_num=max_token_num)


configer = None

def get_configer():
    global configer
    if configer is None:
        configer = AutoConfig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'finetune.csv'),
                              os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'deploy.csv'))
    return configer
