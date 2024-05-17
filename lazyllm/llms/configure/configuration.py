import csv
import os
from .rule import Configurations, Rule
from typing import Dict, List, Union, Any, TypeVar
from dataclasses import dataclass, asdict, fields


@dataclass(frozen=True)
class HardwareConfiguration:
    gpu_type: str
    gpu_num: int
    model_name: str
    ctx_len: int
    batch_size: int
    trainable_params: int

    def to_dict(self):
        return { key.upper() : value for key, value in asdict(self).items() }


@dataclass(frozen=True)
class TrainingConfiguration:
    framework: str
    tp: int
    pp: int
    zero: bool
    gradient_step: int
    lora_r: int
    sp: int 
    ddp: int
    micro_batch_size: int
    memory_usage_gb: int
    tgs: int
    additional_arguments: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        tp = data["TP"]
        pp = data["PP"]
        gpu_num = data["GPU_NUM"]
        batch_size = data["BATCH_SIZE"]
        gradient_step = data["GRADIENT_STEP"]

        ddp = gpu_num // tp // pp
        micro_batch_size = batch_size * tp * pp // gpu_num // gradient_step
        assert ddp > 0, f"(gpu num {gpu_num} / tp {tp} / pp {pp}) must be greater than 0"
        assert micro_batch_size > 0, f"(batch size {batch_size} * tp {tp} * pp {pp} / gpu number {gpu_num} / gradient step {gradient_step}) must be greater than 0"

        data.update(SP=1, MICRO_BATCH_SIZE=micro_batch_size, DDP=ddp)

        keys = set(x.name.upper() for x in fields(cls))
        data = { key.lower(): value for key, value in data.items() if key in keys }
        return cls(**data)


@dataclass(frozen=True)
class DeployConfiguration:
    framework: str
    tp: int
    max_token_num: int
    tgs: int
    additional_arguments: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        keys = set(x.name.upper() for x in fields(cls))
        data = { key.lower(): value for key, value in data.items() if key in keys }
        return cls(**data)


OutputConfiguration = TypeVar('OutputConfiguration', bound=Union[TrainingConfiguration, DeployConfiguration])


class ConfigurationDatabase(object):
    def __init__(self, url: Union[str, os.PathLike[str]], rules: List[Rule]):
        self.configurations = Configurations(rules)
        with open(url) as file:
            reader = csv.reader(file)
            self.configurations.parse_header(next(reader))
            self.configurations.parse_values(reader)

    def query(self, hc: HardwareConfiguration, clazz: type[OutputConfiguration]) -> List[OutputConfiguration]:
        return [ clazz.from_dict(arguments) for arguments in self.configurations.lookup(hc.to_dict()) ]
