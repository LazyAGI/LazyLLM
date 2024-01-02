from .base import LazyLLMFinetuneBase
import os

class AlpacaloraFinetune(LazyLLMFinetuneBase):
    def cmd(self, trainset, valset=None) -> str:
        # TODO(wangzhihong): modify cmd
        cmd = f'python alpaca-lora/finetune.py --weights={self.base_model} --target={self.target_path} --trainset={trainset}, --valset={valset}'
        return cmd