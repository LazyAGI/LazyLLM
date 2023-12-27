from .base import LazyLLMFinetuneBase
import os

class AlpacaloraFinetune(LazyLLMFinetuneBase):
    def cmd(self) -> str:
        cmd = f'python alpaca-lora/finetune.py --weights={self.base_model}'
        return cmd