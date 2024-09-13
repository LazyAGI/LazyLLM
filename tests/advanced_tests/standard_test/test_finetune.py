import os
import shutil
import pytest

import lazyllm
from lazyllm import finetune
from lazyllm.launcher import cleanup

class TestFinetune(object):

    def setup_method(self):
        self.data = 'alpaca/alpaca_data_zh_128.json'
        self.model_path = 'qwen1.5-0.5b-chat'
        self.save_path = os.path.join(os.getcwd(), '.temp')

    @pytest.fixture(autouse=True)
    def run_around_tests(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        yield
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        cleanup()

    def has_bin_file(self, path):
        if not os.path.exists(path):
            raise RuntimeError(f"Cannot find model path: {path}")
        for filename in os.listdir(path):
            if filename.endswith('.bin') or filename.endswith('.safetensors'):
                return True
        return False

    def test_finetune_llamafactory(self):
        ppl = lazyllm.pipeline(
            lambda: 'alpaca/alpaca_data_zh_128.json',
            finetune.llamafactory(
                base_model='qwen1.5-0.5b-chat',
                target_path=self.save_path,
            )
        )
        ppl()
        assert self.has_bin_file(os.path.join(self.save_path, 'lazyllm_lora'))
        assert self.has_bin_file(os.path.join(self.save_path, 'lazyllm_merge'))
