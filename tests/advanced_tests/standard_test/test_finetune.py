import os
import json
import shutil
import pytest

import lazyllm
from lazyllm import finetune
from lazyllm.launcher import cleanup
from lazyllm.components.formatter import encode_query_with_filepaths


class TestFinetune(object):

    def setup_method(self):
        self.llm_data = 'alpaca/alpaca_data_zh_128.json'
        self.llm_path = 'qwen1.5-0.5b-chat'
        self.vlm_data = 'ci_data/vqa_rad/train.json'
        self.vlm_path = 'qwen2.5-vl-3b-instruct'
        self.embed_data = os.path.join(lazyllm.config['data_path'], 'sft_embeding/embedding.json')
        self.embed_path = 'bge-m3'
        self.rerank_data = os.path.join(lazyllm.config['data_path'], 'sft_embeding/rerank.jsonl')
        self.rerank_path = 'bge-reranker-large'
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
            lambda: self.llm_data,
            finetune.llamafactory(
                base_model=self.llm_path,
                target_path=self.save_path,
            )
        )
        ppl()
        assert self.has_bin_file(os.path.join(self.save_path, 'lazyllm_lora'))
        assert self.has_bin_file(os.path.join(self.save_path, 'lazyllm_merge'))

    def test_finetune_vlm_llamafactory(self):
        m = lazyllm.TrainableModule(self.vlm_path, self.save_path)\
            .mode('finetune')\
            .trainset(self.vlm_data)\
            .finetune_method(
                (lazyllm.finetune.llamafactory, {
                    'learning_rate': 1e-4,
                    'cutoff_len': 5120,
                    'max_samples': 20000,
                    'val_size': 0.01,
                    'num_train_epochs': 1.0,
                    'per_device_train_batch_size': 4,
                }))
        m.update()
        assert self.has_bin_file(m.finetuned_model_path)
        image_path = os.path.join(lazyllm.config['data_path'], 'ci_data/vqa_rad/imgs/train_image_0.jpg')
        res = m(encode_query_with_filepaths('are regions of the brain infarcted?', image_path))
        assert type(res) is str

    def test_finetune_embedding(self):
        m = lazyllm.TrainableModule(self.embed_path, self.save_path)\
            .mode('finetune').trainset(self.embed_data)\
            .finetune_method(finetune.flagembedding)
        m.update()
        assert self.has_bin_file(m.finetuned_model_path)
        res = m('你好啊')
        vect = json.loads(res)
        assert type(vect) is list
        assert len(vect) == 1024

    def test_finetune_reranker(self):
        m = lazyllm.TrainableModule(self.rerank_path, self.save_path)\
            .mode('finetune').trainset(self.rerank_data)\
            .finetune_method(finetune.flagembedding)
        m.update()
        assert self.has_bin_file(m.finetuned_model_path)
        res = m('hi', documents=['go', 'hi', 'hello', 'how'], top_n=2)
        assert type(res) is list
        assert len(res) == 2
        assert type(res[0]) is tuple
