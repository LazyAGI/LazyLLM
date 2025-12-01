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
        self.grpo_train_data = 'ci_data/math-json-200/train200.json'
        self.grpo_test_data = 'ci_data/math-json-200/test100.json'
        self.grpo_llm = 'qwen2-0.5b-instruct'
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
            raise RuntimeError(f'Cannot find model path: {path}')
        for filename in os.listdir(path):
            if filename.endswith('.bin') or filename.endswith('.safetensors'):
                return True
        return False

    @pytest.mark.run_on_change(
        'lazyllm/components/auto/autofinetune.py',
        'lazyllm/components/finetune/llamafactory.py')
    def test_finetune_auto_llm(self):
        m = lazyllm.TrainableModule(self.llm_path, self.save_path)\
            .mode('finetune').trainset(self.llm_data)
        tasks = m._impl._get_train_tasks_impl()
        assert isinstance(tasks[-1], lazyllm.finetune.llamafactory)

    @pytest.mark.run_on_change(
        'lazyllm/components/auto/autofinetune.py',
        'lazyllm/components/finetune/llamafactory.py')
    def test_finetune_auto_vlm(self):
        m = lazyllm.TrainableModule(self.vlm_path, self.save_path)\
            .mode('finetune').trainset(self.vlm_data)
        tasks = m._impl._get_train_tasks_impl()
        assert isinstance(tasks[-1], lazyllm.finetune.llamafactory)

    @pytest.mark.run_on_change(
        'lazyllm/components/auto/autofinetune.py',
        'lazyllm/components/finetune/flagembedding.py')
    def test_finetune_auto_embedding(self):
        m = lazyllm.TrainableModule(self.embed_path, self.save_path)\
            .mode('finetune').trainset(self.embed_data)
        tasks = m._impl._get_train_tasks_impl()
        assert isinstance(tasks[-1], lazyllm.finetune.flagembedding)

    @pytest.mark.run_on_change(
        'lazyllm/components/auto/autofinetune.py',
        'lazyllm/components/finetune/flagembedding.py')
    def test_finetune_auto_rerank(self):
        m = lazyllm.TrainableModule(self.rerank_path, self.save_path)\
            .mode('finetune').trainset(self.rerank_data)
        tasks = m._impl._get_train_tasks_impl()
        assert isinstance(tasks[-1], lazyllm.finetune.flagembedding)

    @pytest.mark.run_on_change(
        'lazyllm/components/auto/autofinetune.py',
        'lazyllm/components/finetune/llamafactory.py')
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

    @pytest.mark.run_on_change(
        'lazyllm/components/auto/autofinetune.py',
        'lazyllm/components/finetune/llamafactory.py')
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

    @pytest.mark.run_on_change(
        'lazyllm/components/auto/autofinetune.py',
        'lazyllm/components/finetune/flagembedding.py')
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

    @pytest.mark.run_on_change(
        'lazyllm/components/auto/autofinetune.py',
        'lazyllm/components/finetune/flagembedding.py')
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

    @pytest.mark.run_on_change(
        r'lazyllm/components/finetune/easy_r1/.*',
        r'lazyllm/components/finetune/easyr1\.json',
        regex_mode=True)
    def test_grpo_easyr1(self):
        m = lazyllm.TrainableModule(self.grpo_llm, self.save_path)\
            .mode('finetune')\
            .trainset(lambda: lazyllm.package(self.grpo_train_data, self.grpo_test_data))\
            .finetune_method(
                (lazyllm.finetune.easyr1, {
                    'data.rollout_batch_size': 64,
                    'data.val_batch_size': 32,
                    'worker.actor.global_batch_size': 32,
                    'trainer.save_model_only': True,
                    'trainer.total_epochs': 1,
                    'worker.rollout.tensor_parallel_size': 2,
                    'launcher': lazyllm.launchers.remote(ngpus=2, sync=True),
                }))
        m.update()
        assert self.has_bin_file(m.finetuned_model_path)
        res = m('hi')
        assert type(res) is str
