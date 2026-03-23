import os
import shutil
import tempfile
import pytest

from lazyllm import config
from lazyllm.tools.data.pipelines.pt_text_ppl import (
    build_text_pt_pipeline,
    build_phi4_pt_pipeline,
)
from lazyllm.tools.data.pipelines.pt_img_ppl import build_mm_pt_pipeline


class MockModel:
    def __init__(self, mock_response):
        self.mock_response = mock_response

    def __call__(self, string, **kwargs):
        return self.mock_response

    def prompt(self, prompt):
        return self

    def formatter(self, formatter):
        return self

    def share(self):
        return self

    def start(self):
        return self


class TestPretrainPipelines:

    def setup_method(self):
        self.root_dir = tempfile.mkdtemp()
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)

    def _test_image_file(self, name):
        data_path = config['data_path']
        return os.path.join(data_path, 'ci_data', name)

    def test_text_pt_pipeline(self):
        ppl = build_text_pt_pipeline(
            content_key='content',
            language='zh',
            min_chars=10,
            max_chars=100000,
            min_words=5,
            max_words=100000,
            max_tokens=512,
            min_tokens=1,
        )

        content = (
            '今天是晴天。这是一个用于测试的预训练样本文本，有足够的长度与句子数量。'
            '整体内容健康、正常，适合作为预训练语料。'
            'https://example.com 链接会被移除。HTML <b>标签</b> 也会移除。'
            '表情😀会被去掉。多余  空格  会合并。' * 3
        )
        res = ppl([{'content': content}])
        assert isinstance(res, list)
        assert len(res) >= 1
        for item in res:
            assert 'content' in item
            out = item['content']
            assert out.strip()
            assert 'https://' not in out
            assert '<b>' not in out and '</b>' not in out

    def test_phi4_pt_pipeline(self):
        mock_response = {
            'score': 1,
            'reason': 'suitable',
            'qa_pairs': [
                {'query': 'What is it?', 'answer': 'A test context.'},
            ],
        }
        llm = MockModel(mock_response)

        ppl = build_phi4_pt_pipeline(
            context_key='context',
            image_key=None,
            llm=llm,
            num_qa=1,
        )

        context = (
            'This is a simple test context. '
            'It is used to verify the phi4 pretraining pipeline. '
            'The text is long enough and contains multiple sentences.'
        )
        data = [{'context': context}]

        res = ppl(data)

        assert isinstance(res, list)
        assert len(res) == 1
        assert 'qa_pairs' in res[0]
        assert isinstance(res[0]['qa_pairs'], list)

    def test_mm_pt_pipeline(self):
        ji = self._test_image_file('ji.jpg')
        if not os.path.exists(ji):
            pytest.skip(f'Test image not found, skipping test: {ji}')

        ppl = build_mm_pt_pipeline(
            image_key='image_path',
            text_key='text',
            vlm=None,
            min_width=1,
            min_height=1,
            max_side=4096,
            relevance_threshold=0.0,
            use_dedup=False,
        )

        data = [{'text': 'A descriptive caption for the image.', 'image_path': ji}]

        res = ppl(data)

        assert isinstance(res, list)
        assert len(res) == 1
        assert 'image_path' in res[0]
