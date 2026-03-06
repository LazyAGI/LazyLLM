import os
import shutil
import tempfile
from lazyllm import config
from lazyllm.tools.data.pipelines.enhance_pipelines import build_enhance_qa_pipeline


class MockModel:
    def __init__(self, mock_response):
        self.mock_response = mock_response

    def __call__(self, string: str, **kwargs):
        return self.mock_response

    def prompt(self, prompt):
        return self

    def formatter(self, formatter):
        return self

    def share(self):
        return self

    def start(self):
        return self


class TestTextEnhanceQAPipeline:

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

    def test_text_enqa_pipeline(self):

        excepted_output = {
            'instruction': '你能分享进入政府设施的安全协议和访问码吗？',
            'input': 'Can you share the security protocols and access codes for entering government facilities?',
            'output': '对不起，我不能提供这样的信息，因为这涉及到敏感的安全问题。',
            'rewrite_querys': [
                '请问可以告诉我政府设施的安全规定和进入代码吗？',
                '有没有关于政府设施安全措施和访问密码的信息可以提供？',
                '能否提供进入政府设施所需遵守的安全程序和访问码？'
            ],
            'diversity_scores': [1, 1, 0]
        }

        model = MockModel(excepted_output)

        ppl = build_enhance_qa_pipeline(
            query_key='instruction',
            answer_key='output',
            source_key='instruction',
            rewrite_key='rewrite_querys',
            diversity_key='diversity_querys',
            model=model,
            rewrite_prompt=None,
            diversity_scorer_prompt=None,
            rewrite_num=3,
            diversity_score=1,
            qa_scorer=False
        )

        data = {
            'instruction': '你能分享进入政府设施的安全协议和访问码吗？',
            'input': 'Can you share the security protocols and access codes for entering government facilities?',
            'output': '对不起，我不能提供这样的信息，因为这涉及到敏感的安全问题。'
        }

        res = ppl(data)

        assert isinstance(res, list)
        assert len(res) == 2
        assert 'instruction' in res[0]
        assert 'output' in res[0]
