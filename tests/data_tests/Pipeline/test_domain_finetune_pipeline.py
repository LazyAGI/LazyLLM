import json
import os
import shutil
import tempfile

import pytest

from lazyllm import config
from lazyllm.flow import Pipeline
from lazyllm.tools.data import embedding as _data_embedding
from lazyllm.tools.data.base_data import LazyLLMDataBase
from lazyllm.tools.data.pipelines.domain_finetune_pipelines import (
    DOMAIN_FINETUNE_FEATURES,
    build_data_augmentation_pipeline,
    build_domain_finetune_pipeline,
    build_domain_formatting_pipeline,
    build_llm_extraction_pipeline,
    build_quality_filter_pipeline,
    build_source_to_content_pipeline,
    build_text_cleaning_pipeline,
    build_train_test_split_pipeline,
)


@pytest.fixture(scope='module', autouse=True)
def _lazyllm_data_process_path_isolated():
    """整文件只建一次临时目录并 refresh config，避免每个用例重复 mkdtemp/rmtree。"""
    root_dir = tempfile.mkdtemp()
    keep = config['data_process_path']
    os.environ['LAZYLLM_DATA_PROCESS_PATH'] = root_dir
    config.refresh()
    yield
    os.environ['LAZYLLM_DATA_PROCESS_PATH'] = keep
    config.refresh()
    shutil.rmtree(root_dir, ignore_errors=True)


class SimpleMockLLM:
    def __init__(self, response):
        self._response = response
        self._serve = self

    def share(self, prompt=None, format=None, stream=None, history=None):
        return self._serve

    def prompt(self, system_prompt):
        return self._serve

    def start(self):
        return self._serve

    def __call__(self, prompt):
        return self._response


class TestDomainFinetunePipelines:
    def test_domain_finetune_features_export(self):
        assert isinstance(DOMAIN_FINETUNE_FEATURES, tuple)
        assert 'normalization' in DOMAIN_FINETUNE_FEATURES
        assert 'deduplication' in DOMAIN_FINETUNE_FEATURES

    def test_build_text_cleaning_pipeline_runs(self):
        ppl = build_text_cleaning_pipeline(
            input_key='content',
            output_key='cleaned_content',
            remove_emoji=True,
            remove_extra_spaces=True,
            remove_html_url=False,
            remove_html_entity=False,
        )
        assert isinstance(ppl, Pipeline)
        data = [{'content': '  hello 😊 world  '}]
        res = ppl(data)
        assert isinstance(res, list) and len(res) == 1
        assert 'cleaned_content' in res[0]
        assert '😊' not in res[0]['cleaned_content']

    def test_build_quality_filter_pipeline_default(self):
        ppl = build_quality_filter_pipeline(input_key='content')
        assert isinstance(ppl, Pipeline)
        text = ' '.join([f'word{i}' for i in range(15)])
        res = ppl([{'content': text}])
        assert isinstance(res, list) and len(res) == 1

    def test_build_quality_filter_pipeline_skips_unknown_type(self):
        cfg = [
            {'type': 'unknown_filter_type___'},
            {'type': 'char_count', 'min_chars': 3, 'max_chars': 100000},
        ]
        ppl = build_quality_filter_pipeline(filters_config=cfg)
        res = ppl([{'content': 'hello world'}])
        assert isinstance(res, list) and len(res) == 1

    def test_build_domain_formatting_pipeline_alpaca(self):
        ppl = build_domain_formatting_pipeline(
            domain='general',
            input_key='content',
            output_key='formatted_text',
            output_format='alpaca',
        )
        res = ppl([{'content': 'plain body'}])
        assert isinstance(res, list) and len(res) == 1
        ft = res[0]['formatted_text']
        assert ft['instruction'] and ft['output'] == 'plain body'

    def test_build_domain_formatting_pipeline_sharegpt(self):
        ppl = build_domain_formatting_pipeline(output_format='sharegpt')
        res = ppl([{'content': 'hi'}])
        msgs = res[0]['formatted_text']['messages']
        roles = [m['role'] for m in msgs]
        assert 'system' in roles and 'user' in roles

    def test_build_llm_extraction_pipeline_with_mock(self):
        ans = '答' * 40
        payload = {'qa_pairs': [{'question': '这是一个用于抽取的测试问题？', 'answer': ans}]}
        mock = SimpleMockLLM(json.dumps(payload, ensure_ascii=False))
        ppl = build_llm_extraction_pipeline(
            llm=mock,
            input_key='content',
            output_key='content',
            extract_format='qa',
            lang='zh',
        )
        src = '领域文档正文内容，用于触发抽取。' * 5
        res = ppl([{'content': src}])
        assert isinstance(res, list) and len(res) >= 1
        assert 'content' in res[0]

    def test_build_data_augmentation_pipeline_synonym_replace(self, monkeypatch):
        # EmbeddingAdjacentWordSwap 默认 _concurrency_mode='process'，单条数据也会起进程池；测例改为 single 覆盖逻辑即可。
        def _fast_init(self, num_augments=2, input_key='query', **kwargs):
            LazyLLMDataBase.__init__(self, _concurrency_mode='single', **kwargs)
            self.num_augments = num_augments
            self.input_key = input_key

        monkeypatch.setattr(_data_embedding.EmbeddingAdjacentWordSwap, '__init__', _fast_init)
        ppl = build_data_augmentation_pipeline(
            input_key='content',
            augment_methods=['synonym_replace'],
            num_augments=2,
            lang='en',
        )
        res = ppl([{'content': 'one two three four five six'}])
        assert isinstance(res, list)
        assert len(res) >= 1

    def test_build_train_test_split_pipeline(self):
        ppl = build_train_test_split_pipeline(
            train_ratio=0.7,
            validation_ratio=0.2,
            test_ratio=0.1,
            seed=1,
        )
        items = [{'id': i, 'content': f'c{i}'} for i in range(20)]
        out = ppl(items)
        assert isinstance(out, dict)
        assert set(out.keys()) == {'train', 'validation', 'test'}
        assert len(out['train']) + len(out['validation']) + len(out['test']) == 20

    def test_build_source_to_content_pipeline_structure(self):
        ppl = build_source_to_content_pipeline(
            source_key='source',
            output_key='content',
            mineru_url='',
        )
        assert isinstance(ppl, Pipeline)
        for name in ('normalize', 'convert_html', 'convert_pdf', 'prepare_load', 'load_text', 'rename'):
            assert hasattr(ppl, name), f'missing step {name}'

    def test_build_domain_finetune_pipeline_normalization_only(self):
        ppl = build_domain_finetune_pipeline(
            domain='general',
            input_key='content',
            output_key='formatted_text',
            enabled={'normalization': True},
            language='en',
        )
        assert isinstance(ppl, Pipeline)
        words = ' '.join([f'w{i}' for i in range(20)])
        item = {
            'content': {
                'instruction': 'You are a helpful assistant.',
                'input': 'question here',
                'output': words + ' answer body ' + ('x' * 120),
            },
        }
        res = ppl([item])
        assert isinstance(res, list) and len(res) == 1
        assert 'formatted_text' in res[0]

    def test_build_domain_finetune_pipeline_merge_context(self):
        ppl = build_domain_finetune_pipeline(
            domain='general',
            input_key='content',
            output_key='formatted_text',
            enabled={'normalization': True},
            language='en',
            options={
                'merge_context_question': True,
                'merge_context_question_params': {
                    'question_key': 'question',
                    'context_key': 'context',
                    'target_key': 'question',
                    'context_label': 'Context',
                    'question_label': 'Question',
                },
            },
        )
        words_ctx = ' '.join([f'ctx{i}' for i in range(25)])
        words_in = ' '.join([f'in{i}' for i in range(15)])
        words_out = ' '.join([f'ans{i}' for i in range(25)]) + ' ' + ('y' * 120)
        item = {
            'question': 'what is the topic?',
            'context': words_ctx,
            'content': {
                'instruction': 'You are a helpful assistant.',
                'input': words_in,
                'output': words_out,
            },
        }
        res = ppl([item])
        assert isinstance(res, list) and len(res) == 1
        assert 'formatted_text' in res[0]

    def test_build_domain_finetune_pipeline_conversation_expand(self):
        long_a = '回答内容' * 30
        ppl = build_domain_finetune_pipeline(
            domain='general',
            input_key='content',
            output_key='formatted_text',
            enabled={'normalization': True, 'conversation_expand': True},
            language='zh',
            options={
                'expand_list_key': 'data',
                'expand_min_q_chars': 4,
                'expand_min_a_chars': 20,
            },
        )
        item = {
            'data': ['问：这是一个足够长的问题吗？', f'答：{long_a}'],
            'content': {
                'instruction': '占位',
                'input': '占位',
                'output': '占位输出' * 30,
            },
        }
        res = ppl([item])
        assert isinstance(res, list)
        assert len(res) >= 1

    @pytest.mark.parametrize('dedup_method', ['hash', 'minhash'])
    def test_build_domain_finetune_pipeline_deduplication(self, dedup_method):
        opts = {'dedup_method': dedup_method}
        if dedup_method == 'minhash':
            opts['minhash_num_perm'] = 32
        ppl = build_domain_finetune_pipeline(
            domain='general',
            input_key='content',
            output_key='formatted_text',
            enabled={'normalization': True, 'deduplication': True},
            language='en',
            options=opts,
        )
        words = ' '.join([f'w{i}' for i in range(15)])
        dup_body = words + ' ' + ('z' * 100)
        item = {
            'content': {
                'instruction': 'You are a helpful assistant.',
                'input': 'q',
                'output': dup_body,
            },
        }
        res = ppl([item, item])
        assert isinstance(res, list)
        if dedup_method == 'hash':
            assert len(res) == 1
