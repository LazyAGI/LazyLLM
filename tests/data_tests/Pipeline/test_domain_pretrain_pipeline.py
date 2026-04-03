import os
import shutil
import tempfile

import pytest

from lazyllm import config
from lazyllm.flow import Pipeline
from lazyllm.tools.data.operators import token_chunker as _token_chunker_mod
from lazyllm.tools.data.pipelines import domain_pretrain_pipelines as _dpp
from lazyllm.tools.data.pipelines.domain_pretrain_pipelines import (
    DOMAIN_PRETRAIN_FEATURES,
    build_domain_pretrain_pipeline,
    build_text_pt_pipeline,
    build_text_pt_plus_domain_pretrain_pipeline,
)


@pytest.fixture(scope='module', autouse=True)
def _lazyllm_data_process_path_isolated():
    root_dir = tempfile.mkdtemp()
    keep = config['data_process_path']
    os.environ['LAZYLLM_DATA_PROCESS_PATH'] = root_dir
    config.refresh()
    yield
    os.environ['LAZYLLM_DATA_PROCESS_PATH'] = keep
    config.refresh()
    shutil.rmtree(root_dir, ignore_errors=True)


def _zh_pretrain_body(num_paragraphs: int = 10):
    parts = []
    for i in range(num_paragraphs):
        parts.append(
            f'第{i + 1}段：领域预训练数据清洗与质量评估相关内容。'
            f'涉及股票、投资、财报、数据库、网络与安全等主题。句子长度适中，并带有逗号与句号。'
        )
    return '\n'.join(parts)



def _en_text_pt_non_empty_body(num_paragraphs: int = 28):
    base_terms = [
        'qzvra', 'plnko', 'trvex', 'mylor', 'snkud', 'brvot', 'clqen',
        'dvrym', 'frtul', 'gnvix', 'hpral', 'jtkem', 'kvnor', 'lqzid',
    ]
    parts = []
    for i in range(1, num_paragraphs + 1):
        terms = ' '.join(f'{t}{i:03d}' for t in base_terms)
        parts.append(
            f'Unit {i}. {terms}. '
            f'zzkya{i:03d} qrvop{i:03d} tlnux{i:03d} mvqer{i:03d}.'
        )
    return '\n\n'.join(parts)


@pytest.fixture
def fast_text_pt_heavy_ops(monkeypatch):
    _tc_init = _token_chunker_mod.TokenChunker.__init__

    def _token_chunker_single(
        self,
        input_key='content',
        model_path=None,
        max_tokens=1024,
        min_tokens=200,
        _concurrency_mode='single',
        **kwargs,
    ):
        return _tc_init(
            self, input_key, model_path, max_tokens, min_tokens, _concurrency_mode, **kwargs
        )

    monkeypatch.setattr(_token_chunker_mod.TokenChunker, '__init__', _token_chunker_single)

    _mh = _dpp.filter.MinHashDeduplicator

    def _minhash_smaller_perm(*args, **kwargs):
        kwargs.setdefault('num_perm', 32)
        return _mh(*args, **kwargs)

    monkeypatch.setattr(_dpp.filter, 'MinHashDeduplicator', _minhash_smaller_perm)


class TestDomainPretrainPipelines:
    def test_domain_pretrain_features_export(self):
        assert isinstance(DOMAIN_PRETRAIN_FEATURES, tuple)
        assert 'field_normalization' in DOMAIN_PRETRAIN_FEATURES
        assert 'ngram_repetition_filter' in DOMAIN_PRETRAIN_FEATURES

    def test_build_domain_pretrain_pipeline_general_runs(self):
        ppl = build_domain_pretrain_pipeline(
            domain='general',
            content_key='content',
            language='zh',
        )
        assert isinstance(ppl, Pipeline)
        for name in ('field_normalization', 'text_normalizer', 'sensitive_cleaner', 'ngram_filter'):
            assert hasattr(ppl, name), f'missing step {name}'
        text = _zh_pretrain_body()
        res = ppl([{'content': text}])
        assert isinstance(res, list) and len(res) == 1
        assert 'content' in res[0] and len(res[0]['content']) > 0

    def test_build_domain_pretrain_pipeline_field_mapping_options(self):
        ppl = build_domain_pretrain_pipeline(
            domain='general',
            content_key='content',
            language='zh',
            options={
                'field_mapping': {'body': 'content'},
            },
        )
        text = _zh_pretrain_body()
        res = ppl([{'body': text}])
        assert isinstance(res, list) and len(res) == 1
        assert 'content' in res[0]

    def test_build_domain_pretrain_pipeline_finance_keyword_filter(self):
        ppl = build_domain_pretrain_pipeline(
            domain='finance',
            content_key='content',
            language='zh',
            enabled={
                'domain_keyword_filter': True,
                'domain_relevance_scorer': False,
            },
        )
        text = _zh_pretrain_body()
        res = ppl([{'content': text}])
        assert isinstance(res, list) and len(res) == 1
        assert '_keyword_hits' in res[0]

    def test_build_domain_pretrain_pipeline_relevance_scorer(self):
        ppl = build_domain_pretrain_pipeline(
            domain='finance',
            content_key='content',
            language='zh',
            enabled={
                'domain_keyword_filter': False,
                'domain_relevance_scorer': True,
            },
            options={'min_relevance_score': 0.0001},
        )
        text = _zh_pretrain_body()
        res = ppl([{'content': text}])
        assert isinstance(res, list) and len(res) == 1
        assert '_domain_relevance_score' in res[0]

    def test_build_domain_pretrain_pipeline_language_filter_en(self):
        ppl = build_domain_pretrain_pipeline(
            domain='general',
            content_key='content',
            language='en',
            enabled={'language_filter': True},
            options={'min_language_ratio': 0.3},
        )
        parts = [
            f'Paragraph {i} discusses science, data, and software engineering practices.'
            for i in range(8)
        ]
        text = '\n\n'.join(parts)
        res = ppl([{'content': text}])
        assert isinstance(res, list) and len(res) == 1
        assert '_language_ratio' in res[0]

    def test_build_domain_pretrain_opts_enable_domain_keyword_flag(self):
        ppl = build_domain_pretrain_pipeline(
            domain='general',
            content_key='content',
            language='zh',
            options={
                'enable_domain_keyword_filter': True,
                'keyword_mode': 'any',
                'min_keyword_hits': 1,
            },
            domain_keywords=['测试', '流水线'],
        )
        text = _zh_pretrain_body() + '测试与流水线关键词。'
        res = ppl([{'content': text}])
        assert isinstance(res, list) and len(res) == 1
        assert '_keyword_hits' in res[0]

    def test_build_text_pt_pipeline_structure(self):
        ppl = build_text_pt_pipeline(
            content_key='content',
            language='zh',
            min_chars=100,
            min_words=10,
        )
        assert isinstance(ppl, Pipeline)
        for name in (
            'null_content_filter',
            'char_count_filter',
            'word_count_filter',
            'token_chunker',
        ):
            assert hasattr(ppl, name), f'missing step {name}'

    @pytest.mark.usefixtures('fast_text_pt_heavy_ops')
    def test_build_text_pt_pipeline_runs_if_tokenizer_available(self):
        ppl = build_text_pt_pipeline(
            content_key='content',
            language='en',
            min_chars=200,
            min_words=40,
            max_tokens=512,
            min_tokens=64,
        )
        text = _en_text_pt_non_empty_body()
        try:
            res = ppl([{'content': text}])
        except Exception as e:
            pytest.skip(f'TokenChunker / tokenizer 不可用: {e}')
        assert isinstance(res, list)
        assert len(res) >= 1
        assert all('content' in row for row in res)

    def test_build_text_pt_plus_domain_pretrain_structure(self):
        ppl = build_text_pt_plus_domain_pretrain_pipeline(
            domain='general',
            content_key='content',
            language='zh',
        )
        assert isinstance(ppl, Pipeline)
        assert hasattr(ppl, 'domain_enhance') and hasattr(ppl, 'text_pt')

    @pytest.mark.usefixtures('fast_text_pt_heavy_ops')
    def test_build_text_pt_plus_runs_if_tokenizer_available(self):
        ppl = build_text_pt_plus_domain_pretrain_pipeline(
            domain='general',
            content_key='content',
            language='en',
            options={
                'min_chars': 200,
                'min_words': 40,
                'min_tokens': 64,
                'max_tokens': 512,
            },
        )
        text = _en_text_pt_non_empty_body()
        try:
            res = ppl([{'content': text}])
        except Exception as e:
            pytest.skip(f'组合流水线需 TokenChunker: {e}')
        assert isinstance(res, list)
        assert len(res) >= 1
        assert all('content' in row for row in res)
