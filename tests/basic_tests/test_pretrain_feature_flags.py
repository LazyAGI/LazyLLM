import copy

import pytest

import lazyllm
from lazyllm.tools.data.base_data import LazyLLMDataBase
from lazyllm.tools.data.pipelines.domain_pretrain_pipelines import (
    DOMAIN_PRETRAIN_FEATURES, build_domain_pretrain_pipeline,
)


CASES = [
    ('field_normalization', {'text': 'source'}, {'field_mapping': {'text': 'content'}},
     {'text': 'source', 'content': 'source'}),
    ('text_normalization', {'content': '  Ａ  B  '}, {}, {'content': 'A B'}),
    ('sensitive_info_cleaning', {'content': 'contact test@example.com'}, {},
     {'content': 'contact [REDACTED]'}),
    ('ngram_repetition_filter', {'content': 'alpha beta gamma'}, {'ngram_n': 2},
     {'content': 'alpha beta gamma', '_ngram_repetition_ratio': 0.0}),
]


@pytest.mark.parametrize('feature, row, options, transformed', CASES)
@pytest.mark.parametrize('enabled', [False, True])
def test_pretrain_feature_switches(tmp_path, feature, row, options, transformed, enabled):
    flags = dict.fromkeys(DOMAIN_PRETRAIN_FEATURES, False)
    flags[feature] = enabled
    with lazyllm.config.temp('data_process_path', str(tmp_path)):
        operation = build_domain_pretrain_pipeline(language='en', enabled=flags, options=options)
        operation.for_each(lambda stage: isinstance(stage, LazyLLMDataBase),
                           lambda stage: setattr(stage, '_max_workers', 1))
        result = operation([copy.deepcopy(row)])
    assert result == [transformed if enabled else row]


def test_default_stages_remain_enabled(tmp_path):
    with lazyllm.config.temp('data_process_path', str(tmp_path)):
        operation = build_domain_pretrain_pipeline(language='en')
        operation.for_each(lambda stage: isinstance(stage, LazyLLMDataBase),
                           lambda stage: setattr(stage, '_max_workers', 1))
        result = operation([{'text': '  Ａ test@example.com  '}])
    assert result == [{'text': '  Ａ test@example.com  ', 'content': 'A [REDACTED]',
                       '_ngram_repetition_ratio': 0.0}]
