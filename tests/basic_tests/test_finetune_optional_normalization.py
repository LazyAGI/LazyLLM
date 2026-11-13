import pytest

import lazyllm
from lazyllm.tools.data.base_data import LazyLLMDataBase
from lazyllm.tools.data.pipelines.domain_finetune_pipelines import build_domain_finetune_pipeline


def limit_workers(operation):
    operation.for_each(lambda stage: isinstance(stage, LazyLLMDataBase),
                       lambda stage: setattr(stage, '_max_workers', 1))
    return operation


@pytest.mark.parametrize('enabled', [None, {}, {'normalization': False}, {'deduplication': True}])
def test_raw_text_uses_cleaning_without_normalization(tmp_path, enabled):
    with lazyllm.config.temp('data_process_path', str(tmp_path)):
        operation = limit_workers(build_domain_finetune_pipeline(
            language='en', enabled=enabled, instruction='configured',
            filters_config=[{'type': 'char_count', 'min_chars': 1}],
        ))
        result = operation([{'content': '  useful  text 😀  '}])
    assert len(result) == 1
    assert result[0]['formatted_text'] == {'instruction': 'configured', 'input': '', 'output': 'useful text'}


@pytest.mark.parametrize('flag', ['normalization', 'adaptive_normalization'])
def test_structured_normalization_still_preserves_pair(tmp_path, flag):
    with lazyllm.config.temp('data_process_path', str(tmp_path)):
        operation = limit_workers(build_domain_finetune_pipeline(
            language='en', enabled={flag: True}, instruction='configured',
            normalization_instruction='normalized', filters_config=[{'type': 'char_count', 'min_chars': 1}],
        ))
        result = operation([{'question': 'question', 'answer': 'answer'}])
    assert result[0]['formatted_text'] == {'instruction': 'normalized', 'input': 'question', 'output': 'answer'}


def test_default_builder_keeps_valid_long_raw_text(tmp_path):
    text = 'This document contains useful training material with many distinct words and complete sentences. ' * 20
    with lazyllm.config.temp('data_process_path', str(tmp_path)):
        operation = limit_workers(build_domain_finetune_pipeline(language='en'))
        result = operation([{'content': text}])
    assert len(result) == 1
    assert result[0]['formatted_text']['output'] == text.strip()
