import json

import pytest

import lazyllm
from lazyllm.tools.data.base_data import LazyLLMDataBase
from lazyllm.tools.data.pipelines.kc_pipelines import build_multihop_qa_pipeline


@pytest.mark.parametrize('extension', ['json', 'jsonl'])
@pytest.mark.parametrize('field', ['cleaned_chunk', 'custom_text'])
def test_pipeline_preprocess_uses_selected_field(tmp_path, extension, field):
    text = ('This is the selected premise with sufficiently detailed information. '
            'This is the selected intermediate fact with enough useful context. '
            'This is the selected conclusion derived from both facts.')
    decoy = 'This unrelated default field should never be selected. ' * 4
    record = {'cleaned_chunk': decoy, field: text}
    path = tmp_path / ('chunks.' + extension)
    path.write_text(json.dumps([record]) if extension == 'json' else json.dumps(record) + chr(10))
    with lazyllm.config.temp('data_process_path', str(tmp_path / 'results')):
        operation = build_multihop_qa_pipeline(input_key='source_path', ext_field=field, llm=None)
        prefix = lazyllm.pipeline(operation.load, operation.preprocess, operation.extract)
        prefix.for_each(lambda stage: isinstance(stage, LazyLLMDataBase),
                        lambda stage: setattr(stage, '_max_workers', 1))
        result = prefix([{'source_path': str(path)}])
    assert result[0]['_processed_chunks'][0]['text'] == text
    assert result[0]['_info_pairs'][0]['premise'].startswith('This is the selected premise')
    assert result[0]['_processed_chunks'][0]['original_data'] == record


def test_missing_selected_field_does_not_fall_back_to_default(tmp_path):
    path = tmp_path / 'chunks.json'
    path.write_text(json.dumps([{'cleaned_chunk': 'Unrelated content. ' * 20}]))
    with lazyllm.config.temp('data_process_path', str(tmp_path / 'results')):
        operation = build_multihop_qa_pipeline(ext_field='custom_text', llm=None)
        prefix = lazyllm.pipeline(operation.load, operation.preprocess, operation.extract)
        prefix.for_each(lambda stage: isinstance(stage, LazyLLMDataBase),
                        lambda stage: setattr(stage, '_max_workers', 1))
        result = prefix([{'chunk_path': str(path)}])
    assert result[0]['_processed_chunks'] == []
    assert result[0]['_info_pairs'] == []
