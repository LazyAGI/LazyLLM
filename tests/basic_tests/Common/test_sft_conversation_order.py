import json

import pytest
from datasets import Dataset, DatasetDict

import lazyllm
from lazyllm.module.llms.utils import INPUT_SPLIT, alpaca2openai, openai2alpaca, uniform_sft_dataset


def conversation(turns):
    messages = [{'role': 'system', 'content': 'system'}]
    pairs = [[f'question {index}{INPUT_SPLIT}input {index}', f'answer {index}'] for index in range(turns)]
    for question, answer in pairs:
        messages.extend([{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}])
    row = {'system': 'system', 'instruction': f'question {turns - 1}',
           'input': f'input {turns - 1}', 'output': f'answer {turns - 1}'}
    if turns > 1:
        row['history'] = pairs[:-1]
    return messages, row


@pytest.mark.parametrize('turns', [1, 2, 3])
def test_openai_current_turn_follows_its_history(turns):
    messages, expected = conversation(turns)
    data = DatasetDict(train=Dataset.from_list([{'messages': messages}]))
    assert openai2alpaca(data) == [expected]


@pytest.mark.parametrize('turns', [1, 2, 3])
def test_alpaca_history_precedes_current_turn(turns):
    expected, row = conversation(turns)
    data = DatasetDict(train=Dataset.from_list([row]))
    assert alpaca2openai(data) == [{'messages': expected}]


@pytest.mark.parametrize('suffix', ['json', 'jsonl'])
def test_actual_local_file_conversion_preserves_turn_order(tmp_path, suffix):
    messages, expected = conversation(3)
    source = tmp_path / f'conversation.{suffix}'
    row = {'messages': messages}
    source.write_text(json.dumps([row] if suffix == 'json' else row), encoding='utf-8')
    with lazyllm.config.temp('temp_dir', str(tmp_path / 'converted')):
        alpaca = uniform_sft_dataset(str(source), 'alpaca')
        assert json.loads(open(alpaca, encoding='utf-8').read()) == [expected]
        openai = uniform_sft_dataset(alpaca, 'openai')
        assert json.loads(open(openai, encoding='utf-8').read()) == {'messages': messages}
