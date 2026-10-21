import pytest

from lazyllm.tools.data import domain_finetune


def record(format, system=None, multiple=False):
    turns = [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'answer'}]
    if multiple:
        turns += [{'role': 'user', 'content': 'follow up'}]
    if system is not None:
        turns.insert(0, {'role': 'system', 'content': system})
    if format == 'messages':
        return {'messages': turns}
    return {'conversations': [{'from': turn['role'], 'value': turn['content']} for turn in turns]}


def instruction(result):
    content = result['content']
    return content['messages'][0]['content'] if 'messages' in content else content['instruction']


@pytest.mark.parametrize('format', ['messages', 'conversations'])
@pytest.mark.parametrize('mode', ['single', 'thread'])
@pytest.mark.parametrize('multiple', [False, True])
def test_same_instance_keeps_system_instruction_per_record(format, mode, multiple):
    operation = domain_finetune.DatasetFormatNormalizer(
        instruction='configured default', _concurrency_mode=mode, _max_workers=1, _save_data=False,
    )
    result = operation([record(format, 'record-specific', multiple), record(format, multiple=multiple)])
    assert [instruction(row) for row in result] == ['record-specific', 'configured default']
    assert operation.instruction == 'configured default'
    later = operation([{'question': 'another question', 'answer': 'another answer'}])
    assert instruction(later[0]) == 'configured default'


@pytest.mark.parametrize('format', ['messages', 'conversations'])
def test_keep_system_false_uses_default(format):
    operation = domain_finetune.DatasetFormatNormalizer(
        instruction='configured default', keep_system=False, _concurrency_mode='single', _save_data=False,
    )
    assert instruction(operation([record(format, 'ignored')])[0]) == 'configured default'


@pytest.mark.parametrize('format', ['messages', 'conversations'])
def test_process_mode_behavior_unchanged(format):
    operation = domain_finetune.DatasetFormatNormalizer(
        instruction='configured default', _concurrency_mode='process', _max_workers=1, _save_data=False,
    )
    result = operation([record(format, 'record-specific'), record(format)])
    assert [instruction(row) for row in result] == ['record-specific', 'configured default']
