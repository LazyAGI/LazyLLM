from collections import Counter
import copy
import json
import random

import pytest

from lazyllm.tools.data import embedding


def samples(sizes):
    return [{'id': f'{group}-{i}', 'kind': group, 'query': f'query {group} {i}'}
            for group, size in enumerate(sizes) for i in range(size)]


@pytest.mark.parametrize('seed', [0, 1, 2, 3])
def test_stratify_preserves_exact_class_proportions(seed):
    rows = samples([5, 15])
    operation = embedding.EmbeddingTrainTestSplitter(test_size=0.2, seed=seed, stratify_key='kind', _save_data=False)
    result = operation(copy.deepcopy(rows))
    assert Counter(row['kind'] for row in result if row['split'] == 'test') == {0: 1, 1: 3}
    assert sorted(row['id'] for row in result) == sorted(row['id'] for row in rows)
    assert result == operation(copy.deepcopy(rows))


def test_stratified_rounding_keeps_global_split_size():
    rows = samples([3, 3, 3])
    result = embedding.EmbeddingTrainTestSplitter(test_size=0.2, stratify_key='kind', _save_data=False)(rows)
    train = [row for row in result if row['split'] == 'train']
    assert len(train) == int(9 * 0.8)
    counts = Counter(row['kind'] for row in train)
    assert sorted(counts.values()) == [2, 2, 3]


def test_missing_key_is_a_stratum():
    rows = samples([5, 5])
    for row in rows[:5]:
        del row['kind']
    result = embedding.EmbeddingTrainTestSplitter(test_size=0.2, stratify_key='kind', _save_data=False)(rows)
    assert Counter(row.get('kind') for row in result if row['split'] == 'test') == {None: 1, 1: 1}


def test_unstratified_shuffle_matches_existing_order():
    rows = samples([5, 15])
    expected = copy.deepcopy(rows)
    random.Random(42).shuffle(expected)
    result = embedding.EmbeddingTrainTestSplitter(test_size=0.2, _save_data=False)(rows)
    assert [row['id'] for row in result] == [row['id'] for row in expected]
    assert [row['split'] for row in result] == ['train'] * 16 + ['test'] * 4


def test_stratified_jsonl_exports_and_empty_input(tmp_path):
    train, test = tmp_path / 'train.jsonl', tmp_path / 'test.jsonl'
    operation = embedding.EmbeddingTrainTestSplitter(
        test_size=0.2, stratify_key='kind', train_output_file=str(train), test_output_file=str(test), _save_data=False,
    )
    rows = samples([5, 15])
    operation(copy.deepcopy(rows))
    exported_train = [json.loads(line) for line in train.read_text().splitlines()]
    exported_test = [json.loads(line) for line in test.read_text().splitlines()]
    assert Counter(row['kind'] for row in exported_test) == {0: 1, 1: 3}
    assert all('split' not in row for row in exported_train + exported_test)
    assert sorted(row['id'] for row in exported_train + exported_test) == sorted(row['id'] for row in rows)
    assert operation([]) == []
