import pytest

from lazyllm.tools.data import filter


@pytest.mark.parametrize('texts', [['a', 'b'], ['cat', 'dog'], ['中文', '文本'], ['😀', '😎']])
def test_distinct_short_texts_survive_and_duplicates_are_removed(texts):
    operation = filter.MinHashDeduplicator(_save_data=False)
    rows = [{'content': text, 'id': i} for i, text in enumerate(texts + [texts[0]])]
    assert operation(rows) == rows[:2]


def test_custom_ngram_and_cross_batch_state():
    operation = filter.MinHashDeduplicator(input_key='text', ngram=8, _save_data=False)
    first = {'text': 'first'}
    second = {'text': 'second'}
    assert operation([first]) == [first]
    assert operation([first, second]) == [second]


@pytest.mark.parametrize('use_n_gram', [True, False])
def test_existing_long_text_deduplication(use_n_gram):
    operation = filter.MinHashDeduplicator(use_n_gram=use_n_gram, _save_data=False)
    first = {'content': 'abcdefghijklmno', 'id': 1}
    duplicate = {'content': first['content'], 'id': 2}
    assert operation([first, duplicate]) == [first]


def test_invalid_and_empty_rows_still_excluded():
    operation = filter.MinHashDeduplicator(_save_data=False)
    valid = {'content': 'valid'}
    assert operation([None, {}, {'content': ''}, {'content': '  '}, {'content': 1}, valid]) == [valid]
