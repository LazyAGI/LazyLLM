import os

import pytest

from lazyllm.tools.data.operators.token_chunker import TokenChunker


@pytest.fixture
def tokenizer_path():
    pytest.importorskip('transformers')
    return os.environ.get('LAZYLLM_TEST_TOKENIZER_PATH', 'Qwen/Qwen2.5-0.5B-Instruct')


@pytest.mark.parametrize('max_tokens', [16, 32])
@pytest.mark.parametrize('text', [
    'alpha beta gamma delta ' * 40,
    '人工智能与自然语言处理' * 40,
    '😀😎🚀🌍' * 30,
    'short paragraph.' + chr(10) * 2 + 'long unpunctuated text ' * 40 + '. ending.',
    'first sentence. second sentence! third sentence? ' * 30,
])
def test_long_chunks_obey_budget_and_preserve_text(tokenizer_path, max_tokens, text):
    operation = TokenChunker(model_path=tokenizer_path, max_tokens=max_tokens, min_tokens=0,
                             _concurrency_mode='single', _save_data=False)
    rows = operation([{'content': text, 'meta_data': {'source': 'fixture'}}])
    assert len(rows) > 1
    assert ''.join(row['content'] for row in rows) == text
    for index, row in enumerate(rows):
        assert len(operation.tokenizer.encode(row['content'])) <= max_tokens
        assert row['meta_data'] == {'source': 'fixture', 'index': index, 'total': len(rows),
                                    'length': len(row['content'])}


def test_short_text_stays_unchanged(tokenizer_path):
    operation = TokenChunker(model_path=tokenizer_path, max_tokens=16, min_tokens=0,
                             _concurrency_mode='single', _save_data=False)
    rows = operation([{'content': 'hello world'}])
    assert [row['content'] for row in rows] == ['hello world']


def test_too_small_budget_does_not_corrupt_unicode(tokenizer_path):
    operation = TokenChunker(model_path=tokenizer_path, max_tokens=1, min_tokens=0,
                             _concurrency_mode='single', _save_data=False)
    character = chr(0x10FFFF)
    assert len(operation.tokenizer.encode(character)) > 1
    with pytest.raises(ValueError, match='complete character'):
        operation.forward({'content': character})
