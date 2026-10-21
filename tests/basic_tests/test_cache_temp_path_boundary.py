import os

import pytest

import lazyllm
from lazyllm.components.formatter.formatterbase import (
    decode_query_with_filepaths, encode_query_with_filepaths, path_absolute_to_relative,
)
from lazyllm.module.module import ModuleCache


@pytest.mark.parametrize('nested', [False, True])
def test_memory_cache_preserves_sibling_attachment_path(tmp_path, nested):
    root = tmp_path / 'cache'
    sibling = tmp_path / 'cache-sibling'
    root.mkdir()
    sibling.mkdir()
    attachment = sibling / 'result.txt'
    attachment.write_text('actual result')
    decoy = root / '-sibling' / 'result.txt'
    decoy.parent.mkdir()
    decoy.write_text('unrelated result')
    encoded = encode_query_with_filepaths('query', [str(attachment)])
    value = {'result': [encoded]} if nested else encoded
    with lazyllm.config.temp('temp_dir', str(root)):
        cache = ModuleCache('memory')
        try:
            cache.set('result', ('input',), {}, value)
            returned = cache.get('result', ('input',), {})
        finally:
            cache.close()
    result = returned['result'][0] if nested else returned
    decoded = decode_query_with_filepaths(result)
    assert decoded['files'] == [str(attachment)]
    assert open(decoded['files'][0]).read() == 'actual result'


@pytest.mark.parametrize('kind', ['inside', 'sibling', 'same', 'relative', 'root'])
def test_temp_path_boundaries(tmp_path, kind):
    root = tmp_path / 'cache'
    root.mkdir()
    path, expected = {
        'inside': (str(root / 'result.txt'), 'result.txt'),
        'sibling': (str(tmp_path / 'cache-sibling' / 'result.txt'),
                    str(tmp_path / 'cache-sibling' / 'result.txt')),
        'same': (str(root), ''),
        'relative': ('result.txt', 'result.txt'),
        'root': (str(root / 'result.txt'), str(root / 'result.txt').lstrip(os.sep)),
    }[kind]
    with lazyllm.config.temp('temp_dir', os.path.abspath(os.sep) if kind == 'root' else str(root)):
        assert path_absolute_to_relative(path) == expected
