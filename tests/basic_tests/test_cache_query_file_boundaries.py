import hashlib

import pytest

from lazyllm.components.formatter.formatterbase import encode_query_with_filepaths, file_content_hash
from lazyllm.module.module import CacheNotFoundError, ModuleCache


def write_file(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


@pytest.mark.parametrize('kind', ['query', 'files', 'empty', 'missing'])
def test_distinct_requests_do_not_share_cached_results(tmp_path, kind):
    if kind == 'query':
        left = encode_query_with_filepaths('ab', [write_file(tmp_path, 'left', b'c')])
        right = encode_query_with_filepaths('a', [write_file(tmp_path, 'right', b'bc')])
    elif kind == 'files':
        left = encode_query_with_filepaths('q', [
            write_file(tmp_path, 'left1', b'ab'), write_file(tmp_path, 'left2', b'c'),
        ])
        right = encode_query_with_filepaths('q', [
            write_file(tmp_path, 'right1', b'a'), write_file(tmp_path, 'right2', b'bc'),
        ])
    elif kind == 'empty':
        first = write_file(tmp_path, 'first', b'content')
        left = encode_query_with_filepaths('q', [first])
        right = encode_query_with_filepaths('q', [first, write_file(tmp_path, 'empty', b'')])
    else:
        missing = str(tmp_path / 'missing')
        left = encode_query_with_filepaths('q', [missing])
        right = encode_query_with_filepaths('q', [write_file(tmp_path, 'path-content', missing.encode())])
    cache = ModuleCache('memory')
    try:
        cache.set('output', (left,), {}, 'left result')
        with pytest.raises(CacheNotFoundError):
            cache.get('output', (right,), {})
        cache.set('output', (right,), {}, 'right result')
        assert cache.get('output', (left,), {}) == 'left result'
        assert cache.get('output', (right,), {}) == 'right result'
    finally:
        cache.close()


def test_equal_content_remains_path_independent(tmp_path):
    left = encode_query_with_filepaths('q', [write_file(tmp_path, 'left', b'same')])
    right = encode_query_with_filepaths('q', [write_file(tmp_path, 'right', b'same')])
    assert file_content_hash(left) == file_content_hash(right)


def test_attachment_order_and_content_changes(tmp_path):
    first = write_file(tmp_path, 'first', b'one')
    second = write_file(tmp_path, 'second', b'two')
    request = encode_query_with_filepaths('q', [first, second])
    initial = file_content_hash(request)
    assert initial != file_content_hash(encode_query_with_filepaths('q', [second, first]))
    write_file(tmp_path, 'first', b'changed')
    assert initial != file_content_hash(request)


def test_plain_string_and_file_hashes_unchanged(tmp_path):
    assert file_content_hash('plain query') == hashlib.md5(b'plain query').hexdigest()
    assert file_content_hash(write_file(tmp_path, 'file', b'contents')) == hashlib.md5(b'contents').hexdigest()
