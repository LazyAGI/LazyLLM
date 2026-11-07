import pytest

from lazyllm.tools.rag.dataReader import SimpleDirectoryReader


@pytest.mark.parametrize('workers', [None, 1])
@pytest.mark.parametrize('metadata_kind', ['global', 'per_file', 'none'])
def test_metadata_shapes_preserve_all_input_files(tmp_path, workers, metadata_kind):
    paths = [tmp_path / 'first.txt', tmp_path / 'second.txt']
    for index, path in enumerate(paths):
        path.write_text('document-' + str(index))
    metadata = {'category': 'shared'} if metadata_kind == 'global' else (
        [{'category': 'first'}, {'category': 'second'}] if metadata_kind == 'per_file' else None)
    reader = SimpleDirectoryReader(input_files=[str(path) for path in paths], metadatas=metadata)
    nodes = reader(num_workers=workers)
    assert [node.get_text() for node in nodes] == ['document-0', 'document-1']
    expected = ['shared', 'shared'] if metadata_kind == 'global' else (
        ['first', 'second'] if metadata_kind == 'per_file' else [None, None])
    assert [node.global_metadata.get('category') for node in nodes] == expected
    if metadata_kind == 'global':
        assert metadata == {'category': 'shared'}


@pytest.mark.parametrize('workers', [None, 1])
def test_call_time_global_metadata_overrides_constructor(tmp_path, workers):
    paths = [tmp_path / 'first.txt', tmp_path / 'second.txt']
    for path in paths:
        path.write_text('document')
    reader = SimpleDirectoryReader(input_files=[str(path) for path in paths], metadatas={'category': 'default'})
    nodes = reader(num_workers=workers, metadatas={'category': 'override'})
    assert len(nodes) == 2
    assert all(node.global_metadata['category'] == 'override' for node in nodes)
