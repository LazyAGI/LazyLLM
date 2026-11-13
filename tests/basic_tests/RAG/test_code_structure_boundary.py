import pytest

from lazyllm.tools.rag.doc_node import DocNode
from lazyllm.tools.rag.transform import GeneralCodeSplitter


@pytest.mark.parametrize('body_lines', [198, 199, 200, 201, 405])
@pytest.mark.parametrize('blank_line', [False, True])
def test_structure_threshold_consumes_each_line_once(body_lines, blank_line):
    lines = ['def large_function():'] + [f'    value_{i:04d} = {i}' for i in range(body_lines)]
    if blank_line:
        lines.insert(100, '')
    lines += ['def following_function():', '    return 123']
    source = chr(10).join(lines)
    splitter = GeneralCodeSplitter(chunk_size=100000, overlap=0, filetype='python')
    nodes = splitter.split_text(source, metadata_size=0)
    emitted = [line for node in nodes for line in node.text.split(chr(10))]
    assert emitted == lines
    assert all(node.metadata['filetype'] == 'python' for node in nodes)


def test_public_node_transform_keeps_boundary_line_once():
    lines = ['class LargeClass:'] + [f'    field_{i:04d} = {i}' for i in range(205)]
    source = chr(10).join(lines)
    splitter = GeneralCodeSplitter(chunk_size=100000, overlap=0, filetype='python')
    nodes = splitter(DocNode(text=source))
    assert [line for node in nodes for line in node.text.split(chr(10))] == lines
