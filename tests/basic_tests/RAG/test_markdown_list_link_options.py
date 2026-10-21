import pytest

from lazyllm.tools.rag.doc_node import DocNode
from lazyllm.tools.rag.transform import MarkdownSplitter


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('kind, fragment', [
    ('list', '- first item' + chr(10) + '- second item'),
    ('link', '[reference](https://example.com/reference)'),
])
def test_list_link_option_controls_public_chunks(enabled, kind, fragment):
    splitter = MarkdownSplitter(chunk_size=1024, overlap=0, keep_trace=True, keep_headers=True,
                                **{'keep_lists' if kind == 'list' else 'keep_links': enabled})
    text = '## Section' + chr(10) + 'Before.' + chr(10) * 2 + fragment + chr(10) * 2 + 'After.'
    nodes = splitter.split_text(text, metadata_size=0)
    selected = [node for node in nodes if node.metadata['type'] == kind]
    assert [node.text for node in selected] == ([fragment] if enabled else [])
    assert all(node.metadata['path'] == ['Section'] and node.metadata['header'] == 'Section' for node in nodes)
    assert all(node.metadata['level'] == 2 for node in nodes)


@pytest.mark.parametrize('kind, text, expected', [
    ('python', '```python' + chr(10) + 'value = "[reference](https://example.com)"' + chr(10) + '```',
     'value = "[reference](https://example.com)"'),
    ('table', '| Name | Value |' + chr(10) + '| --- | --- |' + chr(10)
     + '| Entry | [reference](https://example.com) |', None),
    ('list', '- [reference](https://example.com)' + chr(10) + '- second item', None),
    ('image', '![caption](https://example.com/image.png)', None),
])
def test_new_options_preserve_existing_element_boundaries(kind, text, expected):
    splitter = MarkdownSplitter(chunk_size=1024, overlap=0, keep_code_blocks=True, keep_tables=True,
                                keep_images=True, keep_lists=True, keep_links=True)
    nodes = splitter(DocNode(text=text))
    assert [(node.metadata['type'], node.text) for node in nodes] == [(kind, expected or text)]


def test_both_options_keep_surrounding_text_and_distinct_elements():
    text = 'Before.' + chr(10) * 2 + '- first item' + chr(10) + '- second item' + chr(10) * 2
    text += 'Between. [reference](https://example.com) After.'
    splitter = MarkdownSplitter(chunk_size=1024, overlap=0, keep_lists=True, keep_links=True)
    nodes = splitter.split_text(text, metadata_size=0)
    assert [(node.metadata['type'], node.text) for node in nodes] == [
        ('content', 'Before.'), ('list', '- first item' + chr(10) + '- second item'),
        ('content', 'Between.'), ('link', '[reference](https://example.com)'), ('content', 'After.'),
    ]
