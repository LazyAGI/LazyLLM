import pytest

from lazyllm.tools.writer.data_models.writer_ir import WriterBlock, WriterDocument
from lazyllm.tools.writer.utils.conversion import render_document_markdown, writer_document_from_markdown
from lazyllm.tools.writer.utils.serialization import parse_document_markdown


@pytest.mark.parametrize('parse', [parse_document_markdown, writer_document_from_markdown])
@pytest.mark.parametrize('position', ['before', 'after'])
def test_insert_paragraph_preserves_existing_id(parse, position):
    old = '<a id="block-doc-paragraph-1"></a>\n\nOld'
    source = f'New\n\n{old}' if position == 'before' else f'{old}\n\nNew'
    document = parse(source, document_id='doc')
    by_content = {block.content: block.node_id for block in document.iter_blocks()}
    assert by_content == {'Old': 'doc-paragraph-1', 'New': 'doc-paragraph-2'}


@pytest.mark.parametrize('parse', [parse_document_markdown, writer_document_from_markdown])
def test_round_trip_insertion_preserves_targets_and_references(parse):
    source = '<a id="block-doc-heading-1"></a>\n## Original\n\nBody'
    original = parse(source, document_id='doc')
    targets = {block.content: block.node_id for block in original.iter_blocks() if block.type in {'heading', 'image'}}
    for index in range(3):
        source = f'## New {index}\n\n' + render_document_markdown(original)
        source += f'\n\n[Original](#block-{targets["Original"]})'
        original = parse(source, document_id='doc')
        blocks = list(original.iter_blocks())
        assert len({block.node_id for block in blocks}) == len(blocks)
        assert all(next(block.node_id for block in blocks if block.content == content) == node_id
                   for content, node_id in targets.items())
        assert any(span.style.get('link', {}).get('target_node_id') == targets['Original']
                   for block in blocks for span in block.spans)


@pytest.mark.parametrize('parse', [parse_document_markdown, writer_document_from_markdown])
def test_new_image_does_not_take_a_later_images_anchor(parse):
    document = parse(
        '![New](new.png)\n\n<a id="block-doc-image-1"></a>\n![Old](old.png)', document_id='doc',
    )
    assert [(block.node_id, block.content) for block in document.blocks] == [
        ('doc-image-2', 'New'), ('doc-image-1', 'Old'),
    ]


@pytest.mark.parametrize('parse', [parse_document_markdown, writer_document_from_markdown])
def test_explicit_duplicate_still_fails(parse):
    with pytest.raises(ValueError, match='duplicate Markdown anchor target'):
        parse('<a id="block-same"></a>\n\nFirst\n\n<a id="block-same"></a>\n\nSecond', document_id='doc')


@pytest.mark.parametrize('parse', [parse_document_markdown, writer_document_from_markdown])
def test_code_examples_do_not_reserve_ids(parse):
    source = 'Before\n\n```html\n<a id="block-doc-paragraph-1"></a>\n```\n\n`<a id="block-doc-code-2"></a>`'
    document = parse(source, document_id='doc')
    assert document.blocks[0].node_id == 'doc-paragraph-1'
    assert document.blocks[1].node_id == 'doc-code-2'


def test_outline_ids_are_reserved_without_stealing_later_explicit_anchors():
    outline = WriterDocument(document_id='outline', blocks=[
        WriterBlock(node_id='doc-heading-1', type='heading', content='Original'),
    ])
    document = parse_document_markdown('## New\n\n## Original', 'doc', outline=outline)
    assert [block.node_id for block in document.blocks] == ['doc-heading-2', 'doc-heading-1']
    document = parse_document_markdown(
        '## Original\n\n<a id="block-doc-heading-1"></a>\n\n## Explicit', 'doc', outline=outline,
    )
    assert [block.node_id for block in document.blocks] == ['doc-heading-2', 'doc-heading-1']
