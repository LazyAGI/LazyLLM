import pytest

from lazyllm.tools.writer.data_models.writer_ir import WriterBlock, WriterDocument
from lazyllm.tools.writer.numbering import (
    apply_numbering_update_ir,
    apply_numbering_update_markdown,
    build_numbering_view_from_ir,
    build_numbering_view_from_markdown,
    compute_numbering,
    dematerialize_markdown,
    ensure_markdown_heading_anchors,
    format_target_number,
    materialize_markdown,
)
from lazyllm.tools.writer.utils.conversion import (
    render_document_markdown,
    writer_document_from_markdown,
)
from lazyllm.tools.writer.utils.serialization import parse_document_markdown


def _numbering(markdown: str):
    return compute_numbering(build_numbering_view_from_markdown(markdown))


def _materialize(markdown: str) -> str:
    view = build_numbering_view_from_markdown(markdown)
    return materialize_markdown(markdown, view, compute_numbering(view))


def test_markdown_style_restart_and_unordered_heading_round_trip():
    source = '\n'.join([
        '<!-- heading-numbering: {"ordered_style":"chinese"} -->',
        '# 标题',
        '<a id="block-a"></a>',
        '## Alpha',
        '<a id="block-b"></a>',
        '### Beta',
        '<a id="block-c" numbering="restart"></a>',
        '### Gamma',
        '<a id="block-d" numbering="mode=unordered"></a>',
        '## Delta',
        '<a id="block-e"></a>',
        '## Epsilon',
    ])
    numbering = _numbering(source)

    assert [format_target_number(numbering[node_id]) for node_id in 'abcde'] == [
        '一、', '（一）', '（一）', '', '二、',
    ]
    materialized = _materialize(source)
    assert '### （一） Gamma' in materialized
    assert '## Delta' in materialized
    assert dematerialize_markdown(materialized, numbering) == source


def test_markdown_dematerialization_survives_new_editor_anchor_ids():
    base = '\n'.join([
        '# 标题',
        '<a id="block-sec-001"></a>',
        '## 预备知识',
        '### 时间自动机基础',
    ])
    submitted = '\n'.join([
        '# 标题',
        '<a id="block-sec-001"></a>',
        '## 1. 预备知识',
        '<a id="block-user-new"></a>',
        '### 1.1. 时间自动机基础',
    ])

    assert dematerialize_markdown(submitted, _numbering(base)) == '\n'.join([
        '# 标题',
        '<a id="block-sec-001"></a>',
        '## 预备知识',
        '<a id="block-user-new"></a>',
        '### 时间自动机基础',
    ])


def test_generated_markdown_anchor_avoids_existing_id():
    markdown = ensure_markdown_heading_anchors(
        '# 标题\n<a id="block-sec-002"></a>\n## 已有锚点\n## 缺少锚点\n'
    )
    assert [target.id for target in build_numbering_view_from_markdown(markdown).targets] \
        == ['sec-002', 'sec-003']


def test_parenthesized_style_reaches_roman_heading_levels():
    source = '\n'.join([
        '<!-- heading-numbering: {"ordered_style":"parenthesized"} -->',
        '# Title',
        *(
            line
            for node_id, level, title in zip('abcde', range(2, 7), 'ABCDE')
            for line in (f'<a id="block-{node_id}"></a>', f'{"#" * level} {title}')
        ),
    ])
    numbering = _numbering(source)

    assert [numbering[node_id].label for node_id in 'abcde'] == [
        '(1)', '(a)', '(i)', '(A)', '(I)',
    ]
    assert '#### (i) C' in _materialize(source)


def test_global_ir_style_and_local_unordered_heading_are_independent():
    document = WriterDocument(
        document_id='document',
        metadata={'heading_numbering': {'ordered_style': 'parenthesized'}},
        blocks=[
            WriterBlock(node_id='a', type='heading', numbering={'level': 1}),
            WriterBlock(
                node_id='b', type='heading',
                numbering={'level': 1, 'mode': 'unordered'},
            ),
            WriterBlock(node_id='c', type='heading', numbering={'level': 1}),
        ],
    )

    numbering = compute_numbering(build_numbering_view_from_ir(document))
    assert [numbering[node_id].label for node_id in 'abc'] == ['(1)', '', '(2)']


def test_markdown_numbering_metadata_survives_ir_conversion():
    source = '\n'.join([
        '<!-- heading-numbering: {"ordered_style":"parenthesized"} -->',
        '# 标题',
        '<a id="block-a" numbering="mode=unordered"></a>',
        '## Alpha',
    ])
    document = writer_document_from_markdown(source)

    assert document.metadata['heading_numbering'] == {'ordered_style': 'parenthesized'}
    assert document.blocks[0].numbering == {'level': 1, 'mode': 'unordered'}
    parsed = parse_document_markdown(source, 'document')
    assert parsed.metadata['heading_numbering'] == document.metadata['heading_numbering']
    assert parsed.blocks[0].numbering == document.blocks[0].numbering

    document.metadata.pop('markdown_source')
    document.metadata.pop('markdown_signature')
    rendered = render_document_markdown(document)
    assert '"ordered_style":"parenthesized"' in rendered
    assert 'numbering="mode=unordered"' in rendered


def test_html_image_keeps_its_anchor_and_does_not_steal_the_next_heading():
    source = '\n'.join([
        '# 标题',
        '<a id="block-sec-002"></a>',
        '## 深入',
        '<a id="block-sec-002-001"></a>',
        '### 证据与谜团',
        '[因果链](#block-IMAGE-1)',
        '',
        '<a id="block-IMAGE-1"></a>',
        '',
        '<img height="712" width="712" alt="恐惧递进因果链" src="/data/chain.jpg" />',
        '',
        '<a id="block-sec-002-002"></a>',
        '### 不可名状的征兆',
    ])

    view = build_numbering_view_from_markdown(source)
    numbering = compute_numbering(view)
    assert [(target.id, target.kind) for target in view.targets] == [
        ('sec-002', 'section'),
        ('sec-002-001', 'section'),
        ('IMAGE-1', 'figure'),
        ('sec-002-002', 'section'),
    ]
    assert numbering['sec-002-002'].label == '1.2.'

    document = parse_document_markdown(source, 'document')
    blocks = list(document.iter_blocks())
    assert any(block.node_id == 'IMAGE-1' and block.type == 'image' for block in blocks)
    assert any(
        block.node_id == 'sec-002-002'
        and block.type == 'heading'
        and block.content == '不可名状的征兆'
        for block in blocks
    )
    assert any(
        span.style.get('link', {}).get('target_node_id') == 'IMAGE-1'
        for block in blocks for span in block.spans
    )

    materialized = materialize_markdown(source, view, numbering)
    assert '<img height="712" width="712"' in materialized
    assert dematerialize_markdown(materialized, numbering) == source


def test_backend_updates_metadata_and_recomputes_labels():
    source = '\n'.join([
        '# 标题',
        '<a id="block-a"></a>',
        '## Alpha',
        '<a id="block-b"></a>',
        '## Beta',
        '<a id="block-c"></a>',
        '## Gamma',
    ])
    clean = dematerialize_markdown(_materialize(source), _numbering(source))
    clean = apply_numbering_update_markdown(clean, {
        'type': 'ordered_style', 'ordered_style': 'parenthesized',
    })
    clean = apply_numbering_update_markdown(clean, {
        'type': 'heading', 'target_id': 'b', 'mode': 'unordered',
    })

    assert [_numbering(clean)[node_id].label for node_id in 'abc'] == ['(1)', '', '(2)']
    assert '<!-- heading-numbering: {"ordered_style":"parenthesized"} -->' in clean
    assert '<a id="block-b" numbering="mode=unordered"></a>' in clean


def test_backend_applies_ir_restart_without_mutating_input():
    document = WriterDocument(
        document_id='document',
        blocks=[
            WriterBlock(node_id='a', type='heading', numbering={'level': 1}),
            WriterBlock(node_id='b', type='heading', numbering={'level': 1}),
        ],
    )
    changed = apply_numbering_update_ir(document, {
        'type': 'heading', 'target_id': 'b', 'restart': True,
    })

    assert document.blocks[1].numbering == {'level': 1}
    assert changed.blocks[1].numbering == {'level': 1, 'restart': True}
    numbering = compute_numbering(build_numbering_view_from_ir(changed))
    assert [numbering[node_id].label for node_id in 'ab'] == ['1.', '1.']


def test_invalid_markdown_numbering_metadata_is_rejected():
    with pytest.raises(ValueError, match='invalid Markdown heading numbering config'):
        build_numbering_view_from_markdown(
            '<!-- heading-numbering: {"headings":{"a":{"restart":true}}} -->',
        )
