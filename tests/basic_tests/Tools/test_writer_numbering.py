import pytest

from lazyllm.tools.writer.data_models.writer_ir import WriterBlock, WriterDocument
from lazyllm.tools.writer.numbering import (
    apply_numbering_update_ir,
    apply_numbering_update_markdown,
    build_numbering_view_from_ir,
    build_numbering_view_from_markdown,
    compute_numbering,
    dematerialize_ir,
    dematerialize_markdown,
    ensure_markdown_heading_anchors,
    format_target_number,
    materialize_ir,
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


@pytest.mark.parametrize('parse', [
    writer_document_from_markdown,
    lambda source: parse_document_markdown(source, 'document'),
])
def test_markdown_tables_use_structured_ir(parse):
    source = '| 指标 | 数量 |\n| :--- | ---: |\n| DAU | 100 |'

    table = parse(source).blocks[0]

    assert table.type == 'table'
    assert table.content == ''
    assert [row.type for row in table.children] == ['table_row', 'table_row']
    assert [cell.content for cell in table.children[0].children] == ['指标', '数量']
    assert [cell.type for cell in table.children[1].children] == [
        'table_cell', 'table_cell',
    ]
    assert table.children[0].children[0].numbering == {
        'header': True, 'align': 'left',
    }
    assert table.children[0].children[1].numbering == {
        'header': True, 'align': 'right',
    }
    assert table.children[1].children[1].content == '100'
    assert '| DAU | 100 |' in render_document_markdown(parse(source))


def test_markdown_table_entries_preserve_the_same_rich_semantics():
    source = '| 字段 | 值 |\n| :--- | ---: |\n| **粗体** | [A\\|B](https://example.com) |'
    first = writer_document_from_markdown(source).blocks[0]
    second = parse_document_markdown(source, 'document').blocks[0]

    def semantics(block):
        return {
            'type': block.type,
            'content': block.content,
            'spans': [span.model_dump() for span in block.spans],
            'numbering': block.numbering,
            'children': [semantics(child) for child in block.children],
        }

    assert semantics(first) == semantics(second)
    rendered = render_document_markdown(writer_document_from_markdown(source))
    assert '**粗体**' in rendered
    assert '[A\\|B](https://example.com)' in rendered


def test_table_numbering_changes_only_the_caption():
    table = WriterBlock(
        node_id='table', type='table', content='Metrics', children=[WriterBlock(
            node_id='row', type='table_row', children=[WriterBlock(
                node_id='cell', type='table_cell', content='DAU',
            )],
        )],
    )
    document = WriterDocument(document_id='doc', blocks=[table])
    numbering = compute_numbering(build_numbering_view_from_ir(document))

    numbered = materialize_ir(document, numbering)
    restored = dematerialize_ir(numbered, numbering)

    assert numbered.blocks[0].content == '表1 Metrics'
    assert numbered.blocks[0].children[0].children[0].content == 'DAU'
    assert restored == document


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


def test_dematerialize_markdown_accepts_escaped_heading_prefix_when_enabled():
    source = '<a id="block-a"></a>\n## 第一章\n'
    numbering = _numbering(source)
    materialized = '<a id="block-a"></a>\n## 1\\. 第一章\n'

    default_result = dematerialize_markdown(materialized, numbering)
    assert '## 1\\. 第一章' in default_result
    assert dematerialize_markdown(
        materialized,
        numbering,
        allow_escaped_prefix=True,
    ) == source


def test_dematerialize_markdown_preserves_escaped_mode_line_endings():
    source = '<a id="block-a"></a>\r\n## 第一章\r\n'
    numbering = _numbering(source)
    materialized = '<a id="block-a"></a>\r\n## 1\\. 第一章\r\n'

    assert dematerialize_markdown(
        materialized,
        numbering,
        allow_escaped_prefix=True,
    ) == source


def test_empty_parent_heading_preserves_child_numbering():
    source = '\n'.join([
        '# Title',
        '<a id="block-a"></a>',
        '##',
        '<a id="block-a-1"></a>',
        '### Child',
        '<a id="block-b"></a>',
        '## Sibling',
        '<a id="block-b-1"></a>',
        '### Sibling child',
    ])

    numbering = _numbering(source)

    assert numbering['a'].caption is None
    assert [numbering[node_id].label for node_id in ('a', 'a-1', 'b', 'b-1')] == [
        '1.', '1.1.', '2.', '2.1.',
    ]
    assert dematerialize_markdown(_materialize(source), numbering) == source


def test_unordered_parent_preserves_ordered_child_hierarchy():
    source = '\n'.join([
        '# 标题',
        '<a id="block-parent" numbering="mode=unordered"></a>',
        '## 无序父标题',
        '<a id="block-first"></a>',
        '### 子标题一',
        '<a id="block-second"></a>',
        '### 子标题二',
    ])

    numbering = _numbering(source)

    assert [
        format_target_number(numbering[node_id])
        for node_id in ('parent', 'first', 'second')
    ] == ['', '1.1.', '1.2.']
    materialized = _materialize(source)
    assert '## 无序父标题' in materialized
    assert '### 1.1. 子标题一' in materialized
    assert '### 1.2. 子标题二' in materialized

    document = WriterDocument(document_id='document', blocks=[
        WriterBlock(
            node_id='parent', type='heading', content='无序父标题',
            numbering={'level': 1, 'mode': 'unordered'},
        ),
        WriterBlock(
            node_id='first', type='heading', content='子标题一',
            numbering={'level': 2},
        ),
        WriterBlock(
            node_id='second', type='heading', content='子标题二',
            numbering={'level': 2},
        ),
    ])
    ir_numbering = compute_numbering(build_numbering_view_from_ir(document))

    assert [block.content for block in materialize_ir(document, ir_numbering).blocks] == [
        '无序父标题', '1.1. 子标题一', '1.2. 子标题二',
    ]


def test_unordered_parent_continues_previous_child_numbering():
    document = WriterDocument(document_id='document', blocks=[
        WriterBlock(node_id='chapter', type='heading', numbering={'level': 1}),
        WriterBlock(node_id='first', type='heading', numbering={'level': 2}),
        WriterBlock(node_id='second', type='heading', numbering={'level': 2}),
        WriterBlock(
            node_id='bridge', type='heading',
            numbering={'level': 1, 'mode': 'unordered'},
        ),
        WriterBlock(node_id='third', type='heading', numbering={'level': 2}),
        WriterBlock(node_id='fourth', type='heading', numbering={'level': 2}),
        WriterBlock(node_id='fifth', type='heading', numbering={'level': 2}),
    ])

    numbering = compute_numbering(build_numbering_view_from_ir(document))

    assert [
        format_target_number(numbering[node_id])
        for node_id in ('chapter', 'first', 'second', 'bridge', 'third', 'fourth', 'fifth')
    ] == ['1.', '1.1.', '1.2.', '', '1.3.', '1.4.', '1.5.']


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
