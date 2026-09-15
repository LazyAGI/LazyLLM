from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from lazyllm.tools.writer.adapter.feishu import FeishuWriterAdapter
from lazyllm.tools.writer.adapter.base import WriterAdapterBase
from lazyllm.tools.writer.data_models import (
    MediaAsset,
    MediaAssetLibrary,
    PatchHunk,
    PatchSet,
    TargetDocument,
    WriterBlock,
    WriterDocument,
    WriterSpan,
)
from lazyllm.tools.writer.tools import WriterResourceTools
from lazyllm.tools.writer.utils import load_artifact_json, parse_document_markdown
from lazyllm.tools.fs.supplier.feishu import prepare_docx_clone_descendants
from lazyllm.tools.writer.provider.obsidian import ObsidianWriterProvider
from lazyllm.tools.writer.provider.feishu import FeishuWriterProvider
from lazyllm.tools.writer.provider.base import WriterProviderRevisionError


class _CaptionlessAdapter(WriterAdapterBase):
    provider = 'captionless'

    def blocks_to_ir(self, blocks, *, external_document_id, stage='final',
                     title='', uri=None, revision=None):
        raise NotImplementedError

    def ir_to_blocks(self, document, media_assets=None):
        raise NotImplementedError

    def patch_to_operation(self, patch, document, media_assets=None):
        raise NotImplementedError

    def merge_refreshed_document(self, previous_document, refreshed_document, **kwargs):
        return previous_document


def test_bind_written_document_does_not_require_materialized_table_captions():
    source = WriterDocument(
        document_id='document-1',
        blocks=[WriterBlock(node_id='node-1', type='paragraph', content='text')],
    )
    bound = _CaptionlessAdapter().bind_written_document(
        source,
        [{'temporary_block_id': 'node-1', 'block_id': 'provider-block-1'}],
        source.model_copy(deep=True),
    )

    assert bound.blocks[0].provider_binding == {
        'provider': 'captionless',
        'block_id': 'provider-block-1',
    }


def _block(block_id, content, *, parent='doc-1', children=None, heading=False):
    field = 'heading1' if heading else 'text'
    block = {
        'block_id': block_id,
        'block_type': 3 if heading else 2,
        'parent_id': parent,
        field: {'elements': [{'text_run': {'content': content}}]},
    }
    if children:
        block['children'] = children
    return block


def _move_blocks():
    return [
        _block('heading-1', '章节一', children=['paragraph-1'], heading=True),
        _block('paragraph-1', '段落一', parent='heading-1'),
        _block('heading-2', '章节二', children=['paragraph-2'], heading=True),
        _block('paragraph-2', '段落二', parent='heading-2'),
    ]


def test_feishu_tables_use_structured_ir_and_convert_back_to_grid():
    blocks = [{
        'block_id': 'table-1', 'block_type': 31, 'parent_id': 'doc-1',
        'table': {'property': {'row_size': 2, 'column_size': 2}},
        'children': ['cell-1', 'cell-2', 'cell-3', 'cell-4'],
    }]
    values = ['指标', '数量', 'DAU', '100']
    for index, value in enumerate(values, start=1):
        blocks.extend([
            {
                'block_id': f'cell-{index}', 'block_type': 32,
                'parent_id': 'table-1', 'table_cell': {},
                'children': [f'text-{index}'],
            },
            _block(f'text-{index}', value, parent=f'cell-{index}'),
        ])

    adapter = FeishuWriterAdapter()
    document = adapter.blocks_to_ir(blocks, external_document_id='doc-1')
    table = document.blocks[0]

    assert [row.type for row in table.children] == ['table_row', 'table_row']
    assert [[cell.content for cell in row.children] for row in table.children] == [
        ['指标', '数量'], ['DAU', '100'],
    ]

    native = adapter.ir_to_blocks(document)
    assert len(native) == 1
    assert native[0]['table']['property'] == {'row_size': 2, 'column_size': 2}
    assert native[0]['_table_cells'][1][1][0]['text_run']['content'] == '100'

    current = table.children[1].children[1]
    updated = current.model_copy(update={
        'content': '200', 'spans': [WriterSpan(text='200', style={'bold': True})],
    })
    operation = adapter.patch_to_operation(PatchHunk(
        target_node_id=current.node_id,
        modify_type='update',
        block=updated,
    ), document)
    assert operation.params['requests'] == [{
        'block_id': 'text-4',
        'update_text_elements': {'elements': [{
            'text_run': {'content': '200', 'text_element_style': {'bold': True}},
        }]},
    }]


def test_feishu_read_does_not_guess_preceding_paragraph_is_table_caption():
    blocks = [
        _block('caption-1', '表1 指标表'),
        {
            'block_id': 'table-1', 'block_type': 31, 'parent_id': 'doc-1',
            'table': {'property': {'row_size': 1, 'column_size': 1}},
            'children': ['cell-1'],
        },
        {
            'block_id': 'cell-1', 'block_type': 32, 'parent_id': 'table-1',
            'table_cell': {}, 'children': ['text-1'],
        },
        _block('text-1', 'DAU', parent='cell-1'),
    ]

    document = FeishuWriterAdapter().blocks_to_ir(
        blocks, external_document_id='doc-1')

    assert [block.type for block in document.blocks] == ['paragraph', 'table']
    assert document.blocks[0].content == '表1 指标表'
    assert document.blocks[1].content == ''

    adapter = FeishuWriterAdapter()
    previous = document.model_copy(deep=True)
    caption = previous.blocks.pop(0)
    previous.blocks[0].content = '指标表'
    previous.blocks[0].provider_payload['table_caption'] = {
        'content': caption.content, 'provider_binding': caption.provider_binding,
        'provider_payload': caption.provider_payload,
    }
    document = adapter.merge_refreshed_document(previous, document)
    assert len(document.blocks) == 1
    assert document.blocks[0].content == '指标表'
    after = adapter.blocks_to_ir([_block('after', '正文')], external_document_id='doc-1').blocks[0]
    document.blocks.append(after)
    delete_after = adapter.patch_to_operation(PatchHunk(
        target_node_id=after.node_id, modify_type='delete'), document)
    assert (delete_after.params['start_index'], delete_after.params['end_index']) == (2, 3)
    delete_table = adapter.patch_to_operation(PatchHunk(
        target_node_id=document.blocks[0].node_id, modify_type='delete'), document)
    assert (delete_table.params['start_index'], delete_table.params['end_index']) == (0, 2)


def test_create_and_delete_build_native_operations():
    adapter = FeishuWriterAdapter()
    document = adapter.blocks_to_ir(_move_blocks()[:2], external_document_id='doc-1')
    paragraph = document.blocks[0].children[0]
    created = WriterBlock(
        node_id='new-block',
        type='paragraph',
        content='新增段落',
        spans=[WriterSpan(text='新增段落', style={'bold': True})],
        stage='final',
    )

    create = adapter.patch_to_operation(PatchHunk(
        target_node_id=created.node_id,
        modify_type='create',
        block=created,
        parent_node_id=None,
        index=1,
    ), document)
    assert create.operation == 'create'
    assert (create.params['parent_block_id'], create.params['index']) == ('doc-1', 1)
    assert create.params['blocks'][0]['text']['elements'][0][
        'text_run']['text_element_style'] == {'bold': True}

    delete = adapter.patch_to_operation(PatchHunk(
        target_node_id=paragraph.node_id,
        modify_type='delete',
    ), document)
    assert delete.operation == 'delete'
    assert delete.params == {
        'parent_block_id': 'heading-1',
        'start_index': 0,
        'end_index': 1,
    }


def test_image_create_operation_carries_private_media_binding_metadata(tmp_path):
    image_path = tmp_path / 'image.png'
    image_path.write_bytes(b'image')
    media_assets = MediaAssetLibrary(
        library_id='library-1',
        assets={'asset-1': MediaAsset(
            media_asset_id='asset-1',
            asset_type='image',
            source_type='image_generation',
            local_path=str(image_path),
        )},
    )
    adapter = FeishuWriterAdapter()
    document = adapter.blocks_to_ir(
        [_block('paragraph-1', '段落一')], external_document_id='doc-1')
    image = WriterBlock(
        node_id='new-image',
        type='image',
        content='图片说明',
        stage='final',
        references=[{'type': 'media_asset', 'id': 'asset-1'}],
    )

    operation = adapter.patch_to_operation(PatchHunk(
        target_node_id=image.node_id,
        modify_type='create',
        block=image,
        index=1,
    ), document, media_assets=media_assets)

    assert operation.params['blocks'][0]['block_type'] == 27
    assert operation.params['blocks'][0]['_media']['media_asset_id'] == 'asset-1'


def test_update_maps_styles_and_block_type_changes():
    adapter = FeishuWriterAdapter()
    document = adapter.blocks_to_ir(
        [_block('paragraph-1', '段落一')],
        external_document_id='doc-1',
    )
    source = document.blocks[0]

    styled = source.model_copy(deep=True)
    styled.spans[0].style = {
        'bold': True,
        'text_color': 5,
        'background_color': 2,
        'font_size': 16,
    }
    update = adapter.patch_to_operation(PatchHunk(
        target_node_id=source.node_id,
        modify_type='update',
        block=styled,
    ), document)
    assert update.operation == 'update'
    assert update.params['requests'][0]['update_text_elements']['elements'][0][
        'text_run']['text_element_style'] == styled.spans[0].style

    heading = styled.model_copy(deep=True)
    heading.type = 'heading'
    heading.numbering = {'level': 4}
    replace = adapter.patch_to_operation(PatchHunk(
        target_node_id=source.node_id,
        modify_type='update',
        block=heading,
    ), document)
    assert replace.operation == 'replace'
    assert replace.params['replacement_block']['block_type'] == 6
    assert 'heading4' in replace.params['replacement_block']


def test_move_uses_parent_and_final_index():
    adapter = FeishuWriterAdapter()
    document = adapter.blocks_to_ir(_move_blocks(), external_document_id='doc-1')
    source = document.blocks[0].children[0]
    target_parent = document.blocks[1]

    operation = adapter.patch_to_operation(PatchHunk(
        target_node_id=source.node_id,
        modify_type='move',
        parent_node_id=target_parent.node_id,
        index=1,
    ), document)

    assert operation.operation == 'move'
    assert operation.params == {
        'source_parent_block_id': 'heading-1',
        'source_block_id': 'paragraph-1',
        'source_index': 0,
        'target_parent_block_id': 'heading-2',
        'target_index': 1,
    }


def test_obsidian_write_back_drops_writer_system_anchors():
    restored = ObsidianWriterProvider()._from_writer_markdown(
        '# Title\n\n<a id="block-sec-001"></a>\n## Section\n',
        {},
        None,
        MagicMock(),
        None,
    )

    assert restored == '# Title\n\n## Section\n'


def test_obsidian_ir_write_back_serializes_media_asset_path():
    document = WriterDocument(
        document_id='writer-doc',
        stage='final',
        title='Report',
        blocks=[WriterBlock(
            node_id='image-1',
            type='image',
            content='Architecture',
            references=[{'type': 'media_asset', 'id': 'asset-1'}],
        )],
    )
    media_assets = MediaAssetLibrary(
        library_id='library-1',
        assets={'asset-1': MediaAsset(
            media_asset_id='asset-1',
            asset_type='image',
            source_type='input_resource',
            local_path='/tmp/architecture.png',
        )},
    )

    markdown = ObsidianWriterProvider()._serialize_writer_document(document, media_assets)

    assert '](/tmp/architecture.png)' in markdown
    assert document.blocks[0].references == [{'type': 'media_asset', 'id': 'asset-1'}]


def test_obsidian_write_back_uses_the_structured_table_markdown_path():
    document = parse_document_markdown(
        '| Metric | Value |\n| --- | ---: |\n| DAU | 100 |',
        'document',
    )

    markdown = ObsidianWriterProvider()._serialize_writer_document(document, None)

    assert markdown.count('| DAU | 100 |') == 1
    assert document.blocks[0].type == 'table'
    assert document.blocks[0].content == ''


def test_write_result_preserves_provider_fields(tmp_path):
    result = WriterResourceTools(llm=None, artifact_store=str(tmp_path))._save_write_result(
        document_id='vault:note.md',
        adapter='obsidian',
        locator='obsidian://vault/note.md',
        block_count=3,
        provider_result={
            'local_path': '/Users/example/Documents/vault/note.md',
            'warnings': ['source changed'],
        },
    )

    write_result = load_artifact_json(result['artifact_path'], validate_schema=False)

    assert write_result['local_path'] == '/Users/example/Documents/vault/note.md'
    assert write_result['warnings'] == ['source changed']


def test_merge_refreshed_move_restores_writer_identity():
    adapter = FeishuWriterAdapter()
    previous = adapter.blocks_to_ir(_move_blocks(), external_document_id='doc-1')
    source = previous.blocks[0]
    patch = PatchHunk(
        target_node_id=source.node_id,
        modify_type='move',
        parent_node_id=None,
        index=1,
    )
    operation = adapter.patch_to_operation(patch, previous)

    refreshed_raw = deepcopy(_move_blocks()[2:] + _move_blocks()[:2])
    refreshed_raw[2]['block_id'] = 'moved-heading'
    refreshed_raw[2]['children'] = ['moved-paragraph']
    refreshed_raw[3]['block_id'] = 'moved-paragraph'
    refreshed_raw[3]['parent_id'] = 'moved-heading'
    refreshed = adapter.blocks_to_ir(refreshed_raw, external_document_id='doc-1')

    moved = adapter.merge_refreshed_document(
        previous,
        refreshed,
        patch=patch,
        operation=operation,
        operation_result={
            'provider_id_remap': {
                'heading-1': 'moved-heading',
                'paragraph-1': 'moved-paragraph',
            },
        },
    ).blocks[1]
    assert moved.node_id == source.node_id
    assert moved.children[0].node_id == source.children[0].node_id
    assert moved.provider_binding['block_id'] == 'moved-heading'


def test_merge_refreshed_document_rebases_internal_reference_targets():
    adapter = FeishuWriterAdapter()
    raw = [
        _block('target', '目标章节', heading=True),
        {
            **_block('source', '参见目标章节和外部文档'),
            'text': {'elements': [
                {'text_run': {'content': '参见'}},
                {'text_run': {
                    'content': '目标章节',
                    'text_element_style': {
                        'link': {'url': 'https://feishu.cn/docx/doc-1#target'},
                    },
                }},
                {'text_run': {
                    'content': '和外部文档',
                    'text_element_style': {
                        'link': {'url': 'https://example.com/document'},
                    },
                }},
            ]},
        },
    ]
    previous = adapter.blocks_to_ir(raw, external_document_id='doc-1')
    previous.blocks[0].node_id = 'writer-owned-target'
    previous.blocks[1].spans[1].style['link']['target_node_id'] = 'writer-owned-target'
    refreshed = adapter.blocks_to_ir(raw, external_document_id='doc-1')

    merged = adapter.merge_refreshed_document(previous, refreshed)

    assert merged.blocks[0].node_id == 'writer-owned-target'
    assert merged.blocks[1].spans[1].style['link'] == {
        'type': 'internal_ref',
        'target_node_id': 'writer-owned-target',
    }
    assert merged.blocks[1].spans[2].style['link'] == {
        'url': 'https://example.com/document',
    }


def test_move_clone_descendants_preserve_raw_format_and_children():
    blocks = _move_blocks()[:2]
    blocks[0]['heading1']['style'] = {'align': 2}
    blocks[0]['heading1']['elements'][0]['text_run']['text_element_style'] = {
        'bold': True,
        'text_color': 5,
    }

    children_id, descendants, id_map, _ = \
        prepare_docx_clone_descendants(blocks, 'heading-1')

    assert children_id == ['move-block-0']
    assert id_map == {
        'move-block-0': 'heading-1',
        'move-block-1': 'paragraph-1',
    }
    assert descendants[0]['heading1'] == blocks[0]['heading1']
    assert descendants[0]['children'] == ['move-block-1']


def _cell_batch_setup():
    rows = []
    for r in range(2):
        cells = []
        for c in range(2):
            node_id = f'cell-{r}-{c}'
            cells.append(WriterBlock(node_id=node_id, type='table_cell', content=node_id,
                provider_binding={'provider': 'feishu', 'block_id': node_id},
                provider_payload={'table_content_blocks': [{'block_id': f'text-{r}-{c}', 'block_type': 2,
                'text': {'elements': [{'text_run': {'content': node_id}}]}}]}))
        rows.append(WriterBlock(node_id=f'row-{r}', type='table_row', children=cells))
    document = WriterDocument(document_id='writer-doc', revision='10', provider_binding={'provider': 'feishu',
        'document_id': 'doc'}, blocks=[WriterBlock(node_id='table', type='table', children=rows,
        provider_binding={'provider': 'feishu', 'block_id': 'table', 'parent_block_id': 'doc'}),
        WriterBlock(node_id='paragraph', type='paragraph', content='end', provider_binding={'provider': 'feishu',
        'block_id': 'paragraph', 'parent_block_id': 'doc'}, provider_payload={'raw_block': {'block_type': 2,
        'text': {}}})])
    instance = FeishuWriterProvider()
    adapter = FeishuWriterAdapter()
    fs = MagicMock()
    fs.update_block.side_effect = [{'document_revision_id': value} for value in range(11, 30)]
    instance._resolve_document_target = MagicMock(return_value=('feishu', '/doc', fs, adapter, '/doc', 'doc'))
    instance._document_metadata = MagicMock(return_value={'revision_id': 10})
    instance._read_persisted_document = MagicMock(side_effect=lambda **kw: kw['source_document'].model_copy(deep=True))
    return (instance, fs, document)


def _cell_edit(document, node_id, value, hunk_id=None):
    return PatchHunk(hunk_id=hunk_id or f'edit-{node_id}', target_node_id=node_id, modify_type='update',
        block=document.block_by_id(node_id).model_copy(update={'content': value, 'spans': [WriterSpan(text=value,
        style={'bold': True})]}))


def test_feishu_load_document_records_remote_revision():
    provider = FeishuWriterProvider()
    fs = MagicMock()
    fs.get_document_metadata.return_value = {'revision_id': 12, 'title': 'Project'}
    fs.get_doc_blocks.return_value = [_block('paragraph', 'content')]
    provider._resolve_document_target = MagicMock(return_value=(
        'feishu', '/project', fs, FeishuWriterAdapter(), 'feishu:/project', 'doc-1',
    ))

    result = provider.load_document(TargetDocument(uri='feishu:/project', adapter='feishu'))

    assert result['source_document'].revision == '12'
    assert result['source_document'].title == 'Project'
    assert result['target_document'].meta['revision_id'] == 12


def test_feishu_patch_rejects_a_stale_loaded_revision_before_writing():
    provider, fs, document = _cell_batch_setup()
    provider._document_metadata.return_value = {'revision_id': 11}

    with pytest.raises(WriterProviderRevisionError) as exc_info:
        _apply_cell_hunks(provider, document, [_cell_edit(document, 'cell-0-0', 'new')])

    assert exc_info.value.expected == '10'
    assert exc_info.value.actual == '11'
    fs.update_block.assert_not_called()


def _apply_cell_hunks(instance, document, hunks):
    return instance.apply_patch_to_document(PatchSet(target_doc_id=document.document_id, hunks=hunks), document,
        TargetDocument(doc_id='doc', adapter=instance.provider, uri='/doc'))


def test_feishu_cells_merge_by_provider_granularity_and_repeated_edit_keeps_last():
    instance, fs, document = _cell_batch_setup()
    before = document.model_dump()
    hunks = [_cell_edit(document, 'cell-0-0', 'first', 'h1'), _cell_edit(document, 'cell-1-0', 'other row', 'h2'),
        _cell_edit(document, 'cell-0-1', 'same row', 'h3'), _cell_edit(document, 'cell-0-0', 'last', 'h4')]
    result = _apply_cell_hunks(instance, document, hunks)
    assert fs.update_block.call_count == 1
    assert result['patch_result'].applied_hunks == ['h1', 'h2', 'h3', 'h4']
    assert document.model_dump() == before
    assert result['persisted_document'].block_by_id('cell-0-0').content == 'last'
    instance._read_persisted_document.assert_called_once()
    requests = fs.update_block.call_args.kwargs['requests']
    assert len(requests) == 3
    by_id = {item['block_id']: item['update_text_elements']['elements'][0]['text_run'] for item in requests}
    assert by_id['text-0-0']['content'] == 'last'
    assert by_id['text-0-1']['content'] == 'same row'
    assert by_id['text-0-0']['text_element_style']['bold'] is True


def test_feishu_move_flushes_cells_and_following_batch_uses_new_ids():
    instance, fs, document = _cell_batch_setup()
    ids = ['table'] + [f'cell-{r}-{c}' for r in range(2) for c in range(2)]
    ids += [f'text-{r}-{c}' for r in range(2) for c in range(2)]
    relations = {key: key + '-moved' for key in ids}
    fs.move_block.return_value = {'provider_id_remap': relations, 'document_revision_id': 20}
    _apply_cell_hunks(instance, document, [_cell_edit(document, 'cell-0-0', 'before'), PatchHunk(hunk_id='move',
        target_node_id='table', modify_type='move', index=1), _cell_edit(document, 'cell-0-1', 'after')])
    assert [call[0] for call in fs.method_calls] == ['update_block', 'move_block', 'update_block']
    assert fs.update_block.call_args.kwargs['requests'][0]['block_id'] == 'text-0-1-moved'
    assert fs.update_block.call_args.kwargs['document_revision_id'] == 20


def test_feishu_splits_on_native_block_count_and_propagates_revision():
    instance, fs, document = _cell_batch_setup()
    cell = document.block_by_id('cell-0-0')
    raw = cell.provider_payload['table_content_blocks'][0]
    cell.provider_payload['table_content_blocks'] = [dict(deepcopy(raw), block_id=f'text-{i}') for i in range(201)]
    result = _apply_cell_hunks(instance, document, [_cell_edit(document, cell.node_id, 'new content')])
    assert [len(call.kwargs['requests']) for call in fs.update_block.call_args_list] == [200, 1]
    assert [call.kwargs['document_revision_id'] for call in fs.update_block.call_args_list] == [10, 11]
    assert result['persisted_document'].revision == '12'
    assert fs.update_block.call_args.kwargs['requests'][0]['update_text_elements']['elements'] == []


def test_feishu_math_read_write_preserves_inline_boundaries():
    adapter = FeishuWriterAdapter()
    document = parse_document_markdown('$$\nx^2\n$$\n\nbefore $y^2$ after', 'math-doc')
    assert not document.blocks[0].editable
    raw = adapter.ir_to_blocks(document)
    display = r'\begin{equation}x^2\end{equation}'
    assert raw[0]['text']['elements'] == [{'equation': {'content': display}}]
    raw[1]['text']['elements'][1]['equation']['text_element_style'] = {'bold': True}
    document = adapter.blocks_to_ir(raw, external_document_id='doc-1')
    assert [block.type for block in document.blocks] == ['paragraph', 'paragraph']
    assert document.blocks[0].spans[0].text == '$$' + display + '$$'
    assert document.blocks[1].spans[1].style == {'math_source': True, 'bold': True}
    rewritten = adapter.ir_to_blocks(document)
    assert rewritten[0]['text']['elements'] == raw[0]['text']['elements']
    assert rewritten[1]['text']['elements'] == raw[1]['text']['elements']


def test_feishu_numbered_image_is_read_only_and_roundtrips_unchanged():
    adapter = FeishuWriterAdapter()
    raw = {'block_id': 'image-1', 'block_type': 27,
           'image': {'token': 'image-token', 'caption': {'content': '图1 图片说明'}}}
    document = adapter.blocks_to_ir([raw], external_document_id='doc-1')
    assert not document.blocks[0].editable
    assert document.blocks[0].content == '图片说明'
    assert adapter.ir_to_blocks(document)[0]['image'] == raw['image']
