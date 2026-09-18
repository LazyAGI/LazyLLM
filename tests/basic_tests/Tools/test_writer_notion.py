from contextlib import contextmanager
from copy import deepcopy
from unittest.mock import MagicMock, patch
from uuid import NAMESPACE_URL, uuid5

import pytest

from lazyllm.tools.writer.adapter.notion import NotionWriterAdapter
from lazyllm.tools.writer.data_models import (
    MediaAsset, MediaAssetLibrary, PatchHunk, PatchSet, TargetDocument,
    WriterBlock, WriterDocument, WriterSpan,
)
from lazyllm.tools.writer.provider import (
    NotionWriterProvider,
    get_writer_provider,
    match_writer_provider,
)
from lazyllm.tools.writer.provider.base import WriterProviderRevisionError
from lazyllm.tools.writer.tools.resource_tools import WriterResourceTools
from lazyllm.tools.writer.utils import load_artifact_json


DOC_ID = '01234567-89ab-cdef-0123-456789abcdef'
HEADING_ID = '11111111-1111-1111-1111-111111111111'
PARAGRAPH_ID = '22222222-2222-2222-2222-222222222222'
CHILD_ID = '44444444-4444-4444-4444-444444444444'
PAGE_ID = '3c77b0d7-54e0-80d2-b507-d786256d3b18'
BLOCK_ID = '11111111-1111-1111-1111-111111111111'
PAGE_URL = f'https://www.notion.so/Writer-{PAGE_ID.replace("-", "")}'


def _rich(text, **annotations):
    return {
        'type': 'text',
        'text': {'content': text, 'link': None},
        'annotations': {
            'bold': False,
            'italic': False,
            'strikethrough': False,
            'underline': False,
            'code': False,
            'color': 'default',
            **annotations,
        },
        'plain_text': text,
        'href': None,
    }


def _block(block_id, block_type, payload, *, parent=DOC_ID, has_children=False):
    return {
        'object': 'block',
        'id': block_id,
        'block_id': block_id,
        'type': block_type,
        'block_type': block_type,
        'parent': {'type': 'block_id', 'block_id': parent},
        'parent_id': parent,
        'has_children': has_children,
        block_type: payload,
    }


def _raw_blocks():
    return [{
        'object': 'block',
        'id': BLOCK_ID,
        'block_id': BLOCK_ID,
        'type': 'paragraph',
        'block_type': 'paragraph',
        'parent': {'type': 'page_id', 'page_id': PAGE_ID},
        'parent_id': PAGE_ID,
        'paragraph': {
            'rich_text': [{
                'type': 'text',
                'plain_text': 'Notion content',
                'text': {'content': 'Notion content', 'link': None},
                'annotations': {
                    'bold': False, 'italic': False, 'strikethrough': False,
                    'underline': False, 'code': False, 'color': 'default',
                },
            }],
        },
    }]


def _metadata():
    return {
        'document_id': PAGE_ID,
        'object_type': 'page',
        'title': 'Writer page',
        'browser_url': PAGE_URL,
        'internal_uri': f'notion:/~page/{PAGE_ID}',
        'last_edited_time': '2026-08-28T08:00:00.000Z',
    }


def _make_fs():
    fs = MagicMock()
    fs.get_document_metadata.return_value = _metadata()
    fs.get_doc_blocks.return_value = _raw_blocks()
    fs.replace_doc_blocks.return_value = _raw_blocks()
    fs.write_doc_blocks.return_value = _raw_blocks()
    return fs


@contextmanager
def _route_notion(fs, real_path=None):
    path = real_path or f'/~page/{PAGE_ID}'
    with patch(
        'lazyllm.tools.fs.client.FS._parse',
        return_value=('notion', None, path),
    ), patch(
        'lazyllm.tools.fs.client.FS._get_or_create_fs',
        return_value=fs,
    ):
        yield


class TestNotionAdapter:
    def test_blocks_to_ir_preserves_payload_bindings_and_input(self):
        raw = _block(PARAGRAPH_ID, 'paragraph', {
            'rich_text': [_rich('正文')],
            'color': 'default',
        })
        source = deepcopy(raw)

        document = NotionWriterAdapter().blocks_to_ir(
            [raw], external_document_id=DOC_ID, title='标题',
            uri=f'https://notion.so/{DOC_ID}', revision='revision-1')

        block = document.blocks[0]
        assert document.document_id == NotionWriterAdapter.make_document_id(DOC_ID)
        assert document.title == '标题'
        assert document.metadata == {'source_block_count': 1}
        assert document.provider_binding == {
            'provider': 'notion',
            'document_id': DOC_ID,
            'uri': f'https://notion.so/{DOC_ID}',
            'revision': 'revision-1',
        }
        assert document.ui_editable is False
        assert block.node_id == NotionWriterAdapter.make_node_id(DOC_ID, PARAGRAPH_ID)
        assert block.provider_binding == {
            'provider': 'notion',
            'document_id': DOC_ID,
            'block_id': PARAGRAPH_ID,
            'parent_block_id': DOC_ID,
            'revision': 'revision-1',
        }
        assert block.provider_payload == {'raw_block': source, 'source_index': 0}
        assert block.content == '正文'
        assert block.editable is True
        assert raw == source

    def test_ir_to_blocks_flattens_logical_heading_children_as_notion_siblings(self):
        document = WriterDocument(
            document_id='writer-document',
            provider_binding={'provider': 'notion', 'document_id': DOC_ID},
            blocks=[WriterBlock(
                node_id='section-1', type='heading', content='章节',
                numbering={'level': 1},
                children=[
                    WriterBlock(node_id='paragraph-1', type='paragraph', content='正文一'),
                    WriterBlock(
                        node_id='section-1-1', type='heading', content='子章节',
                        numbering={'level': 2},
                        children=[WriterBlock(
                            node_id='paragraph-2', type='paragraph', content='正文二',
                        )],
                    ),
                ],
            )],
        )

        native = NotionWriterAdapter().ir_to_blocks(document)

        assert [block['type'] for block in native] == [
            'heading_1', 'paragraph', 'heading_2', 'paragraph',
        ]
        assert all('children' not in block[block['type']] for block in native)

    def test_ir_to_blocks_materializes_structured_table_and_caption(self):
        table = WriterBlock(
            node_id='table-1', type='table', content='表1 指标表',
            spans=[WriterSpan(text='表1 指标表')], editable=False,
            children=[
                WriterBlock(node_id='row-1', type='table_row', children=[
                    WriterBlock(node_id='cell-1', type='table_cell', content='名称',
                                numbering={'header': True}),
                    WriterBlock(node_id='cell-2', type='table_cell', content='数量',
                                numbering={'header': True, 'align': 'right'}),
                ]),
                WriterBlock(node_id='row-2', type='table_row', children=[
                    WriterBlock(node_id='cell-3', type='table_cell', content='苹果'),
                    WriterBlock(node_id='cell-4', type='table_cell', content='10'),
                ]),
            ],
        )
        document = WriterDocument(
            document_id='writer-document',
            provider_binding={'provider': 'notion', 'document_id': DOC_ID},
            blocks=[table],
        )

        native = NotionWriterAdapter().ir_to_blocks(document)
        table_payload = native[1]['table']

        assert [block['type'] for block in native] == ['paragraph', 'table']
        assert native[0]['paragraph']['rich_text'][0]['text']['content'] == '表1 指标表'
        assert table_payload['table_width'] == 2
        assert [
            cell[0]['text']['content']
            for cell in table_payload['children'][1]['table_row']['cells']
        ] == ['苹果', '10']

    def test_merge_restores_only_a_previously_recorded_table_caption(self):
        caption_id = '77777777-7777-7777-7777-777777777777'
        table_id = '55555555-5555-5555-5555-555555555555'
        row_id = '66666666-6666-6666-6666-666666666666'
        native = [
            _block(caption_id, 'paragraph', {'rich_text': [_rich('表1 指标表')]}),
            _block(table_id, 'table', {
                'table_width': 2, 'has_column_header': True, 'has_row_header': False,
            }, has_children=True),
            _block(row_id, 'table_row', {
                'cells': [[_rich('名称')], [_rich('数量')]],
            }, parent=table_id),
        ]
        adapter = NotionWriterAdapter()
        refreshed = adapter.blocks_to_ir(native, external_document_id=DOC_ID)
        assert [block.type for block in refreshed.blocks] == ['paragraph', 'table']
        assert refreshed.blocks[1].content == ''
        previous_table = refreshed.blocks[1].model_copy(deep=True)
        previous_table.content = '指标表'
        previous_table.spans = [WriterSpan(text='指标表')]
        previous_table.provider_payload['table_caption'] = {
            'provider_binding': {'provider': 'notion', 'block_id': caption_id},
        }
        previous = WriterDocument(
            document_id=refreshed.document_id,
            provider_binding=deepcopy(refreshed.provider_binding),
            blocks=[previous_table],
        )

        merged = adapter.merge_refreshed_document(previous, refreshed)

        assert len(merged.blocks) == 1
        assert merged.blocks[0].type == 'table'
        assert merged.blocks[0].content == '指标表'
        assert merged.blocks[0].provider_payload['table_caption'][
            'provider_binding']['block_id'] == caption_id

    @pytest.mark.parametrize(('block_type', 'expected_type', 'numbering', 'editable'), [
        ('heading_1', 'heading', {'level': 1}, True),
        ('heading_2', 'heading', {'level': 2}, True),
        ('heading_3', 'heading', {'level': 3}, True),
        ('heading_4', 'heading', {'level': 4}, True),
        ('bulleted_list_item', 'list_item', {'ordered': False}, True),
        ('numbered_list_item', 'list_item', {'ordered': True}, True),
    ])
    def test_maps_headings_and_lists(
            self, block_type, expected_type, numbering, editable):
        raw = _block(PARAGRAPH_ID, block_type, {'rich_text': [_rich('内容')]})

        block = NotionWriterAdapter().blocks_to_ir(
            [raw], external_document_id=DOC_ID).blocks[0]

        assert block.type == expected_type
        assert block.numbering == numbering
        assert block.editable is editable

    @pytest.mark.parametrize('internal', [False, True])
    def test_long_text_spans_preserve_styles_and_links(self, internal):
        length = 2001
        text = '中' * length
        link = ({'type': 'internal_ref', 'target_node_id': 'target'} if internal
                else {'url': 'https://example.com'})
        block = WriterBlock(node_id='text', type='paragraph', spans=[
            WriterSpan(text=text, style={'bold': True, 'text_color': 'red', 'link': link}),
            WriterSpan(text='尾部', style={'italic': True}),
        ])
        before = block.model_dump()
        rich = NotionWriterAdapter._spans_to_rich_text(
            block, lambda span: 'https://example.com/ref' if internal and span.style.get('link') else None)
        assert ''.join(item['text']['content'] for item in rich) == text + '尾部'
        assert all(len(item['text']['content']) <= 2000 for item in rich)
        assert len(rich) == (length + 1999) // 2000 + 1
        for item in rich[:-1]:
            assert item['annotations']['bold'] is True
            assert item['annotations']['color'] == 'red'
            assert item['text']['link'] == (
                {'url': 'https://example.com/ref', '_target_node_id': 'target'} if internal
                else {'url': 'https://example.com'})
        assert rich[-1]['annotations']['italic'] is True
        assert block.model_dump() == before

    def test_maps_rich_text_styles_links_mentions_and_equations(self):
        rich_text = [
            {
                **_rich('样式', bold=True, italic=True, underline=True,
                        strikethrough=True, code=True, color='red_background'),
                'text': {'content': '样式', 'link': {'url': 'https://example.com'}},
                'href': 'https://example.com',
            },
            {
                'type': 'mention',
                'mention': {'type': 'page', 'page': {'id': HEADING_ID}},
                'annotations': {'color': 'blue'},
                'plain_text': '页面',
                'href': None,
            },
            {
                'type': 'equation',
                'equation': {'expression': 'E=mc^2'},
                'annotations': {'color': 'default'},
                'plain_text': 'E=mc^2',
                'href': None,
            },
        ]
        raw = _block(PARAGRAPH_ID, 'paragraph', {'rich_text': rich_text})

        block = NotionWriterAdapter().blocks_to_ir(
            [raw], external_document_id=DOC_ID).blocks[0]

        assert block.content == '样式页面E=mc^2'
        assert block.spans[0].style == {
            'bold': True,
            'italic': True,
            'strikethrough': True,
            'underline': True,
            'inline_code': True,
            'background_color': 'red_background',
            'link': {'url': 'https://example.com'},
        }
        assert block.spans[1].style == {
            'text_color': 'blue',
            'notion:rich_text_type': 'mention',
            'notion:mention': {'type': 'page', 'page': {'id': HEADING_ID}},
        }
        assert block.spans[2].style == {
            'notion:rich_text_type': 'equation',
            'notion:equation': {'expression': 'E=mc^2'},
        }

    @pytest.mark.parametrize(
        ('block_type', 'payload', 'plain_text', 'expected_type', 'expected_content'), [
            ('code', {'rich_text': [_rich('print(1)')], 'language': 'python'},
             '', 'code', 'print(1)'),
            ('image', {'caption': [_rich('图注')]}, '', 'image', '图注'),
            ('divider', {}, '', 'divider', ''),
            ('link_preview', {'url': 'https://example.com'}, '',
             'link_preview', 'https://example.com'),
        ])
    def test_maps_special_blocks(
            self, block_type, payload, plain_text, expected_type, expected_content):
        raw = _block(PARAGRAPH_ID, block_type, payload)
        raw['plain_text'] = plain_text

        block = NotionWriterAdapter().blocks_to_ir(
            [raw], external_document_id=DOC_ID).blocks[0]

        assert block.type == expected_type
        assert block.content == expected_content
        if block_type == 'code':
            assert block.provider_payload['code_language'] == 'python'

    def test_normalizes_notion_table_rows_to_cells(self):
        table_id = '55555555-5555-5555-5555-555555555555'
        row_id = '66666666-6666-6666-6666-666666666666'
        table = _block(table_id, 'table', {
            'table_width': 2,
            'has_column_header': True,
            'has_row_header': False,
        }, has_children=True)
        linked = _rich('B1')
        linked['href'] = f'https://notion.so/{DOC_ID}#{HEADING_ID.replace("-", "")}'
        row = _block(row_id, 'table_row', {
            'cells': [[_rich('A1', bold=True)], [linked]],
        }, parent=table_id)
        row['plain_text'] = 'A1 | B1'
        target = _block(HEADING_ID, 'heading_1', {'rich_text': [_rich('目标')]})

        document = NotionWriterAdapter().blocks_to_ir(
            [target, table, row], external_document_id=DOC_ID)

        table_block = document.blocks[1]
        row_block = table_block.children[0]
        assert table_block.type == 'table'
        assert row_block.type == 'table_row'
        assert row_block.provider_binding['block_id'] == row_id
        assert row_block.content == ''
        assert [cell.content for cell in row_block.children] == ['A1', 'B1']
        assert row_block.children[0].spans[0].style['bold'] is True
        assert row_block.children[1].spans[0].style['link']['url'] == linked['href']
        assert row_block.provider_payload['table_cells'][0][0]['annotations']['bold'] is True
        assert row_block.provider_payload['table_cells'][1][0]['href'] == linked['href']

        row_block.children[1].content = 'B2'
        row_block.children[1].spans = []
        operation = NotionWriterAdapter().patch_to_operation(PatchHunk(
            target_node_id=row_block.children[1].node_id,
            modify_type='update',
            block=row_block.children[1],
        ), document)
        assert operation.params['block']['table_row']['cells'][0][0]['text']['content'] == 'A1'
        assert operation.params['block']['table_row']['cells'][1][0]['text']['content'] == 'B2'
        native = NotionWriterAdapter().ir_to_blocks(document)
        native_table = next(block for block in native if block['type'] == 'table')
        native_row = native_table['table']['children'][0]
        assert native_row['table_row']['cells'][1][0]['text']['content'] == 'B2'
        assert 'children' not in native_row['table_row']

    def test_ir_to_blocks_flattens_merged_cells_for_notion(self):
        table = WriterBlock(node_id='table', type='table', children=[
            WriterBlock(node_id='row-1', type='table_row', children=[
                WriterBlock(node_id='a', type='table_cell', content='A'),
                WriterBlock(node_id='b', type='table_cell', content='B'),
            ]),
            WriterBlock(node_id='row-2', type='table_row', children=[
                WriterBlock(
                    node_id='c', type='table_cell', content='C', numbering={'row_span': 2},
                ),
                WriterBlock(node_id='d', type='table_cell', content='D'),
            ]),
            WriterBlock(node_id='row-3', type='table_row', children=[
                WriterBlock(node_id='e', type='table_cell', content='E'),
            ]),
        ])
        document = WriterDocument(
            document_id='document', blocks=[table],
            provider_binding={'provider': 'notion', 'document_id': DOC_ID},
        )

        native = NotionWriterAdapter().ir_to_blocks(document)
        rows = native[0]['table']['children']

        assert native[0]['table']['table_width'] == 2
        assert [len(row['table_row']['cells']) for row in rows] == [2, 2, 2]
        assert rows[2]['table_row']['cells'][0] == []
        assert rows[2]['table_row']['cells'][1][0]['text']['content'] == 'E'

    def test_ir_to_blocks_round_trips_text_styles_payload_and_children(self):
        child = _block(CHILD_ID, 'paragraph', {'rich_text': [_rich('子段落')]},
                       parent=PARAGRAPH_ID)
        parent = _block(PARAGRAPH_ID, 'to_do', {
            'rich_text': [_rich('任务', bold=True)],
            'checked': True,
            'color': 'yellow_background',
        }, has_children=True)
        document = NotionWriterAdapter().blocks_to_ir(
            [parent, child], external_document_id=DOC_ID,
            uri=f'https://www.notion.so/Page-{DOC_ID.replace("-", "")}')

        blocks = NotionWriterAdapter().ir_to_blocks(document)

        assert blocks == [{
            'object': 'block',
            'type': 'to_do',
            'to_do': {
                'rich_text': [{
                    'type': 'text',
                    'annotations': {
                        'bold': True, 'italic': False, 'strikethrough': False,
                        'underline': False, 'code': False, 'color': 'default',
                    },
                    'text': {'content': '任务', 'link': None},
                }],
                'checked': True,
                'color': 'yellow_background',
                'children': [{
                    'object': 'block',
                    'type': 'paragraph',
                    'paragraph': {
                        'rich_text': [{
                            'type': 'text',
                            'annotations': {
                                'bold': False, 'italic': False,
                                'strikethrough': False, 'underline': False,
                                'code': False, 'color': 'default',
                            },
                            'text': {'content': '子段落', 'link': None},
                        }],
                    },
                }],
            },
        }]

    def test_ir_to_blocks_attaches_local_image_for_supplier_upload(self, tmp_path):
        image_path = tmp_path / 'image.png'
        image_path.write_bytes(b'png')
        document = WriterDocument(
            document_id='writer-doc',
            provider_binding={'provider': 'notion', 'document_id': DOC_ID},
            blocks=[WriterBlock(
                node_id='new-image',
                type='image',
                content='图片说明',
                references=[{'type': 'media_asset', 'id': 'asset-1'}],
            )],
        )
        media = MediaAssetLibrary(
            library_id='library-1',
            assets={
                'asset-1': MediaAsset(
                    media_asset_id='asset-1',
                    asset_type='image',
                    source_type='input_resource',
                    local_path=str(image_path),
                ),
            },
        )

        output = NotionWriterAdapter().ir_to_blocks(document, media_assets=media)

        assert output == [{
            'object': 'block',
            'type': 'image',
            'image': {
                'caption': [{
                    'type': 'text',
                    'annotations': {
                        'bold': False, 'italic': False, 'strikethrough': False,
                        'underline': False, 'code': False, 'color': 'default',
                    },
                    'text': {'content': '图片说明', 'link': None},
                }],
            },
            '_media': {
                'media_asset_id': 'asset-1',
                'local_path': str(image_path),
                'file_name': 'image.png',
            },
        }]

    @pytest.mark.parametrize('block_type', [
        'paragraph', 'heading_1', 'heading_2', 'heading_3', 'heading_4',
        'bulleted_list_item', 'numbered_list_item', 'to_do', 'quote',
        'callout', 'code',
    ])
    def test_update_patch_supports_all_editable_native_text_blocks(self, block_type):
        adapter = NotionWriterAdapter()
        payload = {'rich_text': [_rich('old')]}
        if block_type == 'code':
            payload['language'] = 'python'
        document = adapter.blocks_to_ir(
            [_block(PARAGRAPH_ID, block_type, payload)], external_document_id=DOC_ID)
        target = document.blocks[0]
        desired = target.model_copy(deep=True)
        desired.content = 'new'
        desired.spans = [WriterSpan(text='new', style={'bold': True})]
        operation = adapter.patch_to_operation(PatchHunk(
            target_node_id=target.node_id, modify_type='update', block=desired),
            document)

        assert operation.operation == 'update'
        assert operation.params['block_id'] == PARAGRAPH_ID
        assert operation.params['block']['type'] == block_type
        assert operation.params['block'][block_type]['rich_text'][0][
            'text']['content'] == 'new'
        assert operation.params['block'][block_type]['rich_text'][0][
            'annotations']['bold'] is True

    def test_create_patch_serializes_nested_subtree_with_temporary_ids(self):
        document = NotionWriterAdapter().blocks_to_ir(
            [_block(PARAGRAPH_ID, 'paragraph', {'rich_text': [_rich('existing')]})],
            external_document_id=DOC_ID,
            uri=f'https://www.notion.so/Page-{DOC_ID.replace("-", "")}',
        )
        created = WriterBlock(
            node_id='new-callout', type='callout', content='New container',
            children=[WriterBlock(
                node_id='new-paragraph', type='paragraph', content='Child',
                spans=[WriterSpan(text='Existing', style={
                    'link': {
                        'type': 'internal_ref',
                        'target_node_id': document.blocks[0].node_id,
                    },
                })],
            )],
        )

        operation = NotionWriterAdapter().patch_to_operation(PatchHunk(
            target_node_id=created.node_id, modify_type='create',
            block=created, index=1,
        ), document)

        assert operation.operation == 'create'
        assert operation.params['parent_block_id'] == DOC_ID
        assert operation.params['index'] == 1
        assert len(operation.params['blocks']) == 1
        native = operation.params['blocks'][0]
        assert native['_temporary_node_id'] == 'new-callout'
        child = native['callout']['children'][0]
        assert child['_temporary_node_id'] == 'new-paragraph'
        assert child['paragraph']['rich_text'][0]['text']['link']['url'].endswith(
            f'#{PARAGRAPH_ID.replace("-", "")}')

    def test_move_patch_clones_bound_subtree_to_requested_position(self):
        child = _block(CHILD_ID, 'paragraph', {'rich_text': [_rich('child')]},
                       parent=PARAGRAPH_ID)
        source = _block(PARAGRAPH_ID, 'callout', {
            'rich_text': [_rich('source')],
        }, has_children=True)
        sibling = _block(HEADING_ID, 'paragraph', {'rich_text': [_rich('sibling')]})
        document = NotionWriterAdapter().blocks_to_ir(
            [source, child, sibling], external_document_id=DOC_ID)
        source_node = document.blocks[0]

        operation = NotionWriterAdapter().patch_to_operation(PatchHunk(
            target_node_id=source_node.node_id, modify_type='move', index=1,
        ), document)

        assert operation.operation == 'move'
        assert operation.params['source_block_id'] == PARAGRAPH_ID
        assert operation.params['source_parent_block_id'] == DOC_ID
        assert operation.params['source_index'] == 0
        assert operation.params['target_parent_block_id'] == DOC_ID
        assert operation.params['target_index'] == 1
        assert operation.params['block']['_temporary_node_id'] == source_node.node_id
        child_native = operation.params['block']['callout']['children'][0]
        assert child_native['_temporary_node_id'] == source_node.children[0].node_id


class TestNotionProvider:
    def test_notion_provider_is_registered_and_matches_supported_locators(self):
        assert isinstance(get_writer_provider('notion'), NotionWriterProvider)
        assert isinstance(match_writer_provider(PAGE_URL), NotionWriterProvider)
        assert isinstance(
            match_writer_provider(f'notion:/~page/{PAGE_ID}'),
            NotionWriterProvider,
        )
        assert NotionWriterProvider().resolve(PAGE_URL) == TargetDocument(
            uri=PAGE_URL,
            adapter='notion',
        )

    def test_notion_provider_loads_ir_with_page_metadata(self, tmp_path):
        fs = _make_fs()
        with _route_notion(fs):
            result = WriterResourceTools(
                artifact_store=str(tmp_path),
            ).load_document({'uri': PAGE_URL, 'adapter': 'notion'})

        source = load_artifact_json(
            result['metadata']['artifact_paths']['source_document'],
            WriterDocument,
        )
        target = load_artifact_json(
            result['metadata']['artifact_paths']['target_document'],
            TargetDocument,
        )
        assert result['representation'] == 'ir'
        assert source.title == 'Writer page'
        assert source.revision == '2026-08-28T08:00:00.000Z'
        assert source.provider_binding['provider'] == 'notion'
        assert source.provider_binding['document_id'] == PAGE_ID
        assert target.doc_id == PAGE_ID
        assert target.meta['internal_uri'] == f'notion:/~page/{PAGE_ID}'
        fs.get_doc_blocks.assert_called_once_with(
            f'/~page/{PAGE_ID}', with_descendants=True)

    def test_notion_provider_replaces_writer_ir_and_syncs_title(self, tmp_path):
        fs = _make_fs()
        provider = NotionWriterProvider()
        with _route_notion(fs):
            loaded = provider.load_document(
                TargetDocument(uri=PAGE_URL, adapter='notion'),
            )
            document = loaded['source_document']
            result = provider.replace_document(
                document,
                loaded['target_document'],
            )

        assert result == {
            'doc_id': PAGE_ID,
            'adapter': 'notion',
            'locator': PAGE_URL,
            'block_count': 1,
            'warnings': [],
        }
        fs.update_page_title.assert_called_once_with(PAGE_ID, 'Writer page')
        fs.replace_doc_blocks.assert_called_once()
        assert fs.replace_doc_blocks.call_args.args[0] == PAGE_ID
        assert fs.replace_doc_blocks.call_args.args[1][0]['type'] == 'paragraph'

    def test_notion_provider_rejects_replace_after_remote_revision_changes(self):
        fs = _make_fs()
        provider = NotionWriterProvider()
        with _route_notion(fs):
            loaded = provider.load_document(
                TargetDocument(uri=PAGE_URL, adapter='notion'),
            )
            document = loaded['source_document']
            fs.get_document_metadata.return_value = {
                **_metadata(),
                'last_edited_time': '2026-08-28T08:01:00.000Z',
            }
            with pytest.raises(
                WriterProviderRevisionError,
                match='document changed since it was loaded',
            ):
                provider.replace_document(document, loaded['target_document'])

        fs.update_page_title.assert_not_called()
        fs.replace_doc_blocks.assert_not_called()

    def test_notion_provider_converts_markdown_to_ir_before_writing(self):
        pytest.importorskip('mistune')
        fs = _make_fs()
        with _route_notion(fs):
            result = NotionWriterProvider().replace_document(
                '# Safety guide\n\n## Before\n\nPrepare supplies.',
                TargetDocument(uri=PAGE_URL, adapter='notion'),
            )

        assert result['doc_id'] == PAGE_ID
        native = fs.replace_doc_blocks.call_args.args[1]
        # The Markdown H1 becomes the document title; its H2 section becomes the
        # first provider heading and the body remains a paragraph.
        assert [block['type'] for block in native] == ['heading_1', 'paragraph']
        assert native[0]['heading_1']['rich_text'][0]['text'][
            'content'] == '1. Before'
        fs.update_page_title.assert_called_once_with(PAGE_ID, 'Safety guide')

    def test_notion_provider_materializes_heading_and_figure_numbers_on_write(
            self, tmp_path):
        image_path = tmp_path / 'figure.png'
        image_path.write_bytes(b'fake image content')
        media_assets = MediaAssetLibrary(
            library_id='numbering-media',
            assets={
                'asset-1': MediaAsset(
                    media_asset_id='asset-1', asset_type='generated_image',
                    source_type='image_generation', local_path=str(image_path),
                ),
                'asset-2': MediaAsset(
                    media_asset_id='asset-2', asset_type='generated_image',
                    source_type='image_generation', local_path=str(image_path),
                ),
            },
        )
        document = WriterDocument(
            document_id='writer-numbering', stage='final',
            title='Numbered document',
            provider_binding={
                'provider': 'notion', 'document_id': PAGE_ID, 'uri': PAGE_URL},
            blocks=[
                WriterBlock(
                    node_id='section-1', type='heading', content='Preparation',
                    numbering={'level': 1},
                    children=[
                        WriterBlock(
                            node_id='section-1-1', type='heading',
                            content='Supplies', numbering={'level': 2},
                            children=[WriterBlock(
                                node_id='figure-1', type='image',
                                content='Emergency supplies',
                                references=[
                                    {'type': 'media_asset', 'id': 'asset-1'}],
                            )],
                        ),
                    ],
                ),
                WriterBlock(
                    node_id='section-2', type='heading', content='Recovery',
                    numbering={'level': 1},
                    children=[WriterBlock(
                        node_id='figure-2', type='image',
                        content='Safety checklist',
                        references=[
                            {'type': 'media_asset', 'id': 'asset-2'}],
                    )],
                ),
            ],
        )
        fs = _make_fs()

        with _route_notion(fs):
            NotionWriterProvider().replace_document(
                document, TargetDocument(uri=PAGE_URL, adapter='notion'),
                media_assets=media_assets,
            )

        native = fs.replace_doc_blocks.call_args.args[1]
        assert [block['type'] for block in native] == [
            'heading_1', 'heading_2', 'image', 'heading_1', 'image',
        ]
        assert native[0]['heading_1']['rich_text'][0]['text'][
            'content'] == '1. Preparation'
        assert native[1]['heading_2']['rich_text'][0]['text'][
            'content'] == '1.1. Supplies'
        assert native[2]['image']['caption'][0]['text'][
            'content'] == '图1 Emergency supplies'
        assert native[3]['heading_1']['rich_text'][0]['text'][
            'content'] == '2. Recovery'
        assert native[4]['image']['caption'][0]['text'][
            'content'] == '图2 Safety checklist'
        # Publishing uses a materialized copy and does not pollute the canonical IR.
        assert document.blocks[0].content == 'Preparation'
        assert document.blocks[0].children[0].children[0].content == (
            'Emergency supplies')

    def test_notion_numbering_sync_updates_supported_blocks_and_skips_table(self):
        heading = WriterBlock(
            node_id='heading-1', type='heading', content='台风期间', stage='final',
            numbering={'level': 1}, editable=True,
            provider_payload={'raw_block': {
                'type': 'heading_1',
                'heading_1': {'rich_text': [{
                    'type': 'text', 'text': {'content': '台风期间'},
                }]},
            }},
        )
        image = WriterBlock(
            node_id='image-1', type='image', content='安全示意图', stage='final',
            editable=False,
            provider_payload={'raw_block': {
                'type': 'image',
                'image': {'caption': [{
                    'type': 'text', 'text': {'content': '安全示意图'},
                }]},
            }},
        )
        code = WriterBlock(
            node_id='code-1', type='code', content='print("safe")', stage='final',
            editable=True,
            provider_payload={'raw_block': {
                'type': 'code',
                'code': {
                    'rich_text': [{
                        'type': 'text',
                        'text': {'content': 'print("safe")'},
                    }],
                    'caption': [],
                    'language': 'python',
                },
            }},
        )
        table = WriterBlock(
            node_id='table-1', type='table', content='Safety table', stage='final',
            editable=False,
            provider_payload={'raw_block': {'type': 'table', 'table': {}}},
        )
        persisted = WriterDocument(
            document_id='writer-doc', stage='final',
            blocks=[heading, image, code, table])
        numbered = persisted.model_copy(deep=True)
        numbered.blocks[0].content = '1. 台风期间'
        numbered.blocks[1].content = '图1 安全示意图'
        numbered.blocks[2].provider_payload['numbering_caption'] = '代码1'
        numbered.blocks[3].content = '表1 Safety table'

        hunks = NotionWriterProvider._numbering_sync_hunks(numbered, persisted)

        assert [hunk.hunk_id for hunk in hunks] == [
            'heading-numbering-sync-heading-1',
            'image-numbering-sync-image-1',
            'code-numbering-sync-code-1',
        ]
        assert hunks[1].meta == {
            'source': 'system_numbering', 'update_scope': 'caption',
        }
        assert hunks[2].meta == {
            'source': 'system_numbering', 'update_scope': 'caption',
        }

    def test_notion_provider_applies_image_numbering(self):
        provider = NotionWriterProvider()
        image = NotionWriterAdapter().blocks_to_ir([
            _block(BLOCK_ID, 'image', {
                'type': 'external', 'external': {'url': 'https://example.com/image.png'},
                'caption': [_rich('Safety')],
            }, parent=PAGE_ID),
        ], external_document_id=PAGE_ID, stage='final', title='Before', uri=PAGE_URL)
        target = TargetDocument(uri=PAGE_URL, adapter='notion')
        fs = _make_fs()
        fs.update_block.return_value = {'block_id': BLOCK_ID, 'block_type': 'image'}
        provider._resolve_document_target = MagicMock(return_value=(
            'notion', f'/~page/{PAGE_ID}', fs, NotionWriterAdapter(), PAGE_URL, PAGE_ID,
        ))
        provider._read_persisted_document = MagicMock(
            side_effect=lambda **kwargs: kwargs['source_document'].model_copy(deep=True))
        result = provider.apply_patch_to_document(
            PatchSet(target_doc_id=image.document_id, new_title='After'), image, target)
        fs.update_block.assert_called_once()
        assert result['persisted_document'].blocks[0].content == '图1 Safety'
        caption = fs.update_block.call_args.kwargs['block']['image']['caption']
        assert caption[0]['text']['content'] == '图1 Safety'

    def test_notion_provider_creates_private_workspace_page_without_parent(self):
        fs = _make_fs()
        fs.create_document.return_value = {
            'document_id': PAGE_ID,
            'title': 'Private page',
            'browser_url': PAGE_URL,
            'internal_uri': f'notion:/~page/{PAGE_ID}',
            'last_edited_time': '2026-08-28T08:00:00.000Z',
        }
        with _route_notion(fs, real_path='/'):
            target = NotionWriterProvider().create_document('Private page')

        assert target.adapter == 'notion'
        assert target.doc_id == PAGE_ID
        assert target.uri == PAGE_URL
        assert target.meta['parent_uri'] == ''
        fs.create_document.assert_called_once_with('Private page', '/')

    def test_notion_provider_applies_update_and_refreshes_persisted_document(self):
        fs = _make_fs()
        updated = _raw_blocks()
        updated[0]['paragraph']['rich_text'][0]['plain_text'] = 'Updated'
        updated[0]['paragraph']['rich_text'][0]['text']['content'] = 'Updated'
        fs.get_doc_blocks.side_effect = [_raw_blocks(), updated]
        fs.get_document_metadata.side_effect = [
            _metadata(), _metadata(), _metadata(), _metadata(),
            {**_metadata(), 'last_edited_time': '2026-08-28T08:01:00.000Z'},
        ]
        provider = NotionWriterProvider()
        with _route_notion(fs):
            loaded = provider.load_document(
                TargetDocument(uri=PAGE_URL, adapter='notion'))
            source = loaded['source_document']
            desired = source.blocks[0].model_copy(update={
                'content': 'Updated', 'spans': [WriterSpan(text='Updated')],
            })
            result = provider.apply_patch_to_document(PatchSet(
                target_doc_id=source.document_id,
                hunks=[PatchHunk(
                    hunk_id='update-paragraph', target_node_id=desired.node_id,
                    modify_type='update', block=desired,
                )],
            ), source, loaded['target_document'])

        fs.update_block.assert_called_once()
        assert result['patch_result'].success is True
        assert result['patch_result'].applied_hunks == ['update-paragraph']
        assert result['persisted_document'].blocks[0].content == 'Updated'
        assert result['persisted_document'].blocks[0].node_id == (
            source.blocks[0].node_id)
        assert result['persisted_document'].revision == (
            '2026-08-28T08:01:00.000Z')

    def test_notion_provider_creates_then_updates_without_intermediate_refresh(self):
        fs = _make_fs()
        created_id = '22222222-2222-2222-2222-222222222222'
        created_raw = _block(created_id, 'paragraph', {
            'rich_text': [_rich('Updated after create')],
        }, parent=PAGE_ID)
        fs.get_doc_blocks.side_effect = [_raw_blocks(), [*_raw_blocks(), created_raw]]
        fs.create_block.return_value = {
            'block_id': created_id,
            'block_id_relations': [{
                'temporary_block_id': 'new-paragraph', 'block_id': created_id,
            }],
        }
        provider = NotionWriterProvider()
        with _route_notion(fs):
            loaded = provider.load_document(TargetDocument(uri=PAGE_URL, adapter='notion'))
            source = loaded['source_document']
            created = WriterBlock(node_id='new-paragraph', type='paragraph', content='Created')
            updated = created.model_copy(update={'content': 'Updated after create'})
            result = provider.apply_patch_to_document(PatchSet(
                target_doc_id=source.document_id,
                hunks=[
                    PatchHunk(hunk_id='create', target_node_id=created.node_id,
                              modify_type='create', block=created, index=1),
                    PatchHunk(hunk_id='update', target_node_id=created.node_id,
                              modify_type='update', block=updated),
                ],
            ), source, loaded['target_document'])

        fs.create_block.assert_called_once()
        call = fs.create_block.call_args.kwargs
        assert (call['document_id'], call['parent_block_id'], call['index']) == (PAGE_ID, PAGE_ID, 1)
        assert len(call['blocks']) == 1
        assert call['blocks'][0]['_temporary_node_id'] == created.node_id
        fs.update_block.assert_called_once()
        assert fs.update_block.call_args.kwargs['block_id'] == created_id
        assert fs.get_doc_blocks.call_count == 2  # Initial load and final confirmation.
        assert result['patch_result'].applied_hunks == ['create', 'update']
        persisted = result['persisted_document'].blocks[1]
        assert (persisted.node_id, persisted.content) == (created.node_id, updated.content)

    def test_notion_provider_moves_block_and_preserves_writer_node_id(self):
        fs = _make_fs()
        second_id = '22222222-2222-2222-2222-222222222222'
        moved_id = '33333333-3333-3333-3333-333333333333'
        second = deepcopy(_raw_blocks()[0])
        second['id'] = second['block_id'] = second_id
        second['paragraph']['rich_text'][0]['plain_text'] = 'Second'
        second['paragraph']['rich_text'][0]['text']['content'] = 'Second'
        moved = deepcopy(_raw_blocks()[0])
        moved['id'] = moved['block_id'] = moved_id
        fs.get_doc_blocks.side_effect = [
            [*_raw_blocks(), second], [second, moved]]
        fs.get_document_metadata.side_effect = [
            _metadata(), _metadata(), _metadata(), _metadata(),
            {**_metadata(), 'last_edited_time': '2026-08-28T08:01:00.000Z'},
        ]
        provider = NotionWriterProvider()
        fs.move_block.return_value = {
            'block_id': moved_id,
            'source_block_id': BLOCK_ID,
            'block_id_relations': [{
                'temporary_block_id': '', 'block_id': moved_id,
            }],
        }
        with _route_notion(fs):
            loaded = provider.load_document(
                TargetDocument(uri=PAGE_URL, adapter='notion'))
            source = loaded['source_document']
            moved_node_id = source.blocks[0].node_id
            fs.move_block.return_value['block_id_relations'][0][
                'temporary_block_id'] = moved_node_id
            result = provider.apply_patch_to_document(PatchSet(
                target_doc_id=source.document_id,
                hunks=[PatchHunk(
                    hunk_id='move-paragraph', target_node_id=moved_node_id,
                    modify_type='move', index=1,
                )],
            ), source, loaded['target_document'])

        fs.move_block.assert_called_once()
        assert result['patch_result'].applied_hunks == ['move-paragraph']
        assert [block.content for block in result[
            'persisted_document'].blocks] == ['Second', 'Notion content']
        assert result['persisted_document'].blocks[1].node_id == moved_node_id


def _table_native_id(value):
    return str(uuid5(NAMESPACE_URL, value))


def _cell_batch_setup():
    rows = []
    for r in range(2):
        cells = []
        for c in range(2):
            node_id = f'cell-{r}-{c}'
            cells.append(WriterBlock(node_id=node_id, type='table_cell', content=node_id,
                                     provider_binding={}, provider_payload={}))
        rows.append(WriterBlock(
            node_id=f'row-{r}', type='table_row', children=cells,
            provider_binding={'provider': 'notion', 'block_id': _table_native_id(f'row-{r}')},
            provider_payload={'raw_block': {'type': 'table_row', 'table_row': {}}},
        ))
    document = WriterDocument(
        document_id='writer-doc', revision='10',
        provider_binding={'provider': 'notion', 'document_id': 'doc'},
        blocks=[
            WriterBlock(node_id='table', type='table', children=rows,
                        provider_binding={'provider': 'notion', 'block_id': _table_native_id('table'),
                                          'parent_block_id': 'doc'}),
            WriterBlock(node_id='paragraph', type='paragraph', content='end',
                        provider_binding={'provider': 'notion', 'block_id': _table_native_id('paragraph'),
                                          'parent_block_id': 'doc'},
                        provider_payload={'raw_block': {'type': 'paragraph', 'paragraph': {}}}),
        ],
    )
    instance = NotionWriterProvider()
    adapter = NotionWriterAdapter()
    fs = MagicMock()
    fs.update_block.side_effect = [{'document_revision_id': value} for value in range(11, 30)]
    instance._resolve_document_target = MagicMock(return_value=('notion', '/doc', fs, adapter, '/doc', 'doc'))
    instance._document_metadata = MagicMock(return_value={'last_edited_time': '10'})
    instance._read_persisted_document = MagicMock(side_effect=lambda **kw: kw['source_document'].model_copy(deep=True))
    return (instance, fs, document)


def _cell_edit(document, node_id, value, hunk_id=None):
    return PatchHunk(
        hunk_id=hunk_id or f'edit-{node_id}', target_node_id=node_id, modify_type='update',
        block=document.block_by_id(node_id).model_copy(update={
            'content': value, 'spans': [WriterSpan(text=value, style={'bold': True})],
        }),
    )


def _apply_cell_hunks(instance, document, hunks):
    return instance.apply_patch_to_document(PatchSet(target_doc_id=document.document_id, hunks=hunks), document,
                                            TargetDocument(doc_id='doc', adapter=instance.provider, uri='/doc'))


def test_notion_cells_merge_by_provider_granularity_and_repeated_edit_keeps_last():
    instance, fs, document = _cell_batch_setup()
    before = document.model_dump()
    hunks = [_cell_edit(document, 'cell-0-0', 'first', 'h1'), _cell_edit(document, 'cell-1-0', 'other row', 'h2'),
             _cell_edit(document, 'cell-0-1', 'same row', 'h3'), _cell_edit(document, 'cell-0-0', 'last', 'h4')]
    result = _apply_cell_hunks(instance, document, hunks)
    assert fs.update_block.call_count == 2
    assert result['patch_result'].applied_hunks == ['h1', 'h2', 'h3', 'h4']
    assert document.model_dump() == before
    assert result['persisted_document'].block_by_id('cell-0-0').content == 'last'
    instance._read_persisted_document.assert_called_once()
    calls = {call.kwargs['block_id']: call.kwargs['block']['table_row']['cells']
             for call in fs.update_block.call_args_list}
    assert [
        [cell[0]['text']['content'] for cell in calls[_table_native_id(row)]]
        for row in ['row-0', 'row-1']
    ] == [['last', 'same row'], ['other row', 'cell-1-1']]
    assert calls[_table_native_id('row-0')][0][0]['annotations']['bold'] is True


def test_notion_move_flushes_cells_and_following_batch_uses_new_ids():
    instance, fs, document = _cell_batch_setup()
    relations = [{'temporary_block_id': key, 'block_id': _table_native_id(key + '-moved')} for key in ['table',
                                                                                                       'row-0', 'row-1']]
    fs.move_block.return_value = {'block_id_relations': relations, 'document_revision_id': 20}
    _apply_cell_hunks(instance, document, [
        _cell_edit(document, 'cell-0-0', 'before'),
        PatchHunk(hunk_id='move', target_node_id='table', modify_type='move', index=1),
        _cell_edit(document, 'cell-0-1', 'after'),
    ])
    assert [call[0] for call in fs.method_calls] == ['update_block', 'move_block', 'update_block']
    assert fs.update_block.call_args.kwargs['block_id'] == _table_native_id('row-0-moved')
    cells = fs.update_block.call_args.kwargs['block']['table_row']['cells']
    assert [cell[0]['text']['content'] for cell in cells] == ['before', 'after']


def test_notion_math_is_read_only_and_inline_equations_roundtrip():
    adapter = NotionWriterAdapter()
    raw = [
        _block(HEADING_ID, 'equation', {'expression': 'x^2'}),
        _block(PARAGRAPH_ID, 'paragraph', {'rich_text': [
            _rich('before '), {'type': 'equation', 'equation': {'expression': 'y^2'}}, _rich(' after'),
        ]}),
    ]
    document = adapter.blocks_to_ir(raw, external_document_id=DOC_ID)
    assert document.blocks[0].type == 'math'
    assert not document.blocks[0].editable
    assert adapter.ir_to_blocks(document)[0]['equation']['expression'] == 'x^2'
    with pytest.raises(ValueError, match='does not support updates'):
        adapter.patch_to_operation(PatchHunk(
            target_node_id=document.blocks[0].node_id, modify_type='update',
            block=document.blocks[0].model_copy(update={'content': '$$x^3$$'})), document)
    rich = adapter.ir_to_blocks(document)[1]['paragraph']['rich_text']
    assert rich[1]['equation']['expression'] == 'y^2'
    assert rich[0]['text']['content'] == 'before '
    assert rich[-1]['text']['content'] == ' after'
