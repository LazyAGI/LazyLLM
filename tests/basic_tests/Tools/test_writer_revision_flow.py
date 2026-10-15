'''Core revision regressions with minimal documents and mocked model responses.'''
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lazyllm.tools.writer.data_models import WriterBlock, WriterDocument, WritingContext, WritingTask
from lazyllm.tools.writer.data_models.revision import PatchSet
from lazyllm.tools.writer.tools.revision_tools import WriterRevisionTools
from lazyllm.tools.writer.utils import load_artifact_json


IMAGE = '![image](image_(1).png)'
MARKDOWN = '# A\n\nFirst. Last.\n\nSecond.\n\nOld.\n\n' + IMAGE + '\n\n|A|\n|---|\n|1|\n\n[^r]: ref\n'


def _edit(operation, **fields):
    return {'modify_type': operation, 'content_ref': {'heading_path': ['A']},
            'instruction': 'Apply the requested change only.', **fields}


def _revise(tmp_path, source, instructions, generated=None, *, title=False):
    original = deepcopy(source)
    responses = [{'target_title': title, 'targets': [{'content_ref': item['content_ref']} for item in instructions]}]
    if instructions:
        responses.append({'scope': 'block', 'instructions': [
            {'instruction_id': str(i), **item} for i, item in enumerate(instructions, 1)
        ]})
    if generated is not None:
        responses.append(generated)
    model = MagicMock(side_effect=responses)
    tool = WriterRevisionTools(llm=MagicMock(), artifact_store=str(tmp_path))
    task, context = WritingTask(task_type='revise', query='Revise the document.'), WritingContext(context_id='test')
    with patch.object(tool, '_build_structured_llm', return_value=model):
        located = tool.locate_revision_target(task, source, context)
        plan = tool.generate_modify_plan(task, source, located['artifact_path'], context)
        changes = tool.generate_revision_set(source, plan['artifact_path'], context)
        applied = tool.apply_revision(source, changes['artifact_path'], context)
    assert model.call_count == len(responses)
    assert source == original
    assert load_artifact_json(applied['artifact_path'], validate_schema=False)['success']
    path = applied['metadata']['artifact_paths']['revised_document']
    revised = Path(path).read_text() if isinstance(source, str) else load_artifact_json(path, WriterDocument)
    return revised, changes['artifact_path']


@pytest.mark.parametrize('scope,position,expected', [
    ('paragraph', 'before', 'New.\n\nFirst. Last.'),
    ('paragraph', 'after', 'First. Last.\n\nNew.'),
    ('fragment', 'before', 'First. New. Last.'),
    ('fragment', 'after', 'First. Last. New.'),
])
def test_markdown_insertion_boundary(tmp_path, scope, position, expected):
    revised, _ = _revise(tmp_path, MARKDOWN, [_edit(
        'create', target_scope=scope, locator_text='Last.', position=position,
    )], {'insert_text': 'New.' if scope == 'paragraph' else ('New. ' if position == 'before' else ' New.')})
    assert revised == MARKDOWN.replace('First. Last.', expected)


def test_markdown_batch_preserves_distinct_targets_and_unrelated_content(tmp_path):
    revised, _ = _revise(tmp_path, MARKDOWN, [
        _edit('create', target_scope='paragraph', locator_text='First.', position='after'),
        _edit('create', target_scope='paragraph', locator_text='Second.', position='after'),
        _edit('update', target_scope='paragraph', locator_text='Old.'),
    ], {'insertions': {'instruction-1': {'insert_text': 'One.'}, 'instruction-2': {'insert_text': 'Two.'}},
        'contents': {'instruction-3': {'new_string': 'Updated.'}}})
    expected = MARKDOWN.replace('First. Last.', 'First. Last.\n\nOne.').replace('Second.', 'Second.\n\nTwo.')
    assert revised == expected.replace('Old.', 'Updated.')


def test_markdown_deletes_complete_image_only(tmp_path):
    revised, _ = _revise(tmp_path, MARKDOWN, [_edit('delete', target_scope='image', locator_text='image')])
    assert revised == MARKDOWN.replace(IMAGE, '')


def _document(flat=False):
    body = [
        WriterBlock(node_id='p', type='paragraph', content='Body', references=[{'type': 'citation', 'id': 'r'}]),
        WriterBlock(node_id='image', type='image', content='Image', editable=False),
        WriterBlock(node_id='table', type='table', children=[WriterBlock(node_id='row', type='table_row', children=[
            WriterBlock(node_id='cell', type='table_cell', content='1'),
        ])]),
        WriterBlock(node_id='sub', type='heading', numbering={'level': 3}, children=[
            WriterBlock(node_id='sub-body', type='paragraph', content='Sub body'),
        ]),
    ]
    source = WriterDocument(document_id='doc', title='Title', blocks=[
        WriterBlock(node_id='a', type='heading', content='A', numbering={'level': 2}, children=body),
        WriterBlock(node_id='b', type='heading', content='B', numbering={'level': 2}),
    ])
    if flat:
        source.blocks = [block for block in source.iter_blocks() if block.type not in {'table_row', 'table_cell'}]
        for block in source.blocks:
            if block.type == 'heading':
                block.children = []
    return source


@pytest.mark.parametrize('document_title', [False, True], ids=['heading', 'document-title'])
def test_ir_title_change_preserves_other_content(tmp_path, document_title):
    source = _document()
    instructions = [] if document_title else [_edit('update', content_ref={'node_id': 'a'})]
    generated = {'new_title': 'Updated'} if document_title else {'changes': {'1': [{
        'type': 'heading', 'content': 'Updated', 'children': [{'type': 'paragraph', 'content': 'Extra'}],
    }]}}
    revised, _ = _revise(tmp_path, source, instructions, generated, title=document_title)
    expected = source.model_copy(deep=True)
    if document_title:
        expected.title = 'Updated'
    else:
        expected.blocks[0].content = 'Updated'
    assert revised == expected


@pytest.mark.parametrize('flat', [False, True], ids=['nested', 'flat'])
def test_ir_insertion_distinguishes_block_and_section_without_moving_content(tmp_path, flat):
    source = _document(flat)
    revised, changes = _revise(tmp_path, source, [
        _edit('create', content_ref={'node_id': 'a'}, position='after', anchor_scope='block'),
        _edit('create', content_ref={'node_id': 'a'}, position='after', anchor_scope='section'),
    ], {'changes': {
        '1': [{'type': 'paragraph', 'content': 'Note'}],
        '2': [{'type': 'heading', 'content': 'New', 'numbering': {'level': 2},
               'children': [{'type': 'paragraph', 'content': 'New body'}]}],
    }})
    patch_set = load_artifact_json(changes, PatchSet)
    assert [item.modify_type for item in patch_set.hunks] == ['create', 'create']
    added = {block.content: block for block in revised.iter_blocks()}
    assert added['New'].numbering == {'level': 2}
    assert [block.content for block in added['New'].children] == ['New body']
    expected = source.model_copy(deep=True)
    if flat:
        expected.blocks.insert(1, added['Note'])
    else:
        expected.blocks[0].children.insert(0, added['Note'])
    expected.blocks.insert(-1, added['New'])
    assert revised == expected
