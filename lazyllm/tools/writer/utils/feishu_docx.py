# Copyright (c) 2026 LazyAGI. All rights reserved.
from typing import Any, Dict, List, Tuple


DOCX_BLOCK_TYPE_FIELDS: Dict[int, str] = {
    1: 'page', 2: 'text', 3: 'heading1', 4: 'heading2', 5: 'heading3', 6: 'heading4',
    7: 'heading5', 8: 'heading6', 9: 'heading7', 10: 'heading8', 11: 'heading9',
    12: 'bullet', 13: 'ordered', 14: 'code', 15: 'quote', 17: 'todo',
    19: 'callout', 22: 'divider', 24: 'grid', 25: 'grid_column',
    31: 'table', 32: 'table_cell', 34: 'quote_container',
}


def prepare_docx_descendants(
    blocks: List[Dict[str, Any]],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    '''Build Feishu descendant-create payloads from a flat block tree.'''
    content_blocks = [block for block in blocks if block.get('block_type') != 1]
    raw_by_id = {block['block_id']: block for block in content_blocks}
    children_by_parent: Dict[str, List[str]] = {}
    for block in content_blocks:
        parent_id = block.get('parent_id')
        if parent_id in raw_by_id:
            children_by_parent.setdefault(parent_id, []).append(block['block_id'])

    root_block_ids: List[str] = []
    descendants: List[Dict[str, Any]] = []
    for block in content_blocks:
        block_id = block['block_id']
        block_type = block.get('block_type')
        content_key = DOCX_BLOCK_TYPE_FIELDS.get(block_type)
        if content_key is None:
            raise ValueError(
                f'Feishu block type {block_type!r} is not supported for structured writing.')
        content = block.get(content_key)
        if block_type == 22 and content is None:
            content = {}
        content = dict(content)
        if block_type == 31:
            content.pop('cells', None)
            content.pop('merge_info', None)
            prop = dict(content.get('property') or {})
            prop.pop('merge_info', None)
            content['property'] = prop

        descendant = {
            'block_id': block_id,
            'block_type': block_type,
            content_key: content,
        }
        children = children_by_parent.get(block_id)
        if children:
            descendant['children'] = children
        descendants.append(descendant)
        if block.get('parent_id') not in raw_by_id:
            root_block_ids.append(block_id)
    return root_block_ids, descendants


__all__ = ['DOCX_BLOCK_TYPE_FIELDS', 'prepare_docx_descendants']
