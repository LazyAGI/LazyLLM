# Copyright (c) 2026 LazyAGI. All rights reserved.
from copy import deepcopy
from typing import Any, Dict, List, Set, Tuple


DOCX_BLOCK_TYPE_FIELDS: Dict[int, str] = {
    1: 'page', 2: 'text', 3: 'heading1', 4: 'heading2', 5: 'heading3', 6: 'heading4',
    7: 'heading5', 8: 'heading6', 9: 'heading7', 10: 'heading8', 11: 'heading9',
    12: 'bullet', 13: 'ordered', 14: 'code', 15: 'quote', 17: 'todo',
    19: 'callout', 22: 'divider', 24: 'grid', 25: 'grid_column',
    31: 'table', 32: 'table_cell', 34: 'quote_container',
}

_DOCX_DEFAULT_CLONE_FIELDS = {
    'align': 1,
    'done': False,
    'folded': False,
    'wrap': False,
    'language': 1,
    'indentation_level': 'NoIndent',
    'bold': False,
    'italic': False,
    'strikethrough': False,
    'underline': False,
    'inline_code': False,
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


def _normalize_clone_value(value: Any, normalized_fields: Set[str], path: str = '') -> Any:
    if isinstance(value, list):
        normalized = [
            _normalize_clone_value(item, normalized_fields, path)
            for item in value
        ]
        return [item for item in normalized if item not in (None, {}, [])]
    if not isinstance(value, dict):
        return value

    normalized: Dict[str, Any] = {}
    for key, item in value.items():
        field_path = f'{path}.{key}' if path else key
        if key == 'comment_ids':
            normalized_fields.add(field_path)
            continue
        if key == 'title' and path.endswith('mention_doc'):
            normalized_fields.add(field_path)
            continue
        if path.endswith('elements') and key in {
            'file', 'inline_block', 'undefined', 'undefined_element',
        }:
            normalized_fields.add(field_path)
            continue
        if key in _DOCX_DEFAULT_CLONE_FIELDS \
                and item == _DOCX_DEFAULT_CLONE_FIELDS[key]:
            normalized_fields.add(field_path)
            continue
        normalized_item = _normalize_clone_value(item, normalized_fields, field_path)
        if normalized_item not in (None, {}, []):
            normalized[key] = normalized_item
    return normalized


def normalize_docx_clone_content(
    block_type: Any,
    content: Any,
) -> Tuple[Dict[str, Any], List[str]]:
    normalized_fields: Set[str] = set()
    normalized = _normalize_clone_value(
        deepcopy(content) if isinstance(content, dict) else {},
        normalized_fields,
    )
    if block_type == 31:
        if normalized.pop('cells', None) is not None:
            normalized_fields.add('table.cells')
        if normalized.pop('merge_info', None) is not None:
            normalized_fields.add('table.merge_info')
        prop = dict(normalized.get('property') or {})
        if prop.pop('merge_info', None) is not None:
            normalized_fields.add('table.property.merge_info')
        if prop:
            normalized['property'] = prop
        else:
            normalized.pop('property', None)
    return normalized, sorted(normalized_fields)


def prepare_docx_clone_descendants(
    blocks: List[Dict[str, Any]],
    root_block_id: str,
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, str], List[str]]:
    raw_by_id = {
        block.get('block_id'): block
        for block in blocks
        if isinstance(block, dict) and isinstance(block.get('block_id'), str)
    }
    if root_block_id not in raw_by_id:
        raise ValueError(f'Feishu move source block {root_block_id!r} does not exist.')

    children_by_parent: Dict[str, List[str]] = {}
    for block_id, block in raw_by_id.items():
        raw_children = block.get('children') or []
        if not isinstance(raw_children, list):
            raise TypeError(f'Feishu block {block_id!r}.children must be a list.')
        missing_children = [
            child_id for child_id in raw_children if child_id not in raw_by_id
        ]
        if missing_children:
            raise ValueError(
                f'Feishu move subtree is missing child blocks: {missing_children}.')
        children = list(raw_children)
        if children:
            children_by_parent[block_id] = children
    for block_id, block in raw_by_id.items():
        parent_id = block.get('parent_id')
        if parent_id in raw_by_id and block_id not in children_by_parent.get(parent_id, []):
            children_by_parent.setdefault(parent_id, []).append(block_id)

    ordered_ids: List[str] = []
    visiting: Set[str] = set()
    visited: Set[str] = set()

    def visit(block_id: str) -> None:
        if block_id in visiting:
            raise ValueError(f'cycle detected in Feishu move subtree at {block_id!r}.')
        if block_id in visited:
            raise ValueError(
                f'Feishu move subtree contains block {block_id!r} more than once.')
        visiting.add(block_id)
        ordered_ids.append(block_id)
        for child_id in children_by_parent.get(block_id, []):
            visit(child_id)
        visiting.remove(block_id)
        visited.add(block_id)

    visit(root_block_id)
    source_by_temporary_id = {
        f'move-block-{index}': block_id
        for index, block_id in enumerate(ordered_ids)
    }
    temporary_by_source_id = {
        source_id: temporary_id
        for temporary_id, source_id in source_by_temporary_id.items()
    }

    normalized_fields: Set[str] = set()
    descendants: List[Dict[str, Any]] = []
    for source_id in ordered_ids:
        raw = raw_by_id[source_id]
        block_type = raw.get('block_type')
        content_key = DOCX_BLOCK_TYPE_FIELDS.get(block_type)
        if content_key is None:
            raise ValueError(
                f'Feishu block type {block_type!r} cannot be cloned safely.')
        content, fields = normalize_docx_clone_content(block_type, raw.get(content_key))
        normalized_fields.update(f'{source_id}.{field}' for field in fields)
        normalized_fields.update(
            f'{source_id}.{field}'
            for field in raw
            if field not in {
                'block_id', 'block_type', 'parent_id', 'children',
                'plain_text', content_key,
            }
        )
        descendant = {
            'block_id': temporary_by_source_id[source_id],
            'block_type': block_type,
            content_key: content,
        }
        children = children_by_parent.get(source_id)
        if children:
            descendant['children'] = [
                temporary_by_source_id[child_id] for child_id in children
            ]
        descendants.append(descendant)

    return (
        [temporary_by_source_id[root_block_id]],
        descendants,
        source_by_temporary_id,
        sorted(normalized_fields),
    )


__all__ = [
    'DOCX_BLOCK_TYPE_FIELDS',
    'normalize_docx_clone_content',
    'prepare_docx_clone_descendants',
    'prepare_docx_descendants',
]
