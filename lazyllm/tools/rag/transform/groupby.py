import copy
import functools
import re
from typing import Any, List, Optional, Callable

from ..doc_node import DocNode
from .base import NodeTransform, RuleSet


class GroupNodeParser(NodeTransform):
    __requires_all_nodes__ = True

    def __init__(self, num_workers: int = 0, max_length: int = 2048,
                 merge_title: bool = True, return_trace: bool = False, **kwargs):
        rules = RuleSet()
        super().__init__(rules=rules, return_trace=return_trace, **kwargs)

        self.max_length = max_length
        self.merge_title = merge_title
        self._number_workers = num_workers

        self.on_match = self._default_group_handler
        self.on_miss = self._default_group_handler

    def forward(self, document: List[DocNode], **kwargs) -> List[DocNode]:
        return self._parse_nodes(document, **kwargs)

    def _default_group_handler(self, node, match_result_or_ctx, ctx_or_none=None):
        return node

    def process(self, nodes: List[Any], on_match: Optional[Callable] = None,
                on_miss: Optional[Callable] = None) -> List[Any]:
        if on_match is not None or on_miss is not None:
            return super().process(nodes, on_match=on_match, on_miss=on_miss)
        return self._parse_nodes(nodes, max_length=self.max_length, merge_title=self.merge_title)

    def _parse_nodes(self, nodes: List[DocNode], max_length: int = 2048,
                     merge_title: bool = True, **kwargs: Any) -> List[DocNode]:

        def _group_by_level(node_groups: List[List[DocNode]], node: DocNode) -> List[List[DocNode]]:
            text_level = node.metadata.get('text_level', 0)

            if text_level > 0:
                node_groups.append([node])
            else:
                if not node_groups:
                    node_groups.append([node])
                else:
                    node_groups[-1].append(node)

            return node_groups

        node_groups = functools.reduce(_group_by_level, nodes, [])

        res = []
        for group in node_groups:
            res.extend(self._process_group(group, max_length=max_length, merge_title=merge_title))
        return res

    def _process_group(self, nodes: List[DocNode], max_length: int = 2048, merge_title: bool = True) -> List[DocNode]:
        if not nodes:
            return []

        title_node, content_nodes = self._split_title_and_content(nodes, merge_title)
        if title_node:
            title_node.metadata['lines'] = self._split_text_into_lines(
                title_node.text, title_node.metadata.get('lines', []))

        if not content_nodes:
            return [title_node] if title_node else []

        total_length = sum(len(node._content) for node in content_nodes)
        title_text = title_node.text if title_node else None

        if total_length <= max_length:
            merged_node = self._merge_nodes(content_nodes)
            if not merged_node:
                return [title_node] if title_node else []
            if title_text:
                merged_node._metadata['title'] = title_text
            return [title_node, merged_node] if title_node else [merged_node]

        groups = self._group_nodes_by_type(content_nodes)
        result = [title_node] if title_node else []

        for group in groups:
            processed_nodes = self._process_node_group(group, max_length, title_text)
            result.extend(processed_nodes)

        return result

    def _group_nodes_by_type(self, nodes: List[DocNode]) -> List[List[DocNode]]:
        attach_types = {'list', 'image', 'code', 'equation'}
        groups = []
        current_group = []

        for node in nodes:
            node_type = node.metadata.get('type', 'text')
            if node_type in attach_types:
                current_group.append(node)
            else:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                if node_type == 'table':
                    groups.append([node])
                else:
                    current_group = [node]

        if current_group:
            groups.append(current_group)

        return groups

    def _process_node_group(self, group: List[DocNode], max_length: int, title_text: str = None) -> List[DocNode]:
        group_length = sum(len(n._content) for n in group)

        if group_length <= max_length:
            merged = self._merge_nodes(group)
            if merged:
                if title_text:
                    merged._metadata['title'] = title_text
                return [merged]
            return []

        result = []
        for node in group:
            node.metadata['lines'] = self._split_text_into_lines(
                node._content, node.metadata.get('lines', []))

            if len(node._content) > max_length:
                split_nodes = self._split_large_node(node, max_length=max_length)
            else:
                split_nodes = [node]

            if title_text:
                for n in split_nodes:
                    n._metadata['title'] = title_text
            result.extend(split_nodes)

        return result

    def _split_title_and_content(self, nodes: List[DocNode], merge_title: bool) -> tuple:
        if merge_title:
            return None, nodes

        first_node = nodes[0]
        if first_node.metadata.get('text_level', 0) > 0:
            return first_node, nodes[1:]
        return None, nodes

    @staticmethod
    def _merge_line_bbox(bbox_a, bbox_b):
        if not bbox_a:
            return bbox_b
        if not bbox_b:
            return bbox_a
        if len(bbox_a) < 4 or len(bbox_b) < 4:
            return bbox_a or bbox_b
        return [
            min(bbox_a[0], bbox_b[0]),
            min(bbox_a[1], bbox_b[1]),
            max(bbox_a[2], bbox_b[2]),
            max(bbox_a[3], bbox_b[3]),
        ]

    @staticmethod
    def _normalize_for_match(content: str) -> str:
        return re.sub(r'\s+', '', content)

    @staticmethod
    def _extract_image_text(para: str) -> str:
        match = re.match(r'!\[([^\]]*)\]\([^\)]+\)', para.strip())
        if match:
            return match.group(1)
        return para

    def _is_content_in_para(self, content: str, para: str) -> bool:
        if not content:
            return True

        para_text = self._extract_image_text(para)

        norm_content = self._normalize_for_match(content.replace('$', ''))
        norm_para = self._normalize_for_match(para_text.replace('$', ''))

        return len(norm_content) <= len(norm_para) and norm_para.startswith(norm_content)

    def _merge_group_lines(self, group_lines: List[dict]) -> dict:
        merged_content = ''.join(line.get('content') or '' for line in group_lines)
        merged_bbox = None
        for line in group_lines:
            merged_bbox = self._merge_line_bbox(merged_bbox, line.get('bbox'))

        return {
            'content': merged_content,
            'bbox': merged_bbox,
            'type': group_lines[0].get('type', 'text'),
            'page': group_lines[0].get('page', 0),
        }

    def _split_text_into_lines(self, text: str, lines: List) -> List:
        original_lines = lines if isinstance(lines, list) else []
        if not text or not original_lines:
            return original_lines or []

        paragraphs = [p for p in text.split('\n') if p.strip()]
        if not paragraphs:
            return original_lines

        result_lines = []
        line_idx = 0

        for para in paragraphs:
            group_lines = []
            accumulated_content = ''

            while line_idx < len(original_lines):
                line = original_lines[line_idx]
                line_content = line.get('content') or ''
                test_content = accumulated_content + line_content

                if self._is_content_in_para(test_content, para):
                    group_lines.append(line)
                    accumulated_content = test_content
                    line_idx += 1
                else:
                    break

            if not group_lines:
                continue

            merged_line = self._merge_group_lines(group_lines)
            result_lines.append(merged_line)

        for line in original_lines[line_idx:]:
            result_lines.append({
                'content': line.get('content') or '',
                'bbox': line.get('bbox'),
                'type': line.get('type', 'text'),
                'page': line.get('page', 0),
            })

        return result_lines

    def _merge_nodes(self, nodes: List[DocNode]) -> DocNode:
        if not nodes:
            return None

        bboxs = []
        contents = []
        all_lines = []

        for node in nodes:
            if not node._content.strip():
                continue

            contents.append(node._content)

            page = node.metadata.get('page', 0)
            bbox = node.metadata.get('bbox', [])
            if bbox:
                bboxs.append([page] + bbox)

            node_lines = node.metadata.get('lines', [])
            if node_lines:
                if isinstance(node_lines, list):
                    all_lines.extend(node_lines)
                else:
                    all_lines.append({
                        'content': node._content,
                        'bbox': node.metadata.get('bbox', []),
                        'type': node.metadata.get('type', 'text'),
                        'page': node.metadata.get('page', 0),
                    })

        if not contents:
            return None

        merged_text = '\n'.join(contents)

        merged_bbox = None
        if bboxs:
            merged_bbox = self._merge_bbox(bboxs)

        merged_metadata = copy.deepcopy(nodes[0].metadata)
        merged_metadata['lines'] = all_lines
        if merged_bbox:
            merged_metadata['bbox'] = merged_bbox

        node = DocNode(text=merged_text, metadata=merged_metadata)

        node.metadata['lines'] = self._split_text_into_lines(
            node.text, node.metadata.get('lines', []))

        return node

    def _merge_bbox(self, bboxs):
        if not bboxs:
            return None

        page_groups = {}
        for bbox_item in bboxs:
            if len(bbox_item) < 5:
                continue
            page = bbox_item[0]
            if page not in page_groups:
                page_groups[page] = []
            page_groups[page].append(bbox_item[1:5])

        if not page_groups:
            return None

        first_page = min(page_groups.keys())
        first_page_bboxs = page_groups[first_page]

        x_mins = [b[0] for b in first_page_bboxs]
        y_mins = [b[1] for b in first_page_bboxs]
        x_maxs = [b[2] for b in first_page_bboxs]
        y_maxs = [b[3] for b in first_page_bboxs]

        return [min(x_mins), min(y_mins), max(x_maxs), max(y_maxs)]

    def _split_large_node(self, node: DocNode, max_length: int = 2048) -> List[DocNode]:
        content = node._content
        result_nodes = []

        lines = content.split('\n')
        current_chunks = []

        for line in lines:
            if len(line) > max_length:
                if current_chunks:
                    result_nodes.append(self._create_split_node(current_chunks, node.metadata))
                    current_chunks = []

                for i in range(0, len(line), max_length):
                    chunk_text = line[i:i + max_length]
                    result_nodes.append(self._create_split_node([chunk_text], node.metadata))
            else:
                test_chunks = current_chunks + [line]
                test_content = '\n'.join(test_chunks)

                if len(test_content) > max_length:
                    if current_chunks:
                        result_nodes.append(self._create_split_node(current_chunks, node.metadata))
                    current_chunks = [line]
                else:
                    current_chunks.append(line)

        if current_chunks:
            result_nodes.append(self._create_split_node(current_chunks, node.metadata))

        return result_nodes

    def _create_split_node(self, current_chunks, metadata):
        content = '\n'.join(current_chunks)

        is_table = metadata.get('type', 'text') == 'table'
        if is_table:
            table_caption = metadata.get('table_caption', '')
            table_footnote = metadata.get('table_footnote', '')
            if table_caption and not content.lstrip().startswith('table_caption'):
                content = f'{table_caption}\n{content}'
            if table_footnote and not content.rstrip().endswith('table_footnote'):
                content = f'{content.rstrip()}\n\n{table_footnote}'

        new_node = DocNode(text=content, metadata=copy.deepcopy(metadata))
        new_node._content = content
        new_node.metadata['lines'] = [
            {
                'content': content,
                'bbox': metadata.get('bbox', []),
                'type': metadata.get('type', 'text'),
                'page': metadata.get('page', 0)
            }
        ]

        return new_node
