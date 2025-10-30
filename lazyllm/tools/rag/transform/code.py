from .base import _TextSplitterBase, _TokenTextSplitter
import xml.etree.ElementTree as ET
from typing import List, Optional
from lazyllm.tools.rag.doc_node import DocNode
from lazyllm import LOG

class CodeSplitter(_TextSplitterBase):
    def __init__(self, chunk_size: int = 1024, overlap: int = 200, num_workers: int = 0, filetype: Optional[str] = None,
                 keep_trace: bool = False, keep_tags: bool = False, **kwargs):
        super().__init__(chunk_size=chunk_size, overlap=overlap, num_workers=num_workers)
        self.token_splitter = _TokenTextSplitter(chunk_size=chunk_size, overlap=overlap)
        self._keep_trace = keep_trace
        self._keep_tags = keep_tags
        self._dispatch_list = {
            'xml': self._split_xml,
            'python': self._split_code,
            'c': self._split_code,
            'c++': self._split_code,
        }
        self._filetype = filetype

    def transform(self, node: DocNode, **kwargs) -> List[DocNode]:
        return self.split_text(
            node.get_text(),
            metadata_size=self._get_metadata_size(node)
        )

    def split_text(self, text: str, metadata_size: int) -> List[DocNode]:
        if text == '':
            return [DocNode(text='')]
        effective_chunk_size = self._chunk_size - metadata_size
        if effective_chunk_size <= 0:
            raise ValueError(
                f'Metadata length ({metadata_size}) is longer than chunk size '
                f'({self._chunk_size}). Consider increasing the chunk size or '
                'decreasing the size of your metadata to avoid this.'
            )
        elif effective_chunk_size < 50:
            LOG.warning(
                f'Metadata length ({metadata_size}) is close to chunk size '
                f'({self._chunk_size}). Resulting chunks are less than 50 tokens. '
                f'Consider increasing the chunk size or decreasing the size of '
                f'your metadata to avoid this.'
            )

        if not self._filetype:
            LOG.warning('Filetype not specified, cannot determine split method')
            return [DocNode(text=text, metadata={'tag': 'unknown_type'})]

        filetype = self._filetype.lower()
        if filetype not in self._dispatch_list:
            LOG.warning(f'Unsupported file type: {filetype}, fallback to default code splitter')
            _split = self._split_code
        else:
            _split = self._dispatch_list[filetype]
        splits = _split(text, effective_chunk_size)

        return splits

    def _split_xml(self, text: str, chunk_size: int) -> List[DocNode]:  # noqa: C901
        '''
        Split XML text into DocNode list based on XML tags.
        Each XML element becomes a DocNode with tag in metadata and trace path in metadata.
        No parent-child relationships are established between DocNodes.
        '''
        try:
            root = ET.fromstring(text)
        except ET.ParseError as e:
            LOG.warning(f'Failed to parse XML: {e}. Returning original text as a single DocNode.')
            return [DocNode(text=text, metadata={'tag': 'xml_error', 'error': str(e), 'trace': []})]

        def _format_tag_with_attrs(tag_name: str, attributes: dict) -> str:
            if not attributes:
                return tag_name

            attr_strs = []
            for attr_name, attr_value in attributes.items():
                attr_strs.append(f'{attr_name}="{attr_value}"')

            return f'{tag_name} {" ".join(attr_strs)}'

        def _parse_element(element: ET.Element, trace: List[str] = None) -> List[DocNode]:
            if trace is None:
                trace = []
            tag_name = element.tag
            attributes = dict(element.attrib) if element.attrib else {}
            tag_with_attrs = _format_tag_with_attrs(tag_name, attributes)
            current_trace = trace + [tag_with_attrs]
            text_content = (element.text or '').strip() if element.text else ''

            metadata = {
                'tag': tag_name,
                'xml_tag': tag_name,
                'trace': current_trace.copy(),
            }

            for attr_name, attr_value in attributes.items():
                metadata[f'attr_{attr_name}'] = attr_value

            if attributes:
                metadata['attributes'] = attributes

            all_nodes = []
            has_children = len(list(element)) > 0
            has_text = bool(text_content)

            if has_text or not has_children:
                node = DocNode(
                    text=text_content if text_content else '',
                    metadata=metadata
                )
                all_nodes.append(node)

            for child_element in element:
                child_nodes = _parse_element(child_element, trace=current_trace)
                all_nodes.extend(child_nodes)

            return all_nodes

        all_nodes = []

        if root.tag and root.tag not in ['', None]:
            root_trace = [_format_tag_with_attrs(root.tag, dict(root.attrib) if root.attrib else {})]
            nodes = _parse_element(root, trace=[])
            all_nodes.extend(nodes)
        else:
            for child in root:
                nodes = _parse_element(child, trace=[])
                all_nodes.extend(nodes)

        if not all_nodes:
            root_tag = root.tag if root.tag else 'root'
            root_attrs = dict(root.attrib) if root.attrib else {}
            root_trace = [_format_tag_with_attrs(root_tag, root_attrs)]
            all_nodes = [DocNode(
                text=text,
                metadata={
                    'tag': root_tag,
                    'xml_tag': root_tag,
                    'trace': root_trace
                }
            )]

        if not self._keep_trace:
            for node in all_nodes:
                node.metadata.pop('trace')

        if not self._keep_tags:
            for node in all_nodes:
                node.metadata.pop('tag')
                node.metadata.pop('xml_tag')
        all_nodes = self._sub_split(all_nodes, chunk_size)
        return all_nodes

    def _split_json(self, text: str, chunk_size: int) -> List[DocNode]:
        pass

    def _split_yaml(self, text: str, chunk_size: int) -> List[DocNode]:
        pass

    def _split_html(self, text: str, chunk_size: int) -> List[DocNode]:
        pass

    def _split_code(self, text: str, chunk_size: int) -> List[DocNode]:  # noqa: C901
        if not text.strip():
            return [DocNode(text='', metadata={'code_type': 'empty'})]

        lines = text.split('\n')
        nodes = []

        def _is_code_structure_start(line: str) -> bool:
            stripped = line.strip()
            structure_keywords = [
                'def ', 'class ', 'if ', 'elif ', 'else:', 'for ', 'while ',
                'with ', 'try:', 'except ', 'finally:', 'async def ',
                'namespace ', 'struct ', 'union ', 'enum ', 'function ',
                'public ', 'private ', 'protected '
            ]
            return any(stripped.startswith(kw) for kw in structure_keywords)

        def _get_indent_level(line: str) -> int:
            return len(line) - len(line.lstrip())

        def _create_node(chunk_text: str, chunk_type: str = 'code_block') -> DocNode:
            if not chunk_text.strip():
                return None

            metadata = {
                'code_type': chunk_type,
                'filetype': self._filetype if self._filetype else 'code'
            }

            return DocNode(text=chunk_text, metadata=metadata)

        current_chunk_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]

            if current_chunk_lines and _is_code_structure_start(line):
                chunk_text = '\n'.join(current_chunk_lines)
                node = _create_node(chunk_text, 'code_block')
                if node:
                    nodes.append(node)
                current_chunk_lines = []

            current_chunk_lines.append(line)
            if _is_code_structure_start(line):
                base_indent = _get_indent_level(line)
                structure_lines = [line]
                j = i + 1

                while j < len(lines):
                    next_line = lines[j]
                    next_stripped = next_line.strip()

                    if not next_stripped:
                        structure_lines.append(next_line)
                        j += 1
                        continue

                    next_indent = _get_indent_level(next_line)

                    if next_indent <= base_indent and _is_code_structure_start(next_line):
                        break

                    structure_lines.append(next_line)

                    if len(structure_lines) > 200:
                        break

                    j += 1
                structure_text = '\n'.join(structure_lines)
                node = _create_node(structure_text, 'code_structure')
                if node:
                    nodes.append(node)

                current_chunk_lines = []
                i = j
                continue

            i += 1

            if len(current_chunk_lines) > 100:
                chunk_text = '\n'.join(current_chunk_lines)
                node = _create_node(chunk_text, 'code_block')
                if node:
                    nodes.append(node)
                current_chunk_lines = []

        if current_chunk_lines:
            chunk_text = '\n'.join(current_chunk_lines)
            node = _create_node(chunk_text, 'code_block')
            if node:
                nodes.append(node)

        if not nodes:
            nodes.append(DocNode(
                text=text,
                metadata={
                    'code_type': 'code_file',
                    'filetype': self._filetype if self._filetype else 'code'
                }
            ))

        nodes = self._sub_split(nodes, chunk_size)

        return nodes

    def _sub_split(self, nodes: List[DocNode], chunk_size: int) -> List[DocNode]:
        result = []
        for node in nodes:
            metadata_size = self._get_metadata_size(node)
            text_size = self._token_size(node.text)
            if text_size + metadata_size > chunk_size:
                splits = self.token_splitter.split_text(node.text, metadata_size)
                for split in splits:
                    new_node = DocNode(text=split, metadata=node.metadata.copy())
                    result.append(new_node)
            else:
                result.append(node)
        return result
