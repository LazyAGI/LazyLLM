from .base import _TextSplitterBase
from .recursive import RecursiveSplitter
from lazyllm.thirdparty import xml
from lazyllm.thirdparty import bs4
from typing import List, Optional, Dict, Type
from lazyllm.tools.rag.doc_node import DocNode
from lazyllm import LOG
import copy


class _LanguageSplitterBase(_TextSplitterBase):
    def __init__(self, chunk_size: int = 1024, overlap: int = 200, num_workers: int = 0,
                 filetype: Optional[str] = None, **kwargs):
        super().__init__(chunk_size=chunk_size, overlap=overlap, num_workers=num_workers)
        self._recursive_splitter = RecursiveSplitter(chunk_size=chunk_size, overlap=overlap)
        self._filetype = filetype
        self._extra_params = kwargs

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

        return self._do_split(text, effective_chunk_size)

    def _do_split(self, text: str, chunk_size: int) -> List[DocNode]:
        raise NotImplementedError("Subclasses must implement _do_split method")

    def _sub_split(self, nodes: List[DocNode], chunk_size: int) -> List[DocNode]:
        result = []
        for node in nodes:
            metadata_size = self._get_metadata_size(node)
            text_size = self._token_size(node.text)
            if text_size + metadata_size > chunk_size:
                splits = self._recursive_splitter.split_text(node.text, metadata_size)
                for split in splits:
                    new_node = DocNode(text=split, metadata=node.metadata.copy())
                    result.append(new_node)
            else:
                result.append(node)
        return result


# ========== XMLSplitter ==========
class XMLSplitter(_LanguageSplitterBase):
    def __init__(self, chunk_size: int = 1024, overlap: int = 200, num_workers: int = 0,
                 filetype: Optional[str] = None, keep_trace: bool = False, keep_tags: bool = False, **kwargs):
        super().__init__(chunk_size=chunk_size, overlap=overlap, num_workers=num_workers,
                         filetype=filetype, **kwargs)
        self._keep_trace = keep_trace
        self._keep_tags = keep_tags

    def _do_split(self, text: str, chunk_size: int) -> List[DocNode]:  # noqa: C901
        try:
            root = xml.etree.ElementTree.fromstring(text)
        except xml.etree.ElementTree.ParseError as e:
            LOG.warning(f'Failed to parse XML: {e}. Returning original text as a single DocNode.')
            return [DocNode(text=text, metadata={'tag': 'xml_error', 'error': str(e), 'trace': []})]

        def _format_tag_with_attrs(tag_name: str, attributes: dict) -> str:
            if not attributes:
                return tag_name

            attr_strs = []
            for attr_name, attr_value in attributes.items():
                attr_strs.append(f'{attr_name}="{attr_value}"')

            return f'{tag_name} {" ".join(attr_strs)}'

        def _parse_element(element: xml.etree.ElementTree.Element, trace: List[str] = None) -> List[DocNode]:
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
                node.metadata.pop('trace', None)

        if not self._keep_tags:
            for node in all_nodes:
                node.metadata.pop('tag', None)
                node.metadata.pop('xml_tag', None)

        all_nodes = self._sub_split(all_nodes, chunk_size)
        return all_nodes


# ========== ProgrammingSplitter ==========
class ProgrammingSplitter(_LanguageSplitterBase):
    def _do_split(self, text: str, chunk_size: int) -> List[DocNode]:  # noqa: C901
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


# ========== JSONSplitter ==========
class JSONSplitter(_LanguageSplitterBase):
    def _do_split(self, text: str, chunk_size: int) -> List[DocNode]:
        LOG.warning('JSONSplitter not fully implemented yet, returning as single node')
        return [DocNode(text=text, metadata={'filetype': 'json'})]


# ========== YAMLSplitter ==========
class YAMLSplitter(_LanguageSplitterBase):
    def _do_split(self, text: str, chunk_size: int) -> List[DocNode]:
        return [DocNode(text=text, metadata={'filetype': 'yaml'})]


# ========== HTMLSplitter ==========
class HTMLSplitter(_LanguageSplitterBase):
    def __init__(self, chunk_size: int = 1024, overlap: int = 200, num_workers: int = 0,
                 filetype: Optional[str] = None, keep_sections: bool = False, keep_tags: bool = False, **kwargs):
        super().__init__(chunk_size=chunk_size, overlap=overlap, num_workers=num_workers,
                         filetype=filetype, **kwargs)
        self._keep_sections = keep_sections
        self._keep_tags = keep_tags

    def _do_split(self, text: str, chunk_size: int) -> List[DocNode]:
        try:
            soup = bs4.BeautifulSoup(text, 'html.parser')
        except Exception as e:
            LOG.warning(f'Failed to parse HTML: {e}. Returning original text as a single DocNode.')
            return [DocNode(text=text, metadata={'filetype': 'html', 'error': str(e)})]

        sections = self._extract_sections(soup)

        if not sections:
            content = self._extract_content(soup)
            return [DocNode(text=content, metadata={'filetype': 'html', 'section_type': 'full_document'})]

        chunks = []
        for sec_info in sections:
            blocks = self._split_by_heading(sec_info)

            for blk_info in blocks:
                content = self._extract_content(blk_info['element'])
                metadata = blk_info['metadata'].copy()
                metadata['filetype'] = 'html'

                if not content.strip():
                    continue

                if self._token_size(content) > chunk_size:
                    splits = self._recursive_splitter.split_text(content, 0)
                    for split in splits:
                        new_node = DocNode(text=split, metadata=metadata.copy())
                        chunks.append(new_node)
                else:
                    chunks.append(DocNode(text=content, metadata=metadata))

        all_nodes = self._sub_split(chunks, chunk_size)
        return all_nodes if all_nodes else [DocNode(text=text, metadata={'filetype': 'html'})]

    def _extract_sections(self, soup: bs4.BeautifulSoup) -> List[dict]:
        sections = []

        semantic_tags = ['section', 'article', 'main', 'header', 'footer', 'aside', 'nav']
        for tag in semantic_tags:
            elements = soup.find_all(tag)
            for elem in elements:
                sections.append({
                    'element': elem,
                    'metadata': {
                        'section_type': tag,
                        'section_id': elem.get('id', ''),
                        'section_class': ' '.join(elem.get('class', [])),
                    }
                })

        if not sections:
            container_patterns = ['container', 'content', 'wrapper', 'main-content',
                                  'page-content', 'article-content', 'post-content']

            for pattern in container_patterns:
                divs = soup.find_all('div', class_=lambda x: x and any(p in str(x).lower() for p in [pattern]))
                for div in divs:
                    sections.append({
                        'element': div,
                        'metadata': {
                            'section_type': 'div',
                            'section_id': div.get('id', ''),
                            'section_class': ' '.join(div.get('class', [])),
                            'container_pattern': pattern,
                        }
                    })

                if sections:
                    break

        if not sections:
            body = soup.find('body')
            if body:
                sections.append({
                    'element': body,
                    'metadata': {
                        'section_type': 'body',
                        'section_id': body.get('id', ''),
                        'section_class': ' '.join(body.get('class', [])),
                    }
                })
            else:
                sections.append({
                    'element': soup,
                    'metadata': {
                        'section_type': 'document',
                        'section_id': '',
                        'section_class': '',
                    }
                })

        return sections

    def _split_by_heading(self, section: dict) -> List[dict]:
        blocks = []
        section_elem = section['element']
        section_metadata = section['metadata']

        headings = section_elem.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

        if not headings:
            blocks.append({
                'element': section_elem,
                'metadata': {
                    **section_metadata,
                    'has_heading': False,
                    'heading_level': 0,
                    'heading_text': '',
                }
            })
            return blocks

        for i, heading in enumerate(headings):
            heading_level = int(heading.name[1])
            heading_text = heading.get_text(strip=True)

            content_elements = [heading]

            for sibling in heading.find_next_siblings():
                if sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    sibling_level = int(sibling.name[1])
                    if sibling_level <= heading_level:
                        break

                content_elements.append(sibling)

            block_soup = bs4.BeautifulSoup('', 'html.parser')
            for elem in content_elements:
                if isinstance(elem, bs4.Tag):
                    block_soup.append(copy.copy(elem))

            blocks.append({
                'element': block_soup,
                'metadata': {
                    **section_metadata,
                    'has_heading': True,
                    'heading_level': heading_level,
                    'heading_text': heading_text,
                    'heading_id': heading.get('id', ''),
                    'block_index': i,
                }
            })

        return blocks

    def _extract_content(self, element: bs4.BeautifulSoup) -> str:
        if element is None:
            return ''

        for script in element.find_all(['script', 'style']):
            script.decompose()

        text = element.get_text(separator='\n', strip=True)

        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]

        return '\n'.join(lines)


class CodeSplitter(_TextSplitterBase):
    _SPLITTER_REGISTRY: Dict[str, Type[_LanguageSplitterBase]] = {
        'xml': XMLSplitter,
        'json': JSONSplitter,
        'yaml': YAMLSplitter,
        'yml': YAMLSplitter,
        'html': HTMLSplitter,
        'htm': HTMLSplitter,
    }

    def __init__(self, chunk_size: int = 1024, overlap: int = 200,
                 num_workers: int = 0, filetype: Optional[str] = None, **kwargs):
        super().__init__(chunk_size=chunk_size, overlap=overlap, num_workers=num_workers)
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._num_workers = num_workers
        self._filetype = filetype
        self._extra_params = kwargs
        self._splitter: Optional[_LanguageSplitterBase] = None

        if filetype:
            self._splitter = self.from_language(filetype)

    def from_language(self, filetype: str) -> _LanguageSplitterBase:
        filetype_lower = filetype.lower()
        splitter_class = self._SPLITTER_REGISTRY.get(filetype_lower)

        if splitter_class is None:
            splitter_class = ProgrammingSplitter

        return splitter_class(
            chunk_size=self._chunk_size,
            overlap=self._overlap,
            num_workers=self._num_workers,
            filetype=filetype,
            **self._extra_params
        )

    def transform(self, node: DocNode, **kwargs) -> List[DocNode]:
        if self._splitter is None:
            LOG.warning('Filetype not specified, cannot determine split method')
            return [DocNode(text=node.get_text(), metadata={'tag': 'unknown_type'})]

        return self._splitter.transform(node, **kwargs)

    def split_text(self, text: str, metadata_size: int = 0) -> List[DocNode]:
        if self._splitter is None:
            LOG.warning('Filetype not specified, cannot determine split method')
            return [DocNode(text=text, metadata={'tag': 'unknown_type'})]

        return self._splitter.split_text(text, metadata_size)

    @classmethod
    def register_splitter(cls, filetype: str, splitter_class: Type[_LanguageSplitterBase]):
        cls._SPLITTER_REGISTRY[filetype.lower()] = splitter_class

    @classmethod
    def get_supported_filetypes(cls) -> List[str]:
        return list(cls._SPLITTER_REGISTRY.keys())
