from .base import _TextSplitterBase, _UNSET
from .recursive import RecursiveSplitter
from lazyllm.thirdparty import xml
from lazyllm.thirdparty import bs4
from typing import List, Optional, Dict, Type
from lazyllm.tools.rag.doc_node import DocNode
from lazyllm import LOG
import copy
import json
import yaml

class _LanguageSplitterBase(_TextSplitterBase):
    def __init__(self, chunk_size: int = _UNSET, overlap: int = _UNSET, num_workers: int = _UNSET,
                 filetype: Optional[str] = _UNSET, **kwargs):
        super().__init__(chunk_size=chunk_size, overlap=overlap, num_workers=num_workers)
        filetype = self._get_param_value('filetype', filetype, None)

        self._recursive_splitter = RecursiveSplitter(chunk_size=self._chunk_size, overlap=self._overlap)
        self._filetype = filetype
        self._extra_params = kwargs

    def transform(self, node: DocNode, **kwargs) -> List[DocNode]:
        return self.split_text(
            node.get_text(),
            metadata_size=self._get_metadata_size(node)
        )

    def split_text(self, text: str, metadata_size: int) -> List[DocNode]:
        if text == '':
            return [DocNode(text='', metadata={'code_type': 'empty'})]
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
        raise NotImplementedError('Subclasses must implement _do_split method')

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
    def __init__(self, chunk_size: int = _UNSET, overlap: int = _UNSET, num_workers: int = _UNSET,
                 filetype: Optional[str] = 'xml', keep_trace: bool = _UNSET, keep_tags: bool = _UNSET, **kwargs):
        super().__init__(chunk_size=chunk_size, overlap=overlap, num_workers=num_workers,
                         filetype=filetype, **kwargs)
        keep_trace = self._get_param_value('keep_trace', keep_trace, False)
        keep_tags = self._get_param_value('keep_tags', keep_tags, False)

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


# ========== GeneralCodeSplitter ==========
class GeneralCodeSplitter(_LanguageSplitterBase):
    def __init__(self, chunk_size: int = _UNSET, overlap: int = _UNSET, num_workers: int = _UNSET,
                 filetype: str = 'code', **kwargs):
        super().__init__(chunk_size=chunk_size, overlap=overlap, num_workers=num_workers,
                         filetype=filetype, **kwargs)

        self._filetype = filetype
        self._extra_params = kwargs

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
    def __init__(self, chunk_size: int = _UNSET, overlap: int = _UNSET, num_workers: int = _UNSET,
                 filetype: str = 'json', compact_output: bool = _UNSET, **kwargs):
        super().__init__(chunk_size=chunk_size, overlap=overlap, num_workers=num_workers,
                         filetype=filetype, **kwargs)
        compact_output = self._get_param_value('compact_output', compact_output, True)

        self._compact_output = compact_output
        self._max_depth = 20

    def _do_split(self, text: str, chunk_size: int) -> List[DocNode]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            LOG.warning(f'Failed to parse JSON: {e}. Returning original text as a single DocNode.')
            return [self._make_node(
                text=text,
                filetype='json',
                error=str(e),
            )]

        return self._split_json_data(data, chunk_size, 'json', path=[], depth=0)

    def _make_node(self, text, filetype, **meta):
        md = {
            'filetype': filetype,
        }
        md.update(meta)
        return DocNode(text=text, metadata=md)

    def _to_json_str(self, data):
        if self._compact_output:
            return json.dumps(data, ensure_ascii=False)
        else:
            return json.dumps(data, ensure_ascii=False, indent=2)

    def _split_json_data(self, data, chunk_size, filetype, path=None, depth=0):
        if path is None:
            path = []

        if depth > self._max_depth:
            LOG.warning(f"Max depth {self._max_depth} reached at path {'/'.join(path)}")
            raw = self._to_json_str(data)
            return [self._make_node(
                text=raw,
                filetype=filetype,
                type=type(data).__name__,
                path='/'.join(path) if path else 'root',
                depth=depth,
                warning='max_depth_exceeded'
            )]

        raw = self._to_json_str(data)
        if self._token_size(raw) <= chunk_size:
            return [self._make_node(
                text=raw,
                filetype=filetype,
                type=type(data).__name__,
                path='/'.join(path) if path else 'root',
                depth=depth,
                is_complete=True
            )]

        if isinstance(data, dict):
            return self._split_dict(data, chunk_size, filetype, path, depth)

        if isinstance(data, list):
            return self._split_list(data, chunk_size, filetype, path, depth)

        if isinstance(data, str):
            return self._split_string(data, chunk_size, filetype, path, depth)

        return [self._make_node(
            text=raw,
            filetype=filetype,
            type=type(data).__name__,
            path='/'.join(path) if path else 'root',
            depth=depth,
            is_complete=True
        )]

    def _split_dict(self, data: Dict, chunk_size, filetype, path, depth):
        nodes = []
        current = {}

        for key, val in data.items():
            test_dict = {**current, key: val}
            test_str = self._to_json_str(test_dict)
            test_size = self._token_size(test_str)

            if len(current) == 0 and test_size > chunk_size:
                child_nodes = self._split_json_data(
                    val, chunk_size, filetype,
                    path=path + [key],
                    depth=depth + 1
                )

                for i, node in enumerate(child_nodes):
                    node.metadata['parent_field'] = key
                    if len(child_nodes) > 1:
                        node.metadata['part'] = f'{i+1}/{len(child_nodes)}'

                nodes.extend(child_nodes)
                continue

            if test_size > chunk_size:
                if current:
                    nodes.append(self._make_node(
                        text=self._to_json_str(current),
                        filetype=filetype,
                        type='dict',
                        path='/'.join(path) if path else 'root',
                        depth=depth,
                        keys=list(current.keys()),
                        is_complete=False
                    ))
                current = {key: val}
            else:
                current[key] = val

        if current:
            nodes.append(self._make_node(
                text=self._to_json_str(current),
                filetype=filetype,
                type='dict',
                path='/'.join(path) if path else 'root',
                depth=depth,
                keys=list(current.keys()),
                is_complete=(len(nodes) == 0)
            ))

        return nodes

    def _split_list(self, data: List, chunk_size, filetype, path, depth):
        nodes = []
        current = []

        for idx, item in enumerate(data):
            test_list = current + [item]
            test_str = self._to_json_str(test_list)
            test_size = self._token_size(test_str)

            if len(current) == 0 and test_size > chunk_size:
                child_nodes = self._split_json_data(
                    item, chunk_size, filetype,
                    path=path + [f'[{idx}]'],
                    depth=depth + 1
                )

                for i, node in enumerate(child_nodes):
                    node.metadata['list_index'] = idx
                    if len(child_nodes) > 1:
                        node.metadata['part'] = f'{i+1}/{len(child_nodes)}'

                nodes.extend(child_nodes)
                continue

            if test_size > chunk_size:
                if current:
                    nodes.append(self._make_node(
                        text=self._to_json_str(current),
                        filetype=filetype,
                        type='list',
                        path='/'.join(path) if path else 'root',
                        depth=depth,
                        length=len(current),
                        is_complete=False
                    ))
                current = [item]
            else:
                current.append(item)

        if current:
            nodes.append(self._make_node(
                text=self._to_json_str(current),
                filetype=filetype,
                type='list',
                path='/'.join(path) if path else 'root',
                depth=depth,
                length=len(current),
                is_complete=(len(nodes) == 0)
            ))

        return nodes

    def _split_string(self, data, chunk_size, filetype, path, depth):
        splits = self._recursive_splitter.split_text(data, metadata_size=0)

        nodes = []
        for i, s in enumerate(splits):
            nodes.append(self._make_node(
                text=s,
                filetype=filetype,
                type='string',
                path='/'.join(path) if path else 'root',
                depth=depth,
                part=f'{i+1}/{len(splits)}' if len(splits) > 1 else None,
                is_complete=(len(splits) == 1)
            ))

        return nodes


# ========== YAMLSplitter ==========
class YAMLSplitter(JSONSplitter):
    def __init__(self, chunk_size: int = _UNSET, overlap: int = _UNSET, num_workers: int = _UNSET,
                 filetype: str = 'yaml', compact_output: bool = _UNSET, **kwargs):
        super().__init__(chunk_size=chunk_size, overlap=overlap, num_workers=num_workers,
                         filetype=filetype, compact_output=compact_output, **kwargs)

    def _do_split(self, text: str, chunk_size: int) -> List[DocNode]:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            LOG.warning(f'Failed to parse YAML: {e}. Returning original text as a single DocNode.')
            return [self._make_node(text=text, filetype='yaml', error=str(e))]

        return self._split_json_data(data, chunk_size, 'yaml', path=[], depth=0)


# ========== HTMLSplitter ==========
class HTMLSplitter(_LanguageSplitterBase):
    def __init__(self, chunk_size: int = _UNSET, overlap: int = _UNSET, num_workers: int = _UNSET,
                 filetype: str = 'html', keep_sections: bool = _UNSET, keep_tags: bool = _UNSET, **kwargs):
        super().__init__(chunk_size=chunk_size, overlap=overlap, num_workers=num_workers,
                         filetype=filetype, **kwargs)
        keep_sections = self._get_param_value('keep_sections', keep_sections, False)
        keep_tags = self._get_param_value('keep_tags', keep_tags, False)

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

        merged_nodes = self._merge(chunks, chunk_size)
        all_nodes = self._sub_split(merged_nodes, chunk_size)
        return all_nodes if all_nodes else [DocNode(text=text, metadata={'filetype': 'html'})]

    def _extract_child_divs(self, parent_elem, parent_metadata: dict) -> List[dict]:
        child_sections = []
        direct_children = []
        for child in parent_elem.children:
            if hasattr(child, 'name') and child.name == 'div':
                direct_children.append(child)

        if len(direct_children) > 1:
            for idx, child_div in enumerate(direct_children):
                child_sections.append({
                    'element': child_div,
                    'metadata': {
                        'section_type': 'div',
                        'section_id': child_div.get('id', ''),
                        'section_class': ' '.join(child_div.get('class', [])) if child_div.get('class') else '',
                        'parent_id': parent_metadata.get('section_id', ''),
                        'child_index': idx,
                    }
                })

        return child_sections

    def _extract_sections(self, soup: bs4.BeautifulSoup) -> List[dict]:  # noqa: C901
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

            wrapper_divs = []
            for pattern in container_patterns:
                divs_by_class = soup.find_all('div', class_=lambda x: x and any(p in str(x).lower() for p in [pattern]))  # noqa B023
                divs_by_id = soup.find_all('div', id=lambda x: x and any(p in str(x).lower() for p in [pattern]))  # noqa B023

                all_divs = list(dict.fromkeys(divs_by_class + divs_by_id))

                for div in all_divs:
                    wrapper_metadata = {
                        'section_type': 'div',
                        'section_id': div.get('id', ''),
                        'section_class': ' '.join(div.get('class', [])),
                        'container_pattern': pattern,
                    }
                    wrapper_divs.append({'element': div, 'metadata': wrapper_metadata})

                if wrapper_divs:
                    break

            if wrapper_divs:
                for wrapper in wrapper_divs:
                    child_divs = self._extract_child_divs(wrapper['element'], wrapper['metadata'])
                    if child_divs:
                        sections.extend(child_divs)
                    else:
                        sections.append(wrapper)

            if not sections:
                body = soup.find('body')
                if body:
                    top_level_divs = body.find_all('div', recursive=False)
                    for div in top_level_divs:
                        if div.get('id') or div.get('class'):
                            sections.append({
                                'element': div,
                                'metadata': {
                                    'section_type': 'div',
                                    'section_id': div.get('id', ''),
                                    'section_class': ' '.join(div.get('class', [])) if div.get('class') else '',
                                }
                            })

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

        heading_parents = [h.parent for h in headings]
        all_same_parent = len(set(id(p) for p in heading_parents)) == 1

        if all_same_parent and len(headings) > 1:
            processed_headings = set()

            for i, heading in enumerate(headings):
                if id(heading) in processed_headings:
                    continue

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

                processed_headings.add(id(heading))
        else:
            heading_level = int(headings[0].name[1])
            heading_text = headings[0].get_text(strip=True)

            blocks.append({
                'element': section_elem,
                'metadata': {
                    **section_metadata,
                    'has_heading': True,
                    'heading_level': heading_level,
                    'heading_text': heading_text,
                    'heading_id': headings[0].get('id', ''),
                    'block_index': 0,
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

    def _merge(self, chunks: List[DocNode], chunk_size: int) -> List[DocNode]:
        if not chunks or len(chunks) <= 1:
            return chunks

        result = []
        i = 0

        while i < len(chunks):
            current = chunks[i]
            current_size = self._token_size(current.text)

            if i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                next_size = self._token_size(next_chunk.text)

                can_merge = (
                    current.metadata.get('parent_id') == next_chunk.metadata.get('parent_id')
                    and current.metadata.get('section_type') == next_chunk.metadata.get('section_type')
                    and current_size < chunk_size * 0.5 and current_size + next_size <= chunk_size
                )

                if can_merge:
                    merged_text = current.text + '\n\n' + next_chunk.text
                    merged_metadata = current.metadata.copy()

                    if next_chunk.metadata.get('has_heading') and not current.metadata.get('has_heading'):
                        merged_metadata['has_heading'] = True
                        merged_metadata['heading_text'] = next_chunk.metadata.get('heading_text', '')
                        merged_metadata['heading_level'] = next_chunk.metadata.get('heading_level', 0)

                    merged_node = DocNode(text=merged_text, metadata=merged_metadata)
                    result.append(merged_node)
                    i += 2
                    continue

            result.append(current)
            i += 1

        return result


class CodeSplitter(_TextSplitterBase):
    _SPLITTER_REGISTRY: Dict[str, Type[_LanguageSplitterBase]] = {
        'xml': XMLSplitter,
        'json': JSONSplitter,
        'yaml': YAMLSplitter,
        'yml': YAMLSplitter,
        'html': HTMLSplitter,
        'htm': HTMLSplitter,
    }

    def __init__(self, chunk_size: int = _UNSET, overlap: int = _UNSET,
                 num_workers: int = _UNSET, filetype: Optional[str] = _UNSET, **kwargs):
        super().__init__(chunk_size=chunk_size, overlap=overlap, num_workers=num_workers)
        self._filetype = self._get_param_value('filetype', filetype, None)
        self._extra_params = kwargs
        self._splitter: Optional[_LanguageSplitterBase] = None

        if self._filetype:
            self._splitter = self.from_language(filetype)

    def from_language(self, filetype: str) -> _LanguageSplitterBase:
        filetype_lower = filetype.lower()
        splitter_class = self._SPLITTER_REGISTRY.get(filetype_lower)

        if splitter_class is None:
            splitter_class = GeneralCodeSplitter

        return splitter_class(
            chunk_size=self._chunk_size,
            overlap=self._overlap,
            num_workers=self._number_workers,
            filetype=filetype_lower,
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
