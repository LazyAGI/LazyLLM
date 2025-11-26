from .base import _TextSplitterBase, _UNSET
from typing import List, Optional, Union, AbstractSet, Collection, Literal, Any
from dataclasses import dataclass
from lazyllm import LOG
import re
from ..doc_node import DocNode

@dataclass
class _MdSplit:
    path: List[str]
    level: int
    header: Optional[str]
    content: str
    token_size: int
    type: str  # header, content, list, code_block, table, image, link, sentence


class MarkdownSplitter(_TextSplitterBase):
    def __init__(self, chunk_size: int = _UNSET, overlap: int = _UNSET, num_workers: int = _UNSET,
                 keep_trace: bool = _UNSET, keep_headers: bool = _UNSET, keep_lists: bool = _UNSET,
                 keep_code_blocks: bool = _UNSET, keep_tables: bool = _UNSET, keep_images: bool = _UNSET,
                 keep_links: bool = _UNSET, **kwargs):
        super().__init__(chunk_size=chunk_size, overlap=overlap, num_workers=num_workers)

        chunk_size = self._chunk_size
        keep_trace = self._get_param_value('keep_trace', keep_trace, False)
        keep_headers = self._get_param_value('keep_headers', keep_headers, False)
        keep_lists = self._get_param_value('keep_lists', keep_lists, False)
        keep_code_blocks = self._get_param_value('keep_code_blocks', keep_code_blocks, False)
        keep_tables = self._get_param_value('keep_tables', keep_tables, False)
        keep_images = self._get_param_value('keep_images', keep_images, False)
        keep_links = self._get_param_value('keep_links', keep_links, False)

        if chunk_size <= 200:
            LOG.warning(f'Chunk size {chunk_size} is too small, may cause unexpected splits')
        self.keep_trace = keep_trace
        self.keep_headers = keep_headers
        self.keep_lists = keep_lists
        self.keep_code_blocks = keep_code_blocks
        self.keep_tables = keep_tables
        self.keep_images = keep_images
        self.keep_links = keep_links

    def from_tiktoken_encoder(self, encoding_name: str = 'gpt2', model_name: Optional[str] = None,
                              allowed_special: Union[Literal['all'], AbstractSet[str]] = None,
                              disallowed_special: Union[Literal['all'], Collection[str]] = None,
                              **kwargs) -> 'MarkdownSplitter':
        return super().from_tiktoken_encoder(encoding_name, model_name, allowed_special, disallowed_special, **kwargs)

    def from_huggingface_tokenizer(self, tokenizer: Any, **kwargs) -> 'MarkdownSplitter':
        return super().from_huggingface_tokenizer(tokenizer, **kwargs)

    def _split(self, text: str, chunk_size: int) -> List[_MdSplit]:
        splits = self.split_markdown_by_semantics(text)
        if self.keep_code_blocks:
            splits = self._keep_code_blocks(splits)
        if self.keep_tables:
            splits = self._keep_tables(splits)
        if self.keep_images:
            splits = self._keep_images(splits)
        results = []
        for split in splits:
            results.extend(self._sub_split(split, chunk_size))
        return results

    def _split_code_block_by_lines(self, text: str) -> List[str]:
        lines = text.split('\n')
        return [line for line in lines if line.strip()]

    def _sub_split(self, split: _MdSplit, chunk_size: int) -> List[_MdSplit]:
        token_size = split.token_size
        if token_size <= chunk_size:
            return [split]

        is_code_or_table = (split.type != 'content' and split.type not in ['image', 'link'])

        if is_code_or_table:
            lines = split.content.split('\n')
            text_splits = []
            current_chunk = []
            current_size = 0

            for line in lines:
                line_size = self._token_size(line + '\n')
                if line_size > chunk_size:
                    if current_chunk:
                        text_splits.append('\n'.join(current_chunk))
                        current_chunk = []
                        current_size = 0
                    text_splits.append(line)
                elif current_size + line_size > chunk_size:
                    if current_chunk:
                        text_splits.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_size = line_size
                else:
                    current_chunk.append(line)
                    current_size += line_size

            if current_chunk:
                text_splits.append('\n'.join(current_chunk))
        else:
            text_splits, _ = self._get_splits_by_fns(split.content)

        results = []
        for segment in text_splits:
            token_size = self._token_size(segment)
            if token_size <= chunk_size:
                results.append(_MdSplit(
                    path=split.path, level=split.level,
                    header=split.header, content=segment,
                    token_size=token_size, type=split.type,
                ))
            else:
                new_split = _MdSplit(
                    path=split.path, level=split.level,
                    header=split.header, content=segment,
                    token_size=token_size, type=split.type
                )
                results.extend(self._sub_split(new_split, chunk_size=chunk_size))

        return results

    def _keep_code_blocks(self, splits: List[_MdSplit]) -> List[_MdSplit]:
        pattern = re.compile(
            r'```([\w+-]*)\s*(.*?)```',
            re.DOTALL
        )
        results = self._keep_elements(splits, pattern, 'code')

        return results

    def _keep_images(self, splits: List[_MdSplit]) -> List[_MdSplit]:
        pattern = re.compile(
            r'!\[([^\]]*)\]\(([^\)]+)\)',
            re.MULTILINE
        )
        results = self._keep_elements(splits, pattern, 'image')

        return results

    def _keep_links(self, splits: List[_MdSplit]) -> List[_MdSplit]:
        pattern = re.compile(
            r'(?<!!)\[([^\]]+)\]\(([^\)]+)\)',
            re.MULTILINE
        )
        results = self._keep_elements(splits, pattern, 'link')

        return results

    def _keep_lists(self, splits: List[_MdSplit]) -> List[_MdSplit]:
        pattern = re.compile(
            r'(?P<list>(?:^\s*(?:[-*+]|\d+\.)\s+.*$\n?){1,})',
            re.MULTILINE
        )
        results = self._keep_elements(splits, pattern, 'list')

        return results

    def _get_heading_level(self, line: str) -> int:
        line = line.split('\n')[0].strip()
        match = re.match(r'^(#{1,6})\s+(.*)$', line)
        if not match:
            return 0

        level = len(match.group(1))
        content = match.group(2).strip()
        if not content:
            return 0

        if level == 6 and content.startswith('#'):
            return 0

        return level

    def _get_code_block_ranges(self, text: str) -> List[tuple]:
        code_block_pattern = re.compile(r'```[\w+-]*\s*.*?```', re.DOTALL)
        ranges = []
        for match in code_block_pattern.finditer(text):
            ranges.append((match.start(), match.end()))
        return ranges

    def _is_in_code_block(self, pos: int, code_ranges: List[tuple]) -> bool:
        for start, end in code_ranges:
            if start <= pos < end:
                return True
        return False

    def split_markdown_by_semantics(self, md_text: str) -> List[_MdSplit]:
        code_ranges = self._get_code_block_ranges(md_text)

        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        heading_positions = []

        for match in heading_pattern.finditer(md_text):
            if not self._is_in_code_block(match.start(), code_ranges):
                heading_positions.append(match.start())

        results = []
        path_stack = []
        level_stack = []

        if heading_positions:
            first_heading_pos = heading_positions[0]
            if first_heading_pos > 0:
                content_before = md_text[:first_heading_pos].strip()
                if content_before:
                    results.append(_MdSplit(
                        path=[],
                        level=0,
                        header=None,
                        content=content_before,
                        token_size=self._token_size(content_before),
                        type='content'
                    ))
        else:
            content = md_text.strip()
            if content:
                results.append(_MdSplit(
                    path=[],
                    level=0,
                    header=None,
                    content=content,
                    token_size=self._token_size(content),
                    type='content'
                ))
            return results

        for i, heading_pos in enumerate(heading_positions):
            if i + 1 < len(heading_positions):
                end_pos = heading_positions[i + 1]
            else:
                end_pos = len(md_text)

            block = md_text[heading_pos:end_pos].rstrip('\n')
            lines = block.split('\n')

            heading_line = lines[0]
            level = self._get_heading_level(heading_line)

            if level == 0:
                header = None
                content = block.strip()
            else:
                header = heading_line.strip('#').strip()
                content = '\n'.join(lines[1:]).strip()

                while level_stack and level_stack[-1] >= level:
                    path_stack.pop()
                    level_stack.pop()
                path_stack.append(header)
                level_stack.append(level)

            results.append(_MdSplit(
                path=path_stack.copy(),
                level=level,
                header=header,
                content=content,
                token_size=self._token_size(content),
                type='content'
            ))

        return results

    def _keep_elements(self, splits: List[_MdSplit], pattern: re.Pattern, type: str) -> List[_MdSplit]:
        results = []

        for split in splits:
            content = split.content
            matches = list(pattern.finditer(content))
            if not matches:
                results.append(split)
                continue

            last_end = 0
            for match in matches:
                start, end = match.span()

                if type == 'code':
                    lang = match.group(1).strip() if match.group(1) else 'code'
                    element_part = match.group(2)
                    element_type = lang
                else:
                    element_part = match.group()
                    element_type = type

                if start > last_end:
                    text_part = content[last_end:start].strip()
                    if text_part:
                        results.append(_MdSplit(
                            path=split.path,
                            level=split.level,
                            header=split.header,
                            content=text_part,
                            token_size=self._token_size(text_part),
                            type='content'
                        ))

                element_part = element_part.strip()
                results.append(_MdSplit(
                    path=split.path,
                    level=split.level,
                    header=split.header,
                    content=element_part,
                    token_size=self._token_size(element_part),
                    type=element_type,
                ))

                last_end = end

            if last_end < len(content):
                tail = content[last_end:].strip()
                if tail:
                    results.append(_MdSplit(
                        path=split.path,
                        level=split.level,
                        header=split.header,
                        content=tail,
                        token_size=self._token_size(tail),
                        type='content'
                    ))

        return results

    def _merge(self, splits: List[_MdSplit], chunk_size: int) -> List[DocNode]:
        if not splits:
            return []

        if len(splits) == 1:
            return [self._to_docnode(splits[0])]

        end_split = splits[-1]
        if end_split.token_size == chunk_size and self._overlap > 0:
            splits.pop()

            def cut_split(split: _MdSplit) -> List[_MdSplit]:
                text = split.content
                text_tokens = self.token_encoder(text)
                p_text = self.token_decoder(text_tokens[:len(text_tokens) // 2])
                n_text = self.token_decoder(text_tokens[len(text_tokens) // 2:])
                return [
                    _MdSplit(
                        path=split.path, level=split.level, header=split.header,
                        content=p_text, token_size=self._token_size(p_text), type=split.type
                    ),
                    _MdSplit(
                        path=split.path, level=split.level, header=split.header,
                        content=n_text, token_size=self._token_size(n_text), type=split.type
                    ),
                ]
            splits.extend(cut_split(end_split))
            end_split = splits[-1]
        result = []
        for idx in range(len(splits) - 2, -1, -1):
            start_split = splits[idx]
            if start_split.path == end_split.path and end_split.type == start_split.type:
                if (
                    start_split.token_size <= self._overlap
                    and end_split.token_size <= chunk_size - self._overlap
                ):
                    type = end_split.type
                    token_size = start_split.token_size + end_split.token_size
                    content = start_split.content + end_split.content
                    end_split = _MdSplit(
                        path=start_split.path, level=start_split.level,
                        header=start_split.header, content=content,
                        token_size=token_size, type=type
                    )
                    continue
                else:
                    if end_split.token_size > chunk_size:
                        raise ValueError(f'split token size ({end_split.token_size}) \
                                        is greater than chunk size ({chunk_size}).')
                    else:
                        remaining_space = chunk_size - end_split.token_size
                        overlap_len = min(self._overlap, remaining_space, start_split.token_size)

                        if overlap_len > 0:
                            start_tokens = self.token_encoder(start_split.content)
                            overlap_tokens = start_tokens[-overlap_len:]
                            overlap_text = self.token_decoder(overlap_tokens)

                            type = start_split.type
                            token_size = end_split.token_size + overlap_len
                            content = overlap_text + end_split.content
                            end_split = _MdSplit(
                                path=start_split.path, level=start_split.level,
                                header=start_split.header, content=content,
                                token_size=token_size, type=type
                            )

                        result.insert(0, self._to_docnode(end_split))
                        end_split = start_split
            else:
                result.insert(0, self._to_docnode(end_split))
                end_split = start_split
        result.insert(0, self._to_docnode(end_split))
        return result

    def _to_docnode(self, split: _MdSplit) -> DocNode:
        metadata = {
            'path': split.path if self.keep_trace else None,
            'level': split.level,
            'header': split.header if self.keep_headers else None,
            'token_size': split.token_size,
            'type': split.type,
        }

        return DocNode(
            metadata=metadata,
            content=split.content,
        )
