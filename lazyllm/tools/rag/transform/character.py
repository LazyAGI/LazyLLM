from functools import partial
import re

from typing import List, Union, Tuple, Callable, Optional
from .base import _TextSplitterBase, _TokenTextSplitter, _Split

class CharacterSplitter(_TextSplitterBase):
    def __init__(self, chunk_size: int = 1024, overlap: int = 200, num_workers: int = 0,
                 separator: str = ' ', is_separator_regex: bool = False, keep_separator: bool = False, **kwargs):
        super().__init__(chunk_size=chunk_size, overlap=overlap, num_workers=num_workers)
        self._separator = separator
        self._is_separator_regex = is_separator_regex
        self._keep_separator = keep_separator
        self._character_split_fns = []
        self.token_splitter = _TokenTextSplitter(chunk_size=chunk_size, overlap=overlap)

    def split_text(self, text: str, metadata_size: int) -> List[str]:
        return super().split_text(text, metadata_size)

    def _split(self, text: str, chunk_size: int) -> List[_Split]:
        token_size = self._token_size(text)
        if token_size <= chunk_size:
            return [_Split(text, is_sentence=True, token_size=token_size)]

        text_splits, is_sentence = self._get_splits_by_fns(text)

        if len(text_splits) == 1 and self._token_size(text_splits[0]) > chunk_size:
            token_sub_texts = self.token_splitter.split_text(text_splits[0], metadata_size=0)
            return [
                _Split(s, is_sentence=is_sentence, token_size=self._token_size(s))
                for s in token_sub_texts
            ]

        results = []
        for segment in text_splits:
            token_size = self._token_size(segment)
            if token_size <= chunk_size:
                results.append(_Split(segment, is_sentence=is_sentence, token_size=token_size))
            else:
                sub_results = self._split(segment, chunk_size=chunk_size)
                results.extend(sub_results)

        return results

    def set_split_fns(self, split_fns: List[Callable[[str], List[str]]]):
        self._character_split_fns = split_fns

    def add_split_fn(self, split_fn: Callable[[str], List[str]], index: Optional[int] = None):
        if index is None:
            self._character_split_fns.append(split_fn)
        else:
            self._character_split_fns.insert(index, split_fn)

    def clear_split_fns(self):
        self._character_split_fns = []

    def _get_splits_by_fns(self, text: str) -> Tuple[List[str], bool]:
        sep_pattern = self._get_separator_pattern(self._separator)

        character_split_fns = self._character_split_fns
        if character_split_fns == []:
            character_split_fns = [
                partial(self.default_split, sep_pattern),
                lambda t: t.split(' '),
                list
            ]

        for split_fn in character_split_fns:
            splits = split_fn(text)
            if len(splits) > 1:
                break

        return splits, False

    def default_split(self, sep_pattern: Union[str, set[str]], text: str) -> List[str]:
        splits = re.split(sep_pattern, text)
        results = []
        if self._keep_separator:
            for i in range(0, len(splits) - 1, 2):
                if i + 1 < len(splits):
                    combined = splits[i] + splits[i + 1]
                    if combined:
                        results.append(combined)
            if len(splits) % 2 == 1 and splits[-1]:
                results.append(splits[-1])
        else:
            results = [split for split in splits if split]
        return results

    def _get_separator_pattern(self, separator: str) -> Union[str, set[str]]:
        lookaround_prefixes = ('(?=', '(?<!', '(?<=', '(?!')
        lookaround_pattern = re.compile(r'^\(\?(?:=|<=|!|<!)')

        is_lookaround = (
            self._is_separator_regex
            and (separator.startswith(lookaround_prefixes) or bool(lookaround_pattern.match(separator)))
        )

        if self._is_separator_regex or is_lookaround:
            sep_pattern = separator
        else:
            needs_escape = any(char in separator for char in r'\.^$*+?{}[]|()')
            sep_pattern = re.escape(separator) if needs_escape else separator

        if self._keep_separator:
            sep_pattern = f'({sep_pattern})'
        else:
            sep_pattern = f'(?:{sep_pattern})'

        return sep_pattern
