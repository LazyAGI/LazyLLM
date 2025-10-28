from functools import partial
import re

from typing import List, Union, Tuple
from .base import _TextSplitterBase

class CharacterSplitter(_TextSplitterBase):
    def __init__(self, chunk_size: int = 1024, overlap: int = 200, num_workers: int = 0,
                 separator: str = ' ', is_separator_regex: bool = False, keep_separator: bool = False, **kwargs):
        super().__init__(chunk_size=chunk_size, overlap=overlap, num_workers=num_workers)
        self._separator = separator
        self._is_separator_regex = is_separator_regex
        self._keep_separator = keep_separator

    def split_text(self, text: str, metadata_size: int) -> List[str]:
        return super().split_text(text, metadata_size)

    def _get_splits_by_fns(self, text: str) -> Tuple[List[str], bool]:
        sep_pattern = self._get_separator_pattern(self._separator)

        character_split_fns = getattr(self, 'character_split_fns', None)
        if character_split_fns is None:
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
