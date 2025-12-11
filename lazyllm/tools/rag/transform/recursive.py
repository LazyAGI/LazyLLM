from typing import List, Tuple
from functools import partial
from .character import CharacterSplitter
from .base import _UNSET

class RecursiveSplitter(CharacterSplitter):
    def __init__(self, chunk_size: int = _UNSET, overlap: int = _UNSET, num_workers: int = _UNSET,
                 keep_separator: bool = _UNSET, is_separator_regex: bool = _UNSET,
                 separators: List[str] = _UNSET, **kwargs):
        super().__init__(chunk_size=chunk_size, overlap=overlap, num_workers=num_workers,
                         keep_separator=keep_separator, is_separator_regex=is_separator_regex)
        separators = self._get_param_value('separators', separators, None)

        self._separators = separators if separators else ['\n\n', '\n', ' ', '']
        self._cached_recursive_split_fns = [
            partial(self._default_split, self._get_separator_pattern(sep))
            for sep in self._separators
        ] + [list]

    def _get_splits_by_fns(self, text: str) -> Tuple[List[str], bool]:
        character_split_fns = self._character_split_fns
        if character_split_fns == []:
            character_split_fns = self._cached_recursive_split_fns
        splits = []
        for split_fn in character_split_fns:
            splits = split_fn(text)
            if len(splits) > 1:
                break

        return splits, False
