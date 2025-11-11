from typing import List, Tuple
from functools import partial
from .character import CharacterSplitter

class RecursiveSplitter(CharacterSplitter):
    def __init__(self, chunk_size: int = 1024, overlap: int = 200, num_workers: int = 0,
                 separator: str = ' ', keep_separator: bool = False, is_separator_regex: bool = False,
                 separators: List[str] = None, **kwargs):
        super().__init__(chunk_size=chunk_size, overlap=overlap, num_workers=num_workers,
                         separator=separator, keep_separator=keep_separator, is_separator_regex=is_separator_regex)
        self._separators = separators if separators else ['\n\n', '\n', ' ', '']

    def _get_splits_by_fns(self, text: str) -> Tuple[List[str], bool]:
        character_split_fns = self._character_split_fns
        if character_split_fns == []:
            character_split_fns = [
                partial(self.default_split, self._get_separator_pattern(separator))
                for separator in self._separators
            ] + [list]
        splits = []
        for split_fn in character_split_fns:
            splits = split_fn(text)
            if len(splits) > 1:
                break

        return splits, False
