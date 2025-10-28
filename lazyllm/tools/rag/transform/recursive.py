from typing import List, Tuple
from functools import partial
from .character import CharacterSplitter
from .base import _UNSET
from typing import Optional, Union, AbstractSet, Collection, Literal, Any
from typing import Callable

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

    def from_tiktoken_encoder(self, encoding_name: str = 'gpt2', model_name: Optional[str] = None,
                              allowed_special: Union[Literal['all'], AbstractSet[str]] = None,
                              disallowed_special: Union[Literal['all'], Collection[str]] = None,
                              **kwargs) -> 'RecursiveSplitter':
        return super().from_tiktoken_encoder(encoding_name, model_name, allowed_special, disallowed_special, **kwargs)

    def split_text(self, text: str, metadata_size: int) -> List[str]:
        return super().split_text(text, metadata_size)

    def set_split_fns(self, split_fns: List[Callable[[str], List[str]]]):
        return super().set_split_fns(split_fns)

    def add_split_fn(self, split_fn: Callable[[str], List[str]]):
        return super().add_split_fn(split_fn)

    def clear_split_fns(self):
        return super().clear_split_fns()

    def from_huggingface_tokenizer(self, tokenizer: Any, **kwargs) -> 'RecursiveSplitter':
        return super().from_huggingface_tokenizer(tokenizer, **kwargs)

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
