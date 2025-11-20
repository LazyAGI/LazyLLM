from functools import partial
import re
import inspect

from typing import List, Union, Tuple, Callable, Optional, AbstractSet, Collection, Literal, Any
from .base import _TextSplitterBase, _TokenTextSplitter, _Split, _UNSET

class CharacterSplitter(_TextSplitterBase):
    def __init__(self, chunk_size: int = _UNSET, overlap: int = _UNSET, num_workers: int = _UNSET,
                 separator: str = _UNSET, is_separator_regex: bool = _UNSET, keep_separator: bool = _UNSET, **kwargs):
        super().__init__(chunk_size=chunk_size, overlap=overlap, num_workers=num_workers)
        separator = self._get_param_value('separator', separator, ' ')
        is_separator_regex = self._get_param_value('is_separator_regex', is_separator_regex, False)
        keep_separator = self._get_param_value('keep_separator', keep_separator, False)

        self._separator = separator
        self._is_separator_regex = is_separator_regex
        self._keep_separator = keep_separator
        self._character_split_fns = []
        self._cached_sep_pattern = self._get_separator_pattern(self._separator)
        self._cached_default_split_fns = None

    def from_tiktoken_encoder(self, encoding_name: str = 'gpt2', model_name: Optional[str] = None,
                              allowed_special: Union[Literal['all'], AbstractSet[str]] = None,
                              disallowed_special: Union[Literal['all'], Collection[str]] = None,
                              **kwargs) -> 'CharacterSplitter':
        return super().from_tiktoken_encoder(encoding_name, model_name, allowed_special, disallowed_special, **kwargs)

    def from_huggingface_tokenizer(self, tokenizer: Any, **kwargs) -> 'CharacterSplitter':
        return super().from_huggingface_tokenizer(tokenizer, **kwargs)

    def split_text(self, text: str, metadata_size: int) -> List[str]:
        return super().split_text(text, metadata_size)

    def _split(self, text: str, chunk_size: int) -> List[_Split]:
        token_size = self._token_size(text)
        if token_size <= chunk_size:
            return [_Split(text, is_sentence=True, token_size=token_size)]

        text_splits, is_sentence = self._get_splits_by_fns(text)

        if len(text_splits) == 1 and self._token_size(text_splits[0]) > chunk_size:
            token_splitter = _TokenTextSplitter(chunk_size=chunk_size, overlap=self._overlap)
            token_sub_texts = token_splitter.split_text(text_splits[0], metadata_size=0)
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

    def set_split_fns(self, split_fns: Union[Callable[[str], List[str]], List[Callable[[str], List[str]]]], bind_separator: bool = None):  # noqa: E501
        if not isinstance(split_fns, list):
            split_fns = [split_fns]
        self._character_split_fns = []
        for split_fn in split_fns:
            if bind_separator is None:
                sig = inspect.signature(split_fn)
                has_separator = 'separator' in sig.parameters
                should_bind = has_separator
            else:
                should_bind = bind_separator

            if should_bind:
                fn = partial(split_fn, separator=self._separator)
            else:
                fn = split_fn

            self._character_split_fns.append(fn)

    def add_split_fn(self, split_fn: Callable[[str], List[str]], index: Optional[int] = None, bind_separator: bool = None):  # noqa: E501
        if bind_separator is None:
            sig = inspect.signature(split_fn)
            has_separator = 'separator' in sig.parameters
            should_bind = has_separator
        else:
            should_bind = bind_separator

        if should_bind:
            fn = partial(split_fn, separator=self._separator)
        else:
            fn = split_fn

        if index is None:
            self._character_split_fns.append(fn)
        else:
            self._character_split_fns.insert(index, fn)

    def clear_split_fns(self):
        self._character_split_fns = []

    def _get_splits_by_fns(self, text: str) -> Tuple[List[str], bool]:
        character_split_fns = self._character_split_fns
        if character_split_fns == []:
            if self._cached_default_split_fns is None:
                self._cached_default_split_fns = [
                    partial(self._default_split, self._cached_sep_pattern),
                    lambda t: t.split(' '),
                    list
                ]
            character_split_fns = self._cached_default_split_fns

        splits = []
        for split_fn in character_split_fns:
            splits = split_fn(text)
            if len(splits) > 1:
                break

        return splits, False

    def _default_split(self, sep_pattern: Union[str, set[str]], text: str) -> List[str]:
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

    def set_default(self, **kwargs):
        super().set_default(**kwargs)

    def get_default(self, param_name: Optional[str] = None):
        return super().get_default(param_name)

    def reset_default(self):
        super().reset_default()
