from abc import ABC, abstractmethod
from copy import copy as lite_copy
from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, List, Union, Optional, Tuple, AbstractSet, Collection, Literal
from lazyllm import LOG
from lazyllm.tools.rag.doc_node import DocNode
from lazyllm import ThreadPoolExecutor
import re
from functools import partial
import os
import tiktoken
from lazyllm import config
import nltk

class MetadataMode(str, Enum):
    ALL = auto()
    EMBED = auto()
    LLM = auto()
    NONE = auto()

@dataclass
class _Split:
    text: str
    is_sentence: bool
    token_size: int

def split_text_keep_separator(text: str, separator: str) -> List[str]:
    '''Split text and keep the separator.'''
    parts = text.split(separator)
    result = [separator + s if i > 0 else s for i, s in enumerate(parts)]
    return result[1:] if len(result) > 0 and not result[0] else result


class NodeTransform(ABC):
    def __init__(self, num_workers: int = 0):
        self._number_workers = num_workers
        self._name = None

    def batch_forward(
        self, documents: Union[DocNode, List[DocNode]], node_group: str, **kwargs
    ) -> List[DocNode]:
        documents: List[DocNode] = documents if isinstance(documents, (tuple, list)) else [documents]

        def impl(node: DocNode):
            with node._lock:
                if node_group in node.children: return []
                splits = self(node, **kwargs)
                for s in splits:
                    s.parent = node
                    s._group = node_group
                node.children[node_group] = splits
                return splits

        if getattr(self, '_number_workers', 0) > 0:
            pool = ThreadPoolExecutor(max_workers=self._number_workers)
            fs = [pool.submit(impl, node) for node in documents]
            return sum([f.result() for f in fs], [])
        else:
            return sum([impl(node) for node in documents], [])

    @abstractmethod
    def transform(self, document: DocNode, **kwargs) -> List[Union[str, DocNode]]:
        raise NotImplementedError('Not implemented')

    def with_name(self, name: Optional[str], *, copy: bool = True) -> 'NodeTransform':
        if name is not None:
            if copy: return lite_copy(self).with_name(name, copy=False)
            self._name = name
        return self

    def __call__(self, node: DocNode, **kwargs: Any) -> List[DocNode]:
        # Parent and child should not be set here.
        results = self.transform(node, **kwargs)
        if isinstance(results, (DocNode, str)): results = [results]
        return [DocNode(text=chunk) if isinstance(chunk, str) else chunk for chunk in results if chunk]


class _TextSplitterBase(NodeTransform):
    def __init__(self, chunk_size: int = 1024, overlap: int = 200, num_workers: int = 0):
        super().__init__(num_workers=num_workers)
        if overlap > chunk_size:
            raise ValueError(
                f'Got a larger chunk overlap ({overlap}) than chunk size '
                f'({chunk_size}), should be smaller.'
            )

        assert (
            chunk_size > 0 and overlap >= 0
        ), 'chunk size should > 0 and overlap should >= 0'
        self._chunk_size = chunk_size
        self._overlap = overlap
        self.token_encoder = None
        self.token_decoder = None
        self.kwargs = {}
        self.sentence_split_fns = None
        self.sub_sentence_split_fns = None
        self.from_tiktoken_encoder()

    def from_tiktoken_encoder(
        self,
        encoding_name: str = 'gpt2',
        model_name: Optional[str] = None,
        allowed_special: Union[Literal['all'], AbstractSet[str]] = None,
        disallowed_special: Union[Literal['all'], Collection[str]] = 'all',
        **kwargs: Any
    ) -> '_TextSplitterBase':
        if allowed_special is None:
            allowed_special = set()
        if 'TIKTOKEN_CACHE_DIR' not in os.environ and 'DATA_GYM_CACHE_DIR' not in os.environ:
            path = os.path.join(config['model_path'], 'tiktoken')
            os.makedirs(path, exist_ok=True)
            os.environ['TIKTOKEN_CACHE_DIR'] = '/home/mnt/chenhao7/LazyLLM/lazyllm/tools/rag/transform/tiktoken'
        if model_name is not None:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)

        os.environ.pop('TIKTOKEN_CACHE_DIR')

        def _tiktoken_encoder(text: str):
            return enc.encode(
                text,
                allowed_special=allowed_special,
                disallowed_special=disallowed_special
            )

        def _tiktoken_decoder(text: str):
            return enc.decode(text)
        self.token_encoder = _tiktoken_encoder
        self.token_decoder = _tiktoken_decoder
        self.kwargs.update(kwargs)
        if isinstance(self, _TokenTextSplitter):
            extra_kwargs = {
                'encoding_name': encoding_name,
                'model_name': model_name,
                'allowed_special': allowed_special,
                'disallowed_special': disallowed_special,
            }
            self.kwargs.update(extra_kwargs)

        return self

    @classmethod
    def from_huggingface_tokenizer(self, text: str) -> '_TextSplitterBase':
        pass

    def split_text(self, text: str, metadata_size: int) -> List[str]:
        if text == '':
            return ['']
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
        splits = self._split(text, effective_chunk_size)
        chunks = self._merge(splits, effective_chunk_size)
        return chunks

    def _split(self, text: str, chunk_size: int) -> List[_Split]:
        token_size = self._token_size(text)
        if token_size <= chunk_size:
            return [_Split(text, is_sentence=True, token_size=token_size)]

        text_splits, is_sentence = self._get_splits_by_fns(text)

        results = []
        for segment in text_splits:
            token_size = self._token_size(segment)
            if token_size <= chunk_size:
                results.append(_Split(segment, is_sentence=is_sentence, token_size=token_size))
            else:
                results.extend(self._split(segment, chunk_size=chunk_size))

        return results

    def _merge(self, splits: List[_Split], chunk_size: int) -> List[str]:
        if not splits:
            return []

        if len(splits) == 1:
            return [splits[0].text]

        end_split = splits[-1]
        if end_split.token_size == chunk_size and self._overlap > 0:
            splits.pop()

            def cut_split(split: _Split) -> List[_Split]:
                text = split.text
                text_tokens = self.token_encoder(text)
                p_text = self.token_decoder(text_tokens[:len(text_tokens) // 2])
                n_text = self.token_decoder(text_tokens[len(text_tokens) // 2:])
                return [
                    _Split(p_text, is_sentence=split.is_sentence, token_size=self._token_size(p_text)),
                    _Split(n_text, is_sentence=split.is_sentence, token_size=self._token_size(n_text)),
                ]

            splits.extend(cut_split(end_split))
            end_split = splits[-1]

        result = []
        for idx in range(len(splits) - 2, -1, -1):
            start_split = splits[idx]
            if (
                start_split.token_size <= self._overlap
                and end_split.token_size <= chunk_size - self._overlap
            ):
                is_sentence = start_split.is_sentence and end_split.is_sentence
                token_size = start_split.token_size + end_split.token_size
                text = start_split.text + end_split.text
                end_split = _Split(text, is_sentence=is_sentence, token_size=token_size)
                continue
            else:
                if end_split.token_size > chunk_size:
                    raise ValueError(f'split token size ({end_split.token_size}) \
                                    is greater than chunk size ({chunk_size}).')
                else:
                    remaining_space = chunk_size - end_split.token_size
                    overlap_len = min(self._overlap, remaining_space, start_split.token_size)

                    if overlap_len > 0:
                        start_tokens = self.token_encoder(start_split.text)
                        overlap_tokens = start_tokens[-overlap_len:]
                        overlap_text = self.token_decoder(overlap_tokens)

                        is_sentence = end_split.is_sentence
                        token_size = end_split.token_size + overlap_len
                        text = overlap_text + end_split.text
                        end_split = _Split(text, is_sentence=is_sentence, token_size=token_size)

                    result.insert(0, end_split.text)
                    end_split = start_split

        result.insert(0, end_split.text)
        return result

    def transform(self, node: DocNode, **kwargs) -> List[Union[str, DocNode]]:
        return self.split_text(
            node.get_text(),
            metadata_size=self._get_metadata_size(node),
        )

    def _get_splits_by_fns(self, text: str) -> Tuple[List[str], bool]:
        # Define optional split functions (can be external or registered in __init__)
        sentence_split_fns = getattr(self, 'sentence_split_fns', None)

        if sentence_split_fns is None:
            sentence_split_fns = [
                partial(split_text_keep_separator, separator='\n\n\n'),  # paragraph
                nltk.tokenize.PunktSentenceTokenizer().tokenize,
            ]
        for split_fn in sentence_split_fns:
            splits = split_fn(text)
            if len(splits) > 1:
                return splits, True

        sub_sentence_split_fns = getattr(self, 'sub_sentence_split_fns', None)
        if sub_sentence_split_fns is None:
            sub_sentence_split_fns = [
                lambda t: re.findall(r'[^,.;。？！]+[,.;。？！]?', t),
                partial(split_text_keep_separator, separator=' '),
                list,
            ]
        for split_fn in sub_sentence_split_fns:
            splits = split_fn(text)
            if len(splits) > 1:
                break

        return splits, False

    def _get_metadata_size(self, node: DocNode) -> int:
        return max(
            self._token_size(node.get_metadata_str(mode=MetadataMode.EMBED)),
            self._token_size(node.get_metadata_str(mode=MetadataMode.LLM)),
        )

    def _token_size(self, text: str) -> int:
        return len(self.token_encoder(text))


class _TokenTextSplitter(_TextSplitterBase):
    def __init__(self, chunk_size: int = 1024, overlap: int = 200, num_workers: int = 0):
        super().__init__(chunk_size=chunk_size, overlap=overlap, num_workers=num_workers)
        self.token_encoder = None
        self.token_decoder = None
        self.kwargs = {}
        self.from_tiktoken_encoder()

    def _split(self, text: str, chunk_size: int) -> List[_Split]:
        token_size = self._token_size(text)
        if token_size <= chunk_size:
            return [_Split(text, is_sentence=True, token_size=token_size)]

        results = []
        text = self.token_encoder(text)
        start_idx = 0
        end_idx = min(start_idx + chunk_size, len(text))
        chunk_text = text[start_idx:end_idx]
        while start_idx < len(text):
            results.append(_Split(self.token_decoder(chunk_text), is_sentence=True, token_size=len(chunk_text)))
            if end_idx >= len(text):
                break
            start_idx = min(start_idx + chunk_size - self._overlap, len(text))
            end_idx = min(start_idx + chunk_size, len(text))
            chunk_text = text[start_idx:end_idx]

        return results

    def _merge(self, splits: List[_Split], chunk_size: int) -> List[str]:
        return [split.text for split in splits]
