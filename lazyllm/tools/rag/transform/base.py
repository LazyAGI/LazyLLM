from copy import copy as copy_obj
from enum import Enum
from dataclasses import dataclass, field
from typing import (
    Any, List, Union, Optional, Tuple, AbstractSet,
    Collection, Literal, Callable, Dict, Iterator
)
from lazyllm import LOG
from ..doc_node import DocNode, RichDocNode
from ....common.deprecated import deprecated
from lazyllm import ThreadPoolExecutor
from itertools import chain
import re
from functools import partial
import os
import threading
from lazyllm.thirdparty import tiktoken
from lazyllm import config, ModuleBase
from pathlib import Path
import inspect
from lazyllm.thirdparty import nltk
from lazyllm.thirdparty import transformers

class MetadataMode(str, Enum):
    ALL = 'ALL'
    EMBED = 'EMBED'
    LLM = 'LLM'
    NONE = 'NONE'

@dataclass
class _Split:
    text: str
    is_sentence: bool
    token_size: int

def split_text_keep_separator(text: str, separator: str) -> List[str]:
    if not separator:
        return [text] if text else []

    if separator not in text:
        return [text]

    result = []
    start = 0
    sep_len = len(separator)

    while start < len(text):
        idx = text.find(separator, start)

        if idx == -1:
            result.append(text[start:])
            break

        if idx == 0:
            start = sep_len
            continue

        result.append(text[start:idx + sep_len])
        start = idx + sep_len

    return result


class NodeTransform(ModuleBase):
    __support_rich__ = False

    def __init__(self, num_workers: int = 0, rules: Optional['RuleSet'] = None,
                 return_trace: bool = False, **kwargs):
        super().__init__(return_trace=return_trace, **kwargs)
        self._number_workers = num_workers
        self._name = None
        self._rules = rules or RuleSet()
        self._on_match: Optional[Callable[[Any, Tuple['Rule', Any], '_Context'], Any]] = None
        self._on_miss: Optional[Callable[[Any, '_Context'], Any]] = None

    def _get_ref_nodes(self, node, ref_path):
        current = [node]
        for key in ref_path:
            current = list(
                chain.from_iterable(
                    n.children.get(key, []) for n in current
                )
            )
        return current

    def batch_forward(
        self, documents: Union[DocNode, List[DocNode]], node_group: str, ref_path: List[str] = None, **kwargs
    ) -> List[DocNode]:
        documents: List[DocNode] = documents if isinstance(documents, (tuple, list)) else [documents]

        def impl(node: DocNode):
            with node._lock:
                if node_group in node.children:
                    return []
                if ref_path:
                    ref_nodes = self._get_ref_nodes(node, ref_path)
                    if not ref_nodes:
                        return []
                    forward_kwargs = {**kwargs, 'ref': ref_nodes}
                    if self.__support_rich__:
                        if len(ref_nodes) == 1:
                            splits = self.forward(ref_nodes[0], **forward_kwargs)
                        else:
                            input_node = RichDocNode(nodes=ref_nodes)
                            splits = self.forward(input_node, **forward_kwargs)
                    else:
                        splits = []
                        for n in ref_nodes:
                            splits.extend(self.forward(n, **forward_kwargs))
                else:
                    if isinstance(node, RichDocNode) and not self.__support_rich__:
                        splits = []
                        for sub in node.nodes:
                            splits.extend(self.forward(sub, **kwargs))
                    else:
                        splits = self.forward(node, **kwargs)
                for s in splits:
                    s.parent = node
                    s._group = node_group
                node.children[node_group] = splits
                return splits

        if getattr(self, '_number_workers', 0) > 0:
            with ThreadPoolExecutor(max_workers=self._number_workers) as pool:
                fs = [pool.submit(impl, node) for node in documents]
            return sum([f.result() for f in fs], [])
        else:
            return sum([impl(node) for node in documents], [])

    def forward(self, nodes: DocNode, **kwargs) -> List[DocNode]:
        raise NotImplementedError(
            'Subclasses must implement forward() to process a single DocNode or RichDocNode'
        )

    @deprecated('forward')
    def transform(self, node: DocNode, **kwargs) -> List[Union[str, DocNode]]:
        return self.forward(node, **kwargs)

    def process(self, nodes: List[Any], on_match: Optional[Callable] = None,
                on_miss: Optional[Callable] = None) -> List[Any]:
        instance_match = self._on_match
        instance_miss = self._on_miss

        match_handler = (
            on_match if on_match is not None
            else (instance_match if instance_match is not None else self._default_match_handler)
        )
        miss_handler = (
            on_miss if on_miss is not None
            else (instance_miss if instance_miss is not None else self._default_miss_handler)
        )

        ctx = _Context(total=len(nodes))
        results = []

        for i, node in enumerate(nodes):
            ctx.current_idx = i
            match = self._rules.first(node)
            processed = match_handler(node, match, ctx) if match else miss_handler(node, ctx)
            results.append(processed)
            ctx.prev_node, ctx.prev_result = node, processed

        return results

    def _default_match_handler(self, node, matched, ctx):
        return matched[1]

    def _default_miss_handler(self, node, ctx):
        return node

    def with_name(self, name: Optional[str], *, copy: bool = True) -> 'NodeTransform':
        if name is not None:
            if copy: return copy_obj(self).with_name(name, copy=False)
            self._name = name
        return self

    def __call__(self, node_or_nodes: Union[DocNode, List[DocNode]], **kwargs: Any) -> List[DocNode]:
        if isinstance(node_or_nodes, (list, tuple)):
            nodes = node_or_nodes
            for n in nodes:
                if not isinstance(n, DocNode):
                    raise TypeError(
                        f'__call__() expects DocNode objects, got {type(n).__name__} '
                        f'in list. Use forward() directly if you need to process other types.'
                    )
            results = []
            for n in nodes:
                results.extend(self._forward_single(n, **kwargs))
            return results
        if not isinstance(node_or_nodes, DocNode):
            raise TypeError(
                f'__call__() expects DocNode or RichDocNode, got {type(node_or_nodes).__name__}'
            )
        return self._forward_single(node_or_nodes, **kwargs)

    def _forward_single(self, node: Union[DocNode, RichDocNode], **kwargs: Any) -> List[DocNode]:
        if isinstance(node, RichDocNode) and not self.__support_rich__:
            results = []
            for sub in node.nodes:
                results.extend(self.forward(sub, **kwargs))
            return results
        return self.forward(node, **kwargs)


_tiktoken_env_lock = threading.Lock()

_UNSET = object()

class _TextSplitterBase(NodeTransform):
    _default_params = {}
    _default_params_lock = threading.RLock()

    def __init__(self, chunk_size: int = _UNSET, overlap: int = _UNSET, num_workers: int = _UNSET):
        chunk_size = self._get_param_value('chunk_size', chunk_size, 1024)
        overlap = self._get_param_value('overlap', overlap, 200)
        num_workers = self._get_param_value('num_workers', num_workers, 0)

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
        self.from_tiktoken_encoder()

    @classmethod
    def _get_class_lock(cls):
        if '_default_params_lock' not in cls.__dict__:
            cls._default_params_lock = threading.RLock()
        return cls._default_params_lock

    @classmethod
    def _get_param_value(cls, param_name: str, value, default):
        if value is not _UNSET:
            return value

        lock = cls._get_class_lock()
        with lock:
            if hasattr(cls, '_default_params') and param_name in cls._default_params:
                return cls._default_params[param_name]
        return default

    @classmethod
    def set_default(cls, **kwargs):
        lock = cls._get_class_lock()
        with lock:
            if '_default_params' not in cls.__dict__:
                cls._default_params = {}
            cls._default_params.update(kwargs)

        LOG.info(f'{cls.__name__} default parameters updated: {kwargs}')

    @classmethod
    def get_default(cls, param_name: Optional[str] = None):
        lock = cls._get_class_lock()
        with lock:
            defaults = getattr(cls, '_default_params', {})
            if param_name is None:
                return defaults.copy()
            return defaults.get(param_name)

    @classmethod
    def reset_default(cls):
        lock = cls._get_class_lock()
        with lock:
            if '_default_params' in cls.__dict__:
                cls._default_params.clear()

        LOG.info(f'{cls.__name__} default parameters reset')

    def from_tiktoken_encoder(self, encoding_name: str = 'gpt2', model_name: Optional[str] = None,  # noqa: C901
                              allowed_special: Union[Literal['all'], AbstractSet[str]] = None,
                              disallowed_special: Union[Literal['all'], Collection[str]] = 'all',
                              **kwargs: Any) -> '_TextSplitterBase':
        if allowed_special is None:
            allowed_special = set()

        with _tiktoken_env_lock:
            tiktoken_cache_dir_set = False
            original_value = os.environ.get('TIKTOKEN_CACHE_DIR')
            if 'TIKTOKEN_CACHE_DIR' not in os.environ and 'DATA_GYM_CACHE_DIR' not in os.environ:
                try:
                    model_path = config['model_path']
                except (RuntimeError, KeyError, PermissionError):
                    model_path = None

                if not model_path:
                    model_path = os.path.join(os.path.expanduser('~'), '.lazyllm')

                path = Path(model_path) / 'tiktoken'
                try:
                    if path.exists():
                        if not os.access(path, os.W_OK):
                            raise PermissionError(f'No write permission for existing directory: {path}')
                    else:
                        path.mkdir(parents=True, exist_ok=True)
                    os.environ['TIKTOKEN_CACHE_DIR'] = str(path)
                    tiktoken_cache_dir_set = True
                except PermissionError:
                    fallback_path = os.path.join(os.path.expanduser('~'), '.lazyllm', 'tiktoken')
                    os.makedirs(fallback_path, exist_ok=True)
                    os.environ['TIKTOKEN_CACHE_DIR'] = fallback_path
                    tiktoken_cache_dir_set = True

        try:
            if model_name is not None:
                enc = tiktoken.encoding_for_model(model_name)
            else:
                enc = tiktoken.get_encoding(encoding_name)
        finally:
            with _tiktoken_env_lock:
                if tiktoken_cache_dir_set:
                    os.environ.pop('TIKTOKEN_CACHE_DIR', None)
                    if original_value is not None:
                        os.environ['TIKTOKEN_CACHE_DIR'] = original_value

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

    def from_huggingface_tokenizer(self, tokenizer: Any):
        try:
            if not isinstance(tokenizer, transformers.tokenization_utils_base.PreTrainedTokenizerBase):
                raise ValueError('Tokenizer received was not an instance of PreTrainedTokenizerBase')

            def _huggingface_encoder(text: str):
                return tokenizer.encode(text, add_special_tokens=False)

            def _huggingface_decoder(token_ids):
                if isinstance(token_ids, list) and len(token_ids) == 0:
                    return ''
                return tokenizer.decode(token_ids, skip_special_tokens=True)

            self.token_encoder = _huggingface_encoder
            self.token_decoder = _huggingface_decoder

        except Exception as e:
            raise ValueError(f'Failed to initialize HuggingFace tokenizer: {e}')

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

    def forward(self, node: Union[DocNode, RichDocNode], **kwargs) -> List[DocNode]:
        return [c if isinstance(c, DocNode) else DocNode(text=str(c)) for c in
                self.split_text(node.get_text(), metadata_size=self._get_metadata_size(node)) if c]

    def set_split_fns(self, split_fns: List[Callable[[str], List[str]]],
                      sub_split_fns: Optional[List[Callable[[str], List[str]]]] = None) -> '_TextSplitterBase':
        pass

    def add_split_fn(self, split_fn: Callable[[str], List[str]], index: Optional[int] = None):
        pass

    def clear_split_fns(self):
        pass

    def _get_splits_by_fns(self, text: str) -> Tuple[List[str], bool]:
        sentence_split_fns = [
            partial(split_text_keep_separator, separator='\n\n\n'),  # paragraph
            nltk.tokenize.PunktSentenceTokenizer().tokenize,
        ]
        for split_fn in sentence_split_fns:
            splits = split_fn(text)
            if len(splits) > 1:
                return splits, True

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
        self.kwargs = {}

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


@dataclass(frozen=True)
class Rule:
    name: str
    match: Callable[[Any], Any]
    apply: Callable[..., Any]
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __call__(self, data: Any) -> Optional[Any]:
        match_result = self.match(data)
        if not match_result:
            return None
        try:
            sig = inspect.signature(self.apply)
            params = [
                p for p in sig.parameters.values()
                if p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD
                )
            ]
            if len(params) >= 3:
                return self.apply(data, match_result, self)
        except (ValueError, TypeError):
            pass
        return self.apply(data, self)

    @staticmethod
    def build(name: str, rule: Union[str, Callable[[Any], bool]],
              apply: Callable[[Any, 'Rule'], Any], priority: int = 0) -> 'Rule':
        if isinstance(rule, str):
            compiled = re.compile(rule)
            return Rule(
                name=name,
                match=lambda text: compiled.search(text),
                apply=lambda text, match_result, r: apply(match_result, text),
                priority=priority,
            )
        if callable(rule):
            return Rule(
                name=name,
                match=lambda data: True if rule(data) else None,
                apply=lambda data, _match_result, r: apply(data, r),
                priority=priority,
            )
        raise TypeError('rule must be a pattern string or a predicate callable')

@dataclass
class _Context:
    total: int
    current_idx: int = 0
    prev_node: Optional[Any] = None
    prev_result: Optional[Any] = None
    user_data: Dict[str, Any] = field(default_factory=dict)

class RuleSet:
    def __init__(self, rules: Optional[List[Rule]] = None):
        self._rules: List[Rule] = []
        if rules:
            self.extend(rules)

    def add(self, *rules: Rule) -> 'RuleSet':
        self._rules.extend(rules)
        self._sort()
        return self

    def extend(self, rules: List[Rule]) -> 'RuleSet':
        self._rules.extend(rules)
        self._sort()
        return self

    def _sort(self):
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def first(self, data: Any) -> Optional[tuple[Rule, Any]]:
        for rule in self._rules:
            result = rule(data)
            if result is not None:
                return (rule, result)
        return None

    def all(self, data: Any) -> List[tuple[Rule, Any]]:
        results = []
        for rule in self._rules:
            result = rule(data)
            if result is not None:
                results.append((rule, result))
        return results

    def filter(self, predicate: Callable[[Rule], bool]) -> 'RuleSet':
        return RuleSet([r for r in self._rules if predicate(r)])

    def __iter__(self) -> Iterator[Rule]:
        return iter(self._rules)

    def __len__(self) -> int:
        return len(self._rules)

    def __bool__(self) -> bool:
        return bool(self._rules)
