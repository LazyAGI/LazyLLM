import os
import fnmatch
import hashlib
import inspect
import json

from typing import Any, Dict, List, Union, Optional, Callable

from ..doc_node import DocNode, QADocNode
from lazyllm import LOG
from .base import NodeTransform

from lazyllm.components import AlpacaPrompter
from dataclasses import dataclass, field

from lazyllm.module import LLMBase
from lazyllm.components.formatter import encode_query_with_filepaths
from ..prompts import LLMTransformParserPrompts

def _callable_sig(f: Optional[Callable], name_override: Optional[str] = None) -> str:
    if f is None:
        return '__default__'
    if name_override:
        return name_override
    qualname = getattr(f, '__qualname__', None)
    module = getattr(f, '__module__', None)
    if qualname and '<lambda>' not in qualname and '<locals>' not in qualname:
        return f'{module}.{qualname}' if module else qualname
    try:
        return '__lambda__::' + inspect.getsource(f).strip()
    except (OSError, TypeError):
        raise ValueError(
            f'Cannot compute a stable signature for lambda/closure {f!r}. '
            'Please pass a named function or set TransformArgs.name explicitly.'
        )

@dataclass
class TransformArgs():
    f: Union[str, Callable]
    trans_node: Optional[bool] = None
    num_workers: int = 0
    kwargs: Dict = field(default_factory=dict)
    pattern: Optional[Union[str, Callable[[str], bool]]] = None
    name: Optional[str] = None  # explicit name for signature (useful for lambdas)

    @staticmethod
    def from_dict(d):
        return TransformArgs(f=d['f'], trans_node=d.get('trans_node'), num_workers=d.get(
            'num_workers', 0), kwargs=d.get('kwargs', dict()), pattern=d.get('pattern'),
            name=d.get('name'))

    def __getitem__(self, key):
        if key in self.__dict__: return getattr(self, key)
        raise KeyError(f'Key {key} is not found in transform args')

    def get(self, key):
        if key in self.__dict__: return getattr(self, key)
        return None

    def signature(self) -> str:
        f = self.f
        kw = self.kwargs or {}
        cls = None
        if isinstance(f, str):
            from ..doc_impl import _transmap
            cls = _transmap.get(f.lower())
            type_name = f
        elif inspect.isclass(f):
            cls = f
            type_name = f.__name__
        else:
            return hashlib.sha256(json.dumps({
                'type': '__callable__',
                'func': _callable_sig(f, self.name),
                'trans_node': self.trans_node,
            }, sort_keys=True).encode()).hexdigest()[:16]

        sig_dict = _build_transform_sig(type_name, cls, kw)
        if self.pattern is not None:
            sig_dict['pattern'] = _callable_sig(self.pattern) if callable(self.pattern) else self.pattern
        return hashlib.sha256(json.dumps(sig_dict, sort_keys=True).encode()).hexdigest()[:16]


def _build_splitter_sig(type_name: str, cls, _get) -> Optional[Dict]:
    from .sentence import SentenceSplitter
    from .character import CharacterSplitter
    from .recursive import RecursiveSplitter
    from .markdown import MarkdownSplitter
    from .code import (XMLSplitter, JSONSplitter, JSONLSplitter, YAMLSplitter,
                       HTMLSplitter, GeneralCodeSplitter, CodeSplitter)

    if cls is SentenceSplitter:
        return {'type': type_name, 'chunk_size': _get('chunk_size'), 'chunk_overlap': _get('chunk_overlap')}
    if cls is CharacterSplitter:
        return {'type': type_name, 'chunk_size': _get('chunk_size'), 'overlap': _get('overlap'),
                'separator': _get('separator'), 'is_separator_regex': _get('is_separator_regex'),
                'keep_separator': _get('keep_separator')}
    if cls is RecursiveSplitter:
        return {'type': type_name, 'chunk_size': _get('chunk_size'), 'overlap': _get('overlap'),
                'separators': _get('separators'), 'keep_separator': _get('keep_separator'),
                'is_separator_regex': _get('is_separator_regex')}
    if cls is MarkdownSplitter:
        return {'type': type_name, 'chunk_size': _get('chunk_size'), 'overlap': _get('overlap'),
                'keep_trace': _get('keep_trace'), 'keep_headers': _get('keep_headers'),
                'keep_code_blocks': _get('keep_code_blocks'), 'keep_tables': _get('keep_tables'),
                'keep_images': _get('keep_images')}
    if cls is XMLSplitter:
        return {'type': type_name, 'chunk_size': _get('chunk_size'),
                'keep_trace': _get('keep_trace'), 'keep_tags': _get('keep_tags')}
    if cls in (JSONSplitter, JSONLSplitter, YAMLSplitter):
        return {'type': type_name, 'chunk_size': _get('chunk_size'), 'compact_output': _get('compact_output')}
    if cls is HTMLSplitter:
        return {'type': type_name, 'chunk_size': _get('chunk_size')}
    if cls is GeneralCodeSplitter:
        return {'type': type_name, 'chunk_size': _get('chunk_size'), 'filetype': _get('filetype')}
    if cls is CodeSplitter:
        return {'type': type_name, 'chunk_size': _get('chunk_size'), 'overlap': _get('overlap'),
                'filetype': _get('filetype')}
    return None


def _build_transform_sig(type_name: str, cls, kw: Dict) -> Dict:
    def _get(key, default=None):
        return kw.get(key, default)

    splitter_sig = _build_splitter_sig(type_name, cls, _get)
    if splitter_sig is not None:
        return splitter_sig

    from .groupby import GroupNodeParser
    from .treebuilder import TreeBuilderParser
    from .treefixer import TreeFixerParser
    from .layout import LayoutNodeParser

    if cls is GroupNodeParser:
        return {'type': type_name, 'max_length': _get('max_length'), 'merge_title': _get('merge_title')}
    if cls is TreeBuilderParser:
        return {'type': type_name,
                'get_level_sig': _callable_sig(_get('get_level')),
                'is_valid_child_sig': _callable_sig(_get('is_valid_child'))}
    if cls is TreeFixerParser:
        return {'type': type_name, 'patterns': _get('patterns'),
                'skip_level_under': _get('skip_level_under'), 'extra_patterns': _get('extra_patterns')}
    if cls is LayoutNodeParser:
        return {'type': type_name,
                'rules_sig': _callable_sig(_get('rules')),
                'group_by_sig': _callable_sig(_get('group_by')),
                'sort_by_sig': _callable_sig(_get('sort_by')),
                'post_process_sig': _callable_sig(_get('post_process'))}
    if cls is LLMParser:
        llm = _get('llm')
        llm_sig = type(llm).__name__ if llm is not None else '__none__'
        prompts = _get('prompts')
        if prompts is None:
            prompts_sig = '__default__'
        else:
            try:
                prompts_sig = hashlib.sha256(
                    json.dumps(prompts.__dict__, sort_keys=True).encode()
                ).hexdigest()[:16]
            except Exception:
                prompts_sig = repr(prompts)
        return {'type': type_name, 'llm_sig': llm_sig,
                'language': _get('language'), 'task_type': _get('task_type'), 'prompts_sig': prompts_sig}
    # FuncNodeTransform or unknown
    func = _get('func') or _get('f')
    return {'type': type_name, 'func_sig': _callable_sig(func), 'trans_node': _get('trans_node')}

def build_nodes_from_splits(
    text_splits: List[str], doc: DocNode, node_group: str
) -> List[DocNode]:
    nodes: List[DocNode] = []
    for text_chunk in text_splits:
        if not text_chunk:
            continue
        node = DocNode(
            text=text_chunk,
            group=node_group,
            parent=doc,
        )
        nodes.append(node)

    doc.children[node_group] = nodes
    return nodes

def make_transform(t: Union[TransformArgs, Dict[str, Any]], group_name: Optional[str] = None) -> NodeTransform:
    if isinstance(t, dict): t = TransformArgs.from_dict(t)
    transform, trans_node, num_workers = t['f'], t['trans_node'], t['num_workers']
    num_workers = dict(num_workers=num_workers) if num_workers > 0 else dict()
    return (transform(**t['kwargs'], **num_workers).with_name(group_name, copy=False) if isinstance(transform, type)
            else transform.with_name(group_name) if isinstance(transform, NodeTransform)
            else FuncNodeTransform(transform, trans_node=trans_node, **num_workers).with_name(group_name, copy=False))


class AdaptiveTransform(NodeTransform):
    def __init__(self, transforms: Union[List[Union[TransformArgs, Dict]], Union[TransformArgs, Dict]],
                 num_workers: int = 0):
        super().__init__(num_workers=num_workers)
        if not isinstance(transforms, (tuple, list)): transforms = [transforms]
        self._transformers = [(t.get('pattern'), make_transform(t)) for t in transforms]

    def forward(self, node: DocNode, **kwargs) -> List[DocNode]:
        if not isinstance(node, DocNode):
            LOG.warning(f'Invalid document type {type(node)} got')
            return []
        for pt, transform in self._transformers:
            if pt and isinstance(pt, str) and not pt.startswith('*'):
                pt = os.path.join(os.getcwd(), pt)
            if not pt or (callable(pt) and pt(node.docpath)) or (
                    isinstance(pt, str) and fnmatch.fnmatch(node.docpath, pt)):
                chunks = transform(node, **kwargs)
                return list(chunks) if not isinstance(chunks, list) else chunks
        LOG.warning(f'No transform found for document {node.docpath} with group name `{self._name}`')
        return []


class FuncNodeTransform(NodeTransform):
    def __init__(self, func: Union[Callable[[str], List[str]], Callable[[DocNode], List[DocNode]]],
                 trans_node: bool = None, num_workers: int = 0):
        super(__class__, self).__init__(num_workers=num_workers)
        self._func, self._trans_node = func, trans_node
        self._need_ref = 'ref' in inspect.signature(func).parameters

    def forward(self, node: DocNode, **kwargs) -> List[DocNode]:
        if ref := kwargs.get('ref', None):
            assert self._need_ref, 'if node group has ref, the transform function must support ref parameter.'
            kwargs['ref'] = ref if self._trans_node else [r.get_text() for r in ref]
        chunks = self._func(node if self._trans_node else node.get_text(), **kwargs)
        chunks = chunks if isinstance(chunks, list) else [chunks]
        return [c if isinstance(c, DocNode) else DocNode(text=str(c)) for c in chunks if c]


class LLMParser(NodeTransform):
    supported_languages = {'en': 'English', 'zh': 'Chinese'}

    def __init__(self, llm: LLMBase, language: str, task_type: str,
                 prompts: Optional[LLMTransformParserPrompts] = None, num_workers: int = 30):
        super(__class__, self).__init__(num_workers=num_workers)
        assert language in self.supported_languages, f'Not supported language {language}'
        assert task_type in ['summary', 'keywords', 'qa', 'qa_img'], f'Not supported task_type {task_type}'
        self._task_type = task_type
        self._prompts = prompts or LLMTransformParserPrompts()
        task_prompt_tempalte = getattr(self._prompts, self._task_type)
        task_prompt = task_prompt_tempalte.format(language=self.supported_languages[language])
        if self._task_type == 'qa_img':
            prompt = dict(system=task_prompt, user='{input}')
        else:
            prompt = dict(system=task_prompt, user='#input:\n{input}\n#output:\n')
        self._llm = llm.share(prompt=AlpacaPrompter(prompt), stream=False, format=self._format)
        self._task_type = task_type

    def forward(self, node: DocNode, **kwargs) -> List[DocNode]:
        if self._task_type == 'qa_img':
            inputs = encode_query_with_filepaths('Extract QA pairs from images.', [node.image_path])
        else:
            inputs = node.get_text()
        chunks = self._llm(inputs)
        chunks = [chunks] if isinstance(chunks, str) else chunks
        return [c if isinstance(c, DocNode) else DocNode(text=str(c)) for c in chunks if c]

    def _format(self, input):
        if isinstance(input, dict):
            input = input.get('output', input.get('text', input.get('content', str(input))))

        if not isinstance(input, str):
            input = str(input)

        if self._task_type == 'keywords':
            return [s.strip() for s in input.split(',')]
        elif self._task_type in ('qa', 'qa_img'):
            return [QADocNode(query=q.strip()[3:].strip(), answer=a.strip()[3:].strip()) for q, a in zip(
                list(filter(None, map(str.strip, input.split('\n'))))[::2],
                list(filter(None, map(str.strip, input.split('\n'))))[1::2])]
        return input
