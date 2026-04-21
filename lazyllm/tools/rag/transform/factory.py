import os
import fnmatch
import hashlib
import inspect
import json

from typing import Any, Dict, List, Union, Optional, Callable
from dataclasses import dataclass, field

from lazyllm import LOG
from lazyllm.components import AlpacaPrompter
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.module import LLMBase

from ..doc_impl import _transmap
from ..doc_node import DocNode, QADocNode
from ..prompts import LLMTransformParserPrompts
from .base import NodeTransform

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

        instance = cls(**kw) if cls is not None else None
        if instance is not None:
            sig_dict = {'type': type_name, **instance.sig_fields()}
        else:
            sig_dict = {'type': type_name}
        if self.pattern is not None:
            sig_dict['pattern'] = _callable_sig(self.pattern) if callable(self.pattern) else self.pattern
        return hashlib.sha256(json.dumps(sig_dict, sort_keys=True).encode()).hexdigest()[:16]


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

    def sig_fields(self) -> Dict:
        return {'func_sig': _callable_sig(self._func), 'trans_node': self._trans_node}

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
        self._language = language

    def sig_fields(self) -> Dict:
        prompts_sig = '__default__'
        if self._prompts is not None:
            try:
                prompts_sig = hashlib.sha256(
                    json.dumps(self._prompts.__dict__, sort_keys=True).encode()
                ).hexdigest()[:16]
            except Exception:
                prompts_sig = repr(self._prompts)
        return {
            'llm_sig': type(self._llm).__name__,
            'language': self._language,
            'task_type': self._task_type,
            'prompts_sig': prompts_sig,
        }

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
