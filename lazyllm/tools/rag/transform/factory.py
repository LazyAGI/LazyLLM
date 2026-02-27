import os
import fnmatch
import inspect

from typing import Any, Dict, List, Union, Optional, Callable

from ..doc_node import DocNode, QADocNode
from lazyllm import LOG
from .base import NodeTransform

from lazyllm.components import AlpacaPrompter
from dataclasses import dataclass, field

from lazyllm import TrainableModule
from lazyllm.components.formatter import encode_query_with_filepaths
from ..prompts import LLMTransformParserPrompts

@dataclass
class TransformArgs():
    f: Union[str, Callable]
    trans_node: Optional[bool] = None
    num_workers: int = 0
    kwargs: Dict = field(default_factory=dict)
    pattern: Optional[Union[str, Callable[[str], bool]]] = None

    @staticmethod
    def from_dict(d):
        return TransformArgs(f=d['f'], trans_node=d.get('trans_node'), num_workers=d.get(
            'num_workers', 0), kwargs=d.get('kwargs', dict()), pattern=d.get('pattern'))

    def __getitem__(self, key):
        if key in self.__dict__: return getattr(self, key)
        raise KeyError(f'Key {key} is not found in transform args')

    def get(self, key):
        if key in self.__dict__: return getattr(self, key)
        return None

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

    def forward(self, nodes: Union[List[DocNode], DocNode], **kwargs) -> List[DocNode]:
        if not isinstance(nodes, (list, tuple)):
            nodes = [nodes]

        results: List[DocNode] = []
        for document in nodes:
            if not isinstance(document, DocNode):
                LOG.warning(f'Invalid document type {type(document)} got')
                continue

            matched = False
            for pt, transform in self._transformers:
                if pt and isinstance(pt, str) and not pt.startswith('*'):
                    pt = os.path.join(os.getcwd(), pt)
                if not pt or (callable(pt) and pt(document.docpath)) or (
                        isinstance(pt, str) and fnmatch.fnmatch(document.docpath, pt)):
                    chunks = transform([document], **kwargs)
                    results.extend(chunks)
                    matched = True
                    break

            if not matched:
                LOG.warning(f'No transform found for document {document.docpath} with group name `{self._name}`')

        return results


class FuncNodeTransform(NodeTransform):
    def __init__(self, func: Union[Callable[[str], List[str]], Callable[[DocNode], List[DocNode]]],
                 trans_node: bool = None, num_workers: int = 0):
        super(__class__, self).__init__(num_workers=num_workers)
        self._func, self._trans_node = func, trans_node
        self._need_ref = 'ref' in inspect.signature(func).parameters

    def forward(self, nodes: Union[List[DocNode], DocNode], **kwargs) -> List[DocNode]:
        if not isinstance(nodes, (list, tuple)):
            nodes = [nodes]

        results: List[DocNode] = []
        for node in nodes:
            if ref := kwargs.get('ref', None):
                assert self._need_ref, 'if node group has ref, the transform function must support ref parameter.'
                kwargs['ref'] = ref if self._trans_node else [r.get_text() for r in ref]
            chunks = self._func(node if self._trans_node else node.get_text(), **kwargs)
            chunks = chunks if isinstance(chunks, list) else [chunks]
            results.extend(c if isinstance(c, DocNode) else DocNode(text=str(c)) for c in chunks if c)

        return results


class LLMParser(NodeTransform):
    supported_languages = {'en': 'English', 'zh': 'Chinese'}

    def __init__(self, llm: TrainableModule, language: str, task_type: str,
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

    def forward(self, nodes: Union[List[DocNode], DocNode], **kwargs) -> List[DocNode]:
        if not isinstance(nodes, (list, tuple)):
            nodes = [nodes]

        results: List[DocNode] = []
        for node in nodes:
            if self._task_type == 'qa_img':
                inputs = encode_query_with_filepaths('Extract QA pairs from images.', [node.image_path])
            else:
                inputs = node.get_text()
            chunks = self._llm(inputs)
            chunks = [chunks] if isinstance(chunks, str) else chunks
            results.extend(c if isinstance(c, DocNode) else DocNode(text=str(c)) for c in chunks if c)
        return results

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
