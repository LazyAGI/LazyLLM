import os
import fnmatch

from typing import Any, Dict, List, Union, Optional, Callable

from lazyllm.tools.rag.doc_node import DocNode, QADocNode
from lazyllm import LOG
from .base import NodeTransform
from lazyllm.components import AlpacaPrompter
from dataclasses import dataclass, field

from lazyllm import TrainableModule
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.tools.rag.prompts import LLMTransformParserPrompts

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

<<<<<<< HEAD:lazyllm/tools/rag/transform.py

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

    def __call__(self, node: Union[DocNode, List[DocNode]], **kwargs: Any) -> List[DocNode]:
        # Parent and child should not be set here.
        def impl(n):
            results = self.transform(n, **kwargs)
            return [results] if isinstance(results, (DocNode, str)) else results

        results = impl(node) if isinstance(node, DocNode) else [i for n in node for i in impl(n)]
        return [DocNode(text=chunk) if isinstance(chunk, str) else chunk for chunk in results if chunk]


=======
>>>>>>> 0b7658a (reconstruction of transform v1):lazyllm/tools/rag/transform/factory.py
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

    def transform(self, document: DocNode, **kwargs) -> List[Union[str, DocNode]]:
        if not isinstance(document, DocNode): LOG.warning(f'Invalud document type {type(document)} got')
        for pt, transform in self._transformers:
            if pt and isinstance(pt, str) and not pt.startswith('*'): pt = os.path.join(str(os.cwd()), pt)
            if not pt or (callable(pt) and pt(document.docpath)) or (
                    isinstance(pt, str) and fnmatch.fnmatch(document.docpath, pt)):
                return transform(document, **kwargs)
        LOG.warning(f'No transform found for document {document.docpath} with group name `{self._name}`')
        return []

class FuncNodeTransform(NodeTransform):
    '''Used for user defined function.

    Wrapped the transform to: List[Docnode] -> List[Docnode]

    This wrapper supports when trans_node is False:
        1. str -> list: transform=lambda t: t.split('\n')
        2. str -> str: transform=lambda t: t[:3]

    This wrapper supports when trans_node is True:
        1. DocNode -> list: pipeline(lambda x:x, SentenceSplitter)
        2. DocNode -> DocNode: pipeline(LLMParser)
    '''

    def __init__(self, func: Union[Callable[[str], List[str]], Callable[[DocNode], List[DocNode]]],
                 trans_node: bool = None, num_workers: int = 0):
        super(__class__, self).__init__(num_workers=num_workers)
        self._func, self._trans_node = func, trans_node

    def transform(self, node: DocNode, **kwargs) -> List[Union[str, DocNode]]:
        return self._func(node if self._trans_node else node.get_text())


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

    def transform(self, node: DocNode, **kwargs) -> List[str]:
        if self._task_type == 'qa_img':
            inputs = encode_query_with_filepaths('Extract QA pairs from images.', [node.image_path])
        else:
            inputs = node.get_text()
        result = self._llm(inputs)
        return [result] if isinstance(result, str) else result

    def _format(self, input):
        if self._task_type == 'keywords':
            return [s.strip() for s in input.split(',')]
        elif self._task_type in ('qa', 'qa_img'):
            return [QADocNode(query=q.strip()[3:].strip(), answer=a.strip()[3:].strip()) for q, a in zip(
                list(filter(None, map(str.strip, input.split('\n'))))[::2],
                list(filter(None, map(str.strip, input.split('\n'))))[1::2])]
        return input
