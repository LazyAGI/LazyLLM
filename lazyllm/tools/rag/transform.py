from abc import ABC, abstractmethod
from copy import copy as lite_copy
from dataclasses import dataclass, field
import requests
import os
import fnmatch

from functools import partial
import re
from typing import Any, Callable, Dict, List, Tuple, Union, Optional
from lazyllm.components import AlpacaPrompter
from lazyllm.thirdparty import nltk
import tiktoken

from .doc_node import DocNode, MetadataMode, QADocNode
from lazyllm import LOG, TrainableModule, ThreadPoolExecutor, config
from lazyllm.components.formatter import encode_query_with_filepaths


@dataclass
class TransformArgs():
    """
A document transformation parameter container for centralized management of processing configurations.

Args:
    f (Union[str, Callable]): Transformation function or registered function name.Can be either a callable function or a string identifier for registered functions.
    trans_node (bool): Whether to transform node types.When True, modifies the document node structure during processing.
    num_workers (int):Controls parallel processing threads.Values >0.
    kwargs (Dict):Additional parameters passed to the transformation function.
    pattern (Union[str, Callable[[str], bool]]):File name/content matching pattern.


Examples:
    
    >>> from lazyllm.tools import TransformArgs
    >>> args = TransformArgs(f=lambda text: text.lower(),num_workers=4,pattern=r'.*\.md$')
    >>>config = {'f': 'parse_pdf','kwargs': {'engine': 'pdfminer'},'trans_node': True}
    >>>args = TransformArgs.from_dict(config)
    print(args['f'])
    print(args.get('unknown'))
    """
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


@dataclass
class _Split:
    """_Split(text: str, is_sentence: bool, token_size: int)"""
    text: str
    is_sentence: bool
    token_size: int


def split_text_keep_separator(text: str, separator: str) -> List[str]:
    """Split text and keep the separator."""
    parts = text.split(separator)
    result = [separator + s if i > 0 else s for i, s in enumerate(parts)]
    return result[1:] if len(result) > 0 and not result[0] else result


class NodeTransform(ABC):
    """
Processes document nodes in batch, supporting both single-threaded and multi-threaded modes.

Args:
    num_workers(int): Controls whether multi-threading is enabled (enabled when >0).


Examples:
    
    >>> import lazyllm
    >>> from lazyllm.tools import NodeTransform
    >>> node_tran = NodeTransform(num_workers=num_workers)
    >>> doc = lazyllm.Document(dataset_path="/path/to/your/data", embed=m, manager=False)
    >>> nodes = node_tran.batch_forward(doc, "word_split")
    """
    def __init__(self, num_workers: int = 0):
        self._number_workers = num_workers
        self._name = None

    def batch_forward(
        self, documents: Union[DocNode, List[DocNode]], node_group: str, **kwargs
    ) -> List[DocNode]:
        """
Process documents in batch with node group transformation.

Args:
    documents (Union[DocNode, List[DocNode]]): Input node(s) to process.
    node_group (str): Target transformation group name.
    **kwargs: Additional transformation parameters.
"""
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
        """
[Abstract] Core transformation logic to implement.

Args:
    document (DocNode): Input document node.
    **kwargs: Implementation-specific parameters.
"""
        raise NotImplementedError('Not implemented')

    def with_name(self, name: Optional[str], *, copy: bool = True) -> 'NodeTransform':
        """
Set transformer name with optional copying.

Args:
    name (Optional[str]): New name for the transformer.
    copy (bool): Whether to return a copy. Default True.
"""
        if name is not None:
            if copy: return lite_copy(self).with_name(name, copy=False)
            self._name = name
        return self

    def __call__(self, node: DocNode, **kwargs: Any) -> List[DocNode]:
        # Parent and child should not be set here.
        results = self.transform(node, **kwargs)
        if isinstance(results, (DocNode, str)): results = [results]
        return [DocNode(text=chunk) if isinstance(chunk, str) else chunk for chunk in results if chunk]


def make_transform(t: Union[TransformArgs, Dict[str, Any]], group_name: Optional[str] = None) -> NodeTransform:
    if isinstance(t, dict): t = TransformArgs.from_dict(t)
    transform, trans_node, num_workers = t['f'], t['trans_node'], t['num_workers']
    num_workers = dict(num_workers=num_workers) if num_workers > 0 else dict()
    return (transform(**t['kwargs'], **num_workers).with_name(group_name, copy=False) if isinstance(transform, type)
            else transform.with_name(group_name) if isinstance(transform, NodeTransform)
            else FuncNodeTransform(transform, trans_node=trans_node, **num_workers).with_name(group_name, copy=False))


class AdaptiveTransform(NodeTransform):
    """A flexible document transformation system that applies different transforms based on document patterns.

AdaptiveTransform allows you to define multiple transformation strategies and automatically selects the appropriate one based on the document's file path or custom pattern matching. This is particularly useful when you have different types of documents that require different processing approaches.

Args:
    transforms (Union[List[Union[TransformArgs, Dict]], Union[TransformArgs, Dict]]): A list of transform configurations or a single transform configuration. 
    num_workers (int, optional): Number of worker threads for parallel processing. Defaults to 0.


Examples:
    >>> from lazyllm.tools.rag.transform import AdaptiveTransform, DocNode, SentenceSplitter
    >>> doc1 = DocNode(text="这是第一个文档的内容。它包含多个句子。")
    >>> doc2 = DocNode(text="这是第二个文档的内容。")
    >>> transforms = [
    ...     {
    ...         'f': SentenceSplitter,
    ...         'pattern': '*.txt',
    ...         'kwargs': {'chunk_size': 50, 'chunk_overlap': 10}
    ...     },
    ...     {
    ...         'f': SentenceSplitter,
    ...         'pattern': '*.pdf',
    ...         'kwargs': {'chunk_size': 100, 'chunk_overlap': 20}
    ...     }
    ... ]
    >>> adaptive = AdaptiveTransform(transforms)
    >>> results1 = adaptive.transform(doc1)
    >>> print(f"文档1转换结果: {len(results1)} 个块")
    >>> for i, result in enumerate(results1):
    ...     print(f"  块 {i+1}: {result.text}")
    >>> results2 = adaptive.transform(doc2)
    >>> print(f"文档2转换结果: {len(results2)} 个块")
    >>> for i, result in enumerate(results2):
    ...     print(f"  块 {i+1}: {result.text}")      
    """
    def __init__(self, transforms: Union[List[Union[TransformArgs, Dict]], Union[TransformArgs, Dict]],
                 num_workers: int = 0):
        super().__init__(num_workers=num_workers)
        if not isinstance(transforms, (tuple, list)): transforms = [transforms]
        self._transformers = [(t.get('pattern'), make_transform(t)) for t in transforms]

    def transform(self, document: DocNode, **kwargs) -> List[Union[str, DocNode]]:
        """Transform a document using the appropriate transformation strategy based on pattern matching.

This method evaluates each transform configuration in order and applies the first one that matches the document's path pattern. The matching logic supports both glob patterns and custom callable functions.

Args:
    document (DocNode): The document node to be transformed.
    **kwargs: Additional keyword arguments passed to the transform function.

**Returns:**

- List[Union[str, DocNode]]: A list of transformed results (strings or DocNode objects).
"""
        if not isinstance(document, DocNode): LOG.warning(f'Invalud document type {type(document)} got')
        for pt, transform in self._transformers:
            if pt and isinstance(pt, str) and not pt.startswith('*'): pt = os.path.join(str(os.cwd()), pt)
            if not pt or (callable(pt) and pt(document.docpath)) or (
                    isinstance(pt, str) and fnmatch.fnmatch(document.docpath, pt)):
                return transform(document, **kwargs)
        LOG.warning(f'No transform found for document {document.docpath} with group name `{self._name}`')
        return []


class SentenceSplitter(NodeTransform):
    """
Split sentences into chunks of a specified size. You can specify the size of the overlap between adjacent chunks.

Args:
    chunk_size (int): The size of the chunk after splitting.
    chunk_overlap (int): The length of the overlapping content between two adjacent chunks.
    num_workers (int): Controls the number of threads or processes used for parallel processing.


Examples:
    
    >>> import lazyllm
    >>> from lazyllm.tools import Document, SentenceSplitter
    >>> m = lazyllm.OnlineEmbeddingModule(source="glm")
    >>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)
    >>> documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
    """
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200, num_workers: int = 0):
        super(__class__, self).__init__(num_workers=num_workers)
        if chunk_overlap > chunk_size:
            raise ValueError(
                f'Got a larger chunk overlap ({chunk_overlap}) than chunk size '
                f'({chunk_size}), should be smaller.'
            )

        assert (
            chunk_size > 0 and chunk_overlap >= 0
        ), 'chunk size should > 0 and chunk_overlap should >= 0'

        try:
            if 'TIKTOKEN_CACHE_DIR' not in os.environ and 'DATA_GYM_CACHE_DIR' not in os.environ:
                path = os.path.join(config['model_path'], 'tiktoken')
                os.makedirs(path, exist_ok=True)
                os.environ['TIKTOKEN_CACHE_DIR'] = path
            self._tiktoken_tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
            os.environ.pop('TIKTOKEN_CACHE_DIR')
        except requests.exceptions.ConnectionError:
            LOG.error(
                'Unable to download the vocabulary file for tiktoken `gpt-3.5-turbo`. '
                'Please check your internet connection. '
                'Alternatively, you can manually download the file '
                'and set the `TIKTOKEN_CACHE_DIR` environment variable.'
            )
            raise
        except Exception as e:
            LOG.error(f'Unable to build tiktoken tokenizer with error `{e}`')
            raise
        self._punkt_st_tokenizer = nltk.tokenize.PunktSentenceTokenizer()

        self._sentence_split_fns = [
            partial(split_text_keep_separator, separator='\n\n\n'),  # paragraph
            self._punkt_st_tokenizer.tokenize,
        ]

        self._sub_sentence_split_fns = [
            lambda t: re.findall(r'[^,.;。？！]+[,.;。？！]?', t),
            partial(split_text_keep_separator, separator=' '),
            list,  # split by character
        ]

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def transform(self, node: DocNode, **kwargs) -> List[str]:
        return self.split_text(
            node.get_text(),
            metadata_size=self._get_metadata_size(node),
        )

    def _get_metadata_size(self, node: DocNode) -> int:
        # Return the bigger size to ensure chunk_size < limit
        return max(
            self._token_size(node.get_metadata_str(mode=MetadataMode.EMBED)),
            self._token_size(node.get_metadata_str(mode=MetadataMode.LLM)),
        )

    def split_text(self, text: str, metadata_size: int) -> List[str]:
        """Split the input text into multiple chunks based on sentence boundaries and chunk size.

Args:
    text (str): The text to be split.
    metadata_size (int): Length occupied by additional metadata, used to adjust effective chunk size.

**Returns:**

- List[str]: List of resulting text chunks.
"""
        if text == '':
            return ['']
        effective_chunk_size = self.chunk_size - metadata_size
        if effective_chunk_size <= 0:
            raise ValueError(
                f'Metadata length ({metadata_size}) is longer than chunk size '
                f'({self.chunk_size}). Consider increasing the chunk size or '
                'decreasing the size of your metadata to avoid this.'
            )
        elif effective_chunk_size < 50:
            LOG.warning(
                f'Metadata length ({metadata_size}) is close to chunk size '
                f'({self.chunk_size}). Resulting chunks are less than 50 tokens. '
                'Consider increasing the chunk size or decreasing the size of '
                'your metadata to avoid this.'
            )

        splits = self._split(text, effective_chunk_size)
        chunks = self._merge(splits, effective_chunk_size)
        return chunks

    def _split(self, text: str, chunk_size: int) -> List[_Split]:
        """Break text into splits that are smaller than chunk size.

        The order of splitting is:
        1. split by paragraph separator
        2. split by chunking tokenizer
        3. split by second chunking regex
        4. split by default separator (' ')
        5. split by character
        """
        token_size = self._token_size(text)
        if token_size <= chunk_size:
            return [_Split(text, is_sentence=True, token_size=token_size)]

        text_splits_by_fns, is_sentence = self._get_splits_by_fns(text)

        text_splits = []
        for text in text_splits_by_fns:
            token_size = self._token_size(text)
            if token_size <= chunk_size:
                text_splits.append(
                    _Split(
                        text,
                        is_sentence=is_sentence,
                        token_size=token_size,
                    )
                )
            else:
                recursive_text_splits = self._split(text, chunk_size=chunk_size)
                text_splits.extend(recursive_text_splits)
        return text_splits

    def _merge(self, splits: List[_Split], chunk_size: int) -> List[str]:
        chunks: List[str] = []
        cur_chunk: List[Tuple[str, int]] = []  # list of (text, length)
        cur_chunk_len = 0
        is_chunk_new = True

        def close_chunk() -> None:
            nonlocal cur_chunk, cur_chunk_len, is_chunk_new

            chunks.append(''.join([text for text, _ in cur_chunk]))
            last_chunk = cur_chunk
            cur_chunk = []
            cur_chunk_len = 0
            is_chunk_new = True

            # Add overlap to the next chunk using the last one first
            overlap_len = 0
            for text, length in reversed(last_chunk):
                if overlap_len + length > self.chunk_overlap:
                    break
                cur_chunk.append((text, length))
                overlap_len += length
                cur_chunk_len += length
            cur_chunk.reverse()

        i = 0
        while i < len(splits):
            cur_split = splits[i]
            if cur_split.token_size > chunk_size:
                raise ValueError('Single token exceeded chunk size')
            if cur_chunk_len + cur_split.token_size > chunk_size and not is_chunk_new:
                # if adding split to current chunk exceeds chunk size
                close_chunk()
            else:
                if (
                    cur_split.is_sentence
                    or cur_chunk_len + cur_split.token_size <= chunk_size
                    or is_chunk_new  # new chunk, always add at least one split
                ):
                    # add split to chunk
                    cur_chunk_len += cur_split.token_size
                    cur_chunk.append((cur_split.text, cur_split.token_size))
                    i += 1
                    is_chunk_new = False
                else:
                    close_chunk()

        # handle the last chunk
        if not is_chunk_new:
            chunks.append(''.join([text for text, _ in cur_chunk]))

        # Remove whitespace only chunks and remove leading and trailing whitespace.
        return [stripped_chunk for chunk in chunks if (stripped_chunk := chunk.strip())]

    def _token_size(self, text: str) -> int:
        return len(self._tiktoken_tokenizer.encode(text, allowed_special='all'))

    def _get_splits_by_fns(self, text: str) -> Tuple[List[str], bool]:
        for split_fn in self._sentence_split_fns:
            splits = split_fn(text)
            if len(splits) > 1:
                return splits, True

        for split_fn in self._sub_sentence_split_fns:
            splits = split_fn(text)
            if len(splits) > 1:
                break

        return splits, False


class FuncNodeTransform(NodeTransform):
    """
A wrapper class for user-defined functions that transforms document nodes.

This wrapper supports two modes of operation:
    1. When trans_node is False (default): transforms text strings
    2. When trans_node is True: transforms DocNode objects

The wrapper can handle various function signatures:
    - str -> List[str]: transform=lambda t: t.split('\\n')
    - str -> str: transform=lambda t: t[:3]
    - DocNode -> List[DocNode]: pipeline(lambda x:x, SentenceSplitter)
    - DocNode -> DocNode: pipeline(LLMParser)

Args:
    func (Union[Callable[[str], List[str]], Callable[[DocNode], List[DocNode]]]): The user-defined function to be wrapped.
    trans_node (bool, optional): Determines whether the function operates on DocNode objects (True) or text strings (False). Defaults to None.
    num_workers (int): Controls the number of threads or processes used for parallel processing. Defaults to 0.


Examples:
    
    >>> import lazyllm
    >>> from lazyllm.tools.rag import FuncNodeTransform
    >>> from lazyllm.tools import Document, SentenceSplitter
    
    # Example 1: Text-based transformation (trans_node=False)
    >>> def split_by_comma(text):
    ...     return text.split(',')
    >>> text_transform = FuncNodeTransform(split_by_comma, trans_node=False)
    
    # Example 2: Node-based transformation (trans_node=True)
    >>> def custom_node_transform(node):
    ...     # Process the DocNode and return a list of DocNodes
    ...     return [node]  # Simple pass-through
    >>> node_transform = FuncNodeTransform(custom_node_transform, trans_node=True)
    
    # Example 3: Using with Document
    >>> m = lazyllm.OnlineEmbeddingModule(source="glm")
    >>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)
    >>> documents.create_node_group(name="custom", transform=text_transform)
    """

    def __init__(self, func: Union[Callable[[str], List[str]], Callable[[DocNode], List[DocNode]]],
                 trans_node: bool = None, num_workers: int = 0):
        super(__class__, self).__init__(num_workers=num_workers)
        self._func, self._trans_node = func, trans_node

    def transform(self, node: DocNode, **kwargs) -> List[Union[str, DocNode]]:
        """
Transform a document node using the wrapped user-defined function.

This method applies the user-defined function to either the text content of the node (when trans_node=False) or the node itself (when trans_node=True).

Args:
    node (DocNode): The document node to be transformed.
    **kwargs: Additional keyword arguments passed to the transformation function.

**Returns:**

- List[Union[str, DocNode]]: The transformed results, which can be either strings or DocNode objects depending on the function implementation.
"""
        return self._func(node if self._trans_node else node.get_text())


templates = dict(
    en=dict(summary='''
## Role: Text Summarizer
You are a text summarization engine responsible for analyzing user input text and providing a concise summary based on \
the requested task.

## Constraints:
- Respond only with the requested output: a brief summary.
- Do not add any extra fields, explanations, or translations.

## Text Format:
The input is a string contains the user's raw input text

## Example:
#input:
Hello, I am an AI robot developed by SenseTime, named LazyLLM.
My mission is to assist you in building the most powerful large-scale model applications with minimal cost.
#output:
Introduction of AI robot LazyLLM

You should not have any unnecessary output. Lets begin:
''', keywords='''
## Role: Keyword Extractor
You are a text keyword extraction engine responsible for analyzing user input text and providing a extracting relevant \
keywords based on the requested task.

## Constraints:
- Respond only with a list of keywords.
- Do not add any extra fields, explanations, or translations.

## Text Format:
The input is a string contains the user's raw input text

## Example:
#input:
'Hello, I am an AI robot developed by SenseTime, named LazyLLM.
My mission is to assist you in building the most powerful large-scale model applications with minimal cost.'
#output:
LazyLLM, SenseTime, AI robot, large-scale model applications

You should not have any unnecessary output. Lets begin:
''', qa='''
## Role: QA-pair Extractor
You are a question-answer extraction engine responsible for analyzing user input text and providing a extracting \
query and answer based on the requested task.

## Constraints:
- Respond only with a list of question and answer pairs.
- Do not add any extra fields, explanations, or translations.

## Text Format:
The input is a string contains the user's raw input text

## Example:
#input:
'Hello, I am an AI robot developed by SenseTime, named LazyLLM.
My mission is to assist you in building the most powerful large-scale model applications with minimal cost.'
#output:
Q: What is LazyLLM developed by?
A: LazyLLM is developed by SenseTime.
Q: What can LazyLLM do?
A: LazyLLM can assist you in building the most powerful large-scale model applications with minimal cost.

You should not have any unnecessary output. Lets begin:
''', qa_img='''
## Role: Q&A Pair Extraction Engine
You are a Q&A pair extraction engine, responsible for analyzing and extracting Q&A pairs from images.

## Constraints:
- Only reply with the requested output content: extracted Q&A pairs.
- Do not add extra fields, explanations, or translations.

## Example:
Input an image of a pig.
#output:
Q: What color is the pig?
A: The pig is pink.
Q: What is the pig doing?
A: The pig is running on the lawn.

You should not output any extra characters. Let's start now.
'''),
    zh=dict(summary='''
## 角色：文本摘要
你是一个文本摘要引擎，负责分析用户输入的文本，并根据请求任务提供简洁的摘要。

## 约束条件:
- 仅回复请求的输出内容：提供简短摘要。
- 不要添加额外字段、解释或翻译。

## 文本格式:
输入文本为string格式，包含用户的原始输入文本

## 示例:
#input:
你好，我是由商汤开发的人工智能机器人，我叫LazyLLM。我的使命是协助您，用最低的成本，构建最强大的大模型应用。
#output:
人工智能机器人LazyLLM的简介

你不应输出任何多余的字符，现在我们开始吧
''', keywords='''
## 角色：关键词提取引擎
你是一个关键词提取引擎，负责分析用户输入的文本，提取其中的关键词。

## 约束条件:
- 仅回复请求的输出内容：抽取关键词。
- 不要添加额外字段、解释或翻译。

## 文本格式:
输入文本为string格式，包含用户的原始输入文本

## 示例:
#input:
你好，我是由商汤开发的人工智能机器人，我叫LazyLLM。我的使命是协助您，用最低的成本，构建最强大的大模型应用。
#output:
LazyLLM, 商汤, 人工智能机器人, 大模型应用

你不应输出任何多余的字符，现在我们开始吧
''', qa='''
## 角色：问答对提取引擎
你是一个问答对提取引擎，负责分析用户输入的文本，提取其中的问答对。

## 约束条件:
- 仅回复请求的输出内容：抽取问答对。
- 不要添加额外字段、解释或翻译。

## 文本格式:
输入文本为string格式，包含用户的原始输入文本

## 示例:
#input:
你好，我是由商汤开发的人工智能机器人，我叫LazyLLM。我的使命是协助您，用最低的成本，构建最强大的大模型应用。
#output:
Q: LazyLLM是由谁开发的？
A: LazyLLM是由商汤科技开发的。
Q: LazyLLM能做什么？
A: LazyLLM可以协助用户，用最低的成本，构建最强大的大模型应用

你不应输出任何多余的字符，现在我们开始吧
''', qa_img='''
## 角色：问答对提取引擎
你是一个问答对提取引擎，负责分析从图像中提取其中的问答对。

## 约束条件:
- 仅回复请求的输出内容：抽取问答对。
- 不要添加额外字段、解释或翻译。

## 示例:
输入一张小猪的图片。
#output:
Q: 小猪是什么颜色的？
A: 小猪是粉红色的。
Q: 小猪在做啥呢？
A: 小猪在草坪上奔跑。

你不应输出任何多余的字符，现在我们开始吧
'''))

class LLMParser(NodeTransform):
    """
A text summarizer and keyword extractor that is responsible for analyzing the text input by the user and providing concise summaries or extracting relevant keywords based on the requested task.

Args:
    llm (TrainableModule): A trainable module.
    language (str): The language type, currently only supports Chinese (zh) and English (en).
    task_type (str): Currently supports two types of tasks: summary and keyword extraction.
    num_workers (int): Controls the number of threads or processes used for parallel processing.


Examples:
    
    >>> from lazyllm import TrainableModule
    >>> from lazyllm.tools.rag import LLMParser
    >>> llm = TrainableModule("internlm2-chat-7b")
    >>> summary_parser = LLMParser(llm, language="en", task_type="summary")
    """
    def __init__(self, llm: TrainableModule, language: str, task_type: str, num_workers: int = 30):
        super(__class__, self).__init__(num_workers=num_workers)
        assert language in ['en', 'zh'], f'Not supported language {language}'
        assert task_type in ['summary', 'keywords', 'qa', 'qa_img'], f'Not supported task_type {task_type}'
        self._task_type = task_type
        if self._task_type == 'qa_img':
            prompt = dict(system=templates[language][task_type], user='{input}')
        else:
            prompt = dict(system=templates[language][task_type], user='#input:\n{input}\n#output:\n')
        self._llm = llm.share(prompt=AlpacaPrompter(prompt), stream=False, format=self._format)
        self._task_type = task_type

    def transform(self, node: DocNode, **kwargs) -> List[str]:
        """
Perform the set task on the specified document.

Args:
    node (DocNode): The document on which the extraction task needs to be performed.


Examples:
    
    >>> import lazyllm
    >>> from lazyllm.tools import LLMParser
    >>> llm = lazyllm.TrainableModule("internlm2-chat-7b").start()
    >>> m = lazyllm.TrainableModule("bge-large-zh-v1.5").start()
    >>> summary_parser = LLMParser(llm, language="en", task_type="summary")
    >>> keywords_parser = LLMParser(llm, language="en", task_type="keywords")
    >>> documents = lazyllm.Document(dataset_path="/path/to/your/data", embed=m, manager=False)
    >>> rm = lazyllm.Retriever(documents, group_name='CoarseChunk', similarity='bm25', topk=6)
    >>> doc_nodes = rm("test")
    >>> summary_result = summary_parser.transform(doc_nodes[0])
    >>> keywords_result = keywords_parser.transform(doc_nodes[0])
    """
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
