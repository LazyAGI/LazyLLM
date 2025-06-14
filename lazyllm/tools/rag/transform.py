from abc import ABC, abstractmethod
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
    f: Union[str, Callable]
    trans_node: Optional[bool] = None
    num_workers: int = 0
    kwargs: Dict = field(default_factory=dict)
    pattern: Optional[str] = None

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
    text: str
    is_sentence: bool
    token_size: int


def split_text_keep_separator(text: str, separator: str) -> List[str]:
    """Split text and keep the separator."""
    parts = text.split(separator)
    result = [separator + s if i > 0 else s for i, s in enumerate(parts)]
    return result[1:] if len(result) > 0 and not result[0] else result


class NodeTransform(ABC):
    def __init__(self, num_workers: int = 0):
        self._number_workers = num_workers

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

    def __call__(self, node: DocNode, **kwargs: Any) -> List[DocNode]:
        # Parent and child should not be set here.
        results = self.transform(node, **kwargs)
        if isinstance(results, (DocNode, str)): results = [results]
        return [DocNode(text=chunk) if isinstance(chunk, str) else chunk for chunk in results if chunk]


def make_transform(t):
    if isinstance(t, dict): t = TransformArgs.from_dict(t)
    transform, trans_node, num_workers = t['f'], t['trans_node'], t['num_workers']
    num_workers = dict(num_workers=num_workers) if num_workers > 0 else dict()
    return (transform(**t['kwargs'], **num_workers)
            if isinstance(transform, type)
            else transform if isinstance(transform, NodeTransform)
            else FuncNodeTransform(transform, trans_node=trans_node, **num_workers))


class AdaptiveTransform(NodeTransform):
    def __init__(self, transforms: Union[List[TransformArgs], TransformArgs]):
        super().__init__(num_workers=0)
        if not isinstance(transforms, (tuple, list)): transforms = [transforms]
        self._transformers = [(t.get('pattern'), make_transform(t)) for t in transforms]

    def transform(self, document: DocNode, **kwargs) -> List[Union[str, DocNode]]:
        for pt, transform in self._transformers:
            if pt and not pt.startswith('*'): pt = os.path.join(str(os.cwd()), pt)
            if not isinstance(document, DocNode):
                LOG.warning(f'Invalud document type {type(document)} got')
            if not pt or fnmatch.fnmatch(document.docpath, pt):
                return transform(document, **kwargs)
        return []


class SentenceSplitter(NodeTransform):
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
                'your metadata to avoid this.',
                flush=True,
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
    """Used for user defined function.

    Wrapped the transform to: List[Docnode] -> List[Docnode]

    This wrapper supports when trans_node is False:
        1. str -> list: transform=lambda t: t.split('\n')
        2. str -> str: transform=lambda t: t[:3]

    This wrapper supports when trans_node is True:
        1. DocNode -> list: pipeline(lambda x:x, SentenceSplitter)
        2. DocNode -> DocNode: pipeline(LLMParser)
    """

    def __init__(self, func: Union[Callable[[str], List[str]], Callable[[DocNode], List[DocNode]]],
                 trans_node: bool = None, num_workers: int = 0):
        super(__class__, self).__init__(num_workers=num_workers)
        self._func, self._trans_node = func, trans_node

    def transform(self, node: DocNode, **kwargs) -> List[Union[str, DocNode]]:
        return self._func(node if self._trans_node else node.get_text())


templates = dict(
    en=dict(summary="""
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
""", keywords="""
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
"Hello, I am an AI robot developed by SenseTime, named LazyLLM.
My mission is to assist you in building the most powerful large-scale model applications with minimal cost."
#output:
LazyLLM, SenseTime, AI robot, large-scale model applications

You should not have any unnecessary output. Lets begin:
""", qa="""
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
"Hello, I am an AI robot developed by SenseTime, named LazyLLM.
My mission is to assist you in building the most powerful large-scale model applications with minimal cost."
#output:
Q: What is LazyLLM developed by?
A: LazyLLM is developed by SenseTime.
Q: What can LazyLLM do?
A: LazyLLM can assist you in building the most powerful large-scale model applications with minimal cost.

You should not have any unnecessary output. Lets begin:
""", qa_img="""
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
"""),
    zh=dict(summary="""
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
""", keywords="""
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
""", qa="""
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
""", qa_img="""
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
"""))

class LLMParser(NodeTransform):
    def __init__(self, llm: TrainableModule, language: str, task_type: str, num_workers: int = 0):
        super(__class__, self).__init__(num_workers=num_workers)
        assert language in ['en', 'zh'], f'Not supported language {language}'
        assert task_type in ['summary', 'keywords', 'qa', 'qa_img'], f'Not supported task_type {task_type}'
        self._task_type = task_type
        if self._task_type == 'qa_img':
            prompt = dict(system=templates[language][task_type], user='{input}')
        else:
            prompt = dict(system=templates[language][task_type], user='#input:\n{input}\n#output:\n')
        self._llm = llm.share(prompt=AlpacaPrompter(prompt)).formatter(self._format)
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
                list(filter(None, map(str.strip, input.split("\n"))))[::2],
                list(filter(None, map(str.strip, input.split("\n"))))[1::2])]
        return input
