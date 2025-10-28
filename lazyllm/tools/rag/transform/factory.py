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
