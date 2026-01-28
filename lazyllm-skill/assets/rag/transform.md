# Transform的作用

对reader解析出来的文档内容进行更精细的操作，进而使得在检索时获得更好的效果。

## 内置的transform类

- SentenceSplitter:将句子拆分成指定大小的块。可以指定相邻块之间重合部分的大小。

参数:

    chunk_size (int, default: _UNSET ) – 拆分之后的块大小
    chunk_overlap (int, default: _UNSET ) – 相邻两个块之间重合的内容长度
    num_workers (int, default: _UNSET ) – 控制并行处理的线程/进程数量
    **kwargs – 传递给拆分器的额外参数。

- CharacterSplitter:将文本按字符拆分。

参数:

    chunk_size (int, default: _UNSET ) – 拆分之后的块大小
    overlap (int, default: _UNSET ) – 相邻两个块之间重合的内容长度
    num_workers (int, default: _UNSET ) – 控制并行处理的线程/进程数量。
    separator (str, default: _UNSET ) – 用于拆分的分隔符。默认为' '。
    is_separator_regex (bool, default: _UNSET ) – 是否使用正则表达式作为分隔符。默认为False。
    keep_separator (bool, default: _UNSET ) – 是否保留分隔符在拆分后的文本中。默认为False。
    **kwargs – 传递给拆分器的额外参数。

- RecursiveSplitter:递归拆分文本。

参数:

    chunk_size (int, default: _UNSET ) – 拆分之后的块大小
    overlap (int, default: _UNSET ) – 相邻两个块之间重合的内容长度
    num_workers (int, default: _UNSET ) – 控制并行处理的线程/进程数量。
    keep_separator (bool, default: _UNSET ) – 是否保留分隔符在拆分后的文本中。默认为False。
    is_separator_regex (bool, default: _UNSET ) – 是否使用正则表达式作为分隔符。默认为False。
    separators (List[str], default: _UNSET ) – 用于拆分的分隔符列表。默认为['\n', '.', ' ', '']。如果你想按多个分隔符拆分，可以设置这个参数。

- MarkdownSplitter:递归拆分markdown文本。

参数:

    chunk_size (int, default: _UNSET ) – 拆分之后的块大小
    overlap (int, default: _UNSET ) – 相邻两个块之间重合的内容长度
    num_workers (int, default: _UNSET ) – 控制并行处理的线程/进程数量。
    keep_trace (bool, default: _UNSET ) – 是否保留markdown文本的追踪。默认为False。
    keep_headers (bool, default: _UNSET ) – 是否保留headers在拆分后的文本中。默认为False。
    keep_lists (bool, default: _UNSET ) – 是否保留lists在拆分后的文本中。默认为False。
    keep_code_blocks (bool, default: _UNSET ) – 是否保留code blocks在拆分后的文本中。默认为False。
    keep_tables (bool, default: _UNSET ) – 是否保留tables在拆分后的文本中。默认为False。
    keep_images (bool, default: _UNSET ) – 是否保留images在拆分后的文本中。默认为False。
    keep_links (bool, default: _UNSET ) – 是否保留links在拆分后的文本中。默认为False。
    **kwargs – 传递给拆分器的额外参数。

- HTMLSplitter:一个HTML拆分器，负责拆分HTML文本的语义。

参数:
    chunk_size (int, default: _UNSET ) – 拆分之后的块大小
    overlap (int, default: _UNSET ) – 相邻两个块之间重合的内容长度
    num_workers (int, default: _UNSET ) – 控制并行处理的线程/进程数量。
    keep_sections (bool, default: _UNSET ) – 是否保留sections在拆分后的文本中。默认为False。
    keep_tags (bool, default: _UNSET ) – 是否保留tags在拆分后的文本中。默认为False。
    **kwargs – 传递给拆分器的额外参数。

- JSONSplitter:一个JSON拆分器，负责拆分JSON文本的语义。

参数:

    chunk_size (int, default: _UNSET ) – 拆分之后的块大小
    overlap (int, default: _UNSET ) – 相邻两个块之间重合的内容长度
    num_workers (int, default: _UNSET ) – 控制并行处理的线程/进程数量。
    compact_output (bool, default: _UNSET ) – 是否压缩输出。默认为True。
    **kwargs – 传递给拆分器的额外参数。

- JSONLSplitter:一个JSONL拆分器，负责拆分JSONL文本的语义。

参数:
    chunk_size (int, default: _UNSET ) – 拆分之后的块大小
    overlap (int, default: _UNSET ) – 相邻两个块之间重合的内容长度
    num_workers (int, default: _UNSET ) – 控制并行处理的线程/进程数量。
    compact_output (bool, default: _UNSET ) – 是否压缩输出。默认为True。

- YAMLSplitter:一个YAML拆分器，负责拆分YAML文本的语义。

参数:

    chunk_size (int, default: _UNSET ) – 拆分之后的块大小
    overlap (int, default: _UNSET ) – 相邻两个块之间重合的内容长度
    num_workers (int, default: _UNSET ) – 控制并行处理的线程/进程数量。
    compact_output (bool, default: _UNSET ) – 是否压缩输出。默认为True。

- XMLSplitter:一个XML拆分器，负责拆分XML文本的语义。

参数:

    chunk_size (int, default: _UNSET ) – 拆分之后的块大小
    overlap (int, default: _UNSET ) – 相邻两个块之间重合的内容长度
    num_workers (int, default: _UNSET ) – 控制并行处理的线程/进程数量。
    keep_trace (bool, default: _UNSET ) – 是否保留拆分文本中的trace。
    keep_tags (bool, default: _UNSET ) – 是否保留拆分文本中的tags。

- CodeSplitter:一个代码拆分器，负责根据文件类型进行路由选择不同的拆分器。(路由类)

参数:

    chunk_size (int, default: _UNSET ) – 拆分之后的块大小
    overlap (int, default: _UNSET ) – 相邻两个块之间重合的内容长度
    num_workers (int, default: _UNSET ) – 控制并行处理的线程/进程数量。
    filetype (Optional[str], default: _UNSET ) – 要拆分的文件类型。
    **kwargs – 传递给拆分器的额外参数。

- GeneralCodeSplitter:一个通用代码拆分器，负责拆分代码文本的语义。

参数:

    chunk_size (int, default: _UNSET ) – 拆分之后的块大小
    overlap (int, default: _UNSET ) – 相邻两个块之间重合的内容长度
    num_workers (int, default: _UNSET ) – 控制并行处理的线程/进程数量。
    filetype (str, default: 'code' ) – 要拆分的文件类型。

## 基础使用方法

以CharacterSplitter举例，其余切分方法根据参数不同而调整。

将文档根据指定符号拆分成名为 character 的 Node Group:

```python
document.create_node_group(name='character', transform=CharacterSplitter)
```

## 给transform设置全局默认参数

同样以CharacterSplitter举例，其余方法按照参数不同而调整。

```python
from lazyllm.tools.rag import CharacterSplitter

#通过set_default覆盖原先CharacterSplitter的默认值，这样无论在哪里调用都会是我们设置的默认值
CharacterSplitter.set_default(
    chunk_size = 2048,
    overlap = 200,
    separator = '.',
    is_separator_regex = False,
    keep_separator = True,
)

#还可以通过reset_default将我们原先的设置清空，恢复CharacterSplitter原先的默认设置
CharacterSplitter.reset_default()
```

## 设置切分流程

对于CharacterSplitter和RecursiveSplitter可以通过set_split_fns(), add_split_fns()和clear_split_fns()三个方法来管理自定义的切分函数。

```python
def custom_paragraph_split(text, separator):
    chunks = text.split(separator)
    result = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) > 10:
            result.append(chunk)

    return result

def filter_empty_split(text, separator):
    chunks = text.split(separator)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

charactersplitter = CharacterSplitter()
#自定义切分函数或流程（传输一个List[Callable]）
charactersplitter.set_split_fns(custom_paragraph_split)
document.create_node_group(name='character1',
                           transform=charactersplitter,
                           separator='.')

#在自定义的切分流程内的指定位置添加切分函数
charactersplitter.add_split_fn(filter_empty_split, 0)
document.create_node_group(name='character2',
                           transform=charactersplitter,
                           separator='.')

#清空自定义切分函数，使用默认切分流程
charactersplitter.clear_split_fns()
document.create_node_group(name='character3',
                           transform=charactersplitter,
                           separator='.')
```

## 自定义tokenizer

LazyLLM也提供了from_huggingface_tokenizer()方法来进行设置tokenizer

```python
from lazyllm.tools.rag import CharacterSplitter

tokenizer = AutoTokenizer.from_pretrained('gpt2')
charactersplitter = CharacterSplitter()
charactersplitter = charactersplitter.from_huggingface_tokenizer(tokenzier)

document.create_node_group(name='character',
                           transform=charactersplitter,
                           separator='.')
```

## CodeSplitter

CodeSplitter是一个路由类，根据入参选择合适的切分类进行切分。以XML文件举例：

```python
from lazyllm.tools.rag import CodeSplitter

#此时实际使用的是XMLSplitter切分类
document.create_node_group(name='xmlsplitter',
                           transform=CodeSplitter,
                           filetype='xml')

#当然也可以使用CodeSplitter的from_language()方法来指定
splitter = CodeSplitter()
splitter.from_language('xml')
document.create_node_group(name='xmlsplitter',
                           transform=splitter)
```

## 自定义transform

已知 create_node_group 接收的 transform 是 Callable 一个对象，推荐的形式有：

- 函数
- 定义了 __call__ 方法的类
- 匿名函数（lambda函数）

```python
from typing import List, Union
from lazyllm import Document, Retriever
from lazyllm.tools.rag.doc_node import DocNode

docs = Document("/path/to/your/documents")

# 第一种：函数实现直接对字符串进行分块规则
def split_by_sentence1(node: str, **kwargs) -> List[str]:
    """函数接收字符串，返回字符串列表，输入为trans_node=False时调用"""
    return node.split('。')
docs.create_node_group(name='block1', transform=split_by_sentence1)

# 第二种：函数实现获取DocNode对应文本内容后进行分块，并构造DocNode
# 适用于返回非朴素DocNode，例如LazyLLM提供了ImageDocNode等特殊DocNode
def split_by_sentence2(node: DocNode, **kwargs) -> List[DocNode]:
    """函数接收DocNode，返回DocNode列表，输入为trans_node=False时调用"""
    content = node.get_text()
    nodes = []
    for text in content.split('。'):
        nodes.append(DocNode(text=text))
    return nodes
docs.create_node_group(name='block2', transform=split_by_sentence2, trans_node=True)

# 第三种：实现了 __call__ 函数的类
# 优点是一个类用于多种分块，例如这个例子可以通过控制实例化时的参数实现基于多种符号的分块
class SymbolSplitter:
    """实例化后传入Transform，默认情况下接收字符串，trans_node为true时接收DocNode"""
    def __init__(self, splitter="。", trans_node=False):
        self._splitter = splitter
        self._trans_node = trans_node

    def __call__(self, node: Union[str, DocNode]) -> List[Union[str, DocNode]]:
        if self._trans_node:
            return node.get_text().split(self._splitter)
        return node.split(self._splitter)

sentence_splitter_1 = SymbolSplitter()
docs.create_node_group(name='block3', transform=sentence_splitter_1)

# 指定传入 DocNode
sentence_splitter_2 = SymbolSplitter(trans_node=True)
docs.create_node_group(name='block4', transform=sentence_splitter_2, trans_node=True)

# 指定分割符号为 \n
paragraph_splitter = SymbolSplitter(splitter="\n")
docs.create_node_group(name='block5', transform=paragraph_splitter)

# 第四种：直接传入lambda函数，适用于简单规则情况
docs.create_node_group(name='block6', transform=lambda b: b.split('。'))

# 查看节点组内容，此处我们通过一个检索器召回一个节点并打印其中的内容，后续都通过这个方式实现
for i in range(6):
    group_name = f'block{i+1}'
    retriever = Retriever(docs, group_name=group_name, similarity="bm25_chinese", topk=1)
    node = retriever("亚硫酸盐有什么作用？")
    print(f"======= {group_name} =====")
    print(node[0].get_content())
```
