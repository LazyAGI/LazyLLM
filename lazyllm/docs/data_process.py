# flake8: noqa E501
import importlib
from . import utils
import functools

add_chinese_doc = functools.partial(utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools'))
add_english_doc = functools.partial(utils.add_english_doc, module=importlib.import_module('lazyllm.tools'))
add_example = functools.partial(utils.add_example, module=importlib.import_module('lazyllm.tools'))

add_chinese_doc('data.data_register', """\
数据处理算子注册器装饰器 / 工厂，用于将函数或类注册为可复用的数据处理算子。

用法：
     
- 可以用来注册单条数据处理算子（实现 forward 方法或函数）。
- 可以用来注册批处理算子（实现 forward_batch_input 方法或函数）。
- 支持通过参数 rewrite_func 指定注册时替换框架调用的方法（'forward' 或 'forward_batch_input'）。

Args:
    name (str): 注册路径，例如 'data.mygroup'，便于按组查找。
    rewrite_func (str): 可选，指定在注册时框架应当使用的执行接口（'forward' 或 'forward_batch_input'）。
""")

add_english_doc('data.data_register', """\
Decorator / factory for registering data processing operators.

Usage:

- Register functions or classes that process single data items (implementing forward) or batches (implementing forward_batch_input).
- You may specify rewrite_func to indicate which interface to use ('forward' or 'forward_batch_input').

Args:
    name (str): registration path, e.g. 'data.mygroup', used to group operators.
    rewrite_func (str): optional, the method the framework should invoke ('forward' or 'forward_batch_input').
""")

add_example('data.data_register', """\
```python
from lazyllm.tools.data import data_register

Demo = data_register.new_group('Demo')
            
# register a simple batch function
@data_register('data.Demo', rewrite_func='forward_batch_input')
def my_batch_op(data, input_key='text'):
    for item in data:
        item[input_key] = item.get(input_key, '').strip()
    return data

# register a class-based operator
class MyOp(Demo):
    def forward(self, data):
        data['processed'] = True
        return data
```
""")

add_chinese_doc('data.LazyLLMDataBase', """\
数据处理算子基类。为注册到 data_register 的算子提供统一行为，包括并发执行、结果保存/恢复、进度记录和错误收集。

主要方法和行为：

- forward(self, input, **kwargs): 处理单条数据（子类/函数实现）。
- forward_batch_input(self, inputs, **kwargs): 处理批量数据并返回最终结果（子类/函数实现）。
- __call__(self, inputs): 统一入口，会根据子类是否实现 forward 或 forward_batch_input 选择执行逻辑；支持并发执行、断点续传和保存结果。
- set_output(self, path): 设置导出路径，调用后 __call__ 返回导出文件路径而不是内存结果。

构造函数参数:

- _concurrency_mode (str): 并发模式，'process'|'thread'|'single'。
- _save_data (bool): 是否保存中间结果到磁盘以便 Resume。
- _max_workers (int|None): 最大并发工作进程/线程数，None 表示使用默认。
- _ignore_errors (bool): 是否忽略任务异常。
- **kwargs (dict): 其它传递给算子的参数。

配置项（通过 lazyllm.config）:

- data_process_path (str): 存储处理结果的根路径。
- data_process_resume (bool): 是否开启 Resume 功能，从进度文件继续处理。
""")

add_english_doc('data.LazyLLMDataBase', """\
Base class for data processing operators registered via data_register.
Provides concurrency, result persistence/resume, progress tracking, and error collection.

Key methods:

- forward(self, input, **kwargs): implement single-item processing.
- forward_batch_input(self, inputs, **kwargs): implement batch processing and return results.
- __call__(self, inputs): unified entry point; decides execution mode based on implemented methods and handles concurrency, resume and saving.
- set_output(self, path): set export path; when set, __call__ writes results to a file and returns the file path.

Constructor args:

- _concurrency_mode (str): concurrency mode, one of 'process'|'thread'|'single'.
- _save_data (bool): whether to persist intermediate results for resume.
- _max_workers (int|None): maximum workers for concurrency, None means default.
- _ignore_errors (bool): whether to ignore exceptions in tasks.
- **kwargs (dict): additional operator arguments.

Config keys (via lazyllm.config):

- data_process_path (str): root folder to store pipeline outputs.
- data_process_resume (bool): enable resume from previous progress.
""")

add_example('data.LazyLLMDataBase', """\
```python
from lazyllm.tools.data import LazyLLMDataBase

# simple usage: subclass and implement forward
class EchoOp(LazyLLMDataBase):
    def forward(self, data):
        return {'text': data.get('text', '')}

op = EchoOp(_save_data=True)
res = op([{'text': 'hello'}])  # returns list or exported path depending on set_output
```
""")

add_chinese_doc('data.LazyLLMDataBase.set_output', """\
设置输出路径，用于把最终结果导出为 jsonl 文件并返回文件路径。

Args:
    output_path (str): 文件夹路径或具体 .jsonl 文件路径。若为文件夹，则在该文件夹下创建以类名命名的 jsonl 文件。

行为：

- 如果传入的是文件夹路径，则在该文件夹下创建以类名命名的 jsonl 文件。
- 如果传入的是以 .jsonl 结尾的路径，则直接写入该文件（必要时会创建目录）。
- 返回写入的绝对路径字符串。
""")

add_english_doc('data.LazyLLMDataBase.set_output', """\
Set output path for exporting final results to a JSONL file and return the file path.

Args:
    output_path (str): directory path or concrete .jsonl file path. If a directory is provided, a file named <ClassName>.jsonl will be created inside it.

Behavior:
- If a folder path is provided, a file named <ClassName>.jsonl will be created in that folder.
- If a .jsonl file path is provided, results will be written to that file (directories created as needed).
- Returns the absolute path of the exported file.
""")

add_example('data.LazyLLMDataBase.set_output', """\
```python
from lazyllm.tools.data import Demo2

# export to a directory (will create DemoClass.jsonl)
op = Demo2.rich_content(input_key='text').set_output('./out_dir')
path = op([{'text': 'sample'}])
print(path)  # ./out_dir/RichContent.jsonl or similar

# export to a specific file
op = Demo2.rich_content(input_key='text').set_output('./out_dir/results.jsonl')
path = op([{'text': 'sample'}])
print(path)  # ./out_dir/results.jsonl
```
""")  

add_chinese_doc('data.LazyLLMDataBase.forward', """\
子类需要实现的方法，处理单条数据。返回值支持：

- dict: 表示处理后的单条结果。
- list: 表示将一条输入展开为多条输出。
- None: 表示保留原始输入（不修改）。
- 抛出异常或返回错误对象会被记录到错误文件并跳过（依赖配置和调用者）。

Args:
    input (dict): 单条输入数据字典。
    **kwargs (dict): 其它用户传入的参数。
""")

add_english_doc('data.LazyLLMDataBase.forward', """\
Method to implement in subclasses for single-item processing. Supported return types:

- dict: processed single result.
- list: expand one input into multiple outputs.
- None: keep the original input unchanged.
Exceptions or error returns are recorded to the error file and typically skipped from valid results.

Args:
    input (dict): a single input data dict.
    **kwargs (dict): additional user-provided arguments.
""")

add_example('data.LazyLLMDataBase.forward', """\
```python
from lazyllm.tools.data import LazyLLMDataBase

class MyOp(LazyLLMDataBase):
    def forward(self, data):
        # return dict or list or None
        return {'text': data.get('text', '').upper()}

op = MyOp()
print(op([{'text': 'a'}]))
```
""")

add_chinese_doc('data.LazyLLMDataBase.forward_batch_input', """\
子类可实现的批量处理方法，用于在非逐条并发场景下直接接收整个输入列表并返回最终结果列表（可用于自定义批量逻辑或外部服务一次性处理）。

Args:
    inputs (list[dict]): 输入数据列表。
    **kwargs (dict): 其它用户传入的参数。
""")

add_english_doc('data.LazyLLMDataBase.forward_batch_input', """\
Optional batch-processing method for subclasses. Receives the whole input list and returns a final list of results. Useful for custom batching or single-call external services.

Args:
    inputs (list[dict]): list of input data dicts.
    **kwargs (dict): additional user-provided arguments.
""")

add_example('data.LazyLLMDataBase.forward_batch_input', """\
```python
from lazyllm.tools.data import LazyLLMDataBase

class BatchOp(LazyLLMDataBase):
    def forward_batch_input(self, inputs):
        # implement batch processing and return a list
        return [{'text': i.get('text', '').lower()} for i in inputs]

op = BatchOp()
print(op([{'text': 'A'}, {'text': 'B'}]))
```
""")

add_chinese_doc('data.operators.demo_ops.process_uppercase', """\
将输入文本字段转换为大写。适用于单条处理函数注册（forward）。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名，默认 'content'
""")

add_english_doc('data.operators.demo_ops.process_uppercase', """\
Convert the input text field to uppercase. Intended as a single-item processing function.

Args:
    data (dict): a dict representing a single data item.
    input_key (str): key name of the text field, default 'content'.
""")

add_example('data.operators.demo_ops.process_uppercase', """\
```python
from lazyllm.tools.data.operators.demo_ops import process_uppercase

op = process_uppercase(input_key='text')
print(op({'text': 'hello'}))  # {'text': 'HELLO'}
```
""")

add_chinese_doc('data.operators.demo_ops.build_pre_suffix', """\
对输入列表中每项在指定字段前后添加前缀和后缀。此算子以批处理函数注册（forward_batch_input）。

Args:
    data (list[dict]): 输入列表
    input_key (str): 文本字段名
    prefix (str): 要添加的前缀
    suffix (str): 要添加的后缀
""")

add_english_doc('data.operators.demo_ops.build_pre_suffix', """\
Add a prefix and suffix to the specified field of each item in the input list. Registered as a batch operator.

Args:
    data (list[dict]): list of dicts
    input_key (str): key name of the text field
    prefix (str): string to add before the field
    suffix (str): string to add after the field
""")

add_example('data.operators.demo_ops.build_pre_suffix', """\
```python
from lazyllm.tools.data.operators.demo_ops import build_pre_suffix

op = build_pre_suffix(input_key='text', prefix='Hello, ', suffix='!')
print(op([{'text': 'world'}]))
# [{'text': 'Hello, world!'}]
```
""")

add_chinese_doc('data.operators.demo_ops.AddSuffix', """\
通过类方式实现的算子，为指定字段添加后缀。支持并发配置（通过构造参数）。

Args:
    suffix (str): 要添加的后缀
    input_key (str): 文本字段名
    _max_workers (int|None): 可选，最大并发数
    _concurrency_mode (str): 可选，并发模式
    _save_data (bool): 可选，是否保存结果
""")

add_english_doc('data.operators.demo_ops.AddSuffix', """\
Class-based operator that appends a suffix to a specified field. Supports concurrency configuration via constructor args.

Args:
    suffix (str): suffix string to append
    input_key (str): key name of the text field
    _max_workers (int|None): optional max concurrency
    _concurrency_mode (str): optional concurrency mode
    _save_data (bool): optional whether to persist results
""")

add_example('data.operators.demo_ops.AddSuffix', """\
```python
from lazyllm.tools.data.operators.demo_ops import AddSuffix

op = AddSuffix(suffix='!!!', input_key='text', _max_workers=2)
print(op([{'text': 'wow'}]))  # [{'text': 'wow!!!'}]
```
""")

add_chinese_doc('data.operators.demo_ops.rich_content', """\
将单条输入拆分为多条输出，生成富内容表示（原始 + 若干派生）。适用于返回 list 的 forward。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名
""")

add_english_doc('data.operators.demo_ops.rich_content', """\
Split a single input into multiple outputs (original + derived parts). Implemented as a forward that returns a list.

Args:
    data (dict): single data dict
    input_key (str): key name of the text field
""")

add_example('data.operators.demo_ops.rich_content', """\
```python
from lazyllm.tools.data.operators.demo_ops import rich_content

op = rich_content(input_key='text')
print(op({'text': 'This is a test.'}))
# [
#   {'text': 'This is a test.'},
#   {'text': 'This is a test. - part 1'},
#   {'text': 'This is a test. - part 2'}
# ]
```
""")

add_chinese_doc('data.operators.demo_ops.error_prone_op', """\
一个用于测试的算子：在特定输入（content == 'fail'）时抛出异常，否则返回处理后的字典结果。用于验证错误收集与跳过逻辑。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名
""")

add_english_doc('data.operators.demo_ops.error_prone_op', """\
A test operator that raises an exception for specific input (content == 'fail') and otherwise returns a processed dict.
Used to validate error collection and skipping behavior.

Args:
    data (dict): single data dict
    input_key (str): key name of the text field
""")

add_example('data.operators.demo_ops.error_prone_op', """\
```python
from lazyllm.tools.data.operators.demo_ops import error_prone_op

op = error_prone_op(input_key='text', _save_data=True, _concurrency_mode='single')
res = op([{'text': 'ok'}, {'text': 'fail'}, {'text': 'ok2'}])
# valid results skip the failed item; error details written to error file
```
""")

# refine_op
add_chinese_doc('data.operators.refine_op.remove_extra_spaces', """\
将指定字段中的多余空白（多个空格、换行、制表符）归一化为单个空格。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名，默认 'content'
""")

add_english_doc('data.operators.refine_op.remove_extra_spaces', """\
Normalize whitespace by collapsing multiple spaces, newlines and tabs into single spaces.

Args:
    data (dict): single data dict
    input_key (str): key of the text field, default 'content'
""")

add_example('data.operators.refine_op.remove_extra_spaces', """\
```python
from lazyllm.tools.data import refine

func = refine.remove_extra_spaces(input_key='content')
inputs = [{'content': 'hello   world\\\\n\\\\n  foo\\\\tbar'}]
res = func(inputs)
print(res)
# [{'content': 'hello world foo bar'}]
```
""")

add_chinese_doc('data.operators.refine_op.remove_emoji', """\
移除指定字段中的 emoji 字符。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名，默认 'content'
""")

add_english_doc('data.operators.refine_op.remove_emoji', """\
Remove emoji characters from the specified text field.

Args:
    data (dict): single data dict
    input_key (str): key of the text field, default 'content'
""")

add_example('data.operators.refine_op.remove_emoji', """\
```python
from lazyllm.tools.data import refine

func = refine.remove_emoji(input_key='content')
inputs = [{'content': 'Hello 😊 World 🌍!'}]
res = func(inputs)
print(res)
# [{'content': 'Hello  World !'}]
```
""")

add_chinese_doc('data.operators.refine_op.remove_html_url', """\
移除指定字段中的 HTTP/HTTPS 链接和 HTML 标签。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名，默认 'content'
""")

add_english_doc('data.operators.refine_op.remove_html_url', """\
Remove HTTP/HTTPS URLs and HTML tags from the specified text field.

Args:
    data (dict): single data dict
    input_key (str): key of the text field, default 'content'
""")

add_example('data.operators.refine_op.remove_html_url', """\
```python
from lazyllm.tools.data import refine

func = refine.remove_html_url(input_key='content')
inputs = [{'content': 'Check https://example.com and <b>bold</b>'}]
res = func(inputs)
print(res)
# [{'content': 'Check  and bold'}]
```
""")

add_chinese_doc('data.operators.refine_op.remove_html_entity', """\
移除指定字段中的 HTML 实体（如 &nbsp;、&lt;、&amp; 等）。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名，默认 'content'
""")

add_english_doc('data.operators.refine_op.remove_html_entity', """\
Remove HTML entities (e.g. &nbsp;, &lt;, &amp;) from the specified text field.

Args:
    data (dict): single data dict
    input_key (str): key of the text field, default 'content'
""")

add_example('data.operators.refine_op.remove_html_entity', """\
```python
from lazyllm.tools.data import refine

func = refine.remove_html_entity(input_key='content')
inputs = [{'content': 'Hello&nbsp;World &amp; &lt;tag&gt;'}]
res = func(inputs)
print(res)
# [{'content': 'HelloWorld  tag'}]
```
""")

# token_chunker
add_chinese_doc('data.operators.token_chunker.TokenChunker', """\
按 token 数量将长文本切分为多个块。先按段落分隔，再按句子细切，保证每块不超过 max_tokens，过短块可丢弃。

Args:
    input_key (str): 文本字段名，默认 'content'
    model_path (str|None): tokenizer 模型路径，默认使用 Qwen2.5-0.5B-Instruct
    max_tokens (int): 每块最大 token 数，默认 1024
    min_tokens (int): 每块最小 token 数，低于此值的块可能被丢弃，默认 200
    _concurrency_mode (str): 可选，并发模式
    _max_workers (int|None): 可选，最大并发数
""")

add_english_doc('data.operators.token_chunker.TokenChunker', """\
Split long text into chunks by token count. Splits by paragraph first, then by sentence.
Ensures each chunk does not exceed max_tokens; chunks below min_tokens may be discarded.

Args:
    input_key (str): key of the text field, default 'content'
    model_path (str|None): path to tokenizer model, default Qwen2.5-0.5B-Instruct
    max_tokens (int): max tokens per chunk, default 1024
    min_tokens (int): min tokens per chunk, smaller chunks may be discarded, default 200
    _concurrency_mode (str): optional concurrency mode
    _max_workers (int|None): optional max concurrency
""")

add_example('data.operators.token_chunker.TokenChunker', """\
```python
from lazyllm.tools.data import chunker

func = chunker.TokenChunker(input_key='content', max_tokens=50, min_tokens=10)
inputs = [{'content': '人工智能是计算机科学的一个分支。' * 20, 'meta_data': {'source': 'doc_1'}}]
res = func(inputs)
print(res)
# [{'uid': '...', 'content': '...', 'meta_data': {'source': 'doc_1', 'index': 0, 'total': N, 'length': ...}}, ...]
```
""")

# filter_op
add_chinese_doc('data.operators.filter_op.LanguageFilter', """\
使用 FastText 进行语言识别，仅保留指定语言的文本。

Args:
    input_key (str): 文本字段名，默认 'content'
    target_language (str|list): 目标语言代码，如 'zho_Hans'、'eng_Latn'
    threshold (float): 置信度阈值，默认 0.6
    model_path (str|None): FastText 模型路径
    _concurrency_mode (str): 可选，并发模式
""")

add_english_doc('data.operators.filter_op.LanguageFilter', """\
Filter text by language using FastText. Keeps only texts in the specified language(s).

Args:
    input_key (str): key of the text field, default 'content'
    target_language (str|list): target language code(s), e.g. 'zho_Hans', 'eng_Latn'
    threshold (float): confidence threshold, default 0.6
    model_path (str|None): path to FastText model
    _concurrency_mode (str): optional concurrency mode
""")

add_example('data.operators.filter_op.LanguageFilter', """\
```python
from lazyllm.tools.data import filter

func = filter.LanguageFilter(input_key='content', target_language='zho_Hans', threshold=0.3)
inputs = [{'content': '这是一段中文文本。'}, {'content': 'This is English.'}]
res = func(inputs)
print(res)
# [{'content': '这是一段中文文本。'}]
```
""")

add_chinese_doc('data.operators.filter_op.MinHashDeduplicateFilter', """\
使用 MinHash LSH 去除近似重复文本，批处理时保留首次出现的文本。

Args:
    input_key (str): 文本字段名，默认 'content'
    threshold (float): 相似度阈值，默认 0.85
    num_perm (int): MinHash 排列数，默认 128
    use_n_gram (bool): 是否使用 n-gram，默认 True
    ngram (int): n-gram 长度，默认 5
""")

add_english_doc('data.operators.filter_op.MinHashDeduplicateFilter', """\
Remove near-duplicate texts using MinHash LSH. For batch input, keeps first occurrence of each unique text.

Args:
    input_key (str): key of the text field, default 'content'
    threshold (float): similarity threshold, default 0.85
    num_perm (int): number of MinHash permutations, default 128
    use_n_gram (bool): use n-gram, default True
    ngram (int): n-gram size, default 5
""")

add_example('data.operators.filter_op.MinHashDeduplicateFilter', """\
```python
from lazyllm.tools.data import filter

func = filter.MinHashDeduplicateFilter(input_key='content', threshold=0.85)
inputs = [{'uid': '0', 'content': '这是第一段不同的内容。'}, {'uid': '1', 'content': '这是第一段不同的内容。'}]
res = func(inputs)
print(res)
# [{'uid': '0', 'content': '这是第一段不同的内容。'}]
```
""")

add_chinese_doc('data.operators.filter_op.BlocklistFilter', """\
使用 AC 自动机多模式匹配过滤包含敏感词/违禁词超过阈值的文本。

Args:
    input_key (str): 文本字段名，默认 'content'
    blocklist (list|None): 违禁词列表
    blocklist_path (str|None): 违禁词文件路径
    language (str): 语言，'zh' 或 'en'，默认 'zh'
    threshold (int): 允许出现的违禁词最大数量，默认 1
    _concurrency_mode (str): 可选，并发模式
""")

add_english_doc('data.operators.filter_op.BlocklistFilter', """\
Filter text containing more than threshold blocked words using Aho-Corasick automaton.

Args:
    input_key (str): key of the text field, default 'content'
    blocklist (list|None): list of blocked words
    blocklist_path (str|None): path to blocklist file
    language (str): language, 'zh' or 'en', default 'zh'
    threshold (int): max allowed occurrences of blocked words, default 1
    _concurrency_mode (str): optional concurrency mode
""")

add_example('data.operators.filter_op.BlocklistFilter', """\
```python
from lazyllm.tools.data import filter

func = filter.BlocklistFilter(input_key='content', blocklist=['敏感', '违禁'], threshold=0)
inputs = [{'content': '这是正常的文本内容。'}, {'content': '这里包含敏感词。'}]
res = func(inputs)
print(res)
# [{'content': '这是正常的文本内容。'}]
```
""")

add_chinese_doc('data.operators.filter_op.SymbolRatioFilter', """\
过滤指定符号（如 #、...、…）占比过高的文本。

Args:
    input_key (str): 文本字段名，默认 'content'
    max_ratio (float): 符号与词数最大比例，默认 0.3
    symbols (list|None): 要统计的符号列表，默认 ['#', '...', '…']
    _concurrency_mode (str): 可选，并发模式
""")

add_english_doc('data.operators.filter_op.SymbolRatioFilter', """\
Filter text with too high ratio of specified symbols (e.g. #, ..., …) to words.

Args:
    input_key (str): key of the text field, default 'content'
    max_ratio (float): max ratio of symbols to words, default 0.3
    symbols (list|None): symbols to count, default ['#', '...', '…']
    _concurrency_mode (str): optional concurrency mode
""")

add_example('data.operators.filter_op.SymbolRatioFilter', """\
```python
from lazyllm.tools.data import filter

func = filter.SymbolRatioFilter(input_key='content', max_ratio=0.3)
inputs = [{'content': 'Normal text without symbols'}, {'content': '### ... … ###'}]
res = func(inputs)
print(res)
# [{'content': 'Normal text without symbols'}]
```
""")

add_chinese_doc('data.operators.filter_op.StopWordFilter', """\
过滤停用词占比过高的文本（如几乎全为「的了呢」的无效内容）。

Args:
    input_key (str): 文本字段名，默认 'content'
    max_ratio (float): 停用词最大占比，超过则过滤，默认 0.5
    use_tokenizer (bool): 是否使用分词，默认 True
    language (str): 语言，'zh' 或 'en'，默认 'zh'
    _concurrency_mode (str): 可选，并发模式
""")

add_english_doc('data.operators.filter_op.StopWordFilter', """\
Filter text with too high stopword ratio (e.g. invalid content mostly stopwords).

Args:
    input_key (str): key of the text field, default 'content'
    max_ratio (float): max stopword ratio, filter if exceeded, default 0.5
    use_tokenizer (bool): use tokenizer, default True
    language (str): language, 'zh' or 'en', default 'zh'
    _concurrency_mode (str): optional concurrency mode
""")

add_example('data.operators.filter_op.StopWordFilter', """\
```python
from lazyllm.tools.data import filter

func = filter.StopWordFilter(input_key='content', max_ratio=0.5, language='zh')
inputs = [{'content': '这是一段包含实际内容的正常文本。'}, {'content': '的了吗呢吧啊'}]
res = func(inputs)
print(res)
# [{'content': '这是一段包含实际内容的正常文本。'}]
```
""")

add_chinese_doc('data.operators.filter_op.CapitalWordFilter', """\
过滤全大写单词占比过高的文本。

Args:
    input_key (str): 文本字段名，默认 'content'
    max_ratio (float): 全大写单词最大占比，默认 0.5
    use_tokenizer (bool): 是否使用分词，默认 False
    _concurrency_mode (str): 可选，并发模式
""")

add_english_doc('data.operators.filter_op.CapitalWordFilter', """\
Filter text with too high ratio of all-caps words.

Args:
    input_key (str): key of the text field, default 'content'
    max_ratio (float): max ratio of all-caps words, default 0.5
    use_tokenizer (bool): use tokenizer, default False
    _concurrency_mode (str): optional concurrency mode
""")

add_example('data.operators.filter_op.CapitalWordFilter', """\
```python
from lazyllm.tools.data import filter

func = filter.CapitalWordFilter(input_key='content', max_ratio=0.5)
inputs = [{'content': 'Normal text with Some Capitals'}, {'content': 'MOSTLY UPPERCASE'}]
res = func(inputs)
print(res)
# [{'content': 'Normal text with Some Capitals'}]
```
""")

add_chinese_doc('data.operators.filter_op.word_count_filter', """\
按词/字符数量过滤：中文按字符数，英文按单词数，保留在 [min_words, max_words) 范围内的文本。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名，默认 'content'
    min_words (int): 最小词数，默认 10
    max_words (int): 最大词数，默认 10000
    language (str): 语言，'zh' 或 'en'，默认 'zh'
""")

add_english_doc('data.operators.filter_op.word_count_filter', """\
Filter by word/char count: Chinese by char count, English by word count. Keeps text in [min_words, max_words).

Args:
    data (dict): single data dict
    input_key (str): key of the text field, default 'content'
    min_words (int): min count, default 10
    max_words (int): max count, default 10000
    language (str): language, 'zh' or 'en', default 'zh'
""")

add_example('data.operators.filter_op.word_count_filter', """\
```python
from lazyllm.tools.data import filter

func = filter.word_count_filter(input_key='content', min_words=5, max_words=20, language='zh')
inputs = [{'content': '短文本'}, {'content': '这是一段适中长度的中文文本内容。'}]
res = func(inputs)
print(res)
# [{'content': '这是一段适中长度的中文文本内容。'}]
```
""")

add_chinese_doc('data.operators.filter_op.colon_end_filter', """\
过滤以冒号结尾的文本。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名，默认 'content'
""")

add_english_doc('data.operators.filter_op.colon_end_filter', """\
Filter text ending with colon.

Args:
    data (dict): single data dict
    input_key (str): key of the text field, default 'content'
""")

add_example('data.operators.filter_op.colon_end_filter', """\
```python
from lazyllm.tools.data import filter

func = filter.colon_end_filter(input_key='content')
inputs = [{'content': '这是正常结尾。'}, {'content': '这是冒号结尾：'}]
res = func(inputs)
print(res)
# [{'content': '这是正常结尾。'}]
```
""")

add_chinese_doc('data.operators.filter_op.sentence_count_filter', """\
按句子数量过滤，保留在 [min_sentences, max_sentences] 范围内的文本。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名，默认 'content'
    min_sentences (int): 最少句子数，默认 3
    max_sentences (int): 最多句子数，默认 1000
    language (str): 语言，'zh' 或 'en'，默认 'zh'
""")

add_english_doc('data.operators.filter_op.sentence_count_filter', """\
Filter by sentence count. Keeps text with sentences in [min_sentences, max_sentences].

Args:
    data (dict): single data dict
    input_key (str): key of the text field, default 'content'
    min_sentences (int): min sentence count, default 3
    max_sentences (int): max sentence count, default 1000
    language (str): language, 'zh' or 'en', default 'zh'
""")

add_example('data.operators.filter_op.sentence_count_filter', """\
```python
from lazyllm.tools.data import filter

func = filter.sentence_count_filter(input_key='content', min_sentences=2, max_sentences=10, language='zh')
inputs = [{'content': '单句。'}, {'content': '第一句。第二句。'}]
res = func(inputs)
print(res)
# [{'content': '第一句。第二句。'}]
```
""")

add_chinese_doc('data.operators.filter_op.ellipsis_end_filter', """\
过滤以省略号（...、…、……）结尾的行占比过高的文本。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名，默认 'content'
    max_ratio (float): 以省略号结尾的行最大占比，默认 0.3
""")

add_english_doc('data.operators.filter_op.ellipsis_end_filter', """\
Filter text with too many lines ending in ellipsis (...、…、……).

Args:
    data (dict): single data dict
    input_key (str): key of the text field, default 'content'
    max_ratio (float): max ratio of lines ending with ellipsis, default 0.3
""")

add_example('data.operators.filter_op.ellipsis_end_filter', """\
```python
from lazyllm.tools.data import filter

func = filter.ellipsis_end_filter(input_key='content', max_ratio=0.3)
inputs = [{'content': '第一行。\\\\n第二行。\\\\n第三行。'}, {'content': '第一行...\\\\n第二行...'}]
res = func(inputs)
print(res)
# [{'content': '第一行。\\\\n第二行。\\\\n第三行。'}]
```
""")

add_chinese_doc('data.operators.filter_op.null_content_filter', """\
过滤空内容或仅空白字符的文本。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名，默认 'content'
""")

add_english_doc('data.operators.filter_op.null_content_filter', """\
Filter null or whitespace-only content.

Args:
    data (dict): single data dict
    input_key (str): key of the text field, default 'content'
""")

add_example('data.operators.filter_op.null_content_filter', """\
```python
from lazyllm.tools.data import filter

func = filter.null_content_filter(input_key='content')
inputs = [{'content': 'Valid content'}, {'content': ''}, {'content': '   '}]
res = func(inputs)
print(res)
# [{'content': 'Valid content'}]
```
""")

add_chinese_doc('data.operators.filter_op.word_length_filter', """\
按单词平均长度过滤，保留在 [min_length, max_length) 范围内的文本。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名，默认 'content'
    min_length (float): 单词平均最小长度，默认 3
    max_length (float): 单词平均最大长度，默认 20
""")

add_english_doc('data.operators.filter_op.word_length_filter', """\
Filter by average word length. Keeps text with mean word length in [min_length, max_length).

Args:
    data (dict): single data dict
    input_key (str): key of the text field, default 'content'
    min_length (float): min avg word length, default 3
    max_length (float): max avg word length, default 20
""")

add_example('data.operators.filter_op.word_length_filter', """\
```python
from lazyllm.tools.data import filter

func = filter.word_length_filter(input_key='content', min_length=3, max_length=10)
inputs = [{'content': 'I am ok'}, {'content': 'This is a normal sentence'}]
res = func(inputs)
print(res)
# [{'content': 'This is a normal sentence'}]
```
""")

add_chinese_doc('data.operators.filter_op.idcard_filter', """\
过滤包含过多身份证/证件相关词汇的文本。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名，默认 'content'
    threshold (int): 匹配到相关词的最大数量，超过则过滤，默认 3
""")

add_english_doc('data.operators.filter_op.idcard_filter', """\
Filter text containing too many ID card / identity document related terms.

Args:
    data (dict): single data dict
    input_key (str): key of the text field, default 'content'
    threshold (int): max matches of related terms, filter if exceeded, default 3
""")

add_example('data.operators.filter_op.idcard_filter', """\
```python
from lazyllm.tools.data import filter

func = filter.idcard_filter(input_key='content', threshold=1)
inputs = [{'content': '这是正常文本'}, {'content': '请提供身份证号码和ID number'}]
res = func(inputs)
print(res)
# [{'content': '这是正常文本'}]
```
""")

add_chinese_doc('data.operators.filter_op.no_punc_filter', """\
过滤标点之间段路过长的文本（如无标点超长串）。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名，默认 'content'
    max_length_between_punct (int): 标点间最大长度，默认 112
    language (str): 语言，'zh' 或 'en'，默认 'zh'
""")

add_english_doc('data.operators.filter_op.no_punc_filter', """\
Filter text with too long segments between punctuation marks.

Args:
    data (dict): single data dict
    input_key (str): key of the text field, default 'content'
    max_length_between_punct (int): max length between punctuation, default 112
    language (str): language, 'zh' or 'en', default 'zh'
""")

add_example('data.operators.filter_op.no_punc_filter', """\
```python
from lazyllm.tools.data import filter

func = filter.no_punc_filter(input_key='content', max_length_between_punct=20, language='zh')
inputs = [{'content': '这是。正常。文本。'}, {'content': '这是一段没有标点符号的超长文本' * 10}]
res = func(inputs)
print(res)
# [{'content': '这是。正常。文本。'}]
```
""")

add_chinese_doc('data.operators.filter_op.special_char_filter', """\
过滤包含特殊不可见字符的文本（零宽字符、替换字符等）。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名，默认 'content'
""")

add_english_doc('data.operators.filter_op.special_char_filter', """\
Filter text containing special invisible characters (zero-width, replacement char, etc.).

Args:
    data (dict): single data dict
    input_key (str): key of the text field, default 'content'
""")

add_example('data.operators.filter_op.special_char_filter', """\
```python
from lazyllm.tools.data import filter

func = filter.special_char_filter(input_key='content')
inputs = [{'content': 'Normal text 正常文本'}, {'content': 'Text with \u200b zero width'}]
res = func(inputs)
print(res)
# [{'content': 'Normal text 正常文本'}]
```
""")

add_chinese_doc('data.operators.filter_op.watermark_filter', """\
过滤包含版权/水印相关词汇的文本。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名，默认 'content'
    watermarks (list|None): 自定义水印词列表，默认使用内置列表
""")

add_english_doc('data.operators.filter_op.watermark_filter', """\
Filter text containing copyright/watermark related terms.

Args:
    data (dict): single data dict
    input_key (str): key of the text field, default 'content'
    watermarks (list|None): custom watermark terms, default uses built-in list
""")

add_example('data.operators.filter_op.watermark_filter', """\
```python
from lazyllm.tools.data import filter

func = filter.watermark_filter(input_key='content')
inputs = [{'content': 'Normal content'}, {'content': 'This document contains Copyright notice'}]
res = func(inputs)
print(res)
# [{'content': 'Normal content'}]
```
""")

add_chinese_doc('data.operators.filter_op.curly_bracket_filter', """\
过滤花括号 {} 占比过高的文本。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名，默认 'content'
    max_ratio (float): 花括号最大占比，默认 0.08
""")

add_english_doc('data.operators.filter_op.curly_bracket_filter', """\
Filter text with too high ratio of curly brackets {}.

Args:
    data (dict): single data dict
    input_key (str): key of the text field, default 'content'
    max_ratio (float): max ratio of curly brackets, default 0.08
""")

add_example('data.operators.filter_op.curly_bracket_filter', """\
```python
from lazyllm.tools.data import filter

func = filter.curly_bracket_filter(input_key='content', max_ratio=0.08)
inputs = [{'content': 'Normal text'}, {'content': '{{{{{' * 10}]
res = func(inputs)
print(res)
# [{'content': 'Normal text'}]
```
""")

add_chinese_doc('data.operators.filter_op.lorem_ipsum_filter', """\
过滤 Lorem ipsum、占位符等占位文本。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名，默认 'content'
    max_ratio (float): 占位模式最大出现比例，默认 3e-8
""")

add_english_doc('data.operators.filter_op.lorem_ipsum_filter', """\
Filter Lorem ipsum, placeholder text, etc.

Args:
    data (dict): single data dict
    input_key (str): key of the text field, default 'content'
    max_ratio (float): max ratio of placeholder patterns, default 3e-8
""")

add_example('data.operators.filter_op.lorem_ipsum_filter', """\
```python
from lazyllm.tools.data import filter

func = filter.lorem_ipsum_filter(input_key='content')
inputs = [{'content': 'This is real content'}, {'content': 'Lorem ipsum dolor sit amet'}]
res = func(inputs)
print(res)
# [{'content': 'This is real content'}]
```
""")

add_chinese_doc('data.operators.filter_op.unique_word_filter', """\
过滤去重后词数占比过低的文本（重复词过多的无效内容）。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名，默认 'content'
    min_ratio (float): 去重词数最小占比，默认 0.1
    use_tokenizer (bool): 是否使用分词，默认 True
    language (str): 语言，'zh' 或 'en'，默认 'zh'
""")

add_english_doc('data.operators.filter_op.unique_word_filter', """\
Filter text with too low unique word ratio (excessive repetition).

Args:
    data (dict): single data dict
    input_key (str): key of the text field, default 'content'
    min_ratio (float): min unique word ratio, default 0.1
    use_tokenizer (bool): use tokenizer, default True
    language (str): language, 'zh' or 'en', default 'zh'
""")

add_example('data.operators.filter_op.unique_word_filter', """\
```python
from lazyllm.tools.data import filter

func = filter.unique_word_filter(input_key='content', min_ratio=0.4, language='zh')
inputs = [{'content': '这是一段包含多个不同词汇的文本。'}, {'content': '重复重复重复'}]
res = func(inputs)
print(res)
# [{'content': '这是一段包含多个不同词汇的文本。'}]
```
""")

add_chinese_doc('data.operators.filter_op.char_count_filter', """\
按去除空白后的字符数过滤，保留在 [min_chars, max_chars] 范围内的文本。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名，默认 'content'
    min_chars (int): 最小字符数，默认 100
    max_chars (int): 最大字符数，默认 100000
""")

add_english_doc('data.operators.filter_op.char_count_filter', """\
Filter by character count (excluding whitespace). Keeps text in [min_chars, max_chars].

Args:
    data (dict): single data dict
    input_key (str): key of the text field, default 'content'
    min_chars (int): min chars, default 100
    max_chars (int): max chars, default 100000
""")

add_example('data.operators.filter_op.char_count_filter', """\
```python
from lazyllm.tools.data import filter

func = filter.char_count_filter(input_key='content', min_chars=10, max_chars=100)
inputs = [{'content': '短'}, {'content': '这是一段中等长度的文本内容。'}]
res = func(inputs)
print(res)
# [{'content': '这是一段中等长度的文本内容。'}]
```
""")

add_chinese_doc('data.operators.filter_op.bullet_point_filter', """\
过滤子弹点行占比过高的文本（如目录、纯列表）。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名，默认 'content'
    max_ratio (float): 以子弹点开头的行最大占比，默认 0.9
""")

add_english_doc('data.operators.filter_op.bullet_point_filter', """\
Filter text with too many bullet-point lines (e.g. TOC, pure lists).

Args:
    data (dict): single data dict
    input_key (str): key of the text field, default 'content'
    max_ratio (float): max ratio of bullet lines, default 0.9
""")

add_example('data.operators.filter_op.bullet_point_filter', """\
```python
from lazyllm.tools.data import filter

func = filter.bullet_point_filter(input_key='content', max_ratio=0.5)
inputs = [{'content': 'Normal paragraph text'}, {'content': '- Item 1\\\\n- Item 2\\\\n- Item 3'}]
res = func(inputs)
print(res)
# [{'content': 'Normal paragraph text'}]
```
""")

add_chinese_doc('data.operators.filter_op.javascript_filter', """\
过滤含大量 JavaScript 相关模式的文本（如代码、脚本片段）。短文本(<=3行)不检测，直接保留，避免误伤正常短句。

Args:
    data (dict): 单条数据字典
    input_key (str): 文本字段名，默认 'content'
    min_non_script_lines (int): 最少非脚本行数，默认 3
""")

add_english_doc('data.operators.filter_op.javascript_filter', """\
Filter text containing many JavaScript patterns (code, script fragments). Short text (<=3 lines) is passed through to avoid false positives on normal short sentences.

Args:
    data (dict): single data dict
    input_key (str): key of the text field, default 'content'
    min_non_script_lines (int): min non-script lines, default 3
""")

add_example('data.operators.filter_op.javascript_filter', """\
```python
from lazyllm.tools.data import filter

func = filter.javascript_filter(input_key='content', min_non_script_lines=2)
inputs = [{'content': 'Short normal text'}, {'content': 'function() { return 1; }\nconst x = 1;\nvar y = 2;\nlet z = 3;'}]
res = func(inputs)
print(res)
# [{'content': 'Short normal text'}]
```
""")

# pipelines module docs
add_chinese_doc( 'data.pipelines.demo_pipelines.build_demo_pipeline', """\
构建演示用数据处理流水线（Pipeline），包含若干示例算子并展示如何在 pipeline 上组合使用这些算子。

Args:
    input_key (str): 要处理的文本字段名，默认 'text'

**Returns:**\n
    一个可调用的 pipeline 对象，调用时会按顺序执行其中注册的算子。
""")

add_english_doc('data.pipelines.demo_pipelines.build_demo_pipeline', """\
Build a demo data processing pipeline composed of several example operators.

Args:
    input_key (str): the text field name to process, default 'text'

**Returns:**\n
    A callable pipeline object that executes registered operators in sequence.
""")

add_example('data.pipelines.demo_pipelines.build_demo_pipeline', """\
```python
from lazyllm.tools.data.pipelines.demo_pipelines import build_demo_pipeline

ppl = build_demo_pipeline(input_key='text')
data = [{'text': 'lazyLLM'}]
res = ppl(data)
print(res)  # demonstrates how operators are combined and applied
```
""")
