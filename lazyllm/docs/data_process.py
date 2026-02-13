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
from lazyllm.tools.data import Demo1

op = Demo1.process_uppercase(input_key='text')
data = [{'text': 'hello'}]
res = op(data)
print(res)
# [{'text': 'HELLO'}]
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
from lazyllm.tools.data import Demo1

op = Demo1.build_pre_suffix(input_key='text', prefix='Hello, ', suffix='!')
data = [{'text': 'world'}]
res = op(data)
print(res)
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
from lazyllm.tools.data import Demo2

op = Demo2.AddSuffix(suffix='!!!', input_key='text', _max_workers=2)
data = [{'text': 'wow'}]
res = op(data)
print(res)
# [{'text': 'wow!!!'}]
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
from lazyllm.tools.data import Demo2

op = Demo2.rich_content(input_key='text')
data = [{'text': 'This is a test.'}]
res = op(data)
print(res)
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
from lazyllm.tools.data import Demo2

op = Demo2.error_prone_op(input_key='text', _save_data=True, _concurrency_mode='single')
data = [{'text': 'ok'}, {'text': 'fail'}, {'text': 'ok2'}]
res = op(data)
print(res)
# [{'text': 'Processed: ok'}, {'text': 'Processed: ok2'}]
# valid results skip the failed item; error details written to error file
```
""")

# text2qa_ops module docs
add_chinese_doc('data.operators.text2qa_ops.TextToChunks', """\
将输入文本按行切分为多个块（chunk），每条输入可展开为多条输出。支持按 token 数或字符数控制块大小，可选用 tokenizer 或按字符计数。

Args:
    input_key (str): 输入文本字段名，默认 'content'
    output_key (str): 输出块内容写入的字段名，默认 'chunk'
    chunk_size (int): 每块的最大长度（token 数或字符数），默认 10
    tokenize (bool): 是否按 token 计数；为 True 且未提供 tokenizer 时使用默认 Qwen tokenizer
    tokenizer: 可选，用于计数的 tokenizer；None 时若 tokenize=True 则自动加载默认
    **kwargs: 其它基类参数（如 _concurrency_mode、_max_workers 等）
""")

add_english_doc('data.operators.text2qa_ops.TextToChunks', """\
Split input text into chunks by lines, with size controlled by token or character count. One input item may expand into multiple output items. Supports optional tokenizer or character-based length.

Args:
    input_key (str): key of the input text field, default 'content'
    output_key (str): key to write each chunk into, default 'chunk'
    chunk_size (int): max length per chunk (tokens or chars), default 10
    tokenize (bool): whether to count by tokens; if True and tokenizer not provided, uses default Qwen tokenizer
    tokenizer: optional tokenizer for counting; if None and tokenize=True, loads default
    **kwargs: other base-class args (e.g. _concurrency_mode, _max_workers)
""")

add_example('data.operators.text2qa_ops.TextToChunks', """\
```python
from lazyllm.tools.data import Text2qa

op = Text2qa.TextToChunks(input_key='content', output_key='chunk', chunk_size=10, tokenize=False)
data = [{'content': 'line1\nline2\nline3\nline4'}]
res = op(data)
print(res)
# [{'content': 'line1\nline2\nline3\nline4', 'chunk': 'line1\nline2'}, {'content': 'line1\nline2\nline3\nline4', 'chunk': 'line3\nline4'}]
```
""")

add_chinese_doc('data.operators.text2qa_ops.empty_or_noise_filter', """\
过滤空内容或纯噪声数据。若指定字段为空或仅包含非字母/非 CJK 字符则丢弃该条（返回空列表），否则保留原数据。以 forward 单条方式注册。

Args:
    data (dict): 单条数据字典
    input_key (str): 要检查的字段名，默认 'chunk'
""")

add_english_doc('data.operators.text2qa_ops.empty_or_noise_filter', """\
Filter out empty or noise-only items. If the specified field is empty or contains no word/CJK characters, the item is dropped (returns empty list); otherwise the item is kept. Registered as a single-item forward.

Args:
    data (dict): single data dict
    input_key (str): key to check, default 'chunk'
""")

add_example('data.operators.text2qa_ops.empty_or_noise_filter', """\
```python
from lazyllm.tools.data import Text2qa

op = Text2qa.empty_or_noise_filter(input_key='chunk')
data = [{'chunk': 'hello'}, {'chunk': ''}, {'chunk': '\n'}]
res = op(data)
print(res)
# [{'chunk': 'hello'}]
```
""")

add_chinese_doc('data.operators.text2qa_ops.invalid_unicode_cleaner', """\
清除指定文本字段中的无效 Unicode 码位（如 FDD0–FDEF、FFFE/FFFF 及若干 Supplementary Special Purpose 区段），原地修改并返回数据。以 forward 单条方式注册。

Args:
    data (dict): 单条数据字典
    input_key (str): 要清洗的文本字段名，默认 'chunk'
""")

add_english_doc('data.operators.text2qa_ops.invalid_unicode_cleaner', """\
Remove invalid Unicode code points (e.g. FDD0–FDEF, FFFE/FFFF and certain Supplementary Special Purpose ranges) from the specified text field in place. Registered as a single-item forward.

Args:
    data (dict): single data dict
    input_key (str): key of the text field to clean, default 'chunk'
""")

add_example('data.operators.text2qa_ops.invalid_unicode_cleaner', """\
```python
from lazyllm.tools.data import Text2qa

op = Text2qa.invalid_unicode_cleaner(input_key='chunk')
data = {'chunk': 'valid text\uFFFE tail'}
res = op(data)  # 剔除乱码\uFFFE
print(res)
[{'chunk': 'valid text tail'}]
```
""")

add_chinese_doc('data.operators.text2qa_ops.ChunkToQA', """\
基于大模型将每个文本块生成一个 QA 对（问题 + 答案）。使用 JsonFormatter 约束输出格式，可自定义 user_prompt 或使用默认「根据下面文本生成一个 QA 对」。

Args:
    input_key (str): 输入块字段名，默认 'chunk'
    query_key (str): 生成的问题写入的字段名，默认 'query'
    answer_key (str): 生成的答案写入的字段名，默认 'answer'
    model: 可选，TrainableModule 或兼容接口；None 时使用默认 Qwen 模型
    user_prompt (str|None): 可选，用户提示前缀；None 时使用默认
    **kwargs: 其它基类参数
""")

add_english_doc('data.operators.text2qa_ops.ChunkToQA', """\
Use an LLM to generate one QA pair (question + answer) per text chunk. Output format is constrained via JsonFormatter; user_prompt can be customized or left as default.

Args:
    input_key (str): key of the input chunk, default 'chunk'
    query_key (str): key to write the generated question, default 'query'
    answer_key (str): key to write the generated answer, default 'answer'
    model: optional TrainableModule or compatible; None uses default Qwen model
    user_prompt (str|None): optional user prompt prefix; None uses default
    **kwargs: other base-class args
""")

add_example('data.operators.text2qa_ops.ChunkToQA', """\
```python
from lazyllm.tools.data import Text2qa
from lazyllm import OnlineChatModule

llm = OnlineChatModule()
op = Text2qa.ChunkToQA(input_key='chunk', query_key='query', answer_key='answer', model=llm)
data = [{'chunk': '今天是晴天！'}]
res = op(data)
print(res)
# [{'chunk': '今天是晴天！', 'query': '今天的天气怎么样？', 'answer': '今天是晴天！'}]
```
""")

add_chinese_doc('data.operators.text2qa_ops.QAScorer', """\
基于大模型对 QA 对进行打分：判断答案是否严格基于原文，输出 1（基于原文）或 0（否则）。使用 JsonFormatter 约束输出 score 字段。

Args:
    input_key (str): 原文块字段名，默认 'chunk'
    output_key (str): 分数写入的字段名，默认 'score'
    query_key (str): 问题字段名，默认 'query'
    answer_key (str): 答案字段名，默认 'answer'
    model: 可选，TrainableModule 或兼容接口；None 时使用默认 Qwen 模型
    user_prompt (str|None): 可选，用户提示；None 时使用默认规则（严格基于原文→1，否则→0）
    **kwargs: 其它基类参数
""")

add_english_doc('data.operators.text2qa_ops.QAScorer', """\
Use an LLM to score QA pairs: whether the answer is strictly grounded in the source chunk. Outputs 1 (grounded) or 0 (otherwise). Output format constrained via JsonFormatter.

Args:
    input_key (str): key of the source chunk, default 'chunk'
    output_key (str): key to write the score, default 'score'
    query_key (str): key of the question, default 'query'
    answer_key (str): key of the answer, default 'answer'
    model: optional TrainableModule or compatible; None uses default Qwen model
    user_prompt (str|None): optional user prompt; None uses default rules
    **kwargs: other base-class args
""")

add_example('data.operators.text2qa_ops.QAScorer', """\
```python
from lazyllm.tools.data import Text2qa
from lazyllm import OnlineChatModule

llm = OnlineChatModule()
op = Text2qa.QAScorer(input_key='chunk', output_key='score', query_key='query', answer_key='answer', model=llm)
data = [
{'chunk': '今天是晴天！', 'query': '今天的天气怎么样？', 'answer': '今天是晴天！'},
{'chunk': '1+1=2', 'query': '1+1=?', 'answer': '3'}
]
res = op(data)
print(res)
# [{'chunk': '今天是晴天！', 'query': '今天的天气怎么样？', 'answer': '今天是晴天！', 'score': 1}, {'chunk': '1+1=2', 'query': '1+1=?', 'answer': '3', 'score': 0}]
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
