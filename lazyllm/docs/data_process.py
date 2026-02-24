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

# cot_ops module docs
add_chinese_doc('data.operators.cot_ops.CoTGenerator', """\
使用大模型为问题生成带思维链（CoT）的推理过程，要求最终答案用 \\boxed{{ANSWER}} 包裹。输出写入指定字段。

Args:
    input_key (str): 输入问题字段名，默认 'query'
    output_key (str): 输出 CoT 答案字段名，默认 'cot_answer'
    model: 可选，TrainableModule 或兼容接口；None 时使用默认 Qwen 模型
    user_prompt (str|None): 可选，用户提示前缀；None 时使用默认
    **kwargs: 其它基类参数
""")

add_english_doc('data.operators.cot_ops.CoTGenerator', """\
Use an LLM to generate chain-of-thought reasoning for a question, with final answer wrapped in \\boxed{{ANSWER}}. Writes result to the specified output key.

Args:
    input_key (str): key of the input question, default 'query'
    output_key (str): key to write the CoT answer, default 'cot_answer'
    model: optional TrainableModule or compatible; None uses default Qwen model
    user_prompt (str|None): optional user prompt prefix; None uses default
    **kwargs: other base-class args
""")

add_example('data.operators.cot_ops.CoTGenerator', """\
```python
from lazyllm.tools.data import genCot
from lazyllm import OnlineChatModule

llm = OnlineChatModule()
op = genCot.CoTGenerator(input_key='query', output_key='cot_answer', model=llm)
data = {'query': 'What is 2+2?'}
res = op(data)  # each item gets 'cot_answer' with CoT and \\boxed{{4}}
print(res)
# {'query': 'What is 2+2?', 'cot_answer': '首先，我们需要理解加法的基本概念，即两个或多个数值的总和。在这个问题中，我们需要计算 2 和另一个 2 的和。\n\n第一步，我们识别出第一个数值是 2。\n\n第二步，我们识别出第二个数值也是 2。\n\n第三步，我们将这两个数值相加：2 + 2。\n\n第四步，我们进行计算：2 + 2 = 4。\n\n因此，最终答案是 4，使用规定的格式包裹答案。\n\n最终答案：\\boxed{4}'}
```
""")

add_chinese_doc('data.operators.cot_ops.SelfConsistencyCoTGenerator', """\
对同一问题采样多次 CoT，从 \\boxed{{}} 中提取答案并做多数投票，最终保留与多数答案一致的一条 CoT 输出。

Args:
    input_key (str): 输入问题字段名，默认 'query'
    output_key (str): 输出 CoT 答案字段名，默认 'cot_answer'
    num_samples (int): 采样次数，默认 5
    model: 可选；None 时使用默认 Qwen 模型
    user_prompt (str|None): 可选用户提示
    **kwargs: 其它基类参数
""")

add_english_doc('data.operators.cot_ops.SelfConsistencyCoTGenerator', """\
Sample multiple CoT answers for the same question, extract \\boxed{{}} answers, take majority vote, and output one CoT that matches the majority answer.

Args:
    input_key (str): key of the input question, default 'query'
    output_key (str): key to write the CoT answer, default 'cot_answer'
    num_samples (int): number of samples, default 5
    model: optional; None uses default Qwen model
    user_prompt (str|None): optional user prompt
    **kwargs: other base-class args
""")

add_example('data.operators.cot_ops.SelfConsistencyCoTGenerator', """\
```python
from lazyllm.tools.data import genCot
from lazyllm import OnlineChatModule

llm = OnlineChatModule()
op = genCot.SelfConsistencyCoTGenerator(
    input_key='query',
    output_key='cot_answer',
    num_samples=3,
    model=llm
)

data = {'query': 'What is 3*4?'}
res = op(data)
print(res)
# {'query': 'What is 3*4?', 'candidates': ['12', '12', '12'], 'cot_answer': '首先，我们需要理解问题的核心，即计算3乘以4的结果。\n\n1. 确定操作：这是一个乘法问题，我们需要将两个数相乘。\n2. 识别数字：问题中给出的两个数字是3和4。\n3. 执行乘法：将3乘以4，计算过程如下：\n   - 3 * 4 = 12\n\n因此，3乘以4的结果是12。\n\n最终答案为：\\boxed{12}'}
```
""")

add_chinese_doc('data.operators.cot_ops.answer_verify', """\
比较参考答案与模型提取答案是否（数学意义下）相等。使用 math_verify 解析并验证，结果写入指定字段。以 forward 单条方式注册。

Args:
    data (dict): 单条数据字典
    answer_key (str): 参考答案字段名，默认 'reference'
    infer_key (str): 模型提取答案字段名，默认 'llm_extracted'
    output_key (str): 是否相等写入的字段名，默认 'is_equal'
""")

add_english_doc('data.operators.cot_ops.answer_verify', """\
Compare reference answer and model-extracted answer for mathematical equality. Uses math_verify to parse and verify; result written to the specified key. Registered as single-item forward.

Args:
    data (dict): single data dict
    answer_key (str): key of reference answer, default 'reference'
    infer_key (str): key of LLM-extracted answer, default 'llm_extracted'
    output_key (str): key to write equality result, default 'is_equal'
""")

add_example('data.operators.cot_ops.answer_verify', """\
```python
from lazyllm.tools.data import genCot

data = {'reference': '1/2', 'llm_extracted': '0.5'}
op = genCot.answer_verify(answer_key='reference', infer_key='llm_extracted', output_key='is_equal')
print(op(data))  # Add key/value: 'is_equal': True
# {'reference': '1/2', 'llm_extracted': '0.5', 'is_equal': True}
```
""")

# enQa_ops module docs
add_chinese_doc('data.operators.enQa_ops.QueryRewriter', """\
使用大模型将原问题重写为多个语义一致、表达不同的问法，输出列表写入指定字段。

Args:
    input_key (str): 输入问题字段名，默认 'query'
    output_key (str): 重写问题列表写入的字段名，默认 'rewrite_querys'
    rewrite_num (int): 生成的重写数量，默认 3
    model: 可选；None 时使用默认 Qwen 模型
    user_prompt (str|None): 可选用户提示
    **kwargs: 其它基类参数
""")

add_english_doc('data.operators.enQa_ops.QueryRewriter', """\
Use an LLM to rewrite the original question into multiple semantically equivalent formulations. Writes a list to the specified output key.

Args:
    input_key (str): key of the input question, default 'query'
    output_key (str): key to write the list of rewrites, default 'rewrite_querys'
    rewrite_num (int): number of rewrites to generate, default 3
    model: optional; None uses default Qwen model
    user_prompt (str|None): optional user prompt
    **kwargs: other base-class args
""")

add_example('data.operators.enQa_ops.QueryRewriter', """\
```python
from lazyllm.tools.data import EnQA
from lazyllm import OnlineChatModule

llm = OnlineChatModule()
op = EnQA.QueryRewriter(input_key='query', output_key='rewrite_querys', rewrite_num=2, model=llm)
data = {'query': 'What is machine learning?'}
res = op(data)  # data gets 'rewrite_querys': [str, str, ...]
print(res)
# [{'query': 'What is machine learning?', 'rewrite_querys': ['Could you explain what machine learning is?', 'What does the term machine learning refer to?']}]
```
""")

add_chinese_doc('data.operators.enQa_ops.DiversityScorer', """\
对问题列表进行多样性打分，输出与输入顺序一致的列表，每项含 rewritten_query 与 diversity_score（0 相似/1 差异明显）。

Args:
    input_key (str): 问题列表字段名，默认 'rewrite_querys'
    output_key (str): 带多样性分数的列表写入的字段名，默认 'diversity_querys'
    model: 可选；None 时使用默认 Qwen 模型
    user_prompt (str|None): 可选用户提示
    **kwargs: 其它基类参数
""")

add_english_doc('data.operators.enQa_ops.DiversityScorer', """\
Score diversity of a list of questions; output list matches input order, each item has rewritten_query and diversity_score (0 similar / 1 diverse).

Args:
    input_key (str): key of the question list, default 'rewrite_querys'
    output_key (str): key to write the scored list, default 'diversity_querys'
    model: optional; None uses default Qwen model
    user_prompt (str|None): optional user prompt
    **kwargs: other base-class args
""")

add_example('data.operators.enQa_ops.DiversityScorer', """\
```python
from lazyllm.tools.data import EnQA
from lazyllm import OnlineChatModule

llm = OnlineChatModule()
op = EnQA.DiversityScorer(input_key='rewrite_querys', output_key='diversity_querys', model=llm)
data = {'rewrite_querys': ['今天是个好天气', '今天天气不错', 'It is a nice day!']}
res = op(data)
print(data)
# {'rewrite_querys': ['今天是个好天气', '今天天气不错', 'It is a nice day!'], 'diversity_querys': [{'rewritten_query': '今天是个好天气', 'diversity_score': 1}, {'rewritten_query': '今天天气不错', 'diversity_score': 1}, {'rewritten_query': 'It is a nice day!', 'diversity_score': 1}]}
```
""")

add_chinese_doc('data.operators.enQa_ops.post_processor', """\
将指定字段（列表 of dict）展开为多行：每项 dict 与原始 data 合并为一行，原列表字段移除。返回多行时以 list 形式；无数据返回 None。以 forward 单条方式注册。

Args:
    data (dict): 单条数据字典
    input_key (str): 要展开的列表字段名（列表中每项为 dict）
""")

add_english_doc('data.operators.enQa_ops.post_processor', """\
Expand the specified key (list of dicts) into multiple rows: each dict merged with original data as one row, list key removed. Returns list of rows or None if no data. Registered as single-item forward.

Args:
    data (dict): single data dict
    input_key (str): key of the list of dicts to expand
""")

add_example('data.operators.enQa_ops.post_processor', """\
```python
from lazyllm.tools.data import EnQA

data = {'rewrite_querys': ['今天是个好天气', '今天天气不错', 'It is a nice day!'], 'diversity_querys': [{'rewritten_query': '今天是个好天气', 'diversity_score': 1}, {'rewritten_query': '今天天气不错', 'diversity_score': 1}, {'rewritten_query': 'It is a nice day!', 'diversity_score': 1}]}
op = EnQA.post_processor(input_key='diversity_querys')
print(op(data))  
# [{'rewrite_querys': ['今天是个好天气', '今天天气不错', 'It is a nice day!'], 'rewritten_query': '今天是个好天气', 'diversity_score': 1}, {'rewrite_querys': ['今天是个好天气', '今天天气不错', 'It is a nice day!'], 'rewritten_query': '今天天气不错', 'diversity_score': 1}, {'rewrite_querys': ['今天是个好天气', '今天天气不错', 'It is a nice day!'], 'rewritten_query': 'It is a nice day!', 'diversity_score': 1}]
```
""")

add_chinese_doc('data.operators.enQa_ops.diversity_filter', """\
按多样性分数过滤：若 data 中指定字段（分数）小于 min_score 则丢弃该条（返回 []），否则保留（返回 None 表示保留原 data）。以 forward 单条方式注册。

Args:
    data (dict): 单条数据字典
    input_key (str): 分数所在字段名
    min_score: 最小分数阈值
""")

add_english_doc('data.operators.enQa_ops.diversity_filter', """\
Filter by diversity score: if the value at input_key is less than min_score, drop the item (return []); otherwise keep (return None to keep original data). Registered as single-item forward.

Args:
    data (dict): single data dict
    input_key (str): key holding the score
    min_score: minimum score threshold
""")

add_example('data.operators.enQa_ops.diversity_filter', """\
```python
from lazyllm.tools.data import EnQA

data = {'query': 'a and b', 'rewritten_query': 'b', 'diversity_score': 0}
op = EnQA.diversity_filter(input_key='diversity_score', min_score=1)
print(op(data))  # [None] (drop) 
# []
```
""")

# math_ops module docs
add_chinese_doc('data.operators.math_ops.math_answer_extractor', """\
从文本中提取 \\boxed{{}} 内的数学答案，写入指定输出字段。以 forward 单条方式注册。

Args:
    data (dict): 单条数据字典
    input_key (str): 含答案文本的字段名，默认 'answer'
    output_key (str): 提取结果写入的字段名，默认 'math_answer'
""")

add_english_doc('data.operators.math_ops.math_answer_extractor', """\
Extract the math answer inside \\boxed{{}} from text and write to the specified output key. Registered as single-item forward.

Args:
    data (dict): single data dict
    input_key (str): key of the text containing the answer, default 'answer'
    output_key (str): key to write the extracted value, default 'math_answer'
""")

add_example('data.operators.math_ops.math_answer_extractor', """\
```python
from lazyllm.tools.data import MathQA

data = {'answer': 'So the answer is \\\\boxed{{42}}.'}
op = MathQA.math_answer_extractor(input_key='answer', output_key='math_answer')
print(op(data))  # data['math_answer'] == '42'
# [{'answer': 'So the answer is \\\\boxed{{42}}.', 'math_answer': '{42}'}]
```
""")

add_chinese_doc('data.operators.math_ops.MathAnswerGenerator', """\
使用大模型为数学问题生成推理与答案，要求最终结果用 \\boxed{{ANSWER}} 包裹。若已有 answer 且未设置 regenerate 则跳过。

Args:
    input_key (str): 问题字段名，默认 'question'
    output_key (str): 答案写入的字段名，默认 'answer'
    regenerate_key (str): 是否强制重新生成的标志字段，默认 'regenerate'
    model: 可选；None 时使用默认 Qwen 模型
    user_prompt (str|None): 可选用户提示
    **kwargs: 其它基类参数
""")

add_english_doc('data.operators.math_ops.MathAnswerGenerator', """\
Use an LLM to generate reasoning and answer for a math question, with final result in \\boxed{{ANSWER}}. Skips if answer already exists and regenerate is not set.

Args:
    input_key (str): key of the question, default 'question'
    output_key (str): key to write the answer, default 'answer'
    regenerate_key (str): key for force-regenerate flag, default 'regenerate'
    model: optional; None uses default Qwen model
    user_prompt (str|None): optional user prompt
    **kwargs: other base-class args
""")

add_example('data.operators.math_ops.MathAnswerGenerator', """\
```python
from lazyllm.tools.data.operators.math_ops import MathAnswerGenerator

from lazyllm.tools.data import MathQA
from lazyllm import OnlineChatModule

llm = OnlineChatModule()
op = MathQA.MathAnswerGenerator(input_key='question', output_key='answer', model=llm)
data = [{'question': 'Solve 10 * 10'}]
res = op(data) 
print(res)
# [{'question': 'Solve 10 * 10', 'answer': '首先，我们需要计算 \\(10 \times 10\\)。这是一个简单的乘法运算，其中两个乘数都是10。\n\n步骤1：写下乘数10和另一个乘数10。\n步骤2：将两个10相乘。\n\n计算过程如下：\n\\[ 10 \times 10 = 100 \\]\n\n因此，最终结果是 \\(\\boxed{100}\\)。', 'regenerate': False}]
```
""")

add_chinese_doc('data.operators.math_ops.DifficultyEvaluator', """\
使用大模型判断数学问题难度，输出 Easy | Medium | Hard（小学/初中高中/大学及以上）。若已有 difficulty 则跳过。

Args:
    input_key (str): 问题字段名，默认 'question'
    output_key (str): 难度写入的字段名，默认 'difficulty'
    model: 可选；None 时使用默认 Qwen 模型
    user_prompt (str|None): 可选用户提示
    **kwargs: 其它基类参数
""")

add_english_doc('data.operators.math_ops.DifficultyEvaluator', """\
Use an LLM to evaluate math question difficulty; output Easy | Medium | Hard. Skips if difficulty already present.

Args:
    input_key (str): key of the question, default 'question'
    output_key (str): key to write difficulty, default 'difficulty'
    model: optional; None uses default Qwen model
    user_prompt (str|None): optional user prompt
    **kwargs: other base-class args
""")

add_example('data.operators.math_ops.DifficultyEvaluator', """\
```python
from lazyllm.tools.data.operators.math_ops import DifficultyEvaluator

from lazyllm.tools.data import MathQA
from lazyllm import OnlineChatModule

llm = OnlineChatModule()
op = MathQA.DifficultyEvaluator(input_key='question', output_key='difficulty', model=llm)
data = {'question': '1+1=?'}
res = op(data)  # each item gets 'difficulty': 'Easy'|'Medium'|'Hard'
print(res)
# [{'question': '1+1=?', 'difficulty': 'Easy'}]
```
""")

add_chinese_doc('data.operators.math_ops.DifficultyEvaluatorBatch', """\
批处理：统计输入列表中指定字段（难度）的分布，返回包含各难度计数的单元素列表 [{{难度: 数量}}]。以 forward_batch_input 注册。

Args:
    data (list[dict]): 输入数据列表
    input_key (str): 难度字段名，默认 'difficulty'
""")

add_english_doc('data.operators.math_ops.DifficultyEvaluatorBatch', """\
Batch: aggregate counts of the specified key (e.g. difficulty) over the input list; returns a single-element list [{{key: count}}]. Registered as forward_batch_input.

Args:
    data (list[dict]): list of input dicts
    input_key (str): key to aggregate, default 'difficulty'
""")

add_example('data.operators.math_ops.DifficultyEvaluatorBatch', """\
```python
from lazyllm.tools.data import MathQA

op = MathQA.DifficultyEvaluatorBatch(input_key='difficulty')
data = [{'difficulty': 'Easy'}, {'difficulty': 'Hard'}, {'difficulty': 'Easy'}]
print(op(data))  
# [{'Easy': 2, 'Hard': 1}]
```
""")

add_chinese_doc('data.operators.math_ops.QualityEvaluator', """\
使用大模型对问题-答案对做质量打分：0 表示需重新生成，1 表示合格。若已有 output_key 则跳过。

Args:
    question_key (str): 问题字段名，默认 'question'
    answer_key (str): 答案字段名，默认 'answer'
    output_key (str): 分数写入的字段名，默认 'score'
    model: 可选；None 时使用默认 Qwen 模型
    user_prompt (str|None): 可选用户提示
    **kwargs: 其它基类参数
""")

add_english_doc('data.operators.math_ops.QualityEvaluator', """\
Use an LLM to score question-answer quality: 0 = regenerate, 1 = acceptable. Skips if output_key already present.

Args:
    question_key (str): key of the question, default 'question'
    answer_key (str): key of the answer, default 'answer'
    output_key (str): key to write score, default 'score'
    model: optional; None uses default Qwen model
    user_prompt (str|None): optional user prompt
    **kwargs: other base-class args
""")

add_example('data.operators.math_ops.QualityEvaluator', """\
```python
from lazyllm.tools.data import MathQA
from lazyllm import OnlineChatModule

llm = OnlineChatModule()
op = MathQA.QualityEvaluator(question_key='question', answer_key='answer', output_key='score', model=llm)
data = {'question': '今天天气如何', 'answer': '大家好~'}
res = op(data) # 质量低的会被打 0 分
print(res)
# [{'question': '今天天气如何', 'answer': '大家好~', 'score': 0}]
```
""")

add_chinese_doc('data.operators.math_ops.DuplicateAnswerDetector', """\
检测答案是否存在重复/周期/长片段重复：周期重复、句子级重复、或合并问题+答案后的长子串重复则标记为 True。不调用模型。

Args:
    question_key (str): 问题字段名，默认 'question'
    answer_key (str): 答案字段名，默认 'answer'
    output_key (str): 是否重复写入的字段名，默认 'duplicate'
    min_repeat_len (int): 判定长重复的最小子串长度，默认 15
    repeat_threshold (int): 子串出现次数阈值，默认 2
    periodic_min_repeat (int): 周期重复的最小周期重复次数，默认 3
    **kwargs: 其它基类参数
""")

add_english_doc('data.operators.math_ops.DuplicateAnswerDetector', """\
Detect duplicate/periodic/long-repeat in answers: periodic repetition, sentence-level repeat, or long substring repeat in question+answer. Sets output True if detected. No model call.

Args:
    question_key (str): key of the question, default 'question'
    answer_key (str): key of the answer, default 'answer'
    output_key (str): key to write duplicate flag, default 'duplicate'
    min_repeat_len (int): min substring length for long repeat, default 15
    repeat_threshold (int): occurrence threshold for substring, default 2
    periodic_min_repeat (int): min period repeats for periodic, default 3
    **kwargs: other base-class args
""")

add_example('data.operators.math_ops.DuplicateAnswerDetector', """\
```python
from lazyllm.tools.data import MathQA

op = MathQA.DuplicateAnswerDetector(question_key='question', answer_key='answer', output_key='duplicate')
data = {'question': 'Q', 'answer': 'A' * 50}
res = op(data)  # data['duplicate'] True
print(res)
# [{'question': 'Q', 'answer': 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', 'duplicate': True}]
```
""")

add_chinese_doc('data.operators.math_ops.ReasoningAnswerTokenLengthFilter', """\
按 token 或字符长度过滤答案：超过 max_answer_token_length 时清空该字段并返回修改后的 data；未超过时返回 None 保留原样；无内容时返回 []。支持 tokenizer 或字符计数。

Args:
    input_key (str): 答案字段名，默认 'answer'
    max_answer_token_length (int): 最大允许长度，默认 300
    tokenize (bool): 是否按 token 计数；True 且未提供 tokenizer 时使用默认 Qwen tokenizer
    tokenizer: 可选
    **kwargs: 其它基类参数
""")

add_english_doc('data.operators.math_ops.ReasoningAnswerTokenLengthFilter', """\
Filter by answer length (tokens or chars): if over max_answer_token_length, clear the field and return modified data; if within limit return None to keep; if empty return []. Supports tokenizer or char count.

Args:
    input_key (str): key of the answer, default 'answer'
    max_answer_token_length (int): max allowed length, default 300
    tokenize (bool): whether to count by tokens; uses default Qwen tokenizer if True and tokenizer not provided
    tokenizer: optional
    **kwargs: other base-class args
""")

add_example('data.operators.math_ops.ReasoningAnswerTokenLengthFilter', """\
```python
from lazyllm.tools.data import MathQA

op = MathQA.ReasoningAnswerTokenLengthFilter(input_key='answer', max_answer_token_length=100, tokenize=False)
data = [{'answer': 'short'}]
print(op(data))  # less than the max_length, keep the original input
# [{'answer': 'short'}]
```
""")

add_chinese_doc('data.operators.math_ops.QuestionFusionGenerator', """\
使用大模型将多条问题融合为一个新问题并生成推理与 \\boxed{{}} 答案。需要 list_key 下至少 2 个问题。

Args:
    input_key (str): 融合后问题字段名，默认 'question'
    output_key (str): 推理结果/答案写入的字段名，默认 'answer'
    list_key (str): 问题列表字段名，默认 'question_list'
    model: 可选；None 时使用默认 Qwen 模型
    user_prompt (str|None): 可选用户提示
    **kwargs: 其它基类参数
""")

add_english_doc('data.operators.math_ops.QuestionFusionGenerator', """\
Use an LLM to fuse multiple questions into one and generate reasoning with \\boxed{{}} answer. Requires at least 2 questions under list_key.

Args:
    input_key (str): key for fused question, default 'question'
    output_key (str): key to write answer, default 'answer'
    list_key (str): key of the question list, default 'question_list'
    model: optional; None uses default Qwen model
    user_prompt (str|None): optional user prompt
    **kwargs: other base-class args
""")

add_example('data.operators.math_ops.QuestionFusionGenerator', """\
```python
from lazyllm.tools.data import MathQA
from lazyllm import OnlineChatModule

llm = OnlineChatModule()
op = MathQA.QuestionFusionGenerator(input_key='new_question', list_key='question_list', output_key='new_answer', model=llm)
data = {'question_list': [
    {'question': '1加1等于几？', 'answer': '1+1 = 2'}, 
    {'question': '2的平方等于几？', 'answer': '2*2 = 4'}]}
res = op(data) 
print(res)
# [{'question_list': [{'question': '1加1等于几？', 'answer': '1+1 = 2'}, {'question': '2的平方等于几？', 'answer': '2*2 = 4'}], 
# 'new_question': '如果1加1的结果与2的平方相比较，哪个更大？', 
# 'new_answer': '首先，我们解决第一个问题：1加1等于几？计算得到 1+1 = 2。然后，解决第二个问题：2的平方等于几？计算得到 2*2 = 4。最后，我们比较这两个结果，2和4。显然，4大于2。所以，2的平方更大。'}]
```
""")

# pdf_ops module docs
add_chinese_doc('data.operators.pdf_ops.Pdf2Md', """\
将 PDF 转为 Markdown 文档列表。通过 MineruPDFReader（需配置 reader_url）调用后端服务，支持缓存。

Args:
    input_key (str): PDF 路径字段名，默认 'pdf_path'
    output_key (str): 转换得到的文档列表写入的字段名，默认 'docs'
    reader_url: 必填，Mineru 阅读器服务 URL
    backend (str): 后端类型，默认 'vlm-vllm-async-engine'
    upload_mode (bool): 是否上传模式，默认 True
    use_cache (bool): 是否使用缓存，默认 False
    **kwargs: 其它基类参数
""")

add_english_doc('data.operators.pdf_ops.Pdf2Md', """\
Convert PDF to a list of Markdown documents. Uses MineruPDFReader (reader_url required). Supports cache.

Args:
    input_key (str): key of the PDF path, default 'pdf_path'
    output_key (str): key to write the document list, default 'docs'
    reader_url: required, Mineru reader service URL
    backend (str): backend type, default 'vlm-vllm-async-engine'
    upload_mode (bool): whether to use upload mode, default True
    use_cache (bool): whether to use cache, default False
    **kwargs: other base-class args
""")

add_example('data.operators.pdf_ops.Pdf2Md', """\
```python
from lazyllm.tools.data import Pdf2Qa
from lazyllm.tools.data.operators.pdf_ops import Pdf2Md

op = Pdf2Qa.Pdf2Md(input_key='pdf_path', output_key='docs', reader_url='http://...')
data = [{'pdf_path': '/path/to/file.pdf'}]
res = op(data)  # each item gets 'docs' (list of doc content)
```"""
)

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

# =========================
# Embedding Data Formatter
# =========================

add_chinese_doc('data.operators.embedding_synthesis.embedding_data_formatter.EmbeddingFormatFlagEmbedding', """\
将数据格式化为 FlagEmbedding 训练格式的算子。

该算子将输入的 query、pos（正样本）、neg（负样本）格式化为 FlagEmbedding 框架所需的训练数据格式。
支持添加指令（instruction）字段用于有监督的 Embedding 训练。

Args:
    instruction (str, optional): 指令文本，用于有监督训练场景。默认为 None。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含 query、pos、neg 和可选 prompt 字段的字典。
""")

add_english_doc('data.operators.embedding_synthesis.embedding_data_formatter.EmbeddingFormatFlagEmbedding', """\
An operator that formats data into FlagEmbedding training format.

This operator formats the input query, pos (positive samples), and neg (negative samples)
into the training data format required by the FlagEmbedding framework.
Supports adding an instruction field for supervised Embedding training.

Args:
    instruction (str, optional): Instruction text for supervised training scenarios. Defaults to None.
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    dict: A dictionary containing query, pos, neg, and optional prompt fields.
""")

add_example('data.operators.embedding_synthesis.embedding_data_formatter.EmbeddingFormatFlagEmbedding', """\
```python
from lazyllm.tools.data import embedding

op = embedding.EmbeddingFormatFlagEmbedding(instruction='Represent this sentence for searching relevant passages:')
result = op({'query': 'machine learning', 'pos': ['ML tutorial'], 'neg': ['cooking recipe']})
# Returns: {'query': 'machine learning', 'pos': ['ML tutorial'], 'neg': ['cooking recipe'], 'prompt': 'Represent this sentence for searching relevant passages:'}
```
""")

add_chinese_doc('data.operators.embedding_synthesis.embedding_data_formatter.EmbeddingFormatSentenceTransformers', """\
将数据格式化为 SentenceTransformers 三元组训练格式的算子。

该算子将输入的 query、pos（正样本）、neg（负样本）转换为 SentenceTransformers 框架所需的 anchor-positive-negative 三元组格式。
适用于 MultipleNegativesRankingLoss 等损失函数的训练。

Args:
    **kwargs (dict): 可选的参数，传递给父类。

Returns:
    List[dict]: 包含 anchor、positive、negative 字段的字典列表，每对正负样本生成一个三元组。
""")

add_english_doc('data.operators.embedding_synthesis.embedding_data_formatter.EmbeddingFormatSentenceTransformers', """\
An operator that formats data into SentenceTransformers triplet training format.

This operator converts the input query, pos (positive samples), and neg (negative samples)
into the anchor-positive-negative triplet format required by the SentenceTransformers framework.
Suitable for training with losses like MultipleNegativesRankingLoss.

Args:
    **kwargs (dict): Optional arguments passed to the parent class.

Returns:
    List[dict]: A list of dictionaries containing anchor, positive, and negative fields,
    with one triplet generated for each positive-negative pair.
""")

add_example('data.operators.embedding_synthesis.embedding_data_formatter.EmbeddingFormatSentenceTransformers', """\
```python
from lazyllm.tools.data import embedding

op = embedding.EmbeddingFormatSentenceTransformers()
result = op({'query': 'machine learning', 'pos': ['ML basics'], 'neg': ['cooking tips']})
# Returns: [{'anchor': 'machine learning', 'positive': 'ML basics', 'negative': 'cooking tips'}]
```
""")

add_chinese_doc('data.operators.embedding_synthesis.embedding_data_formatter.EmbeddingFormatTriplet', """\
将数据格式化为通用三元组格式的算子。

该算子将输入的 query、pos（正样本）、neg（负样本）转换为标准的三元组格式，
字段名为 query、positive、negative。适用于多种 Embedding 训练框架。

Args:
    **kwargs (dict): 可选的参数，传递给父类。

Returns:
    List[dict]: 包含 query、positive、negative 字段的字典列表，每对正负样本生成一个三元组。
""")

add_english_doc('data.operators.embedding_synthesis.embedding_data_formatter.EmbeddingFormatTriplet', """\
An operator that formats data into generic triplet format.

This operator converts the input query, pos (positive samples), and neg (negative samples)
into a standard triplet format with field names query, positive, and negative.
Compatible with various Embedding training frameworks.

Args:
    **kwargs (dict): Optional arguments passed to the parent class.

Returns:
    List[dict]: A list of dictionaries containing query, positive, and negative fields,
    with one triplet generated for each positive-negative pair.
""")

add_example('data.operators.embedding_synthesis.embedding_data_formatter.EmbeddingFormatTriplet', """\
```python
from lazyllm.tools.data import embedding

op = embedding.EmbeddingFormatTriplet()
result = op({'query': 'deep learning', 'pos': ['neural networks', 'AI'], 'neg': ['history', 'geography']})
# Returns list of triplets combining each positive with each negative
```
""")

add_chinese_doc('data.operators.embedding_synthesis.embedding_data_formatter.EmbeddingTrainTestSplitter', """\
将数据集分割为训练集和测试集的算子。

该算子对输入数据进行随机打乱，并按指定比例分割为训练集和测试集。
支持保存分割后的数据到 JSONL 文件，并可按指定键进行分层抽样。

Args:
    test_size (float): 测试集比例，默认为 0.1（即 10%）。
    seed (int): 随机种子，用于可复现的分割结果，默认为 42。
    stratify_key (str, optional): 分层抽样的键名，默认为 None。
    train_output_file (str, optional): 训练集输出文件路径，默认为 None。
    test_output_file (str, optional): 测试集输出文件路径，默认为 None。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    List[dict]: 包含训练集和测试集的所有样本，每个样本添加了 'split' 字段标记所属集合。
""")

add_english_doc('data.operators.embedding_synthesis.embedding_data_formatter.EmbeddingTrainTestSplitter', """\
An operator that splits dataset into training and test sets.

This operator randomly shuffles the input data and splits it into training and test sets
according to the specified ratio. Supports saving split data to JSONL files
and stratified sampling by a specified key.

Args:
    test_size (float): Proportion of test set, defaults to 0.1 (i.e., 10%).
    seed (int): Random seed for reproducible splitting, defaults to 42.
    stratify_key (str, optional): Key name for stratified sampling, defaults to None.
    train_output_file (str, optional): Output file path for training set, defaults to None.
    test_output_file (str, optional): Output file path for test set, defaults to None.
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    List[dict]: All samples from both training and test sets, with a 'split' field added
to indicate which set each sample belongs to.
""")

add_example('data.operators.embedding_synthesis.embedding_data_formatter.EmbeddingTrainTestSplitter', """\
```python
from lazyllm.tools.data import embedding

op = embedding.EmbeddingTrainTestSplitter(test_size=0.2, seed=123, train_output_file='train.jsonl', test_output_file='test.jsonl')
data = [{'query': 'q1', 'pos': 'p1'}, {'query': 'q2', 'pos': 'p2'}, {'query': 'q3', 'pos': 'p3'}]
result = op(data)
# Returns all samples with 'split' field ('train' or 'test')
# Saves train data to train.jsonl and test data to test.jsonl
```
""")

# =========================
# Embedding Hard Negative Miner
# =========================

add_chinese_doc('data.operators.embedding_synthesis.embedding_hard_negative_miner.build_embedding_corpus', """\
构建 Embedding 训练所需的语料库。

该函数从输入数据中提取正样本和语料字段，构建一个唯一的语料库，并将其保存到文件中。
支持使用外部语料库，如果提供了 corpus 参数，则直接使用外部语料库。

Args:
    inputs (List[dict]): 输入数据列表，每条数据应包含正样本和可选的语料字段。
    input_pos_key (str): 正样本字段名，默认为 'pos'。
    corpus_key (str): 语料字段名，默认为 'passage'。
    corpus (List[str], optional): 外部语料库，如果提供则直接使用。默认为 None。
    corpus_dir (str, optional): 语料库保存目录，默认为临时目录。

Returns:
    List[dict]: 原始输入数据，每条数据添加了 '_corpus' 字段指向语料库文件路径。
""")

add_english_doc('data.operators.embedding_synthesis.embedding_hard_negative_miner.build_embedding_corpus', """\
Build corpus for Embedding training.

This function extracts positive samples and corpus fields from input data to build a unique corpus and saves it to a file.
Supports using external corpus; if corpus parameter is provided, it will be used directly.

Args:
    inputs (List[dict]): List of input data, each should contain positive samples and optional corpus field.
    input_pos_key (str): Key name for positive samples, defaults to 'pos'.
    corpus_key (str): Key name for corpus field, defaults to 'passage'.
    corpus (List[str], optional): External corpus, used directly if provided. Defaults to None.
    corpus_dir (str, optional): Directory to save corpus, defaults to temp directory.

Returns:
    List[dict]: Original input data with '_corpus' field added pointing to corpus file path.
""")

add_example('data.operators.embedding_synthesis.embedding_hard_negative_miner.build_embedding_corpus', """\
```python
from lazyllm.tools.data.operators.embedding_synthesis.embedding_hard_negative_miner import build_embedding_corpus

data = [{'query': 'machine learning', 'pos': ['ML tutorial', 'deep learning']}, {'query': 'cooking', 'pos': ['recipe']}]
result = build_embedding_corpus(data, input_pos_key='pos')
# Returns data with '_corpus' field pointing to corpus file containing unique passages
```
""")

add_chinese_doc('data.operators.embedding_synthesis.embedding_hard_negative_miner.EmbeddingInitBM25', """\
初始化 BM25 索引的算子。

该算子基于语料库构建 BM25 索引，用于后续的关键词检索和困难负样本挖掘。
支持中英文分词，使用 jieba 进行中文分词，Stemmer 进行英文词干提取。

Args:
    language (str): 语言类型，'zh' 表示中文，'en' 表示英文，默认为 'zh'。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    List[dict]: 输入数据，每条数据添加了 BM25 索引和相关配置信息。
""")

add_english_doc('data.operators.embedding_synthesis.embedding_hard_negative_miner.EmbeddingInitBM25', """\
An operator that initializes BM25 index.

This operator builds BM25 index based on corpus for subsequent keyword retrieval and hard negative mining.
Supports Chinese and English tokenization, using jieba for Chinese and Stemmer for English stemming.

Args:
    language (str): Language type, 'zh' for Chinese, 'en' for English, defaults to 'zh'.
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    List[dict]: Input data with BM25 index and related configuration added to each item.
""")

add_example('data.operators.embedding_synthesis.embedding_hard_negative_miner.EmbeddingInitBM25', """\
```python
from lazyllm.tools.data import embedding

# First build corpus, then initialize BM25
corpus_op = embedding.build_embedding_corpus(input_pos_key='pos')
bm25_op = embedding.EmbeddingInitBM25(language='zh')
# Returns data with '_bm25' index and tokenizer configuration
```
""")

add_chinese_doc('data.operators.embedding_synthesis.embedding_hard_negative_miner.EmbeddingInitSemantic', """\
初始化语义嵌入向量的算子。

该算子使用 Embedding 服务计算语料库中所有文档的向量表示，并保存到文件中。
用于后续的语义相似度计算和困难负样本挖掘。

Args:
    embedding_serving (Callable): Embedding 服务调用函数，用于计算文本向量。
    embeddings_dir (str, optional): 向量文件保存目录，默认为语料库所在目录。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    List[dict]: 输入数据，每条数据添加了语义向量文件路径和语料库信息。
""")

add_english_doc('data.operators.embedding_synthesis.embedding_hard_negative_miner.EmbeddingInitSemantic', """\
An operator that initializes semantic embeddings.

This operator uses Embedding service to compute vector representations for all documents in the corpus
and saves them to files. Used for subsequent semantic similarity calculation and hard negative mining.

Args:
    embedding_serving (Callable): Embedding service callable for computing text vectors.
    embeddings_dir (str, optional): Directory to save embedding files, defaults to corpus directory.
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    List[dict]: Input data with semantic embedding file paths and corpus information added.
""")

add_example('data.operators.embedding_synthesis.embedding_hard_negative_miner.EmbeddingInitSemantic', """\
```python
from lazyllm.tools.data import embedding

# Assuming my_embedding_fn is an embedding service
semantic_op = embedding.EmbeddingInitSemantic(embedding_serving=my_embedding_fn)
# Returns data with '_semantic_embeddings_path' pointing to saved embeddings
```
""")

add_chinese_doc('data.operators.embedding_synthesis.embedding_hard_negative_miner.mine_bm25_negatives', """\
使用 BM25 算法挖掘困难负样本的函数。

该函数基于 BM25 索引，检索与查询最相关但不属于正样本的文档作为负样本。
适用于挖掘与查询有词汇重叠但语义不同的困难负样本。

Args:
    data (dict): 单条输入数据，应包含 query、pos 和 BM25 索引信息。
    num_negatives (int): 需要挖掘的负样本数量，默认为 7。
    input_query_key (str): 查询字段名，默认为 'query'。
    input_pos_key (str): 正样本字段名，默认为 'pos'。
    output_neg_key (str): 负样本输出字段名，默认为 'neg'。

Returns:
    dict: 输入数据，添加了挖掘到的负样本列表。
""")

add_english_doc('data.operators.embedding_synthesis.embedding_hard_negative_miner.mine_bm25_negatives', """\
A function that mines hard negative samples using BM25 algorithm.

This function retrieves documents most relevant to the query but not in positive samples as negative samples,
based on BM25 index. Suitable for mining hard negatives that have lexical overlap but different semantics.

Args:
    data (dict): Single input data, should contain query, pos, and BM25 index information.
    num_negatives (int): Number of negative samples to mine, defaults to 7.
    input_query_key (str): Key name for query field, defaults to 'query'.
    input_pos_key (str): Key name for positive samples field, defaults to 'pos'.
    output_neg_key (str): Key name for output negative samples field, defaults to 'neg'.

Returns:
    dict: Input data with mined negative samples list added.
""")

add_example('data.operators.embedding_synthesis.embedding_hard_negative_miner.mine_bm25_negatives', """\
```python
from lazyllm.tools.data.operators.embedding_synthesis.embedding_hard_negative_miner import mine_bm25_negatives

# After building corpus and initializing BM25
data = {'query': 'machine learning', 'pos': ['ML tutorial'], '_bm25': bm25_index, '_bm25_corpus': corpus}
result = mine_bm25_negatives(data, num_negatives=5)
# Returns data with 'neg' field containing BM25-mined negative samples
```
""")

add_chinese_doc('data.operators.embedding_synthesis.embedding_hard_negative_miner.mine_random_negatives', """\
随机挖掘负样本的函数。

该函数从语料库中随机选择不属于正样本的文档作为负样本。
适用于基线对比或需要随机负样本的场景。

Args:
    data (dict): 单条输入数据，应包含 query、pos 和语料库信息。
    num_negatives (int): 需要挖掘的负样本数量，默认为 7。
    seed (int): 随机种子，用于可复现的随机选择，默认为 42。
    input_query_key (str): 查询字段名，默认为 'query'。
    input_pos_key (str): 正样本字段名，默认为 'pos'。
    output_neg_key (str): 负样本输出字段名，默认为 'neg'。

Returns:
    dict: 输入数据，添加了随机选择的负样本列表。
""")

add_english_doc('data.operators.embedding_synthesis.embedding_hard_negative_miner.mine_random_negatives', """\
A function that mines random negative samples.

This function randomly selects documents from corpus that are not in positive samples as negative samples.
Suitable for baseline comparison or scenarios requiring random negatives.

Args:
    data (dict): Single input data, should contain query, pos, and corpus information.
    num_negatives (int): Number of negative samples to mine, defaults to 7.
    seed (int): Random seed for reproducible selection, defaults to 42.
    input_query_key (str): Key name for query field, defaults to 'query'.
    input_pos_key (str): Key name for positive samples field, defaults to 'pos'.
    output_neg_key (str): Key name for output negative samples field, defaults to 'neg'.

Returns:
    dict: Input data with randomly selected negative samples list added.
""")

add_example('data.operators.embedding_synthesis.embedding_hard_negative_miner.mine_random_negatives', """\
```python
from lazyllm.tools.data.operators.embedding_synthesis.embedding_hard_negative_miner import mine_random_negatives

data = {'query': 'machine learning', 'pos': ['ML tutorial'], '_corpus': corpus_path}
result = mine_random_negatives(data, num_negatives=5, seed=123)
# Returns data with 'neg' field containing randomly selected negative samples
```
""")

add_chinese_doc('data.operators.embedding_synthesis.embedding_hard_negative_miner.EmbeddingMineSemanticNegatives', """\
使用语义相似度挖掘困难负样本的算子。

该算子基于语义向量相似度，找出与查询最相似但不属于正样本的文档作为负样本。
适用于挖掘语义相近但实际不相关的困难负样本，通常比 BM25 方法效果更好。

Args:
    num_negatives (int): 需要挖掘的负样本数量，默认为 7。
    embedding_serving (Callable): Embedding 服务调用函数，用于计算查询向量。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 输入数据，添加了基于语义相似度挖掘的负样本列表。
""")

add_english_doc('data.operators.embedding_synthesis.embedding_hard_negative_miner.EmbeddingMineSemanticNegatives', """\
An operator that mines hard negative samples using semantic similarity.

This operator finds documents most similar to the query but not in positive samples based on semantic vector similarity.
Suitable for mining hard negatives that are semantically similar but actually irrelevant,
usually performs better than BM25 method.

Args:
    num_negatives (int): Number of negative samples to mine, defaults to 7.
    embedding_serving (Callable): Embedding service callable for computing query vectors.
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    dict: Input data with negative samples mined based on semantic similarity added.
""")

add_example('data.operators.embedding_synthesis.embedding_hard_negative_miner.EmbeddingMineSemanticNegatives', """\
```python
from lazyllm.tools.data import embedding

# Assuming embeddings are initialized
semantic_miner = embedding.EmbeddingMineSemanticNegatives(num_negatives=5, embedding_serving=my_embedding_fn)
data = {'query': 'machine learning', 'pos': ['ML tutorial'], '_semantic_embeddings_path': emb_path, '_semantic_corpus': corpus}
result = semantic_miner(data)
# Returns data with 'neg' field containing semantically similar negative samples
```
""")


add_chinese_doc('data.operators.embedding_synthesis.embedding_query_generator.EmbeddingGenerateQueries', """\
使用 LLM 生成查询的算子。

该算子调用语言模型服务，基于构建的提示生成查询。返回 JSON 格式的查询响应。

Args:
    llm: LLM 服务实例，用于生成查询。
    num_queries (int): 要生成的查询数量，默认为 3。
    lang (str): 语言，'zh' 表示中文，'en' 表示英文，默认为 'zh'。
    query_types (List[str], optional): 查询类型列表，默认为 ['factual', 'semantic', 'inferential']。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 输入数据，添加了 '_query_response' 字段包含生成的查询响应。
""")

add_english_doc('data.operators.embedding_synthesis.embedding_query_generator.EmbeddingGenerateQueries', """\
An operator that generates queries using LLM.

This operator calls a language model service to generate queries based on the built prompts. Returns the query response in JSON format.

Args:
    llm: LLM service instance for generating queries.
    num_queries (int): Number of queries to generate, defaults to 3.
    lang (str): Language, 'zh' for Chinese, 'en' for English, defaults to 'zh'.
    query_types (List[str], optional): List of query types, defaults to ['factual', 'semantic', 'inferential'].
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    dict: Input data with '_query_response' field added containing the generated query response.
""")

add_example('data.operators.embedding_synthesis.embedding_query_generator.EmbeddingGenerateQueries', """\
```python
from lazyllm.tools.data import embedding

# Assuming llm is an LLM service instance
generator = embedding.EmbeddingGenerateQueries(llm=llm, lang='zh')
data = {'_query_prompt': 'Generate queries for: machine learning tutorial'}
result = generator(data)
# Returns data with '_query_response' field containing JSON queries
```
""")

add_chinese_doc('data.operators.embedding_synthesis.embedding_query_generator.EmbeddingParseQueries', """\
解析生成的查询的算子。

该算子解析 LLM 生成的查询响应，将每条查询展开为独立的数据记录。

Args:
    input_key (str): 输入字段名，默认为 'passage'。
    output_query_key (str): 输出查询字段名，默认为 'query'。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    List[dict]: 解析后的查询列表，每个查询为一个独立的数据记录。
""")

add_english_doc('data.operators.embedding_synthesis.embedding_query_generator.EmbeddingParseQueries', """\
An operator that parses generated queries.

This operator parses the query response generated by LLM and expands each query into an independent data record.

Args:
    input_key (str): Input field name, defaults to 'passage'.
    output_query_key (str): Output query field name, defaults to 'query'.
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    List[dict]: List of parsed queries, each query as an independent data record.
""")

add_example('data.operators.embedding_synthesis.embedding_query_generator.EmbeddingParseQueries', """\
```python
from lazyllm.tools.data import embedding

parser = embedding.EmbeddingParseQueries(input_key='passage', output_query_key='query')
data = {'_query_response': '[{"query": "what is ML?", "type": "factual"}]', 'passage': 'Machine learning is...'}
result = parser(data)
# Returns list of expanded query records with 'query' and 'pos' fields
```
""")

# =========================
# File/URL to Markdown Converter API
# =========================

add_chinese_doc('data.operators.knowledge_cleaning.file_or_url_to_markdown_converter_api.FileOrURLNormalizer', """\
文件或URL标准化算子。

该算子根据输入类型（文件或URL）自动识别文件格式，进行标准化处理。
支持PDF、HTML/XML、TXT/MD等文件格式，以及网页URL。对于网络PDF，会先下载到本地。

Args:
    intermediate_dir (str): 中间文件保存目录，默认为 'intermediate'。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 标准化后的数据，包含以下字段：
    - _type: 文件类型 ('pdf', 'html', 'text', 'invalid', 'unsupported')
    - _raw_path: 本地文件路径（如果有）
    - _url: URL地址（如果是网页）
    - _output_path: 预期的Markdown输出路径
    - _error: 错误信息（如果有）
""")

add_english_doc('data.operators.knowledge_cleaning.file_or_url_to_markdown_converter_api.FileOrURLNormalizer', """\
File or URL normalizer operator.

This operator automatically identifies file format based on input type (file or URL) and performs normalization.
Supports PDF, HTML/XML, TXT/MD files, and web URLs. For network PDFs, they will be downloaded locally first.

Args:
    intermediate_dir (str): Directory for intermediate files, defaults to 'intermediate'.
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    dict: Normalized data containing the following fields:
    - _type: File type ('pdf', 'html', 'text', 'invalid', 'unsupported')
    - _raw_path: Local file path (if available)
    - _url: URL address (if web page)
    - _output_path: Expected Markdown output path
    - _error: Error message (if any)
""")

add_example('data.operators.knowledge_cleaning.file_or_url_to_markdown_converter_api.FileOrURLNormalizer', """\
```python
from lazyllm.tools.data import kbc

normalizer = kbc.FileOrURLNormalizer(intermediate_dir='./temp')

# For file input
data = {'source': '/path/to/document.pdf'}
result = normalizer(data)
# Returns: {'source': '/path/to/document.pdf', '_type': 'pdf', '_raw_path': '/path/to/document.pdf', '_output_path': './temp/document.md'}

# For URL input
data = {'source': 'https://example.com/page.html'}
result = normalizer(data)
# Returns: {'source': 'https://example.com/page.html', '_type': 'html', '_url': 'https://example.com/page.html', '_output_path': './temp/url_xxx.md'}
```
""")

add_chinese_doc('data.operators.knowledge_cleaning.file_or_url_to_markdown_converter_api.HTMLToMarkdownConverter', """\
HTML转Markdown转换器算子。

该算子使用trafilatura库从HTML或XML文件中提取内容并转换为Markdown格式。
支持本地HTML文件和网络URL，会自动处理页面元数据。

Args:
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 转换后的数据，包含以下字段：
    - _markdown_path: 生成的Markdown文件路径
""")

add_english_doc('data.operators.knowledge_cleaning.file_or_url_to_markdown_converter_api.HTMLToMarkdownConverter', """\
HTML to Markdown converter operator.

This operator uses the trafilatura library to extract content from HTML or XML files and convert to Markdown format.
Supports local HTML files and web URLs, automatically handles page metadata.

Args:
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    dict: Converted data containing the following fields:
    - _markdown_path: Path to the generated Markdown file
""")

add_example('data.operators.knowledge_cleaning.file_or_url_to_markdown_converter_api.HTMLToMarkdownConverter', """\
```python
from lazyllm.tools.data import kbc

converter = kbc.HTMLToMarkdownConverter()

# After normalization
data = {'_type': 'html', '_url': 'https://example.com/article', '_output_path': './temp/output.md'}
result = converter(data)
# Returns: {'_type': 'html', '_url': 'https://example.com/article', '_output_path': './temp/output.md', '_markdown_path': './temp/output.md'}
```
""")

add_chinese_doc('data.operators.knowledge_cleaning.file_or_url_to_markdown_converter_api.PDFToMarkdownConverterAPI', """\
PDF转Markdown转换器API算子。

该算子使用MinerU服务将PDF文件（包括扫描件和图片）转换为Markdown格式。
支持通过API调用MinerU进行PDF解析，可配置后端引擎和上传模式。

Args:
    mineru_url (str): MinerU服务URL地址。
    mineru_backend (str): MinerU后端引擎类型，默认为 'vlm-vllm-async-engine'。
    upload_mode (bool): 是否使用上传模式，默认为 True。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 转换后的数据，包含以下字段：
    - _markdown_path: 生成的Markdown文件路径
""")

add_english_doc('data.operators.knowledge_cleaning.file_or_url_to_markdown_converter_api.PDFToMarkdownConverterAPI', """\
PDF to Markdown converter API operator.

This operator uses the MinerU service to convert PDF files (including scanned documents and images) to Markdown format.
Supports calling MinerU via API for PDF parsing, with configurable backend engine and upload mode.

Args:
    mineru_url (str): MinerU service URL address.
    mineru_backend (str): MinerU backend engine type, defaults to 'vlm-vllm-async-engine'.
    upload_mode (bool): Whether to use upload mode, defaults to True.
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    dict: Converted data containing the following fields:
    - _markdown_path: Path to the generated Markdown file
""")

add_example('data.operators.knowledge_cleaning.file_or_url_to_markdown_converter_api.PDFToMarkdownConverterAPI', """\
```python
from lazyllm.tools.data import kbc

converter = kbc.PDFToMarkdownConverterAPI(
    mineru_url='your_mineru_url',
    mineru_backend='vlm-vllm-async-engine',
    upload_mode=True
)

# After normalization
data = {'_type': 'pdf', '_raw_path': '/path/to/doc.pdf', '_output_path': './temp/output.md'}
result = converter(data)
# Returns: {'_type': 'pdf', '_raw_path': '/path/to/doc.pdf', '_output_path': './temp/output.md', '_markdown_path': './temp/output.md'}
```
""")

# =========================
# KBC Chunk Generator Batch
# =========================

add_chinese_doc('data.operators.knowledge_cleaning.kbc_chunk_generator_batch.KBCLoadText', """\
加载文本文件内容的算子。

该算子从指定路径加载文本文件内容，支持多种文件格式：
- .txt, .md, .xml: 直接读取文本内容
- .json, .jsonl: 从指定的文本字段中提取内容并合并

Args:
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含加载结果的数据：
    - _text_content: 加载的文本内容
    - _load_error: 加载错误信息（如果有）
""")

add_english_doc('data.operators.knowledge_cleaning.kbc_chunk_generator_batch.KBCLoadText', """\
Operator for loading text file content.

This operator loads text file content from the specified path, supporting multiple file formats:
- .txt, .md, .xml: Direct text content reading
- .json, .jsonl: Extract and merge content from specified text fields

Args:
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    dict: Data containing loading results:
    - _text_content: Loaded text content
    - _load_error: Loading error message (if any)
""")

add_example('data.operators.knowledge_cleaning.kbc_chunk_generator_batch.KBCLoadText', """\
```python
from lazyllm.tools.data import kbc

loader = kbc.KBCLoadText()

# Load text file
data = {'text_path': '/path/to/document.txt'}
result = loader(data)
# Returns: {'text_path': '/path/to/document.txt', '_text_content': 'file content...'}

# Load JSON file
data = {'text_path': '/path/to/data.json'}
result = loader(data)
# Returns: {'text_path': '/path/to/data.json', '_text_content': 'extracted text...'}
```
""")

add_chinese_doc('data.operators.knowledge_cleaning.kbc_chunk_generator_batch.KBCChunkText', """\
文本分块算子。

该算子将长文本分割成小块（chunks），支持多种分块策略：
- token: 基于Token数量分块
- sentence: 基于句子边界分块
- semantic: 基于语义相似度分块
- recursive: 递归分块

Args:
    chunk_size (int): 每个块的最大大小，默认为 512。
    chunk_overlap (int): 块之间的重叠大小，默认为 50。
    split_method (str): 分块方法，可选 'token', 'sentence', 'semantic', 'recursive'，默认为 'token'。
    tokenizer_name (str): 使用的tokenizer名称，默认为 'bert-base-uncased'。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含分块结果的数据：
    - _chunks: 分块后的文本列表
    - _chunk_error: 分块错误信息（如果有）
""")

add_english_doc('data.operators.knowledge_cleaning.kbc_chunk_generator_batch.KBCChunkText', """\
Text chunking operator.

This operator splits long text into chunks, supporting multiple chunking strategies:
- token: Token-based chunking
- sentence: Sentence boundary-based chunking
- semantic: Semantic similarity-based chunking
- recursive: Recursive chunking

Args:
    chunk_size (int): Maximum size of each chunk, defaults to 512.
    chunk_overlap (int): Overlap size between chunks, defaults to 50.
    split_method (str): Chunking method, options: 'token', 'sentence', 'semantic', 'recursive', defaults to 'token'.
    tokenizer_name (str): Name of the tokenizer to use, defaults to 'bert-base-uncased'.
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    dict: Data containing chunking results:
    - _chunks: List of chunked texts
    - _chunk_error: Chunking error message (if any)
""")

add_example('data.operators.knowledge_cleaning.kbc_chunk_generator_batch.KBCChunkText', """\
```python
from lazyllm.tools.data import kbc

chunker = kbc.KBCChunkText(chunk_size=512, chunk_overlap=50, split_method='token')

data = {'_text_content': 'Long text content that needs to be chunked...'}
result = chunker(data)
# Returns: {'_text_content': 'Long text content...', '_chunks': ['chunk1', 'chunk2', ...]}
```
""")

add_chinese_doc('data.operators.knowledge_cleaning.kbc_chunk_generator_batch.KBCSaveChunks', """\
保存文本分块结果的算子。

该算子将分块后的文本保存为JSON文件，每个分块作为一个JSON对象。
支持指定输出目录，会保留原始文件的相对路径结构。

Args:
    output_dir (str, optional): 输出目录路径，默认为 None（保存到原文件所在目录的 'extract' 子目录）。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含保存结果的数据：
    - chunk_path: 保存的JSON文件路径
""")

add_english_doc('data.operators.knowledge_cleaning.kbc_chunk_generator_batch.KBCSaveChunks', """\
Operator for saving text chunking results.

This operator saves chunked texts as JSON files, with each chunk as a JSON object.
Supports specifying output directory, preserving the relative path structure of the original file.

Args:
    output_dir (str, optional): Output directory path, defaults to None (save to 'extract' subdirectory of the original file's directory).
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    dict: Data containing save results:
    - chunk_path: Path to the saved JSON file
""")

add_example('data.operators.knowledge_cleaning.kbc_chunk_generator_batch.KBCSaveChunks', """\
```python
from lazyllm.tools.data import kbc

saver = kbc.KBCSaveChunks(output_dir='./output')

data = {'text_path': '/path/to/doc.txt', '_chunks': ['chunk1', 'chunk2']}
result = saver(data)
# Returns: {'text_path': '/path/to/doc.txt', 'chunk_path': './output/path/to/doc_chunk.json'}
```
""")

# =========================
# KBC Chunk Generator
# =========================

add_chinese_doc('data.operators.knowledge_cleaning.kbc_chunk_generator.KBCExpandChunks', """\
将分块文本展开为独立记录的算子。

该算子将包含多个文本分块的数据记录展开为多个独立的数据记录，每个记录包含一个分块。
适用于需要将分块后的文本作为独立样本进行后续处理的场景。

Args:
    output_key (str): 输出字段名，用于存储分块文本，默认为 'raw_chunk'。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    List[dict]: 展开后的独立数据记录列表，每个记录包含一个分块。
""")

add_english_doc('data.operators.knowledge_cleaning.kbc_chunk_generator.KBCExpandChunks', """\
Operator that expands chunked text into independent records.

This operator expands data records containing multiple text chunks into multiple independent data records,
with each record containing one chunk. Suitable for scenarios where chunked texts need to be processed
as independent samples.

Args:
    output_key (str): Output key name for storing chunk text, defaults to 'raw_chunk'.
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    List[dict]: List of expanded independent data records, each containing one chunk.
""")

add_example('data.operators.knowledge_cleaning.kbc_chunk_generator.KBCExpandChunks', """\
```python
from lazyllm.tools.data import kbc

expander = kbc.KBCExpandChunks(output_key='raw_chunk')

data = {'text_path': '/path/to/doc.txt', '_chunks': ['chunk1 content', 'chunk2 content', 'chunk3 content']}
result = expander(data)
# Returns: [
#   {'text_path': '/path/to/doc.txt', 'raw_chunk': 'chunk1 content'},
#   {'text_path': '/path/to/doc.txt', 'raw_chunk': 'chunk2 content'},
#   {'text_path': '/path/to/doc.txt', 'raw_chunk': 'chunk3 content'}
# ]
```
""")

# =========================
# KBC MultiHop QA Generator Batch
# =========================

add_chinese_doc('data.operators.knowledge_cleaning.kbc_multihop_qa_generator_batch.KBCLoadChunkFile', """\
加载分块文件算子。

该算子从指定路径加载JSON或JSONL格式的分块文件。
支持从知识库清洗流程中生成的分块结果文件。

Args:
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含分块数据的数据：
    - _chunks_data: 分块数据列表
    - _chunk_path: 分块文件路径
""")

add_english_doc('data.operators.knowledge_cleaning.kbc_multihop_qa_generator_batch.KBCLoadChunkFile', """\
Chunk file loading operator.

This operator loads JSON or JSONL format chunk files from the specified path.
Supports chunk result files generated from the knowledge base cleaning process.

Args:
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    dict: Data containing chunk data:
    - _chunks_data: List of chunk data
    - _chunk_path: Chunk file path
""")

add_example('data.operators.knowledge_cleaning.kbc_multihop_qa_generator_batch.KBCLoadChunkFile', """\
```python
from lazyllm.tools.data import kbc

loader = kbc.KBCLoadChunkFile()

data = {'chunk_path': '/path/to/chunks.json'}
result = loader(data)
# Returns: {'chunk_path': '/path/to/chunks.json', '_chunks_data': [...], '_chunk_path': '/path/to/chunks.json'}
```
""")

add_chinese_doc('data.operators.knowledge_cleaning.kbc_multihop_qa_generator_batch.KBCPreprocessText', """\
文本预处理算子。

该算子对加载的分块文本进行预处理，根据长度过滤分块。
只保留长度在指定范围内的分块，避免处理过短或过长的文本。

Args:
    min_length (int): 最小文本长度，默认为 100。
    max_length (int): 最大文本长度，默认为 200000。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含预处理结果的数据：
    - _processed_chunks: 预处理后的分块列表
""")

add_english_doc('data.operators.knowledge_cleaning.kbc_multihop_qa_generator_batch.KBCPreprocessText', """\
Text preprocessing operator.

This operator preprocesses loaded chunk texts, filtering chunks based on length.
Only retains chunks within the specified length range, avoiding processing text that is too short or too long.

Args:
    min_length (int): Minimum text length, defaults to 100.
    max_length (int): Maximum text length, defaults to 200000.
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    dict: Data containing preprocessing results:
    - _processed_chunks: List of preprocessed chunks
""")

add_example('data.operators.knowledge_cleaning.kbc_multihop_qa_generator_batch.KBCPreprocessText', """\
```python
from lazyllm.tools.data import kbc

processor = kbc.KBCPreprocessText(min_length=50, max_length=10000)

data = {'_chunks_data': [{'cleaned_chunk': 'Short text.'}, {'cleaned_chunk': 'A much longer text that meets the length requirements and will be processed.'}]}
result = processor(data, text_field='cleaned_chunk')
# Returns: {'_chunks_data': [...], '_processed_chunks': [{'text': 'A much longer text...', 'original_data': {...}}]}
```
""")

add_chinese_doc('data.operators.knowledge_cleaning.kbc_multihop_qa_generator_batch.KBCExtractInfoPairs', """\
信息对提取算子。

该算子从预处理后的文本中提取信息对，用于生成多跳问答。
根据语言类型（中文或英文）使用不同的句子分割符，
提取前提-中间-结论三元组和相关上下文。

Args:
    lang (str): 语言类型，'en' 表示英文，'zh' 表示中文，默认为 'en'。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含信息对的数据：
    - _info_pairs: 信息对列表，每个包含 premise、intermediate、conclusion 和 related_contexts
""")

add_english_doc('data.operators.knowledge_cleaning.kbc_multihop_qa_generator_batch.KBCExtractInfoPairs', """\
Information pair extraction operator.

This operator extracts information pairs from preprocessed text for multi-hop QA generation.
Uses different sentence delimiters based on language type (Chinese or English),
extracting premise-intermediate-conclusion triples and related contexts.

Args:
    lang (str): Language type, 'en' for English, 'zh' for Chinese, defaults to 'en'.
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    dict: Data containing information pairs:
    - _info_pairs: List of information pairs, each containing premise, intermediate, conclusion, and related_contexts
""")

add_example('data.operators.knowledge_cleaning.kbc_multihop_qa_generator_batch.KBCExtractInfoPairs', """\
```python
from lazyllm.tools.data import kbc

extractor = kbc.KBCExtractInfoPairs(lang='en')

data = {'_processed_chunks': [{'text': 'First sentence. Second sentence. Third sentence.', 'original_data': {}}]}
result = extractor(data)
# Returns: {'_processed_chunks': [...], '_info_pairs': [{'premise': 'First sentence', 'intermediate': 'Second sentence', 'conclusion': 'Third sentence', 'related_contexts': [], 'original_data': {}}]}
```
""")

add_chinese_doc('data.operators.knowledge_cleaning.kbc_multihop_qa_generator_batch.KBCGenerateMultiHopQA', """\
多跳问答生成算子。

该算子使用LLM根据提取的信息对生成多跳问答对。
多跳问答需要多个推理步骤才能回答，适用于训练复杂的问答模型。

Args:
    llm: LLM服务实例，用于生成问答对。
    lang (str): 语言类型，'en' 表示英文，'zh' 表示中文，默认为 'en'。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含生成的问答结果的数据：
    - _qa_results: 问答结果列表，每个包含 response 和 info_pair
""")

add_english_doc('data.operators.knowledge_cleaning.kbc_multihop_qa_generator_batch.KBCGenerateMultiHopQA', """\
Multi-hop QA generation operator.

This operator uses LLM to generate multi-hop QA pairs based on extracted information pairs.
Multi-hop QA requires multiple reasoning steps to answer, suitable for training complex QA models.

Args:
    llm: LLM service instance for generating QA pairs.
    lang (str): Language type, 'en' for English, 'zh' for Chinese, defaults to 'en'.
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    dict: Data containing generated QA results:
    - _qa_results: List of QA results, each containing response and info_pair
""")

add_example('data.operators.knowledge_cleaning.kbc_multihop_qa_generator_batch.KBCGenerateMultiHopQA', """\
```python
from lazyllm.tools.data import kbc

# Assuming llm is an LLM service instance
generator = kbc.KBCGenerateMultiHopQA(llm=llm, lang='en')

data = {'_info_pairs': [{'premise': 'A', 'intermediate': 'B', 'conclusion': 'C', 'original_data': {}}]}
result = generator(data)
# Returns: {'_info_pairs': [...], '_qa_results': [{'response': {...}, 'info_pair': {...}}]}
```
""")

add_chinese_doc('data.operators.knowledge_cleaning.kbc_multihop_qa_generator_batch.parse_qa_pairs', """\
解析问答对函数。

该函数解析LLM生成的问答响应，提取有效的问答对。
支持多种响应格式（字典、列表、字符串），并将解析结果与原始数据合并。

Args:
    data (dict): 包含问答结果的数据。

Returns:
    dict: 包含解析后的问答对的数据：
    - _qa_pairs: 解析后的问答对列表
""")

add_english_doc('data.operators.knowledge_cleaning.kbc_multihop_qa_generator_batch.parse_qa_pairs', """\
QA pair parsing function.

This function parses LLM-generated QA responses, extracting valid QA pairs.
Supports multiple response formats (dict, list, string) and merges parsing results with original data.

Args:
    data (dict): Data containing QA results.

Returns:
    dict: Data containing parsed QA pairs:
    - _qa_pairs: List of parsed QA pairs
""")

add_example('data.operators.knowledge_cleaning.kbc_multihop_qa_generator_batch.parse_qa_pairs', """\
```python
from lazyllm.tools.data.operators.knowledge_cleaning.kbc_multihop_qa_generator_batch import parse_qa_pairs

data = {'_qa_results': [{'response': {'question': 'What is AI?', 'answer': 'Artificial Intelligence'}, 'info_pair': {'original_data': {'id': 1}}}]}
result = parse_qa_pairs(data)
# Returns: {'_qa_results': [...], '_qa_pairs': [{'id': 1, 'qa_pairs': {'question': 'What is AI?', 'answer': 'Artificial Intelligence'}}]}
```
""")

add_chinese_doc('data.operators.knowledge_cleaning.kbc_multihop_qa_generator_batch.KBCSaveEnhanced', """\
保存增强数据算子。

该算子将生成的问答对与原始分块数据合并，保存为增强后的分块文件。
支持指定输出目录，会保留原始文件的相对路径结构。

Args:
    output_dir (str, optional): 输出目录路径，默认为 None（保存到原文件所在目录）。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含保存结果的数据：
    - enhanced_chunk_path: 增强后的分块文件路径
""")

add_english_doc('data.operators.knowledge_cleaning.kbc_multihop_qa_generator_batch.KBCSaveEnhanced', """\
Enhanced data saving operator.

This operator merges generated QA pairs with original chunk data and saves them as enhanced chunk files.
Supports specifying output directory, preserving the relative path structure of the original file.

Args:
    output_dir (str, optional): Output directory path, defaults to None (save to the original file's directory).
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    dict: Data containing save results:
    - enhanced_chunk_path: Path to the enhanced chunk file
""")

add_example('data.operators.knowledge_cleaning.kbc_multihop_qa_generator_batch.KBCSaveEnhanced', """\
```python
from lazyllm.tools.data import kbc

saver = kbc.KBCSaveEnhanced(output_dir='./enhanced_output')

data = {'_chunk_path': '/path/to/chunks.json', '_chunks_data': [{'id': 1, 'text': 'chunk1'}], '_qa_pairs': [{'id': 1, 'qa_pairs': {'question': 'Q1', 'answer': 'A1'}}]}
result = saver(data, output_key='enhanced_chunk_path')
# Returns: {'enhanced_chunk_path': './enhanced_output/path/to/chunks_enhanced.json'}
```
""")

# =========================
# KBC Text Cleaner Batch
# =========================

add_chinese_doc('data.operators.knowledge_cleaning.kbc_text_cleaner_batch.KBCLoadRAWChunkFile', """\
加载原始分块文件算子。

该算子从指定路径加载包含原始分块（raw_chunk）的JSON或JSONL文件。
用于知识库清洗流程中加载需要清洗的原始分块数据。

Args:
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含原始分块数据的数据：
    - _chunks_data: 原始分块数据列表
    - _chunk_path: 分块文件路径
""")

add_english_doc('data.operators.knowledge_cleaning.kbc_text_cleaner_batch.KBCLoadRAWChunkFile', """\
Raw chunk file loading operator.

This operator loads JSON or JSONL files containing raw chunks (raw_chunk) from the specified path.
Used in the knowledge base cleaning process to load raw chunk data that needs cleaning.

Args:
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    dict: Data containing raw chunk data:
    - _chunks_data: List of raw chunk data
    - _chunk_path: Chunk file path
""")

add_example('data.operators.knowledge_cleaning.kbc_text_cleaner_batch.KBCLoadRAWChunkFile', """\
```python
from lazyllm.tools.data import kbc

loader = kbc.KBCLoadRAWChunkFile()

data = {'chunk_path': '/path/to/raw_chunks.json'}
result = loader(data)
# Returns: {'chunk_path': '/path/to/raw_chunks.json', '_chunks_data': [{'raw_chunk': '...'}], '_chunk_path': '/path/to/raw_chunks.json'}
```
""")

add_chinese_doc('data.operators.knowledge_cleaning.kbc_text_cleaner_batch.KBCGenerateCleanedText', """\
生成清洗后文本的算子。

该算子使用LLM对原始分块文本进行清洗，去除噪声、格式化内容。
支持多语言，当LLM调用失败时会使用原始文本作为回退。

Args:
    llm: LLM服务实例，用于清洗文本。
    lang (str): 语言类型，'en' 表示英文，'zh' 表示中文，默认为 'en'。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含清洗结果的数据：
    - _cleaned_results: 清洗结果列表，每个包含 response、raw_chunk 和 original_item
""")

add_english_doc('data.operators.knowledge_cleaning.kbc_text_cleaner_batch.KBCGenerateCleanedText', """\
Cleaned text generation operator.

This operator uses LLM to clean raw chunk text, removing noise and formatting content.
Supports multiple languages, falls back to original text when LLM call fails.

Args:
    llm: LLM service instance for cleaning text.
    lang (str): Language type, 'en' for English, 'zh' for Chinese, defaults to 'en'.
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    dict: Data containing cleaning results:
    - _cleaned_results: List of cleaning results, each containing response, raw_chunk, and original_item
""")

add_example('data.operators.knowledge_cleaning.kbc_text_cleaner_batch.KBCGenerateCleanedText', """\
```python
from lazyllm.tools.data import kbc

# Assuming llm is an LLM service instance
cleaner = kbc.KBCGenerateCleanedText(llm=llm, lang='en')

data = {'_chunks_data': [{'raw_chunk': 'Noisy text with errors...'}]}
result = cleaner(data)
# Returns: {'_chunks_data': [...], '_cleaned_results': [{'response': 'Cleaned text', 'raw_chunk': '...', 'original_item': {...}}]}
```
""")

add_chinese_doc('data.operators.knowledge_cleaning.kbc_text_cleaner_batch.extract_cleaned_content', """\
提取清洗内容函数。

该函数从LLM清洗结果中提取清洗后的文本内容，处理不同的响应格式。
支持从标签 <cleaned_start> 和 <cleaned_end> 之间提取内容。

Args:
    data (dict): 包含清洗结果的数据。

Returns:
    dict: 包含提取后清洗内容的数据：
    - _cleaned_chunks: 清洗后的分块列表，每个包含 raw_chunk、cleaned_chunk 和 original_item
""")

add_english_doc('data.operators.knowledge_cleaning.kbc_text_cleaner_batch.extract_cleaned_content', """\
Extract cleaned content function.

This function extracts cleaned text content from LLM cleaning results, handling different response formats.
Supports extracting content between <cleaned_start> and <cleaned_end> tags.

Args:
    data (dict): Data containing cleaning results.

Returns:
    dict: Data containing extracted cleaned content:
    - _cleaned_chunks: List of cleaned chunks, each containing raw_chunk, cleaned_chunk, and original_item
""")

add_example('data.operators.knowledge_cleaning.kbc_text_cleaner_batch.extract_cleaned_content', """\
```python
from lazyllm.tools.data.operators.knowledge_cleaning.kbc_text_cleaner_batch import extract_cleaned_content

data = {'_cleaned_results': [{'response': '<cleaned_start>Clean text<cleaned_end>', 'raw_chunk': 'raw', 'original_item': {}}]}
result = extract_cleaned_content(data)
# Returns: {'_cleaned_results': [...], '_cleaned_chunks': [{'raw_chunk': 'raw', 'cleaned_chunk': 'Clean text', 'original_item': {}}]}
```
""")

add_chinese_doc('data.operators.knowledge_cleaning.kbc_text_cleaner_batch.KBCSaveCleaned', """\
保存清洗后数据算子。

该算子将清洗后的分块数据保存为JSON文件，保留原始分块和清洗后分块的对应关系。
支持指定输出目录，会保留原始文件的相对路径结构。

Args:
    output_dir (str, optional): 输出目录路径，默认为 None（保存到原文件所在目录）。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含保存结果的数据：
    - cleaned_chunk_path: 清洗后的分块文件路径
""")

add_english_doc('data.operators.knowledge_cleaning.kbc_text_cleaner_batch.KBCSaveCleaned', """\
Cleaned data saving operator.

This operator saves cleaned chunk data as JSON files, preserving the correspondence between raw and cleaned chunks.
Supports specifying output directory, preserving the relative path structure of the original file.

Args:
    output_dir (str, optional): Output directory path, defaults to None (save to the original file's directory).
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    dict: Data containing save results:
    - cleaned_chunk_path: Path to the cleaned chunk file
""")

add_example('data.operators.knowledge_cleaning.kbc_text_cleaner_batch.KBCSaveCleaned', """\
```python
from lazyllm.tools.data import kbc

saver = kbc.KBCSaveCleaned(output_dir='./cleaned_output')

data = {'_chunk_path': '/path/to/raw_chunks.json', '_cleaned_chunks': [{'raw_chunk': 'raw', 'cleaned_chunk': 'cleaned'}]}
result = saver(data, output_key='cleaned_chunk_path')
# Returns: {'cleaned_chunk_path': './cleaned_output/path/to/raw_chunks_cleaned.json'}
```
""")

# =========================
# KBC Text Cleaner
# =========================

add_chinese_doc('data.operators.knowledge_cleaning.kbc_text_cleaner.KBCGenerateCleanedTextSingle', """\
单条文本清洗生成算子。

该算子使用LLM对单条原始文本进行清洗，去除噪声、格式化内容。
适用于单条数据的实时清洗场景，当LLM调用失败时会使用原始文本作为回退。

Args:
    llm: LLM服务实例，用于清洗文本。
    lang (str): 语言类型，'en' 表示英文，'zh' 表示中文，默认为 'en'。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含清洗响应的数据：
    - _cleaned_response: LLM的清洗响应
""")

add_english_doc('data.operators.knowledge_cleaning.kbc_text_cleaner.KBCGenerateCleanedTextSingle', """\
Single text cleaning generation operator.

This operator uses LLM to clean single raw text, removing noise and formatting content.
Suitable for real-time cleaning of individual data items, falls back to original text when LLM call fails.

Args:
    llm: LLM service instance for cleaning text.
    lang (str): Language type, 'en' for English, 'zh' for Chinese, defaults to 'en'.
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    dict: Data containing cleaning response:
    - _cleaned_response: LLM's cleaning response
""")

add_example('data.operators.knowledge_cleaning.kbc_text_cleaner.KBCGenerateCleanedTextSingle', """\
```python
from lazyllm.tools.data import kbc

# Assuming llm is an LLM service instance
cleaner = kbc.KBCGenerateCleanedTextSingle(llm=llm, lang='en')

data = {'raw_chunk': 'Noisy text with errors...'}
result = cleaner(data, input_key='raw_chunk')
# Returns: {'raw_chunk': '...', '_cleaned_response': 'Cleaned text result'}
```
""")

add_chinese_doc('data.operators.knowledge_cleaning.kbc_text_cleaner.extract_cleaned_content_single', """\
单条清洗内容提取函数。

该函数从单条LLM清洗响应中提取清洗后的文本内容，处理不同的响应格式。
支持从标签 <cleaned_start> 和 <cleaned_end> 之间提取内容，并清理中间字段。

Args:
    data (dict): 包含清洗响应的数据。
    output_key (str): 输出字段名，默认为 'cleaned_chunk'。

Returns:
    dict: 包含提取后清洗内容的数据，添加了 output_key 指定的字段。
""")

add_english_doc('data.operators.knowledge_cleaning.kbc_text_cleaner.extract_cleaned_content_single', """\
Single cleaned content extraction function.

This function extracts cleaned text content from single LLM cleaning response, handling different response formats.
Supports extracting content between <cleaned_start> and <cleaned_end> tags and cleans intermediate fields.

Args:
    data (dict): Data containing cleaning response.
    output_key (str): Output key name, defaults to 'cleaned_chunk'.

Returns:
    dict: Data containing extracted cleaned content with field specified by output_key added.
""")

add_example('data.operators.knowledge_cleaning.kbc_text_cleaner.extract_cleaned_content_single', """\
```python
from lazyllm.tools.data.operators.knowledge_cleaning.kbc_text_cleaner import extract_cleaned_content_single

data = {'_cleaned_response': '<cleaned_start>Clean text<cleaned_end>'}
result = extract_cleaned_content_single(data, output_key='cleaned_chunk')
# Returns: {'cleaned_chunk': 'Clean text'}
```
""")

# =========================
# QA Extract
# =========================

add_chinese_doc('data.operators.knowledge_cleaning.qa_extract.KBCLoadQAData', """\
加载问答数据的算子。

该算子从输入数据或分块文件中加载问答数据。首先检查输入数据中是否已包含问答数据，
如果没有则尝试从增强分块文件、清洗后分块文件或普通分块文件中加载。

Args:
    qa_key (str): 问答数据字段名，默认为 'QA_pairs'。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含问答数据的数据：
    - _qa_data: 加载的问答数据
    - _source_file: 数据来源文件路径（如果从文件加载）
""")

add_english_doc('data.operators.knowledge_cleaning.qa_extract.KBCLoadQAData', """\
QA data loading operator.

This operator loads QA data from input data or chunk files. First checks if QA data already exists in input data,
if not, tries to load from enhanced chunk files, cleaned chunk files, or regular chunk files.

Args:
    qa_key (str): QA data field name, defaults to 'QA_pairs'.
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    dict: Data containing QA data:
    - _qa_data: Loaded QA data
    - _source_file: Data source file path (if loaded from file)
""")

add_example('data.operators.knowledge_cleaning.qa_extract.KBCLoadQAData', """\
```python
from lazyllm.tools.data import kbc

loader = kbc.KBCLoadQAData(qa_key='QA_pairs')

# From existing data
data = {'QA_pairs': [{'question': 'Q1', 'answer': 'A1'}]}
result = loader(data)
# Returns: {'QA_pairs': [...], '_qa_data': [...]}

# From file
data = {'enhanced_chunk_path': '/path/to/enhanced.json'}
result = loader(data)
# Returns: {'enhanced_chunk_path': '...', '_qa_data': [...], '_source_file': '/path/to/enhanced.json'}
```
""")

add_chinese_doc('data.operators.knowledge_cleaning.qa_extract.KBCExtractQAPairs', """\
提取问答对的算子。

该算子从加载的问答数据中提取问答对，并将其转换为标准格式。
支持自定义指令、问题和答案的输出字段名。

Args:
    qa_key (str): 问答数据字段名，默认为 'QA_pairs'。
    instruction (str): 指令文本，默认为 'Please answer the following question based on the provided information.'。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    List[dict]: 提取的问答对列表，每个包含 instruction、input 和 output 字段。
""")

add_english_doc('data.operators.knowledge_cleaning.qa_extract.KBCExtractQAPairs', """\
QA pairs extraction operator.

This operator extracts QA pairs from loaded QA data and converts them to standard format.
Supports customizing output field names for instruction, question, and answer.

Args:
    qa_key (str): QA data field name, defaults to 'QA_pairs'.
    instruction (str): Instruction text, defaults to 'Please answer the following question based on the provided information.'.
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    List[dict]: List of extracted QA pairs, each containing instruction, input, and output fields.
""")

add_example('data.operators.knowledge_cleaning.qa_extract.KBCExtractQAPairs', """\
```python
from lazyllm.tools.data import kbc

extractor = kbc.KBCExtractQAPairs(
    qa_key='QA_pairs',
    instruction='Please answer based on the context.'
)

data = {'_qa_data': {'qa_pairs': [{'question': 'What is AI?', 'answer': 'Artificial Intelligence'}]}}
result = extractor(
    data,
    output_instruction_key='instruction',
    output_question_key='input',
    output_answer_key='output'
)
# Returns: [{'instruction': 'Please answer based on the context.', 'input': 'What is AI?', 'output': 'Artificial Intelligence'}]
```
""")

# =========================
# Reranker Data Formatter
# =========================

add_chinese_doc('data.operators.reranker_synthesis.reranker_data_formatter.validate_reranker_data', """\
验证重排序数据的函数。

该函数验证输入数据是否包含必要的字段（query、正样本），并确保正样本和负样本为列表格式。

Args:
    data (dict): 输入数据，应包含 query、pos 和 neg 字段。
    input_query_key (str): 查询字段名，默认为 'query'。
    input_pos_key (str): 正样本字段名，默认为 'pos'。
    input_neg_key (str): 负样本字段名，默认为 'neg'。

Returns:
    dict: 验证后的数据，包含：
    - _is_valid: 数据是否有效
    - _error: 错误信息（如果无效）
    - _query, _pos, _neg: 标准化后的字段值
""")

add_english_doc('data.operators.reranker_synthesis.reranker_data_formatter.validate_reranker_data', """\
Reranker data validation function.

This function validates if input data contains required fields (query, positive samples) and ensures positive and negative samples are in list format.

Args:
    data (dict): Input data, should contain query, pos, and neg fields.
    input_query_key (str): Query field name, defaults to 'query'.
    input_pos_key (str): Positive samples field name, defaults to 'pos'.
    input_neg_key (str): Negative samples field name, defaults to 'neg'.

Returns:
    dict: Validated data containing:
    - _is_valid: Whether data is valid
    - _error: Error message (if invalid)
    - _query, _pos, _neg: Normalized field values
""")

add_example('data.operators.reranker_synthesis.reranker_data_formatter.validate_reranker_data', """\
```python
from lazyllm.tools.data.operators.reranker_synthesis.reranker_data_formatter import validate_reranker_data

data = {'query': 'machine learning', 'pos': ['ML tutorial'], 'neg': ['cooking recipe']}
result = validate_reranker_data(data)
# Returns: {'query': '...', 'pos': [...], 'neg': [...], '_is_valid': True, '_query': 'machine learning', '_pos': ['ML tutorial'], '_neg': ['cooking recipe']}
```
""")

add_chinese_doc('data.operators.reranker_synthesis.reranker_data_formatter.RerankerFormatFlagReranker', """\
FlagReranker格式转换算子。

该算子将验证后的数据转换为FlagReranker训练格式。确保负样本数量符合训练组大小要求，
如果负样本不足会复制填充，如果过多会截断。

Args:
    train_group_size (int): 训练组大小（包含1个正样本），默认为 8。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    List[dict]: 转换后的数据列表，每个包含 query、pos 和 neg 字段。
""")

add_english_doc('data.operators.reranker_synthesis.reranker_data_formatter.RerankerFormatFlagReranker', """\
FlagReranker format conversion operator.

This operator converts validated data to FlagReranker training format. Ensures the number of negative samples meets training group size requirements, padding with duplicates if insufficient or truncating if excessive.

Args:
    train_group_size (int): Training group size (including 1 positive sample), defaults to 8.
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    List[dict]: List of converted data, each containing query, pos, and neg fields.
""")

add_example('data.operators.reranker_synthesis.reranker_data_formatter.RerankerFormatFlagReranker', """\
```python
from lazyllm.tools.data import reranker

formatter = reranker.RerankerFormatFlagReranker(train_group_size=8)

data = {'_is_valid': True, '_query': 'machine learning', '_pos': ['ML tutorial'], '_neg': ['cooking', 'history']}
result = formatter(data)
# Returns: [{'query': 'machine learning', 'pos': ['ML tutorial'], 'neg': ['cooking', 'history', ...]}]
```
""")

add_chinese_doc('data.operators.reranker_synthesis.reranker_data_formatter.RerankerFormatCrossEncoder', """\
CrossEncoder格式转换算子。

该算子将验证后的数据转换为CrossEncoder训练格式。每个查询-文档对作为一个独立样本，
正样本标记为1，负样本标记为0。

Args:
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    List[dict]: 转换后的数据列表，每个包含 query、document 和 label 字段。
""")

add_english_doc('data.operators.reranker_synthesis.reranker_data_formatter.RerankerFormatCrossEncoder', """\
CrossEncoder format conversion operator.

This operator converts validated data to CrossEncoder training format. Each query-document pair is an independent sample, with positive samples labeled 1 and negative samples labeled 0.

Args:
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    List[dict]: List of converted data, each containing query, document, and label fields.
""")

add_example('data.operators.reranker_synthesis.reranker_data_formatter.RerankerFormatCrossEncoder', """\
```python
from lazyllm.tools.data import reranker

formatter = reranker.RerankerFormatCrossEncoder()

data = {'_is_valid': True, '_query': 'machine learning', '_pos': ['ML tutorial'], '_neg': ['cooking']}
result = formatter(data)
# Returns: [{'query': 'machine learning', 'document': 'ML tutorial', 'label': 1}, {'query': 'machine learning', 'document': 'cooking', 'label': 0}]
```
""")

add_chinese_doc('data.operators.reranker_synthesis.reranker_data_formatter.RerankerFormatPairwise', """\
Pairwise格式转换算子。

该算子将验证后的数据转换为Pairwise训练格式。创建正样本和负样本的成对组合，
用于训练排序模型区分相关和不相关文档。

Args:
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    List[dict]: 转换后的数据列表，每个包含 query、doc_pos 和 doc_neg 字段。
""")

add_english_doc('data.operators.reranker_synthesis.reranker_data_formatter.RerankerFormatPairwise', """\
Pairwise format conversion operator.

This operator converts validated data to Pairwise training format. Creates pairwise combinations of positive and negative samples for training ranking models to distinguish relevant from irrelevant documents.

Args:
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    List[dict]: List of converted data, each containing query, doc_pos, and doc_neg fields.
""")

add_example('data.operators.reranker_synthesis.reranker_data_formatter.RerankerFormatPairwise', """\
```python
from lazyllm.tools.data import reranker

formatter = reranker.RerankerFormatPairwise()

data = {'_is_valid': True, '_query': 'machine learning', '_pos': ['ML tutorial'], '_neg': ['cooking']}
result = formatter(data)
# Returns: [{'query': 'machine learning', 'doc_pos': 'ML tutorial', 'doc_neg': 'cooking'}]
```
""")

add_chinese_doc('data.operators.reranker_synthesis.reranker_data_formatter.RerankerTrainTestSplitter', """\
重排序训练集/测试集分割算子。

该算子将数据集随机分割为训练集和测试集，支持指定分割比例和随机种子。
可以保存训练集和测试集到指定文件，测试集会转换格式以兼容评估需求。

Args:
    test_size (float): 测试集比例，默认为 0.1（即10%）。
    seed (int): 随机种子，用于可复现的分割，默认为 42。
    train_output_file (str, optional): 训练集输出文件路径，默认为 None。
    test_output_file (str, optional): 测试集输出文件路径，默认为 None。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    List[dict]: 分割后的数据列表，每个样本包含 split 字段标记所属集合（'train' 或 'test'）。
""")

add_english_doc('data.operators.reranker_synthesis.reranker_data_formatter.RerankerTrainTestSplitter', """\
Reranker train/test splitter operator.

This operator randomly splits dataset into training and test sets, supporting specified split ratio and random seed. Can save training and test sets to specified files, with test set format converted for evaluation compatibility.

Args:
    test_size (float): Test set proportion, defaults to 0.1 (i.e., 10%).
    seed (int): Random seed for reproducible splitting, defaults to 42.
    train_output_file (str, optional): Training set output file path, defaults to None.
    test_output_file (str, optional): Test set output file path, defaults to None.
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    List[dict]: List of split data, each sample contains split field marking its set ('train' or 'test').
""")

add_example('data.operators.reranker_synthesis.reranker_data_formatter.RerankerTrainTestSplitter', """\
```python
from lazyllm.tools.data import reranker

splitter = reranker.RerankerTrainTestSplitter(
    test_size=0.2,
    seed=123,
    train_output_file='train.jsonl',
    test_output_file='test.jsonl'
)

data = [
    {'query': 'q1', 'pos': ['p1'], 'neg': ['n1']},
    {'query': 'q2', 'pos': ['p2'], 'neg': ['n2']}
]
result = splitter(data)
# Returns: [{'query': 'q1', 'pos': ['p1'], 'neg': ['n1'], 'split': 'train'}, {'query': 'q2', 'pos': ['p2'], 'neg': ['n2'], 'split': 'test'}]
```
""")

# =========================
# Reranker from Embedding Converter
# =========================

add_chinese_doc('data.operators.reranker_synthesis.reranker_from_embedding_converter.validate_reranker_embedding_data', """\
验证Embedding数据用于重排序的函数。

该函数验证输入的Embedding格式数据是否适合转换为重排序格式。检查query和正样本是否存在，
并确保正样本和负样本为列表格式。

Args:
    data (dict): 输入数据，应包含 query、pos 和 neg 字段。
    input_query_key (str): 查询字段名，默认为 'query'。
    input_pos_key (str): 正样本字段名，默认为 'pos'。
    input_neg_key (str): 负样本字段名，默认为 'neg'。

Returns:
    dict: 验证后的数据，包含：
    - _is_valid: 数据是否有效
    - _error: 错误信息（如果无效）
    - _query, _pos, _neg: 标准化后的字段值
""")

add_english_doc('data.operators.reranker_synthesis.reranker_from_embedding_converter.validate_reranker_embedding_data', """\
Validate embedding data for reranker function.

This function validates if input embedding format data is suitable for conversion to reranker format. Checks if query and positive samples exist, and ensures positive and negative samples are in list format.

Args:
    data (dict): Input data, should contain query, pos, and neg fields.
    input_query_key (str): Query field name, defaults to 'query'.
    input_pos_key (str): Positive samples field name, defaults to 'pos'.
    input_neg_key (str): Negative samples field name, defaults to 'neg'.

Returns:
    dict: Validated data containing:
    - _is_valid: Whether data is valid
    - _error: Error message (if invalid)
    - _query, _pos, _neg: Normalized field values
""")

add_example('data.operators.reranker_synthesis.reranker_from_embedding_converter.validate_reranker_embedding_data', """\
```python
from lazyllm.tools.data.operators.reranker_synthesis.reranker_from_embedding_converter import validate_reranker_embedding_data

data = {'query': 'machine learning', 'pos': 'ML tutorial', 'neg': ['cooking']}
result = validate_reranker_embedding_data(data)
# Returns: {'query': '...', 'pos': 'ML tutorial', 'neg': [...], '_is_valid': True, '_query': 'machine learning', '_pos': ['ML tutorial'], '_neg': ['cooking']}
```
""")

add_chinese_doc('data.operators.reranker_synthesis.reranker_from_embedding_converter.RerankerAdjustNegatives', """\
调整重排序负样本数量的算子。

该算子调整负样本数量以匹配目标数量。如果负样本过多则截断，如果不足则通过随机采样进行填充。
使用基于查询内容的确定性随机种子以保证可复现性。

Args:
    adjust_neg_count (int): 目标负样本数量，默认为 7。
    seed (int): 随机种子，用于填充时的随机选择，默认为 42。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 调整后的数据，包含更新后的 _neg 字段。
""")

add_english_doc('data.operators.reranker_synthesis.reranker_from_embedding_converter.RerankerAdjustNegatives', """\
Reranker negative sample adjustment operator.

This operator adjusts the number of negative samples to match the target count. Truncates if there are too many, or pads by random sampling if there are too few. Uses deterministic random seed based on query content for reproducibility.

Args:
    adjust_neg_count (int): Target negative sample count, defaults to 7.
    seed (int): Random seed for random selection during padding, defaults to 42.
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    dict: Adjusted data with updated _neg field.
""")

add_example('data.operators.reranker_synthesis.reranker_from_embedding_converter.RerankerAdjustNegatives', """\
```python
from lazyllm.tools.data import reranker

adjuster = reranker.RerankerAdjustNegatives(adjust_neg_count=5, seed=123)

# Too many negatives
data = {'_is_valid': True, '_query': 'ML', '_neg': ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8']}
result = adjuster(data)
# Returns: {'_is_valid': True, '_query': 'ML', '_neg': ['n1', 'n2', 'n3', 'n4', 'n5']}

# Too few negatives
data = {'_is_valid': True, '_query': 'ML', '_neg': ['n1', 'n2']}
result = adjuster(data)
# Returns: {'_is_valid': True, '_query': 'ML', '_neg': ['n1', 'n2', 'n1', 'n2', 'n1']}
```
""")

add_chinese_doc('data.operators.reranker_synthesis.reranker_from_embedding_converter.RerankerBuildFormat', """\
构建重排序格式的算子。

该算子将验证后的数据转换为标准的重排序训练格式。输出包含 query、pos 和 neg 字段的字典，
不包含提示或指令字段。

Args:
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 重排序格式的数据，包含 query、pos 和 neg 字段。如果数据无效则返回空字典。
""")

add_english_doc('data.operators.reranker_synthesis.reranker_from_embedding_converter.RerankerBuildFormat', """\
Reranker format builder operator.

This operator converts validated data to standard reranker training format. Outputs a dictionary containing query, pos, and neg fields without prompts or instructions.

Args:
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    dict: Reranker format data containing query, pos, and neg fields. Returns empty dict if data is invalid.
""")

add_example('data.operators.reranker_synthesis.reranker_from_embedding_converter.RerankerBuildFormat', """\
```python
from lazyllm.tools.data import reranker

builder = reranker.RerankerBuildFormat()

data = {'_is_valid': True, '_query': 'machine learning', '_pos': ['ML tutorial'], '_neg': ['cooking']}
result = builder(data)
# Returns: {'query': 'machine learning', 'pos': ['ML tutorial'], 'neg': ['cooking']}
```
""")

# =========================
# Reranker Hard Negative Miner
# =========================

add_chinese_doc('data.operators.reranker_synthesis.reranker_hard_negative_miner.build_reranker_corpus', """\
构建重排序语料库的函数。

该函数从输入数据中提取正样本文本构建语料库，并保存到临时文件中供后续使用。

Args:
    inputs (List[dict]): 输入数据列表，每个字典应包含正样本。
    input_pos_key (str): 正样本字段名，默认为 'pos'。
    corpus (List[str], optional): 外部语料库，如果提供则直接使用。默认为 None。
    corpus_dir (str, optional): 语料库文件保存目录，默认为临时目录。

Returns:
    List[dict]: 输入数据列表，每个数据添加了 '_corpus' 字段指向语料库文件路径。
""")

add_english_doc('data.operators.reranker_synthesis.reranker_hard_negative_miner.build_reranker_corpus', """\
Build reranker corpus function.

This function extracts positive sample texts from input data to build a corpus and saves it to a temporary file for later use.

Args:
    inputs (List[dict]): List of input data, each dict should contain positive samples.
    input_pos_key (str): Positive sample field name, defaults to 'pos'.
    corpus (List[str], optional): External corpus, if provided will be used directly. Defaults to None.
    corpus_dir (str, optional): Corpus file save directory, defaults to temp directory.

Returns:
    List[dict]: Input data list, each data adds '_corpus' field pointing to corpus file path.
""")

add_example('data.operators.reranker_synthesis.reranker_hard_negative_miner.build_reranker_corpus', """\
```python
from lazyllm.tools.data.operators.reranker_synthesis.reranker_hard_negative_miner import build_reranker_corpus

inputs = [{'query': 'q1', 'pos': ['doc1', 'doc2']}, {'query': 'q2', 'pos': ['doc2', 'doc3']}]
result = build_reranker_corpus(inputs)
# Returns: [{'query': 'q1', 'pos': [...], '_corpus': '/tmp/reranker_corpus_xxx.json'}, ...]
```
""")

add_chinese_doc('data.operators.reranker_synthesis.reranker_hard_negative_miner.RerankerInitBM25', """\
初始化BM25索引的算子。

该算子基于语料库构建BM25索引，用于基于关键词的负样本挖掘。
支持中英文分词，中文使用jieba，英文使用Stemmer词干提取。

Args:
    language (str): 语言类型，'zh'表示中文，'en'表示英文，默认为'zh'。
    **kwargs (dict): 其他可选参数，传递给父类。

Returns:
    List[dict]: 输入数据列表，每个数据添加了BM25索引和分词器配置。
""")

add_english_doc('data.operators.reranker_synthesis.reranker_hard_negative_miner.RerankerInitBM25', """\
Initialize BM25 index operator.

This operator builds BM25 index based on corpus for keyword-based negative sample mining.
Supports Chinese and English tokenization, Chinese uses jieba, English uses Stemmer stemming.

Args:
    language (str): Language type, 'zh' for Chinese, 'en' for English, defaults to 'zh'.
    **kwargs (dict): Additional optional parameters passed to parent class.

Returns:
    List[dict]: Input data list, each data adds BM25 index and tokenizer configuration.
""")

add_example('data.operators.reranker_synthesis.reranker_hard_negative_miner.RerankerInitBM25', """\
```python
from lazyllm.tools.data import reranker

init_bm25 = reranker.RerankerInitBM25(language='zh')

# 先构建语料库
data_with_corpus = reranker.build_reranker_corpus(inputs)
# 然后初始化BM25
result = init_bm25(data_with_corpus)
```
""")

add_chinese_doc('data.operators.reranker_synthesis.reranker_hard_negative_miner.RerankerInitSemantic', """\
初始化语义向量的算子。

该算子使用embedding服务计算语料库中所有文档的向量表示，并保存到文件中。
用于后续的语义相似度计算和负样本挖掘。

Args:
    embedding_serving (Callable): embedding服务调用函数。
    embeddings_dir (str, optional): 向量文件保存目录，默认为语料库所在目录。
    **kwargs (dict): 其他可选参数，传递给父类。

Returns:
    List[dict]: 输入数据列表，每个数据添加了向量文件路径和语料库信息。
""")

add_english_doc('data.operators.reranker_synthesis.reranker_hard_negative_miner.RerankerInitSemantic', """\
Initialize semantic embeddings operator.

This operator uses embedding service to compute vector representations for all documents in the corpus and saves them to files.
Used for subsequent semantic similarity calculation and negative sample mining.

Args:
    embedding_serving (Callable): Embedding service callable function.
    embeddings_dir (str, optional): Embedding file save directory, defaults to corpus directory.
    **kwargs (dict): Additional optional parameters passed to parent class.

Returns:
    List[dict]: Input data list, each data adds embedding file path and corpus information.
""")

add_example('data.operators.reranker_synthesis.reranker_hard_negative_miner.RerankerInitSemantic', """\
```python
from lazyllm.tools.data import reranker

# 假设 embedding_fn 是embedding服务
init_semantic = reranker.RerankerInitSemantic(embedding_serving=embedding_fn)

# 先构建语料库
data_with_corpus = reranker.build_reranker_corpus(inputs)
# 然后计算语义向量
result = init_semantic(data_with_corpus)
```
""")

add_chinese_doc('data.operators.reranker_synthesis.reranker_hard_negative_miner.RerankerMineRandomNegatives', """\
随机负样本挖掘算子。

该算子从语料库中随机选择不属于正样本的文档作为负样本。
适用于基线对比或需要随机负样本的场景。

Args:
    num_negatives (int): 需要挖掘的负样本数量，默认为 7。
    seed (int): 随机种子，用于可复现的随机选择，默认为 42。
    **kwargs (dict): 其他可选参数，传递给父类。

Returns:
    dict: 输入数据，添加了挖掘到的负样本列表。
""")

add_english_doc('data.operators.reranker_synthesis.reranker_hard_negative_miner.RerankerMineRandomNegatives', """\
Random negative sample mining operator.

This operator randomly selects documents from corpus that are not in positive samples as negative samples.
Suitable for baseline comparison or scenarios requiring random negative samples.

Args:
    num_negatives (int): Number of negative samples to mine, defaults to 7.
    seed (int): Random seed for reproducible selection, defaults to 42.
    **kwargs (dict): Additional optional parameters passed to parent class.

Returns:
    dict: Input data with mined negative samples list added.
""")

add_example('data.operators.reranker_synthesis.reranker_hard_negative_miner.RerankerMineRandomNegatives', """\
```python
from lazyllm.tools.data import reranker

miner = reranker.RerankerMineRandomNegatives(num_negatives=5, seed=123)

data = {'query': 'machine learning', 'pos': ['ML tutorial'], '_corpus': corpus_path}
result = miner(data)
# Returns: {'query': '...', 'pos': [...], '_corpus': '...', 'neg': ['random_neg1', 'random_neg2', ...]}
```
""")

add_chinese_doc('data.operators.reranker_synthesis.reranker_hard_negative_miner.RerankerMineBM25Negatives', """\
BM25负样本挖掘算子。

该算子基于BM25索引，检索与查询最相关但不属于正样本的文档作为负样本。
适用于挖掘与查询有词汇重叠但语义不同的困难负样本。

Args:
    num_negatives (int): 需要挖掘的负样本数量，默认为 7。
    **kwargs (dict): 其他可选参数，传递给父类。

Returns:
    dict: 输入数据，添加了挖掘到的负样本列表。
""")

add_english_doc('data.operators.reranker_synthesis.reranker_hard_negative_miner.RerankerMineBM25Negatives', """\
BM25 negative sample mining operator.

This operator retrieves documents most relevant to the query but not in positive samples based on BM25 index.
Suitable for mining hard negatives that have lexical overlap but different semantics.

Args:
    num_negatives (int): Number of negative samples to mine, defaults to 7.
    **kwargs (dict): Additional optional parameters passed to parent class.

Returns:
    dict: Input data with mined negative samples list added.
""")

add_example('data.operators.reranker_synthesis.reranker_hard_negative_miner.RerankerMineBM25Negatives', """\
```python
from lazyllm.tools.data import reranker

miner = reranker.RerankerMineBM25Negatives(num_negatives=5)

data = {'query': 'machine learning', 'pos': ['ML tutorial'], '_bm25': bm25_index, '_bm25_corpus': corpus}
result = miner(data)
# Returns: {'query': '...', 'pos': [...], 'neg': ['bm25_neg1', 'bm25_neg2', ...]}
```
""")

add_chinese_doc('data.operators.reranker_synthesis.reranker_hard_negative_miner.RerankerMineSemanticNegatives', """\
语义相似度负样本挖掘算子。

该算子基于语义向量相似度，找出与查询最相似但不属于正样本的文档作为负样本。
适用于挖掘语义相近但实际不相关的困难负样本，通常比BM25方法效果更好。

Args:
    num_negatives (int): 需要挖掘的负样本数量，默认为 7。
    embedding_serving (Callable): embedding服务调用函数，用于计算查询向量。
    **kwargs (dict): 其他可选参数，传递给父类。

Returns:
    dict: 输入数据，添加了基于语义相似度挖掘的负样本列表。
""")

add_english_doc('data.operators.reranker_synthesis.reranker_hard_negative_miner.RerankerMineSemanticNegatives', """\
Semantic similarity negative sample mining operator.

This operator finds documents most similar to the query but not in positive samples based on semantic vector similarity.
Suitable for mining hard negatives that are semantically similar but actually irrelevant, usually performs better than BM25 method.

Args:
    num_negatives (int): Number of negative samples to mine, defaults to 7.
    embedding_serving (Callable): Embedding service callable function for computing query vectors.
    **kwargs (dict): Additional optional parameters passed to parent class.

Returns:
    dict: Input data with negative samples mined based on semantic similarity added.
""")

add_example('data.operators.reranker_synthesis.reranker_hard_negative_miner.RerankerMineSemanticNegatives', """\
```python
from lazyllm.tools.data import reranker

# 假设 embedding_fn 是embedding服务
miner = reranker.RerankerMineSemanticNegatives(num_negatives=5, embedding_serving=embedding_fn)

data = {'query': 'machine learning', 'pos': ['ML tutorial'], '_semantic_embeddings_path': emb_path, '_semantic_corpus': corpus}
result = miner(data)
# Returns: {'query': '...', 'pos': [...], 'neg': ['semantic_neg1', 'semantic_neg2', ...]}
```
""")

add_chinese_doc('data.operators.reranker_synthesis.reranker_hard_negative_miner.RerankerMineMixedNegatives', """\
混合策略负样本挖掘算子。

该算子结合BM25和语义相似度两种方法挖掘负样本。按指定比例分别使用两种方法，
可以获得更多样化的困难负样本。

Args:
    embedding_serving (Callable): embedding服务调用函数。
    num_negatives (int): 需要挖掘的负样本数量，默认为 7。
    bm25_ratio (float): BM25方法占比，剩余部分使用语义方法，默认为 0.5。
    **kwargs (dict): 其他可选参数，传递给父类。

Returns:
    dict: 输入数据，添加了混合策略挖掘的负样本列表。
""")

add_english_doc('data.operators.reranker_synthesis.reranker_hard_negative_miner.RerankerMineMixedNegatives', """\
Mixed strategy negative sample mining operator.

This operator combines BM25 and semantic similarity methods to mine negative samples. Uses both methods according to specified ratio to obtain more diverse hard negatives.

Args:
    embedding_serving (Callable): Embedding service callable function.
    num_negatives (int): Number of negative samples to mine, defaults to 7.
    bm25_ratio (float): BM25 method ratio, remaining portion uses semantic method, defaults to 0.5.
    **kwargs (dict): Additional optional parameters passed to parent class.

Returns:
    dict: Input data with mixed strategy mined negative samples list added.
""")

add_example('data.operators.reranker_synthesis.reranker_hard_negative_miner.RerankerMineMixedNegatives', """\
```python
from lazyllm.tools.data import reranker

# 假设 embedding_fn 是embedding服务
miner = reranker.RerankerMineMixedNegatives(
    embedding_serving=embedding_fn,
    num_negatives=6,
    bm25_ratio=0.5  # 3个BM25负样本 + 3个语义负样本
)

data = {
    'query': 'machine learning',
    'pos': ['ML tutorial'],
    '_bm25': bm25_index,
    '_bm25_corpus': corpus,
    '_semantic_embeddings_path': emb_path,
    '_semantic_corpus': corpus
}
result = miner(data)
# Returns: {'query': '...', 'pos': [...], 'neg': [...]} 包含3个BM25负样本和3个语义负样本
```
""")

# =========================
# RerankerGenerateQueries
# =========================

add_chinese_doc('data.operators.reranker_synthesis.RerankerGenerateQueries', """\
基于给定文本生成多条检索查询（query）的算子。

该算子使用 RerankerQueryGeneratorPrompt 构造提示词，
调用 LLM 生成不同难度等级的查询语句。
生成结果通过 JsonFormatter 解析后，
以 JSON 字符串形式保存在 '_query_response' 字段中。

若输入 passage 为空或生成失败，则返回空响应字段。

Args:
    llm_serving: 语言模型服务实例
    lang (str): 查询生成语言，默认 'zh'
    num_queries (int): 生成查询数量，默认 3
    difficulty_levels (List[str]): 查询难度等级列表，默认 ['easy', 'medium', 'hard']
    **kwargs (dict): 其他可选参数，传递给父类。
""")

add_english_doc('data.operators.reranker_synthesis.RerankerGenerateQueries', """\
Generates multiple retrieval queries from a given passage.

This operator builds prompts using RerankerQueryGeneratorPrompt
and calls the LLM to produce queries with different difficulty levels.
The result is parsed by JsonFormatter and stored as a JSON string
in the '_query_response' field.

If the passage is empty or generation fails, an empty response is returned.

Args:
    llm_serving: language model serving instance
    lang (str): language of generated queries, default 'zh'
    num_queries (int): number of queries to generate, default 3
    difficulty_levels (List[str]): list of difficulty levels, default ['easy', 'medium', 'hard']
    **kwargs (dict): Additional optional parameters passed to parent class.
""")

add_example('data.operators.reranker_synthesis.RerankerGenerateQueries', """\
```python
op = RerankerGenerateQueries(
    llm_serving=my_llm,
    lang='en',
    num_queries=5,
    difficulty_levels=['easy', 'hard']
)

result = op({'passage': 'Large language models are widely used in NLP.'})
print(result['_query_response'])
```
""")

# =========================
# RerankerParseQueries
# =========================

add_chinese_doc('data.operators.reranker_synthesis.RerankerParseQueries', """\
解析 LLM 生成的查询结果，并展开为多条训练样本数据。

该算子读取 '_query_response' 字段中的 JSON 内容，
解析得到查询列表（支持 list 或 {'queries': [...]} 结构）。
每条查询会生成一条新的数据记录，包含：

- query: 查询文本
- difficulty: 难度等级（默认 'medium'）
- pos: 正样本文本列表（原始 passage）

同时会清理中间字段 '_query_response' 等。

Args:
    input_key (str): 原始文本字段名，默认 'passage'
    output_query_key (str): 输出查询字段名，默认 'query'
    **kwargs (dict): 其他可选参数，传递给父类。
""")

add_english_doc('data.operators.reranker_synthesis.RerankerParseQueries', """\
Parses LLM-generated query results and expands them into multiple training samples.

It reads the '_query_response' JSON content and extracts the query list
(supporting both list and {'queries': [...]} structures).
Each query generates a new data record containing:

- query: query text
- difficulty: difficulty level (default 'medium')
- pos: positive sample list (original passage)

Intermediate fields like '_query_response' are removed.

Args:
    input_key (str): source passage field name, default 'passage'
    output_query_key (str): output query field name, default 'query'
    **kwargs (dict): Additional optional parameters passed to parent class.
""")

add_example('data.operators.reranker_synthesis.RerankerParseQueries', """\
```python
op = RerankerParseQueries(input_key='passage', output_query_key='query')

data = {
    'passage': 'Large language models are widely used in NLP.',
    '_query_response': '[{"query": "What are LLMs used for?", "difficulty": "easy"}]'
}

rows = op(data)
for row in rows:
    print(row['query'], row['difficulty'], row['pos'])
```
""")
