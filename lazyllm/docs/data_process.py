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

add_chinese_doc('data.operators.preference_ops.IntentExtractor', """\
偏好数据处理算子：意图提取器。

从输入数据 dict 的指定字段中提取“核心意图”，并将结果写回到输出字段中，便于后续生成多候选回复与偏好对构造。

注意：

- 该算子内部使用模型 + JSON 格式化器，期望模型输出为 JSON dict；若无法解析为 dict，则输出为 None。
- 默认并发模式为 thread。

Args:
    model: LazyLLM 模型对象（必需），会被 share() 后复用。
    input_key (str): 输入文本字段名，默认 'content'。
    output_key (str): 输出意图字段名，默认 'intent'。
    **kwargs: 传递给基类算子的其它参数（如 _max_workers、_save_data 等）。
""")

add_english_doc('data.operators.preference_ops.IntentExtractor', """\
Preference operator: intent extractor.

Extracts the core intent from a specified field of the input data dict and writes it to an output field,
so that downstream steps can generate multiple candidate responses and construct preference pairs.

Notes:

- Internally uses a model plus a JSON formatter; it expects the model output to be a JSON dict. If it cannot be parsed as dict, the output is None.
- Default concurrency mode is thread.

Args:
    model: a LazyLLM model object (required), will be shared via share().
    input_key (str): input text field name, default 'content'.
    output_key (str): output intent field name, default 'intent'.
    **kwargs: extra args passed to the base operator (e.g. _max_workers, _save_data).
""")

add_example('data.operators.preference_ops.IntentExtractor', """\
```python
from lazyllm.tools.data.operators.preference_ops import IntentExtractor

# model 需要由你的项目环境提供，例如 lazyllm.xxx(...) 得到的模型对象
op = IntentExtractor(model=model, input_key='content', output_key='intent')
print(op({'content': 'I want to stay at a hotel in Beijing.'}))
# [{
#   'content': 'I want to stay at a hotel in Beijing.',
#   'intent': {
#     'intent': 'book_hotel',
#     'entities': [{'entity': 'location', 'value': 'Beijing'}]
#   }
# }]
```
""")

add_chinese_doc('data.operators.preference_ops.PreferenceResponseGenerator', """\
偏好数据处理算子：多候选回复生成器。

根据上一步得到的意图（或任意指令文本），生成 n 条候选回复列表写入到输出字段中。

Args:
    model: LazyLLM 模型对象（必需），会被 share() 后复用。
    n (int): 生成候选回复条数，默认 3。
    temperature (float): 采样温度，默认 1.0。
    system_prompt (str|None): 可选系统提示词；提供则会对模型调用 .prompt(system_prompt)。
    input_key (str): 输入字段名，默认 'intent'。
    output_key (str): 输出字段名，默认 'responses'。
    **kwargs: 传递给基类算子的其它参数。
""")

add_english_doc('data.operators.preference_ops.PreferenceResponseGenerator', """\
Preference operator: multi-response generator.

Given the intent (or any instruction text), generates n candidate responses and writes them as a list to the output field.

Args:
    model: a LazyLLM model object (required), will be shared via share().
    n (int): number of candidate responses to generate, default 3.
    temperature (float): sampling temperature, default 1.0.
    system_prompt (str|None): optional system prompt; if provided, applies .prompt(system_prompt) to the model.
    input_key (str): input field name, default 'intent'.
    output_key (str): output field name, default 'responses'.
    **kwargs: extra args passed to the base operator.
""")

add_example('data.operators.preference_ops.PreferenceResponseGenerator', """\
```python
from lazyllm.tools.data.operators.preference_ops import PreferenceResponseGenerator

op = PreferenceResponseGenerator(model=model, n=3, temperature=0.8, input_key='intent', output_key='responses')
print(op({'intent': 'book a hotel'}))
# [{
#   'intent': {'intent': 'book a hotel'},
#   'responses': [
#     "<think>Okay, the user wants to book a hotel. ...",
#     "<think>Okay, the user wants to book a hotel. ..."
#   ]
# }]
```
""")

add_chinese_doc('data.operators.preference_ops.ResponseEvaluator', """\
偏好数据处理算子：候选回复评测器。

对同一条指令下的多个候选回复逐一打分，输出每条回复的分数列表，便于后续构造 chosen/rejected。

评分维度（总分 10 分）：

- 有用性 (Helpfulness) 4 分
- 真实性 (Truthfulness) 3 分
- 流畅度 (Fluency) 3 分

注意：

- 该算子内部使用模型 + JSON 格式化器；每条回复都期望输出包含 total_score 的 dict。
- 如果某条回复无法解析 total_score，会记录 warning，并为该条回复记 0 分。

Args:
    model: LazyLLM 模型对象（必需），会被 share() 后复用。
    input_key (str): 指令/原始内容字段名，默认 'content'。
    response_key (str): 候选回复列表字段名，默认 'responses'。
    output_key (str): 输出评分列表字段名，默认 'evaluation'。
    **kwargs: 传递给基类算子的其它参数。
""")

add_english_doc('data.operators.preference_ops.ResponseEvaluator', """\
Preference operator: response evaluator.

Evaluates multiple candidate responses for the same instruction and outputs a score list, which can be used to build chosen/rejected pairs.

Scoring dimensions (total 10):

- Helpfulness: 4
- Truthfulness: 3
- Fluency: 3

Notes:

- Internally uses a model plus a JSON formatter; each evaluation is expected to return a dict with total_score.
- If total_score cannot be extracted, a warning is logged and the score defaults to 0 for that response.

Args:
    model: a LazyLLM model object (required), will be shared via share().
    input_key (str): instruction/raw content field name, default 'content'.
    response_key (str): candidate response list field name, default 'responses'.
    output_key (str): output score list field name, default 'evaluation'.
    **kwargs: extra args passed to the base operator.
""")

add_example('data.operators.preference_ops.ResponseEvaluator', """\
```python
from lazyllm.tools.data.operators.preference_ops import ResponseEvaluator

op = ResponseEvaluator(model=model, input_key='intent', response_key='responses', output_key='evaluation')
data = {
    'intent': {'intent': 'book a hotel'},
    'responses': [
        'I can help you book a hotel in Beijing.',
        'Here are some hotels for you.'
    ],
}
print(op(data))
# [{
#   'intent': {'intent': 'book a hotel'},
#   'responses': [
#     'I can help you book a hotel in Beijing.',
#     'Here are some hotels for you.'
#   ],
#   'evaluation': [10, 8]
# }]
```
""")

add_chinese_doc('data.operators.preference_ops.PreferencePairConstructor', """\
偏好数据处理算子：偏好对构造器（chosen / rejected）。

根据候选回复列表及其评分列表，构造一对 (chosen, rejected)，并输出为偏好数据格式：

- instruction: 指令文本（默认取 intent 字段）
- chosen: 更优的回复
- rejected: 更差的回复

支持两种策略：

- max_min: 选择最高分作为 chosen、最低分作为 rejected（要求最高分 > 最低分）。
- threshold: 从高到低寻找分差 >= threshold 的一对，满足则返回。

注意：若输入为空、长度不一致、或无法构造有效 pair，则返回空列表 []（用于在流水线中过滤无效样本）。

Args:
    strategy (str): 'max_min' 或 'threshold'，默认 'max_min'。
    threshold (float): strategy == 'threshold' 时使用的最小分差，默认 0.5。
    instruction_key (str): 指令字段名，默认 'intent'。
    response_key (str): 候选回复列表字段名，默认 'responses'。
    score_key (str): 评分列表字段名，默认 'evaluation'。
    output_chosen_key (str): 输出 chosen 字段名，默认 'chosen'。
    output_rejected_key (str): 输出 rejected 字段名，默认 'rejected'。
    **kwargs: 传递给基类算子的其它参数。
""")

add_english_doc('data.operators.preference_ops.PreferencePairConstructor', """\
Preference operator: preference pair constructor (chosen / rejected).

Given a list of candidate responses and their score list, constructs a (chosen, rejected) pair and outputs a preference sample:

- instruction: instruction text (by default read from the intent field)
- chosen: better response
- rejected: worse response

Two strategies are supported:

- max_min: choose the highest score as chosen and the lowest as rejected (requires highest > lowest).
- threshold: find a pair with score difference >= threshold, from high to low.

Note: if inputs are empty/mismatched, or no valid pair can be constructed, it returns an empty list [] (useful to filter invalid samples in pipelines).

Args:
    strategy (str): 'max_min' or 'threshold', default 'max_min'.
    threshold (float): minimum score gap when strategy == 'threshold', default 0.5.
    instruction_key (str): instruction field name, default 'intent'.
    response_key (str): candidate response list field name, default 'responses'.
    score_key (str): score list field name, default 'evaluation'.
    output_chosen_key (str): chosen field name, default 'chosen'.
    output_rejected_key (str): rejected field name, default 'rejected'.
    **kwargs: extra args passed to the base operator.
""")

add_example('data.operators.preference_ops.PreferencePairConstructor', """\
```python
from lazyllm.tools.data.operators.preference_ops import PreferencePairConstructor

op = PreferencePairConstructor(strategy='max_min', instruction_key='intent',
                              response_key='responses', score_key='evaluation')
data = {
    'intent': 'book a hotel',
    'responses': ['good response', 'bad response'],
    'evaluation': [10, 6],
}
print(op(data))
# [{
#   'instruction': 'book a hotel',
#   'chosen': 'good response',
#   'rejected': 'bad response'
# }]
```
""")


add_chinese_doc('data.operators.tool_use_ops.SequentialTaskGenerator', """\
工具调用数据生成算子：顺序任务生成器。

基于原子任务列表，生成“后继任务关系”与对应的组合任务列表，用于构造线性或有依赖关系的任务链。

输出 JSON 典型结构：

- items: 列表，每项为：
  - task: 当前原子任务
  - next_task: 紧随其后的任务
  - composed_task: 由 task + next_task 组合而成的描述

Args:
    model: LazyLLM 模型对象（必需）。
    input_key (str): 输入原子任务字段名，默认 'atomic_tasks'。
    output_key (str): 输出顺序任务列表字段名，默认 'sequential_tasks'。
    system_prompt (str|None): 可选系统提示词。
    **kwargs: 传递给基类算子的其它参数。
""")

add_english_doc('data.operators.tool_use_ops.SequentialTaskGenerator', """\
Tool-use data operator: sequential task generator.

Given a list of atomic tasks, generates successor relationships and composed tasks to form linear or dependency-aware task chains.

Typical JSON structure:

- items: list of dicts:
  - task: current atomic task
  - next_task: its successor task
  - composed_task: description combining task and next_task

Args:
    model: a LazyLLM model object (required).
    input_key (str): input atomic task field name, default 'atomic_tasks'.
    output_key (str): output sequential task list field name, default 'sequential_tasks'.
    system_prompt (str|None): optional system prompt.
    **kwargs: extra args passed to the base operator.
""")

add_example('data.operators.tool_use_ops.SequentialTaskGenerator', """\
```python
from lazyllm.tools.data.operators.tool_use_ops import SequentialTaskGenerator

atomic_tasks = [
    {'task': '获取出发地与目的地'},
    {'task': '确认出行日期'},
    {'task': '筛选符合条件的车次'},
]
op = SequentialTaskGenerator(model=model, input_key='atomic_tasks', output_key='sequential_tasks')
print(op({'atomic_tasks': atomic_tasks}))
# {
#   'atomic_tasks': [...],
#   'sequential_tasks': [
#     {'task': '获取出发地与目的地', 'next_task': '确认出行日期', 'composed_task': '先获取站点再确认日期'},
#     {'task': '确认出行日期', 'next_task': '筛选符合条件的车次', 'composed_task': '在已知日期基础上筛选车次'},
#     ...
#   ]
# }
```
""")

add_chinese_doc('data.operators.tool_use_ops.ParaSeqTaskGenerator', """\
工具调用数据生成算子：并行/顺序/混合任务组合生成器。

基于原子任务列表，自动生成三类任务组合：

- parallel_tasks: 可以并行执行的任务组合
- sequential_tasks: 具有明确先后依赖的任务组合
- hybrid_tasks: 同时包含并行与顺序关系的混合任务组合

Args:
    model: LazyLLM 模型对象（必需）。
    input_key (str): 输入原子任务字段名，默认 'atomic_tasks'。
    output_key (str): 输出任务组合字段名，默认 'para_seq_tasks'。
    system_prompt (str|None): 可选系统提示词。
    **kwargs: 传递给基类算子的其它参数。
""")

add_english_doc('data.operators.tool_use_ops.ParaSeqTaskGenerator', """\
Tool-use data operator: parallel/sequential/hybrid task combination generator.

Given atomic tasks, generates three kinds of task compositions:

- parallel_tasks: tasks that can be executed in parallel
- sequential_tasks: tasks with explicit ordering dependencies
- hybrid_tasks: compositions mixing parallel and sequential relations

Args:
    model: a LazyLLM model object (required).
    input_key (str): input atomic task field name, default 'atomic_tasks'.
    output_key (str): output composition field name, default 'para_seq_tasks'.
    system_prompt (str|None): optional system prompt.
    **kwargs: extra args passed to the base operator.
""")

add_example('data.operators.tool_use_ops.ParaSeqTaskGenerator', """\
```python
from lazyllm.tools.data.operators.tool_use_ops import ParaSeqTaskGenerator

atomic_tasks = [
    {'task': '收集出行需求'},
    {'task': '查询可选车次'},
    {'task': '对比价格与时间'},
    {'task': '完成下单支付'},
]
op = ParaSeqTaskGenerator(model=model, input_key='atomic_tasks', output_key='para_seq_tasks')
print(op({'atomic_tasks': atomic_tasks}))
# {
#   'atomic_tasks': [...],
#   'para_seq_tasks': {
#     'parallel_tasks': ['同时查询不同日期/车次方案', ...],
#     'sequential_tasks': ['先确认日期再选车次', ...],
#     'hybrid_tasks': ['并行对比多个方案后统一决策并下单', ...]
#   }
# }
```
""")

add_chinese_doc('data.operators.tool_use_ops.CompositionTaskFilter', """\
工具调用数据生成算子：组合任务可行性过滤器。

对一组“组合任务”进行可运行性与完备性评审，筛选出被认为合理可行的组合任务列表。

模型内部期望的中间 JSON 结构：

- items: 列表，每项包含 composed_task、is_valid、reason 等字段。

在算子输出中，仅保留 is_valid 为 true 且含有 composed_task 的项；如果模型未按预期输出，则尽量回退返回原 items 或原始 parsed 结果。

Args:
    model: LazyLLM 模型对象（必需）。
    composition_key (str): 输入组合任务字段名，默认 'composition_tasks'。
    subtask_key (str): 输入原子任务字段名（可选），默认 'atomic_tasks'。
    output_key (str): 输出过滤后组合任务字段名，默认 'filtered_composition_tasks'。
    system_prompt (str|None): 可选系统提示词。
    **kwargs: 传递给基类算子的其它参数。
""")

add_english_doc('data.operators.tool_use_ops.CompositionTaskFilter', """\
Tool-use data operator: composition task feasibility filter.

Evaluates a list of composed tasks for feasibility and completeness, and filters out invalid ones.

Expected intermediate JSON from the model:

- items: list of dicts with composed_task, is_valid, reason, etc.

On output, only keeps composed_task values where is_valid is true. If the model output does not match the schema, it falls back to returning items or the raw parsed result.

Args:
    model: a LazyLLM model object (required).
    composition_key (str): input composition task field name, default 'composition_tasks'.
    subtask_key (str): input atomic task field name (optional), default 'atomic_tasks'.
    output_key (str): output filtered composition task field name, default 'filtered_composition_tasks'.
    system_prompt (str|None): optional system prompt.
    **kwargs: extra args passed to the base operator.
""")

add_example('data.operators.tool_use_ops.CompositionTaskFilter', """\
```python
from lazyllm.tools.data.operators.tool_use_ops import CompositionTaskFilter

composition_tasks = ['先获取出发地和目的地再筛选车次', '直接随机推荐一个车次']
atomic_tasks = [
    {'task': '获取出发地与目的地'}, {'task': '确认出行日期'}, {'task': '筛选符合条件的车次'}
]
op = CompositionTaskFilter(model=model,
                           composition_key='composition_tasks',
                           subtask_key='atomic_tasks',
                           output_key='filtered_composition_tasks')
print(op({'composition_tasks': composition_tasks, 'atomic_tasks': atomic_tasks}))
# {
#   'composition_tasks': [...],
#   'atomic_tasks': [...],
#   'filtered_composition_tasks': ['先获取出发地和目的地再筛选车次', ...]
# }
```
""")

add_chinese_doc('data.operators.tool_use_ops.FunctionGenerator', """\
工具调用数据生成算子：函数规格生成器。

根据组合任务及其子任务，生成一组适合用于工具调用（function calling）的函数规格列表。

输出 JSON 典型结构：

- functions: 列表，每项包含：
  - name: 函数名称
  - description: 函数用途描述
  - args: 参数列表，每个参数包含 name/type/description
  - returns: 返回值类型与描述

Args:
    model: LazyLLM 模型对象（必需）。
    task_key (str): 输入组合任务字段名，默认 'composition_task'。
    subtask_key (str): 输入原子任务字段名，默认 'atomic_tasks'。
    output_key (str): 输出函数规格列表字段名，默认 'functions'。
    system_prompt (str|None): 可选系统提示词。
    **kwargs: 传递给基类算子的其它参数。
""")

add_english_doc('data.operators.tool_use_ops.FunctionGenerator', """\
Tool-use data operator: function specification generator.

Given a composed task and its subtasks, generates a list of function specifications suitable for tool calling.

Typical JSON structure:

- functions: list of dicts:
  - name: function name
  - description: what the function does
  - args: list of argument specs with name/type/description
  - returns: return type and description

Args:
    model: a LazyLLM model object (required).
    task_key (str): input composition task field name, default 'composition_task'.
    subtask_key (str): input atomic task field name, default 'atomic_tasks'.
    output_key (str): output function spec list field name, default 'functions'.
    system_prompt (str|None): optional system prompt.
    **kwargs: extra args passed to the base operator.
""")

add_example('data.operators.tool_use_ops.FunctionGenerator', """\
```python
from lazyllm.tools.data.operators.tool_use_ops import FunctionGenerator

composition_task = '根据用户出发地、目的地和日期查询可选高铁车次并返回候选列表'
atomic_tasks = [
    {'task': '获取出发地与目的地'},
    {'task': '确认出行日期'},
    {'task': '调用车次查询接口并过滤结果'},
]
op = FunctionGenerator(model=model,
                       task_key='composition_task',
                       subtask_key='atomic_tasks',
                       output_key='functions')
print(op({'composition_task': composition_task, 'atomic_tasks': atomic_tasks}))
# {
#   'composition_task': '根据用户出发地、目的地和日期查询可选高铁车次并返回候选列表',
#   'atomic_tasks': [...],
#   'functions': [
#     {
#       'name': 'query_train_tickets',
#       'description': '根据出发地、目的地与日期查询高铁车次',
#       'args': [{'name': 'from_city', 'type': 'string', ...}, ...],
#       'returns': {'type': 'TrainList', 'description': '符合条件的车次列表'}
#     },
#     ...
#   ]
# }
```
""")

add_chinese_doc('data.operators.tool_use_ops.MultiTurnConversationGenerator', """\
工具调用数据生成算子：多轮对话生成器（含 Tool 调用）。

根据组合任务与可用函数列表，生成带有 User / Assistant / Tool 三种角色的多轮对话 JSON，用于构造工具调用训练数据。

输出 JSON 典型结构：

- messages: 列表，每项为：
  - role: 'user' | 'assistant' | 'tool'
  - content: 文本内容
  - name: 工具名（仅 role == 'tool' 时可选）

Args:
    model: LazyLLM 模型对象（必需）。
    task_key (str): 输入组合任务字段名，默认 'composition_task'。
    functions_key (str): 输入函数列表字段名，默认 'functions'。
    output_key (str): 输出多轮对话字段名，默认 'conversation'。
    n_turns (int): 期望的轮次数量（提示给模型），默认 6。
    system_prompt (str|None): 可选系统提示词。
    **kwargs: 传递给基类算子的其它参数。
""")

add_english_doc('data.operators.tool_use_ops.MultiTurnConversationGenerator', """\
Tool-use data operator: multi-turn conversation generator (with tools).

Given a composed task and a list of available functions, generates a multi-turn conversation JSON involving User, Assistant and Tool roles, suitable for tool-calling training data.

Typical JSON structure:

- messages: list of dicts:
  - role: 'user' | 'assistant' | 'tool'
  - content: text content
  - name: tool name (optional, when role == 'tool')

Args:
    model: a LazyLLM model object (required).
    task_key (str): input composition task field name, default 'composition_task'.
    functions_key (str): input function list field name, default 'functions'.
    output_key (str): output conversation field name, default 'conversation'.
    n_turns (int): desired number of turns (as a hint to the model), default 6.
    system_prompt (str|None): optional system prompt.
    **kwargs: extra args passed to the base operator.
""")

add_example('data.operators.tool_use_ops.MultiTurnConversationGenerator', """\
```python
from lazyllm.tools.data.operators.tool_use_ops import MultiTurnConversationGenerator

composition_task = '根据用户需求查询并推荐合适的高铁车次'
functions = [
    {
        'name': 'query_train_tickets',
        'description': '查询高铁车次',
        'args': [...],
        'returns': {...},
    }
]
op = MultiTurnConversationGenerator(model=model,
                                    task_key='composition_task',
                                    functions_key='functions',
                                    output_key='conversation',
                                    n_turns=6)
print(op({'composition_task': composition_task, 'functions': functions}))
# {
#   'composition_task': '根据用户需求查询并推荐合适的高铁车次',
#   'functions': [...],
#   'conversation': {
#     'messages': [
#       {'role': 'user', 'content': '我想订一张明天下午从北京到上海的高铁票'},
#       {'role': 'assistant', 'content': '好的，我先为您确认出发时间与车次。'},
#       {'role': 'tool', 'name': 'query_train_tickets', 'content': '{...工具返回...}'},
#       ...
#     ]
#   }
# }
```
""")

add_chinese_doc('data.operators.tool_use_ops.ScenarioExtractor', """\
工具调用数据生成算子：场景抽取器。

从一段对话文本中抽取可用于后续任务/工具调用数据生成的“场景信息”，并以结构化 JSON 形式写入输出字段。

输出 JSON 典型结构：

- scene: 一句话场景描述
- domain: 领域/主题
- user_profile: 用户角色/背景（可为空）
- assistant_goal: 助手应完成的目标
- constraints: 约束条件列表
- key_entities: 关键实体列表

Args:
    model: LazyLLM 模型对象（必需），会被 share() 后复用并接 JSON 格式化器。
    input_key (str): 输入对话内容字段名，默认 'content'。
    output_key (str): 输出场景字段名，默认 'scenario'。
    system_prompt (str|None): 可选，自定义系统提示词，不传则使用内置中文提示。
    **kwargs: 传递给基类算子的其它参数（如 _max_workers、_save_data 等）。
""")

add_english_doc('data.operators.tool_use_ops.ScenarioExtractor', """\
Tool-use data operator: scenario extractor.

Extracts high-level scenario information from a conversation text and writes a structured JSON object into the output field.

Typical JSON structure:

- scene: one-sentence scenario description
- domain: domain/topic
- user_profile: user role/profile (optional)
- assistant_goal: goal the assistant should achieve
- constraints: list of constraints
- key_entities: list of key entities

Args:
    model: a LazyLLM model object (required), shared and wrapped with a JSON formatter.
    input_key (str): input conversation field name, default 'content'.
    output_key (str): output scenario field name, default 'scenario'.
    system_prompt (str|None): optional custom system prompt, defaults to a built-in Chinese prompt.
    **kwargs: extra args passed to the base operator (e.g. _max_workers, _save_data).
""")

add_example('data.operators.tool_use_ops.ScenarioExtractor', r"""
from lazyllm.tools.data.operators.tool_use_ops import ScenarioExtractor

op = ScenarioExtractor(model=model, input_key='content', output_key='scenario')
item = {
    'content': 'User: 我想订一张从北京到上海的高铁票，下午出发最好。\\nAssistant: 好的，请问具体日期？'
}
print(op(item))

# Output Example:
# {
#   'content': 'User: 我想订一张从北京到上海的高铁票，下午出发最好。\\nAssistant: 好的，请问具体日期？',
#   'scenario': {
#     'scene': '用户咨询高铁购票服务',
#     'domain': '出行/购票',
#     'user_profile': '普通出行乘客',
#     'assistant_goal': '帮助用户完成车次与时间筛选并完成购票',
#     'constraints': ['出发地为北京', '目的地为上海', '尽量下午出发'],
#     'key_entities': ['北京', '上海', '高铁', '下午']
#   }
# }
""")

add_chinese_doc('data.operators.tool_use_ops.ScenarioExpander', """\
工具调用数据生成算子：场景扩展器。

在已有基础场景的基础上，生成若干个语义相关但细节不同的替代场景列表，便于扩充数据多样性。

输出 JSON 典型结构：

- scenarios: 场景列表，每项为包含 scene/domain/assistant_goal/constraints/key_entities 等字段的字典。

Args:
    model: LazyLLM 模型对象（必需）。
    input_key (str): 输入场景字段名，默认 'scenario'（可为 dict 或 str）。
    output_key (str): 输出扩展场景列表字段名，默认 'expanded_scenarios'。
    n (int): 希望生成的场景数量上限，默认 3。
    system_prompt (str|None): 可选系统提示词。
    **kwargs: 传递给基类算子的其它参数。
""")

add_english_doc('data.operators.tool_use_ops.ScenarioExpander', """\
Tool-use data operator: scenario expander.

Given a base scenario, generates multiple alternative scenarios that are semantically related but differ in details, to enrich data diversity.

Typical JSON structure:

- scenarios: list of scenario dicts, each with fields like scene/domain/assistant_goal/constraints/key_entities.

Args:
    model: a LazyLLM model object (required).
    input_key (str): input scenario field name, default 'scenario' (dict or str).
    output_key (str): output expanded scenario list field name, default 'expanded_scenarios'.
    n (int): maximum number of scenarios to generate, default 3.
    system_prompt (str|None): optional system prompt.
    **kwargs: extra args passed to the base operator.
""")

add_example('data.operators.tool_use_ops.ScenarioExpander', """\
```python
from lazyllm.tools.data.operators.tool_use_ops import ScenarioExpander

base = {
    'scene': '用户咨询高铁购票服务',
    'domain': '出行/购票',
    'assistant_goal': '帮助用户完成车次筛选并购票',
}
op = ScenarioExpander(model=model, input_key='scenario', output_key='expanded_scenarios', n=3)
print(op({'scenario': base}))
# {
#   'scenario': {...},
#   'expanded_scenarios': [
#     {'scene': '用户预订跨城商务出差火车票', ...},
#     {'scene': '用户为家人购买回乡火车票', ...},
#     ...
#   ]
# }
```
""")

add_chinese_doc('data.operators.tool_use_ops.AtomTaskGenerator', """\
工具调用数据生成算子：原子任务生成器。

基于单个场景，生成一组粒度较小、目标单一的“原子任务”列表，用于后续任务编排与工具设计。

输出 JSON 典型结构：

- tasks: 原子任务列表，每项包含：
  - task: 任务描述
  - input: 任务输入（可为空）
  - output: 任务输出（可为空）
  - constraints: 相关约束列表

Args:
    model: LazyLLM 模型对象（必需）。
    input_key (str): 输入场景字段名，默认 'scenario'。
    output_key (str): 输出原子任务列表字段名，默认 'atomic_tasks'。
    n (int): 原子任务数量上限，默认 5。
    system_prompt (str|None): 可选系统提示词。
    **kwargs: 传递给基类算子的其它参数。
""")

add_english_doc('data.operators.tool_use_ops.AtomTaskGenerator', """\
Tool-use data operator: atomic task generator.

Given a scenario, generates a list of fine-grained, single-goal atomic tasks, which can be used for later orchestration and tool design.

Typical JSON structure:

- tasks: list of atomic task dicts:
  - task: task description
  - input: task input (optional)
  - output: task output (optional)
  - constraints: list of constraints

Args:
    model: a LazyLLM model object (required).
    input_key (str): input scenario field name, default 'scenario'.
    output_key (str): output atomic task list field name, default 'atomic_tasks'.
    n (int): maximum number of tasks, default 5.
    system_prompt (str|None): optional system prompt.
    **kwargs: extra args passed to the base operator.
""")

add_example('data.operators.tool_use_ops.AtomTaskGenerator', """\
```python
from lazyllm.tools.data.operators.tool_use_ops import AtomTaskGenerator

scenario = {
    'scene': '用户咨询高铁购票服务',
    'assistant_goal': '帮助用户完成车次筛选并购票',
}
op = AtomTaskGenerator(model=model, input_key='scenario', output_key='atomic_tasks', n=4)
print(op({'scenario': scenario}))
# {
#   'scenario': {...},
#   'atomic_tasks': [
#     {'task': '获取用户出发地和目的地', 'input': '', 'output': '出发地与目的地', 'constraints': [...]},
#     {'task': '确认出行日期与大致时间', ...},
#     ...
#   ]
# }
```
""")

add_chinese_doc('data.operators.tool_use_ops.SequentialTaskGenerator', """\
工具调用数据生成算子：顺序任务生成器。

基于原子任务列表，生成“后继任务关系”与对应的组合任务列表，用于构造线性或有依赖关系的任务链。

输出 JSON 典型结构：

- items: 列表，每项为：
  - task: 当前原子任务
  - next_task: 紧随其后的任务
  - composed_task: 由 task + next_task 组合而成的描述

Args:
    model: LazyLLM 模型对象（必需）。
    input_key (str): 输入原子任务字段名，默认 'atomic_tasks'。
    output_key (str): 输出顺序任务列表字段名，默认 'sequential_tasks'。
    system_prompt (str|None): 可选系统提示词。
    **kwargs: 传递给基类算子的其它参数。
""")

add_english_doc('data.operators.tool_use_ops.SequentialTaskGenerator', """\
Tool-use data operator: sequential task generator.

Given a list of atomic tasks, generates successor relationships and composed tasks to form linear or dependency-aware task chains.

Typical JSON structure:

- items: list of dicts:
  - task: current atomic task
  - next_task: its successor task
  - composed_task: description combining task and next_task

Args:
    model: a LazyLLM model object (required).
    input_key (str): input atomic task field name, default 'atomic_tasks'.
    output_key (str): output sequential task list field name, default 'sequential_tasks'.
    system_prompt (str|None): optional system prompt.
    **kwargs: extra args passed to the base operator.
""")

add_example('data.operators.tool_use_ops.SequentialTaskGenerator', """\
```python
from lazyllm.tools.data.operators.tool_use_ops import SequentialTaskGenerator

atomic_tasks = [
    {'task': '获取出发地与目的地'},
    {'task': '确认出行日期'},
    {'task': '筛选符合条件的车次'},
]
op = SequentialTaskGenerator(model=model, input_key='atomic_tasks', output_key='sequential_tasks')
print(op({'atomic_tasks': atomic_tasks}))
# {
#   'atomic_tasks': [...],
#   'sequential_tasks': [
#     {'task': '获取出发地与目的地', 'next_task': '确认出行日期', 'composed_task': '先获取站点再确认日期'},
#     {'task': '确认出行日期', 'next_task': '筛选符合条件的车次', 'composed_task': '在已知日期基础上筛选车次'},
#     ...
#   ]
# }
```
""")

add_chinese_doc('data.operators.tool_use_ops.ParaSeqTaskGenerator', """\
工具调用数据生成算子：并行/顺序/混合任务组合生成器。

基于原子任务列表，自动生成三类任务组合：

- parallel_tasks: 可以并行执行的任务组合
- sequential_tasks: 具有明确先后依赖的任务组合
- hybrid_tasks: 同时包含并行与顺序关系的混合任务组合

Args:
    model: LazyLLM 模型对象（必需）。
    input_key (str): 输入原子任务字段名，默认 'atomic_tasks'。
    output_key (str): 输出任务组合字段名，默认 'para_seq_tasks'。
    system_prompt (str|None): 可选系统提示词。
    **kwargs: 传递给基类算子的其它参数。
""")

add_english_doc('data.operators.tool_use_ops.ParaSeqTaskGenerator', """\
Tool-use data operator: parallel/sequential/hybrid task combination generator.

Given atomic tasks, generates three kinds of task compositions:

- parallel_tasks: tasks that can be executed in parallel
- sequential_tasks: tasks with explicit ordering dependencies
- hybrid_tasks: compositions mixing parallel and sequential relations

Args:
    model: a LazyLLM model object (required).
    input_key (str): input atomic task field name, default 'atomic_tasks'.
    output_key (str): output composition field name, default 'para_seq_tasks'.
    system_prompt (str|None): optional system prompt.
    **kwargs: extra args passed to the base operator.
""")

add_example('data.operators.tool_use_ops.ParaSeqTaskGenerator', """\
```python
from lazyllm.tools.data.operators.tool_use_ops import ParaSeqTaskGenerator

atomic_tasks = [
    {'task': '收集出行需求'},
    {'task': '查询可选车次'},
    {'task': '对比价格与时间'},
    {'task': '完成下单支付'},
]
op = ParaSeqTaskGenerator(model=model, input_key='atomic_tasks', output_key='para_seq_tasks')
print(op({'atomic_tasks': atomic_tasks}))
# {
#   'atomic_tasks': [...],
#   'para_seq_tasks': {
#     'parallel_tasks': ['同时查询不同日期/车次方案', ...],
#     'sequential_tasks': ['先确认日期再选车次', ...],
#     'hybrid_tasks': ['并行对比多个方案后统一决策并下单', ...]
#   }
# }
```
""")

add_chinese_doc('data.operators.tool_use_ops.CompositionTaskFilter', """\
工具调用数据生成算子：组合任务可行性过滤器。

对一组“组合任务”进行可运行性与完备性评审，筛选出被认为合理可行的组合任务列表。

模型内部期望的中间 JSON 结构：

- items: 列表，每项包含 composed_task、is_valid、reason 等字段。

在算子输出中，仅保留 is_valid 为 true 且含有 composed_task 的项；如果模型未按预期输出，则尽量回退返回原 items 或原始 parsed 结果。

Args:
    model: LazyLLM 模型对象（必需）。
    composition_key (str): 输入组合任务字段名，默认 'composition_tasks'。
    subtask_key (str): 输入原子任务字段名（可选），默认 'atomic_tasks'。
    output_key (str): 输出过滤后组合任务字段名，默认 'filtered_composition_tasks'。
    system_prompt (str|None): 可选系统提示词。
    **kwargs: 传递给基类算子的其它参数。
""")

add_english_doc('data.operators.tool_use_ops.CompositionTaskFilter', """\
Tool-use data operator: composition task feasibility filter.

Evaluates a list of composed tasks for feasibility and completeness, and filters out invalid ones.

Expected intermediate JSON from the model:

- items: list of dicts with composed_task, is_valid, reason, etc.

On output, only keeps composed_task values where is_valid is true. If the model output does not match the schema, it falls back to returning items or the raw parsed result.

Args:
    model: a LazyLLM model object (required).
    composition_key (str): input composition task field name, default 'composition_tasks'.
    subtask_key (str): input atomic task field name (optional), default 'atomic_tasks'.
    output_key (str): output filtered composition task field name, default 'filtered_composition_tasks'.
    system_prompt (str|None): optional system prompt.
    **kwargs: extra args passed to the base operator.
""")

add_example('data.operators.tool_use_ops.CompositionTaskFilter', """\
```python
from lazyllm.tools.data.operators.tool_use_ops import CompositionTaskFilter

composition_tasks = ['先获取出发地和目的地再筛选车次', '直接随机推荐一个车次']
atomic_tasks = [
    {'task': '获取出发地与目的地'}, {'task': '确认出行日期'}, {'task': '筛选符合条件的车次'}
]
op = CompositionTaskFilter(model=model,
                           composition_key='composition_tasks',
                           subtask_key='atomic_tasks',
                           output_key='filtered_composition_tasks')
print(op({'composition_tasks': composition_tasks, 'atomic_tasks': atomic_tasks}))
# {
#   'composition_tasks': [...],
#   'atomic_tasks': [...],
#   'filtered_composition_tasks': ['先获取出发地和目的地再筛选车次', ...]
# }
```
""")

add_chinese_doc('data.operators.tool_use_ops.FunctionGenerator', """\
工具调用数据生成算子：函数规格生成器。

根据组合任务及其子任务，生成一组适合用于工具调用（function calling）的函数规格列表。

输出 JSON 典型结构：

- functions: 列表，每项包含：
  - name: 函数名称
  - description: 函数用途描述
  - args: 参数列表，每个参数包含 name/type/description
  - returns: 返回值类型与描述

Args:
    model: LazyLLM 模型对象（必需）。
    task_key (str): 输入组合任务字段名，默认 'composition_task'。
    subtask_key (str): 输入原子任务字段名，默认 'atomic_tasks'。
    output_key (str): 输出函数规格列表字段名，默认 'functions'。
    system_prompt (str|None): 可选系统提示词。
    **kwargs: 传递给基类算子的其它参数。
""")

add_english_doc('data.operators.tool_use_ops.FunctionGenerator', """\
Tool-use data operator: function specification generator.

Given a composed task and its subtasks, generates a list of function specifications suitable for tool calling.

Typical JSON structure:

- functions: list of dicts:
  - name: function name
  - description: what the function does
  - args: list of argument specs with name/type/description
  - returns: return type and description

Args:
    model: a LazyLLM model object (required).
    task_key (str): input composition task field name, default 'composition_task'.
    subtask_key (str): input atomic task field name, default 'atomic_tasks'.
    output_key (str): output function spec list field name, default 'functions'.
    system_prompt (str|None): optional system prompt.
    **kwargs: extra args passed to the base operator.
""")

add_example('data.operators.tool_use_ops.FunctionGenerator', """\
```python
from lazyllm.tools.data.operators.tool_use_ops import FunctionGenerator

composition_task = '根据用户出发地、目的地和日期查询可选高铁车次并返回候选列表'
atomic_tasks = [
    {'task': '获取出发地与目的地'},
    {'task': '确认出行日期'},
    {'task': '调用车次查询接口并过滤结果'},
]
op = FunctionGenerator(model=model,
                       task_key='composition_task',
                       subtask_key='atomic_tasks',
                       output_key='functions')
print(op({'composition_task': composition_task, 'atomic_tasks': atomic_tasks}))
# {
#   'composition_task': '根据用户出发地、目的地和日期查询可选高铁车次并返回候选列表',
#   'atomic_tasks': [...],
#   'functions': [
#     {
#       'name': 'query_train_tickets',
#       'description': '根据出发地、目的地与日期查询高铁车次',
#       'args': [{'name': 'from_city', 'type': 'string', ...}, ...],
#       'returns': {'type': 'TrainList', 'description': '符合条件的车次列表'}
#     },
#     ...
#   ]
# }
```
""")

add_chinese_doc('data.operators.tool_use_ops.MultiTurnConversationGenerator', """\
工具调用数据生成算子：多轮对话生成器（含 Tool 调用）。

根据组合任务与可用函数列表，生成带有 User / Assistant / Tool 三种角色的多轮对话 JSON，用于构造工具调用训练数据。

输出 JSON 典型结构：

- messages: 列表，每项为：
  - role: 'user' | 'assistant' | 'tool'
  - content: 文本内容
  - name: 工具名（仅 role == 'tool' 时可选）

Args:
    model: LazyLLM 模型对象（必需）。
    task_key (str): 输入组合任务字段名，默认 'composition_task'。
    functions_key (str): 输入函数列表字段名，默认 'functions'。
    output_key (str): 输出多轮对话字段名，默认 'conversation'。
    n_turns (int): 期望的轮次数量（提示给模型），默认 6。
    system_prompt (str|None): 可选系统提示词。
    **kwargs: 传递给基类算子的其它参数。
""")

add_english_doc('data.operators.tool_use_ops.MultiTurnConversationGenerator', """\
Tool-use data operator: multi-turn conversation generator (with tools).

Given a composed task and a list of available functions, generates a multi-turn conversation JSON involving User, Assistant and Tool roles, suitable for tool-calling training data.

Typical JSON structure:

- messages: list of dicts:
  - role: 'user' | 'assistant' | 'tool'
  - content: text content
  - name: tool name (optional, when role == 'tool')

Args:
    model: a LazyLLM model object (required).
    task_key (str): input composition task field name, default 'composition_task'.
    functions_key (str): input function list field name, default 'functions'.
    output_key (str): output conversation field name, default 'conversation'.
    n_turns (int): desired number of turns (as a hint to the model), default 6.
    system_prompt (str|None): optional system prompt.
    **kwargs: extra args passed to the base operator.
""")

add_example('data.operators.tool_use_ops.MultiTurnConversationGenerator', """\
```python
from lazyllm.tools.data.operators.tool_use_ops import MultiTurnConversationGenerator

composition_task = '根据用户需求查询并推荐合适的高铁车次'
functions = [
    {
        'name': 'query_train_tickets',
        'description': '查询高铁车次',
        'args': [...],
        'returns': {...},
    }
]
op = MultiTurnConversationGenerator(model=model,
                                    task_key='composition_task',
                                    functions_key='functions',
                                    output_key='conversation',
                                    n_turns=6)
print(op({'composition_task': composition_task, 'functions': functions}))
# {
#   'composition_task': '根据用户需求查询并推荐合适的高铁车次',
#   'functions': [...],
#   'conversation': {
#     'messages': [
#       {'role': 'user', 'content': '我想订一张明天下午从北京到上海的高铁票'},
#       {'role': 'assistant', 'content': '好的，我先为您确认出发时间与车次。'},
#       {'role': 'tool', 'name': 'query_train_tickets', 'content': '{...工具返回...}'},
#       ...
#     ]
#   }
# }
```
""")

add_chinese_doc('data.operators.text2sql_ops.SQLGenerator', """\
Text2SQL 数据生成算子：SQL 生成器。

基于数据库 Schema 与样例数据，为给定或全部数据库自动生成可执行的 SQL 语句集合，并标注大致复杂度类型。

主要行为：

- 对每个数据库生成 generate_num 条 SQL。
- 内置默认提示词（可自定义 prompt_template），控制难度标签（easy/medium/hard 等）。
- 从模型返回中解析出 ```sql ... ``` 代码块中的 SQL 文本。

Args:
    model: LazyLLM 模型对象（必需），会被 share() 后复用。
    database_manager: 提供数据库 Schema 与样例数据的管理器（必需），需实现：
        - list_databases()
        - get_create_statements_and_insert_statements(db_name)
    generate_num (int): 每个数据库生成的 SQL 数量，默认 300。
    prompt_template: 可选，自定义 prompt 构造器对象，需实现 build_prompt(...)。
    system_prompt (str|None): 可选系统提示词，不传则使用内置英文提示。
    **kwargs: 传递给基类 Text2SQLOps/LazyLLMDataBase 的其它参数。
""")

add_english_doc('data.operators.text2sql_ops.SQLGenerator', """\
Text2SQL data operator: SQLGenerator.

Generates executable SQL queries for one or multiple databases based on their schema and optional sample data, and labels each query with a rough complexity type.

Behavior:

- Generates generate_num SQLs per database.
- Uses a default English system prompt (or a custom prompt_template) to control complexity labels (easy/medium/hard, etc.).
- Parses SQL text from model responses, preferring ```sql ... ``` code blocks.

Args:
    model: a LazyLLM model object (required), shared via share().
    database_manager: database manager (required) implementing:
        - list_databases()
        - get_create_statements_and_insert_statements(db_name)
    generate_num (int): number of SQLs to generate per database, default 300.
    prompt_template: optional custom prompt builder with build_prompt(...).
    system_prompt (str|None): optional system prompt, defaults to a built-in English prompt.
    **kwargs: extra args forwarded to the Text2SQLOps/LazyLLMDataBase base class.
""")

add_example('data.operators.text2sql_ops.SQLGenerator', """\
```python
from lazyllm.tools.data.operators.text2SQL_ops import SQLGenerator

# 假设 database_manager 已封装了你的 SQLite / Postgres 等数据库
op = SQLGenerator(model=model, database_manager=database_manager, generate_num=10)

# 如果 data 中不指定 db_id，则为所有数据库各生成若干条 SQL
res = op({})
print(res[0])
# {
#   'db_id': 'database_1',
#   'SQL': 'SELECT ...',
#   'sql_complexity_type': 'easy'
# }
```
""")

add_chinese_doc('data.operators.text2sql_ops.SQLExecutabilityFilter', """\
Text2SQL 数据过滤算子：SQL 可执行性过滤器。

对每条数据中的 SQL 进行简单语法形态过滤（仅保留 SELECT / WITH 开头的查询），并调用 database_manager 进行 EXPLAIN 校验；只保留可在目标库上成功执行的 SQL。

Args:
    database_manager: 提供数据库连接与 explain 能力的管理器（必需），需实现：
        - database_exists(db_id)
        - batch_explain_queries(list[(db_id, sql)])
    **kwargs: 传递给基类算子的其它参数。
""")

add_english_doc('data.operators.text2sql_ops.SQLExecutabilityFilter', """\
Text2SQL data operator: SQLExecutabilityFilter.

Filters SQL queries by:

1. Keeping only queries that look like SELECT/WITH queries.
2. Calling database_manager to run EXPLAIN (or similar) and keeping only those that execute successfully.

Args:
    database_manager: database manager (required) implementing:
        - database_exists(db_id)
        - batch_explain_queries(list[(db_id, sql)])
    **kwargs: extra args forwarded to the base operator.
""")

add_example('data.operators.text2sql_ops.SQLExecutabilityFilter', """\
```python
from lazyllm.tools.data.operators.text2SQL_ops import SQLExecutabilityFilter

op = SQLExecutabilityFilter(database_manager=database_manager)
item = {'db_id': 'db_1', 'SQL': 'SELECT * FROM users;'}
res = op(item)
print(res)  # 若 SQL 可在 db_1 上 explain 成功，则返回原始 dict；否则返回 None
```
""")

add_chinese_doc('data.operators.text2sql_ops.Text2SQLQuestionGenerator', """\
Text2SQL 数据生成算子：自然语言问题生成器。

基于给定 SQL + 数据库 Schema 以及列注释信息，生成与 SQL 语义对应的自然语言问题，并可附带“外部知识”提示，以支持 Text2SQL 训练。

主要特性：

- 支持多候选问题生成（question_candidates_num），并通过 embedding 去重/多样性选择。
- 内置输出格式标记：[QUESTION-START]/[QUESTION-END] 与 [EXTERNAL-KNOWLEDGE-START]/[...-END]。

Args:
    model: LazyLLM 文本生成模型（必需）。
    embedding_model: 可选向量模型，用于对候选问题做多样性选择；需支持：
        - generate_embedding_from_input(texts) 或直接可调用(texts)。
    database_manager: 提供 Schema 的管理器（必需），需实现：
        - get_create_statements_and_insert_statements(db_id)
    question_candidates_num (int): 每条 SQL 生成候选问题的数量，默认 5。
    prompt_template: 可选，自定义 prompt 构造器。
    system_prompt (str|None): 可选系统提示词，默认简要英文助手提示。
    **kwargs: 其它传递给基类算子的参数。
""")

add_english_doc('data.operators.text2sql_ops.Text2SQLQuestionGenerator', """\
Text2SQL data operator: Text2SQLQuestionGenerator.

Given a SQL query and database schema (with optional column descriptions), generates a natural language question aligned with the SQL semantics, plus optional external knowledge text.

Key features:

- Generates multiple candidate questions (question_candidates_num) and selects one using embeddings-based diversity.
- Uses special markers in model output: [QUESTION-START]/[QUESTION-END] and [EXTERNAL-KNOWLEDGE-START]/[...-END].

Args:
    model: text generation model (required).
    embedding_model: optional embedding model, supporting:
        - generate_embedding_from_input(texts) or callable(texts).
    database_manager: schema provider (required) implementing:
        - get_create_statements_and_insert_statements(db_id)
    question_candidates_num (int): number of question candidates per SQL, default 5.
    prompt_template: optional custom prompt builder.
    system_prompt (str|None): optional system prompt, default simple English helper.
    **kwargs: extra args forwarded to the base operator.
""")

add_example('data.operators.text2sql_ops.Text2SQLQuestionGenerator', """\
```python
from lazyllm.tools.data.operators.text2SQL_ops import Text2SQLQuestionGenerator

op = Text2SQLQuestionGenerator(model=model,
                               embedding_model=embedding_model,
                               database_manager=database_manager,
                               question_candidates_num=5)
item = {'db_id': 'db_1', 'SQL': 'SELECT count(*) FROM orders WHERE status = \\'paid\\';'}
res = op(item)
print(res)
# {
#   'db_id': 'db_1',
#   'SQL': 'SELECT count(*) FROM orders WHERE status = \\'paid\\';',
#   'question_type': 'default',
#   'question': '有多少已支付的订单？',
#   'evidence': '...可选的外部知识...'
# }
```
""")

add_chinese_doc('data.operators.text2sql_ops.Text2SQLCorrespondenceFilter', """\
Text2SQL 数据过滤算子：问句-SQL 一致性过滤器。

给定自然语言问题 + 证据（可选）+ SQL + 数据库 Schema，判断 SQL 是否能够正确回答该问题，保留“正确”的样本。

内部逻辑：

- 调用 database_manager 获取 db_id 对应的 DDL（create_statements）。
- 通过模型生成判断（Yes/No），仅当返回中包含 'yes' 时保留该样本，否则丢弃（返回 None）。

Args:
    model: LazyLLM 模型对象（必需）。
    database_manager: 提供 Schema 的管理器（必需），需实现：
        - get_create_statements_and_insert_statements(db_id)
    prompt_template: 可选，自定义 prompt 构造器。
    system_prompt (str|None): 可选系统提示词，默认英文 Yes/No 判定说明。
    **kwargs: 其它传递给基类算子的参数。
""")

add_english_doc('data.operators.text2sql_ops.Text2SQLCorrespondenceFilter', """\
Text2SQL data operator: Text2SQLCorrespondenceFilter.

Given a natural language question + optional evidence + SQL + database schema, determines whether the SQL correctly answers the question and filters samples accordingly.

Behavior:

- Fetches DDL for the given db_id via database_manager.
- Asks the model to answer Yes/No; only keeps data when the response contains 'yes' (case-insensitive).

Args:
    model: a LazyLLM model object (required).
    database_manager: schema provider (required) implementing:
        - get_create_statements_and_insert_statements(db_id)
    prompt_template: optional custom prompt builder.
    system_prompt (str|None): optional system prompt, defaults to English Yes/No instructions.
    **kwargs: extra args forwarded to the base operator.
""")

add_example('data.operators.text2sql_ops.Text2SQLCorrespondenceFilter', """\
```python
from lazyllm.tools.data.operators.text2SQL_ops import Text2SQLCorrespondenceFilter

op = Text2SQLCorrespondenceFilter(model=model, database_manager=database_manager)
item = {
    'db_id': 'db_1',
    'SQL': 'SELECT count(*) FROM orders WHERE status = \\'paid\\';',
    'question': '有多少已支付的订单？',
    'evidence': ''
}
res = op(item)
print(res)
# {
#   'db_id': 'db_1',
#   'SQL': 'SELECT count(*) FROM orders WHERE status = \\'paid\\';',
#   'question': '有多少已支付的订单？',
#   'evidence': ''
# }
# 如果模型判断不匹配，则返回 None
```
""")

add_chinese_doc('data.operators.text2sql_ops.Text2SQLPromptGenerator', """\
Text2SQL 数据生成算子：Prompt 构造器。

根据数据库 Schema、自然语言问题与证据，构造下游 Text2SQL 模型的输入提示词（prompt）。

行为：

- 优先调用 database_manager.get_db_details(db_id) 获取 Schema 文本；若不存在则回退到 get_create_statements_and_insert_statements。
- 支持自定义 prompt_template；否则使用简单英文模板。

Args:
    database_manager: 提供 Schema 的管理器（必需），需实现：
        - get_db_details(db_id)（可选）
        - get_create_statements_and_insert_statements(db_id)
    prompt_template: 可选，自定义 prompt 构造器。
    **kwargs: 其它传递给基类算子的参数。
""")

add_english_doc('data.operators.text2sql_ops.Text2SQLPromptGenerator', """\
Text2SQL data operator: Text2SQLPromptGenerator.

Builds prompts for downstream Text2SQL models from database schema, natural language question, and evidence.

Behavior:

- Prefers database_manager.get_db_details(db_id); falls back to get_create_statements_and_insert_statements if not available.
- Supports a custom prompt_template; otherwise uses a simple English template.

Args:
    database_manager: schema provider (required), implementing:
        - get_db_details(db_id) (optional)
        - get_create_statements_and_insert_statements(db_id)
    prompt_template: optional custom prompt builder.
    **kwargs: extra args forwarded to the base operator.
""")

add_example('data.operators.text2sql_ops.Text2SQLPromptGenerator', """\
```python
from lazyllm.tools.data.operators.text2SQL_ops import Text2SQLPromptGenerator

op = Text2SQLPromptGenerator(database_manager=database_manager)
item = {
    'db_id': 'db_1',
    'question': '有多少已支付的订单？',
    'evidence': '订单表中 status 字段标记订单状态。'
}
res = op(item)
print(res['prompt'])
# Database Schema:
# CREATE TABLE orders (id INT, status TEXT, ...);
# ...
#
# Question: 有多少已支付的订单？
# Evidence: 订单表中 status 字段标记订单状态。
# Generate a SQL query for postgres.
```
""")

add_chinese_doc('data.operators.text2sql_ops.Text2SQLCoTGenerator', """\
Text2SQL 数据生成算子：CoT 轨迹生成器。

针对给定 (问题, SQL, 数据库 Schema, 证据) 生成若干条“从问题到 SQL 的链式思考（Chain-of-Thought）”文本，用于训练/分析。

Args:
    model: LazyLLM 模型对象（必需）。
    database_manager: 提供 Schema 的管理器（必需），需实现：
        - get_create_statements_and_insert_statements(db_id)
    prompt_template: 可选，自定义 prompt 构造器。
    sampling_num (int): 每条样本生成的 CoT 轨迹数量，默认 3（>=1）。
    **kwargs: 其它传递给基类算子的参数。
""")

add_english_doc('data.operators.text2sql_ops.Text2SQLCoTGenerator', """\
Text2SQL data operator: Text2SQLCoTGenerator.

For each (question, SQL, schema, evidence) item, generates multiple chain-of-thought (CoT) reasoning traces from question to SQL.

Args:
    model: a LazyLLM model object (required).
    database_manager: schema provider (required) implementing:
        - get_create_statements_and_insert_statements(db_id)
    prompt_template: optional custom prompt builder.
    sampling_num (int): number of CoT trajectories per item, default 3 (>=1).
    **kwargs: extra args forwarded to the base operator.
""")

add_example('data.operators.text2sql_ops.Text2SQLCoTGenerator', """\
```python
from lazyllm.tools.data.operators.text2SQL_ops import Text2SQLCoTGenerator

op = Text2SQLCoTGenerator(model=model, database_manager=database_manager, sampling_num=3)
item = {
    'db_id': 'db_1',
    'question': '有多少已支付的订单？',
    'SQL': 'SELECT count(*) FROM orders WHERE status = \\'paid\\';',
    'evidence': ''
}
res = op(item)
print(len(res['cot_responses']))
print(res['cot_responses'][0][:200])  # 打印第一条 CoT 的前 200 个字符
# 3
# "Database Schema: ... Question: 有多少已支付的订单？ ... 推理步骤1：... 推理步骤2：... ```sql SELECT count(*) FROM orders WHERE status = 'paid';```"
```
""")

add_chinese_doc('data.operators.text2sql_ops.Text2SQLCoTVotingGenerator', """\
Text2SQL 数据处理算子：CoT 轨迹投票选择器。

对一组 CoT 轨迹（cot_responses）进行 SQL 解析与执行，基于执行结果的一致性与正确性，从中选出“最佳” CoT 及对应 SQL。

行为：

- 从每条 CoT 中解析 SQL（使用与 SQLGenerator 相同的解析逻辑）。
- 调用 database_manager.batch_execute_queries 执行 SQL，计算结果 signature 与 success。
- 使用投票策略 _vote_select 选择最终 CoT，并将：
  - output_cot_key（默认 'cot_reasoning'）设置为最佳 CoT 文本；
  - 同时覆盖数据中的 'SQL' 字段为最佳 SQL。

Args:
    database_manager: 提供 SQL 执行能力的管理器（必需），需实现：
        - batch_execute_queries(list[(db_id, sql)])
    **kwargs: 其它传递给基类算子的参数。
""")

add_english_doc('data.operators.text2sql_ops.Text2SQLCoTVotingGenerator', """\
Text2SQL data operator: Text2SQLCoTVotingGenerator.

Given multiple CoT traces (cot_responses), parses SQL from each, executes them, and selects the best CoT/SQL pair based on execution consistency and success.

Behavior:

- Parses SQL from each CoT using the same logic as SQLGenerator.
- Calls database_manager.batch_execute_queries to get execution results and signatures.
- Uses a voting strategy (_vote_select) to pick the best candidate, then:
  - sets output_cot_key (default 'cot_reasoning') to the winning CoT,
  - overwrites data['SQL'] with the winning SQL.

Args:
    database_manager: query execution provider (required) implementing:
        - batch_execute_queries(list[(db_id, sql)])
    **kwargs: extra args forwarded to the base operator.
""")

add_example('data.operators.text2sql_ops.Text2SQLCoTVotingGenerator', """\
```python
from lazyllm.tools.data.operators.text2SQL_ops import Text2SQLCoTVotingGenerator

op = Text2SQLCoTVotingGenerator(database_manager=database_manager)
item = {
    'db_id': 'db_1',
    'cot_responses': [
        '...CoT + ```sql SELECT count(*) FROM orders WHERE status = \\'paid\\'```',
        '...CoT + ```sql SELECT count(*) FROM orders```',
    ]
}
res = op(item)
print(res['cot_reasoning'][:200])
print(res['SQL'])
# "...首先识别需要统计已支付订单数量，其次在 orders 表中过滤 status = 'paid' ... ```sql SELECT count(*) FROM orders WHERE status = 'paid';```"
# "SELECT count(*) FROM orders WHERE status = 'paid';"
```
""")

add_chinese_doc('data.operators.text2sql_ops.SQLComponentClassifier', """\
Text2SQL 数据分类算子：SQL 组件难度分类器。

使用 SQL 结构级别的难度评估器（EvalHardness/EvalHardnessLite），根据 SQL 中涉及的组件复杂度对其进行难度打标（easy/medium/hard/extra 等）。

Args:
    difficulty_thresholds (list[int]|None): 难度阈值列表，默认 [2, 4, 6]。
    difficulty_labels (list[str]|None): 难度标签列表，默认 ['easy', 'medium', 'hard', 'extra']。
    **kwargs: 其它传递给基类算子的参数。
""")

add_english_doc('data.operators.text2sql_ops.SQLComponentClassifier', """\
Text2SQL data operator: SQLComponentClassifier.

Classifies SQL difficulty based on structural components using EvalHardness/EvalHardnessLite, assigning labels such as easy/medium/hard/extra.

Args:
    difficulty_thresholds (list[int]|None): thresholds list, default [2, 4, 6].
    difficulty_labels (list[str]|None): label list, default ['easy', 'medium', 'hard', 'extra'].
    **kwargs: extra args forwarded to the base operator.
""")

add_example('data.operators.text2sql_ops.SQLComponentClassifier', """\
```python
from lazyllm.tools.data.operators.text2SQL_ops import SQLComponentClassifier

op = SQLComponentClassifier()
item = {'SQL': 'SELECT count(*) FROM orders WHERE status = \\'paid\\';'}
res = op(item)
print(res)
# {
#   'SQL': 'SELECT count(*) FROM orders WHERE status = \\'paid\\';',
#   'sql_component_difficulty': 'easy'
# }
```
""")

add_chinese_doc('data.operators.text2sql_ops.SQLExecutionClassifier', """\
Text2SQL 数据分类算子：SQL 执行难度分类器。

基于 Text2SQLPromptGenerator 生成的 prompt、多次采样生成 SQL 并与金标 SQL 在数据库上对比执行结果，从“可被模型正确生成的次数”角度对样本执行难度进行分类。

主要流程：

1. 使用输入的 prompt，重复调用模型生成 num_generations 条 SQL，并解析出 SQL 文本。
2. 对每条 SQL 与金标 SQL 组成比较对 (db_id, predicted_sql, gold_sql)，调用 database_manager.batch_compare_queries。
3. 根据匹配次数 cnt_true 与难度阈值 difficulty_thresholds，将样本分类为 easy/medium/hard/extra/gold error。

Args:
    model: LazyLLM 模型对象（必需）。
    database_manager: 提供 batch_compare_queries 能力的管理器（必需）。
    num_generations (int): 采样生成 SQL 的次数，默认 10；若小于最大阈值会被自动上调为某个 5 的倍数。
    difficulty_thresholds (list[int]|None): 难度阈值列表，默认 [2, 5, 9]。
    difficulty_labels (list[str]|None): 难度标签列表，默认 ['extra', 'hard', 'medium', 'easy']。
    system_prompt (str|None): 可选系统提示词。
    **kwargs: 其它传递给基类算子的参数。
""")

add_english_doc('data.operators.text2sql_ops.SQLExecutionClassifier', """\
Text2SQL data operator: SQLExecutionClassifier.

Classifies SQL execution difficulty by repeatedly generating SQL from a prompt, comparing each prediction to the gold SQL on the database, and counting how many generations match.

Workflow:

1. Uses the input prompt to generate num_generations SQL candidates, parsing SQL text from each.
2. Builds comparison tuples (db_id, predicted_sql, gold_sql) and calls database_manager.batch_compare_queries.
3. Maps the number of correct generations (cnt_true) to a difficulty label using difficulty_thresholds and difficulty_labels.

Args:
    model: a LazyLLM model object (required).
    database_manager: provider implementing batch_compare_queries (required).
    num_generations (int): number of SQL generations per item, default 10; may be auto-increased to a multiple of 5.
    difficulty_thresholds (list[int]|None): thresholds list, default [2, 5, 9].
    difficulty_labels (list[str]|None): label list, default ['extra', 'hard', 'medium', 'easy'].
    system_prompt (str|None): optional system prompt.
    **kwargs: extra args forwarded to the base operator.
""")

add_example('data.operators.text2sql_ops.SQLExecutionClassifier', """\
```python
from lazyllm.tools.data.operators.text2SQL_ops import SQLExecutionClassifier

op = SQLExecutionClassifier(model=model, database_manager=database_manager, num_generations=15)
item = {
    'db_id': 'db_1',
    'prompt': 'Database Schema: ... Question: 有多少已支付的订单？',
    'SQL': 'SELECT count(*) FROM orders WHERE status = \\'paid\\';'
}
res = op(item)
print(res)
# {
#   'db_id': 'db_1',
#   'prompt': 'Database Schema: ... Question: 有多少已支付的订单？',
#   'SQL': 'SELECT count(*) FROM orders WHERE status = \\'paid\\';',
#   'sql_execution_difficulty': 'medium'
# }
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
