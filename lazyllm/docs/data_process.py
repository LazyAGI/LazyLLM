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
# =========================
# AgenticRAGGetIdentifier
# =========================

add_chinese_doc('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGGetIdentifier', """\
调用 LLM 从输入文本中抽取内容标识符（identifier）的算子。

Args:
    llm: 语言模型服务实例
    input_key (str): 输入文本字段名，默认 'prompts'
    **kwargs (dict): 其它可选的参数。
""")

add_english_doc('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGGetIdentifier', """\
An operator that extracts a content identifier from the input text using an LLM.


Args:
    llm: language model service instance
    input_key (str): name of the input text field, default 'prompts'
    **kwargs (dict): additional user-provided arguments.
""")

add_example('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGGetIdentifier', """\
```python
from lazyllm.tools.data import agenticrag
op = agenticrag.AgenticRAGGetIdentifier(llm=my_llm, input_key='prompts')
result = op({'prompts': 'What is the third movie in the Avatar series?'})
print('identifier:', result['identifier'])
# {'identifier': 'Avatar series'}
```
""")

# =========================
# AgenticRAGGetConclusion
# =========================

add_chinese_doc('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGGetConclusion', """\
调用 LLM 进行结论提取和关系生成的算子。

该算子根据输入文本构造提示词，并将模型的原始输出
保存至 data['raw_conclusion']，供后续 JSON 解析与任务展开使用。
若生成失败，则写入空字符串。

Args:
    llm: 语言模型服务实例
    input_key (str): 输入文本字段名，默认 'prompts'
    **kwargs (dict): 其它可选的参数。
""")

add_english_doc('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGGetConclusion', """\
An operator that extracts conclusions and generates relationships using an LLM.

It builds prompts from the input text and stores the raw model output
in data['raw_conclusion'] for downstream parsing and task expansion.
If generation fails, an empty string is assigned.

Args:
    llm: language model service instance
    input_key (str): name of the input text field, default 'prompts'
    **kwargs (dict): additional user-provided arguments.
""")

add_example('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGGetConclusion', """\
```python
from lazyllm.tools.data import agenticrag
op = agenticrag.AgenticRAGGetConclusion(llm=my_llm)
result = op({'prompts': 'Some document content'})
print(result['raw_conclusion'])
```
""")

# =========================
# AgenticRAGExpandConclusions
# =========================

add_chinese_doc('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGExpandConclusions', """\
解析 raw_conclusion 字段中的 JSON 结论列表，
并将其展开为多条候选任务数据。

仅保留包含 'conclusion' 和 'R' 字段的条目，
为每个条目生成独立数据行，并写入 candidate_tasks_str。

Args:
    max_per_task (int): 每个样本最多展开的候选任务数量
    **kwargs (dict): 其它可选的参数。
""")

add_english_doc('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGExpandConclusions', """\
Parses the JSON conclusion list in raw_conclusion
and expands it into multiple candidate task records.

Only items containing 'conclusion' and 'R' are kept.
Each valid item produces a new data row with candidate_tasks_str.

Args:
    max_per_task (int): maximum number of candidate tasks per sample
    **kwargs (dict): additional user-provided arguments.
""")

add_example('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGExpandConclusions', """\
```python
from lazyllm.tools.data import agenticrag
op = agenticrag.AgenticRAGExpandConclusions(max_per_task=5)
rows = op({
    'raw_conclusion': '[{"conclusion":"A","R":"rel"}]',
    'identifier': 'doc1'
})
print(rows)
```
""")

# =========================
# AgenticRAGGenerateQuestion
# =========================

add_chinese_doc('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGGenerateQuestion', """\
根据主要内容标识符(ID), 关系(R), 答案(A) 生成问题（question）与标准答案（answer）的算子。

Args:
    llm: 语言模型服务实例
    **kwargs (dict): 其它可选的参数。
""")

add_english_doc('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGGenerateQuestion', """\
Generates a question-answer pair from task identifier (ID), relationship (R), and answer (A).

Args:
    llm: language model service instance
    **kwargs (dict): additional user-provided arguments.
""")

add_example('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGGenerateQuestion', """\
```python
from lazyllm.tools.data import agenticrag
op = agenticrag.AgenticRAGGenerateQuestion(llm=my_llm)
result = op({
    'candidate_tasks_str': '{"conclusion":"Paris","R":"capital_of"}',
    'identifier': 'France'
})
print(result['question'], result['answer'])
```
""")

# =========================
# AgenticRAGCleanQA
# =========================

add_chinese_doc('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGCleanQA', """\
对生成的问答对进行清洗与答案规范化。调用 LLM 生成 refined_answer，用于后续验证与评分。

Args:
    llm: 语言模型服务实例
    **kwargs (dict): 其它可选的参数。
""")

add_english_doc('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGCleanQA', """\
Cleans and refines a generated QA pair by calling the LLM to produce a refined_answer   .

Args:
    llm: language model service instance
    **kwargs (dict): additional user-provided arguments.
""")

add_example('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGCleanQA', """\
```python
from lazyllm.tools.data import agenticrag
op = agenticrag.AgenticRAGCleanQA(llm=my_llm)
result = op({'question': 'What is...', 'answer': 'Raw answer'})
print(result['refined_answer'])
```
""")

# =========================
# AgenticRAGLLMVerify
# =========================

add_chinese_doc('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGLLMVerify', """\
使用 LLM 对问答进行回答与召回评分验证。

先让模型根据 question 生成 llm_answer，
再对 refined_answer 与 llm_answer 进行评分。
若评分 >= 1，则过滤该样本；否则保留。

Args:
    llm: 语言模型服务实例
    **kwargs (dict): 其它可选的参数。
""")

add_english_doc('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGLLMVerify', """\
Verifies QA quality via LLM answering and recall scoring.

The model first answers the question to produce llm_answer,
then scores refined_answer against llm_answer.
If score >= 1, the sample is filtered out; otherwise retained.

Args:
    llm: language model service instance
    **kwargs (dict): additional user-provided arguments.
""")

add_example('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGLLMVerify', """\
```python
from lazyllm.tools.data import agenticrag
op = agenticrag.AgenticRAGLLMVerify(llm=my_llm)
result = op({'question': 'Q?', 'refined_answer': 'A'})
print(result)
```
""")

# =========================
# AgenticRAGGoldenDocAnswer
# =========================

add_chinese_doc('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGGoldenDocAnswer', """\
基于黄金文档生成答案并进行评分验证。

使用 golden_doc 与 question 生成答案，
再与 refined_answer 进行评分。
若评分不足则过滤样本。

Args:
    llm: 语言模型服务实例
    input_key (str): 黄金文档字段名
    **kwargs (dict): 其它可选的参数。
""")

add_english_doc('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGGoldenDocAnswer', """\
Generates answers from a golden document and verifies via recall scoring.

It produces an answer using golden_doc and question,
then scores it against refined_answer.
Samples with insufficient score are filtered out.

Args:
    llm: language model service instance
    input_key (str): golden document field name
    **kwargs (dict): additional user-provided arguments.
""")

add_example('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGGoldenDocAnswer', """\
```python
from lazyllm.tools.data import agenticrag
op = agenticrag.AgenticRAGGoldenDocAnswer(llm=my_llm)
result = op({
    'prompts': 'Golden document text',
    'question': 'Q?',
    'refined_answer': 'Expected A'
})
print(result)
```
""")

# =========================
# AgenticRAGOptionalAnswers
# =========================

add_chinese_doc('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGOptionalAnswers', """\
为标准答案生成多个可选答案。

基于 refined_answer 调用 LLM，
生成语义等价或近似表达的答案列表，
写入 optional_answer 字段。

Args:
    llm: 语言模型服务实例


""")

add_english_doc('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGOptionalAnswers', """\
Generates multiple optional answers for a refined answer.

It calls the LLM to produce semantically equivalent or similar variants,
stored in optional_answer.

Args:
    llm: language model service instance
""")

add_example('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGOptionalAnswers', """\
```python
from lazyllm.tools.data import agenticrag
op = agenticrag.AgenticRAGOptionalAnswers(llm=my_llm)
result = op({'refined_answer': 'Paris'})
print(result['optional_answer'])
```
""")

# =========================
# AgenticRAGGroupAndLimit
# =========================

add_chinese_doc('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGGroupAndLimit', """\
按指定字段分组并限制每组最大问答数量。

对批量数据按 input_key 分组，
每组最多保留 max_question 条，
用于控制同源样本数量。

Args:
    input_key (str): 分组字段名
    max_question (int): 每组最大问答数量
""")

add_english_doc('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGGroupAndLimit', """\
Groups data by a specified key and limits the number of QA pairs per group.

It groups batch input by input_key and retains up to max_question
items per group to control sample distribution.

Args:
    input_key (str): grouping field name
    max_question (int): maximum QA pairs per group
""")

add_example('data.operators.agentic_rag.agenticrag_atomic_task_generator.AgenticRAGGroupAndLimit', """\
```python
from lazyllm.tools.data import agenticrag
op = agenticrag.AgenticRAGGroupAndLimit(input_key='prompts', max_question=2)
result = op([
    {'prompts': 'doc1', 'question': 'Q1'},
    {'prompts': 'doc1', 'question': 'Q2'},
    {'prompts': 'doc1', 'question': 'Q3'}
])
print(result)  # only 2 kept for doc1
```
""")

# =========================
# DepthQAGGetIdentifier
# =========================

add_chinese_doc('data.operators.agentic_rag.agenticrag_depth_qa_generator.DepthQAGGetIdentifier', """\
调用 LLM 从输入文本中抽取内容标识符（identifier）的算子。

如果数据中已存在 identifier 字段，则跳过处理。

Args:
    llm: 语言模型服务实例
    input_key (str): 输入文本字段名，默认 'question'
    **kwargs (dict): 其它可选的参数。
""")

add_english_doc('data.operators.agentic_rag.agenticrag_depth_qa_generator.DepthQAGGetIdentifier', """\
An operator that extracts a content identifier from the input text using an LLM.

If the identifier field already exists in the data, processing is skipped.

Args:
    llm: language model service instance
    input_key (str): name of the input text field, default 'question'
    **kwargs (dict): additional user-provided arguments.
""")

add_example('data.operators.agentic_rag.agenticrag_depth_qa_generator.DepthQAGGetIdentifier', """\
```python
from lazyllm.tools.data import agenticrag
op = agenticrag.DepthQAGGetIdentifier(llm=my_llm, input_key='question')
result = op({'question': 'What is the capital of France?'})
print('identifier:', result['identifier'])
# {'identifier': 'capital of France'}
```
""")

# =========================
# DepthQAGBackwardTask
# =========================

add_chinese_doc('data.operators.agentic_rag.agenticrag_depth_qa_generator.DepthQAGBackwardTask', """\
根据现有标识符生成反向任务，产生新的标识符和关系。

该算子用于从给定的 identifier 反向推理，生成新的 identifier 和对应的 relation，
用于构建深度问答任务。

Args:
    llm: 语言模型服务实例
    identifier_key (str): 原始标识符字段名，默认 'identifier'
    new_identifier_key (str): 新生成的标识符字段名，默认 'new_identifier'
    relation_key (str): 关系字段名，默认 'relation'
    **kwargs (dict): 其它可选的参数。
""")

add_english_doc('data.operators.agentic_rag.agenticrag_depth_qa_generator.DepthQAGBackwardTask', """\
Generates a backward task from the existing identifier, producing a new identifier and relation.

This operator infers backwards from the given identifier to generate a new identifier
and corresponding relation for building depth QA tasks.

Args:
    llm: language model service instance
    identifier_key (str): original identifier field name, default 'identifier'
    new_identifier_key (str): new identifier field name, default 'new_identifier'
    relation_key (str): relation field name, default 'relation'
    **kwargs (dict): additional user-provided arguments.
""")

add_example('data.operators.agentic_rag.agenticrag_depth_qa_generator.DepthQAGBackwardTask', """\
```python
from lazyllm.tools.data import agenticrag

op = agenticrag.DepthQAGBackwardTask(llm=my_llm)
result = op({'identifier': 'machine learning'})
print(result)
# {'identifier': 'machine learning', 'new_identifier': '...', 'relation': '...'}
```
""")

# =========================
# DepthQAGCheckSuperset
# =========================

add_chinese_doc('data.operators.agentic_rag.agenticrag_depth_qa_generator.DepthQAGCheckSuperset', """\
检查新生成的查询是否为原始标识符的超集。

验证 new_identifier 和 relation 组合后是否构成对原始 identifier 的有效超集查询，
若验证通过则保留数据，否则返回空列表过滤掉该样本。

Args:
    llm: 语言模型服务实例
    new_identifier_key (str): 新标识符字段名，默认 'new_identifier'
    relation_key (str): 关系字段名，默认 'relation'
    identifier_key (str): 原始标识符字段名，默认 'identifier'
    **kwargs (dict): 其它可选的参数。
""")

add_english_doc('data.operators.agentic_rag.agenticrag_depth_qa_generator.DepthQAGCheckSuperset', """\
Checks whether the newly generated query is a superset of the original identifier.

Verifies if the combination of new_identifier and relation constitutes a valid superset query
of the original identifier. If validation passes, the data is retained; otherwise,
an empty list is returned to filter out the sample.

Args:
    llm: language model service instance
    new_identifier_key (str): new identifier field name, default 'new_identifier'
    relation_key (str): relation field name, default 'relation'
    identifier_key (str): original identifier field name, default 'identifier'
    **kwargs (dict): additional user-provided arguments.
""")

add_example('data.operators.agentic_rag.agenticrag_depth_qa_generator.DepthQAGCheckSuperset', """\
```python
from lazyllm.tools.data import agenticrag

op = agenticrag.DepthQAGCheckSuperset(llm=my_llm)
result = op({
    'identifier': 'Paris',
    'new_identifier': 'France',
    'relation': 'capital_of'
})
print(result)  # returns data if valid, empty list if invalid
```
""")

# =========================
# DepthQAGGenerateQuestion
# =========================

add_chinese_doc('data.operators.agentic_rag.agenticrag_depth_qa_generator.DepthQAGGenerateQuestion', """\
根据新标识符、关系和原始标识符生成深度问题。

使用 LLM 基于 new_identifier、relation 和 identifier 生成深度问答任务中的问题，
存储在指定的 question_key 字段中。

Args:
    llm: 语言模型服务实例
    new_identifier_key (str): 新标识符字段名，默认 'new_identifier'
    relation_key (str): 关系字段名，默认 'relation'
    identifier_key (str): 原始标识符字段名，默认 'identifier'
    question_key (str): 生成问题存储的字段名，默认 'depth_question'
    **kwargs (dict): 其它可选的参数。
""")

add_english_doc('data.operators.agentic_rag.agenticrag_depth_qa_generator.DepthQAGGenerateQuestion', """\
Generates a depth question based on the new identifier, relation, and original identifier.

Uses an LLM to generate a question for depth QA tasks based on new_identifier, relation,
and identifier, storing the result in the specified question_key field.

Args:
    llm: language model service instance
    new_identifier_key (str): new identifier field name, default 'new_identifier'
    relation_key (str): relation field name, default 'relation'
    identifier_key (str): original identifier field name, default 'identifier'
    question_key (str): field name to store generated question, default 'depth_question'
    **kwargs (dict): additional user-provided arguments.
""")

add_example('data.operators.agentic_rag.agenticrag_depth_qa_generator.DepthQAGGenerateQuestion', """\
```python
from lazyllm.tools.data import agenticrag

op = agenticrag.DepthQAGGenerateQuestion(llm=my_llm)
result = op({
    'identifier': 'Paris',
    'new_identifier': 'France',
    'relation': 'capital_of'
})
print(result['depth_question'])
# 'What is the capital of France?'
```
""")

# =========================
# DepthQAGVerifyQuestion
# =========================

add_chinese_doc('data.operators.agentic_rag.agenticrag_depth_qa_generator.DepthQAGVerifyQuestion', """\
验证生成问题的质量，过滤过于简单的问题。

先让 LLM 回答问题生成 llm_answer，然后与 refined_answer 进行召回评分。
若评分 >= 1（表示问题太简单），则过滤该样本；否则保留数据。

Args:
    llm: 语言模型服务实例
    question_key (str): 问题字段名，默认 'depth_question'
    **kwargs (dict): 其它可选的参数。
""")

add_english_doc('data.operators.agentic_rag.agenticrag_depth_qa_generator.DepthQAGVerifyQuestion', """\
Verifies the quality of generated questions and filters out overly easy ones.

First has the LLM answer the question to produce llm_answer, then calculates a recall score
against refined_answer. If score >= 1 (indicating the question is too easy), the sample
is filtered out; otherwise the data is retained.

Args:
    llm: language model service instance
    question_key (str): question field name, default 'depth_question'
    **kwargs (dict): additional user-provided arguments.
""")

add_example('data.operators.agentic_rag.agenticrag_depth_qa_generator.DepthQAGVerifyQuestion', """\
```python
from lazyllm.tools.data import agenticrag

op = agenticrag.DepthQAGVerifyQuestion(llm=my_llm)
result = op({
    'depth_question': 'What is the capital of France?',
    'refined_answer': 'Paris'
})
# Returns data if question is challenging, empty list if too easy
print(result)
```
""")

# =========================
# WidthQAGMergePairs
# =========================

add_chinese_doc('data.operators.agentic_rag.agenticrag_width_qa_generator.WidthQAGMergePairs', """\
将相邻的问答对合并生成广度问题的算子。

该算子接收批量问答数据，通过 LLM 将相邻的两个问答对合并为一个更复杂的广度问题。
需要至少2条数据才能进行合并操作。

Args:
    llm: 语言模型服务实例
    **kwargs (dict): 其它可选的参数。
""")

add_english_doc('data.operators.agentic_rag.agenticrag_width_qa_generator.WidthQAGMergePairs', """\
An operator that merges adjacent QA pairs to generate width questions.

This operator receives a batch of QA data and uses an LLM to merge adjacent pairs
into more complex width questions. Requires at least 2 items to perform merging.

Args:
    llm: language model service instance
    **kwargs (dict): additional user-provided arguments.
""")

add_example('data.operators.agentic_rag.agenticrag_width_qa_generator.WidthQAGMergePairs', """\
```python
from lazyllm.tools.data import agenticrag

op = agenticrag.WidthQAGMergePairs(llm=my_llm)
result = op([
    {'question': 'What is Paris?', 'golden_answer': 'Capital of France'},
    {'question': 'What is London?', 'golden_answer': 'Capital of UK'}
])
print(result[0]['question'])  # Merged complex question
```
""")

# =========================
# WidthQAGCheckDecomposition
# =========================

add_chinese_doc('data.operators.agentic_rag.agenticrag_width_qa_generator.WidthQAGCheckDecomposition', """\
验证合并后的问题是否有效分解了原始问题的算子。

该算子检查 LLM 生成的复杂问题是否正确地分解和包含了原始问题，
如果验证通过则保留数据，否则返回空列表过滤掉该样本。

Args:
    llm: 语言模型服务实例
    output_question_key (str): 输出生成问题的字段名，默认 'generated_width_task'
    **kwargs (dict): 其它可选的参数。
""")

add_english_doc('data.operators.agentic_rag.agenticrag_width_qa_generator.WidthQAGCheckDecomposition', """\
An operator that verifies whether the merged question effectively decomposes the original questions.

This operator checks if the complex question generated by LLM correctly decomposes
and includes the original questions. If validation passes, the data is retained;
otherwise an empty list is returned to filter out the sample.

Args:
    llm: language model service instance
    output_question_key (str): field name for the generated question, default 'generated_width_task'
    **kwargs (dict): additional user-provided arguments.
""")

add_example('data.operators.agentic_rag.agenticrag_width_qa_generator.WidthQAGCheckDecomposition', """\
```python
from lazyllm.tools.data import agenticrag

op = agenticrag.WidthQAGCheckDecomposition(llm=my_llm)
result = op({
    'question': 'What are the capitals of France and UK?',
    'original_question': ['What is Paris?', 'What is London?'],
    'index': 0
})
print(result)  # Returns data if valid, empty list if invalid
```
""")

# =========================
# WidthQAGVerifyQuestion
# =========================

add_chinese_doc('data.operators.agentic_rag.agenticrag_width_qa_generator.WidthQAGVerifyQuestion', """\
验证生成的问题能否被正确回答的算子。

该算子使用 LLM 尝试回答生成的问题，并将答案存储在 llm_answer 字段中，
供后续评分使用。

Args:
    llm: 语言模型服务实例
    output_question_key (str): 问题字段名，默认 'generated_width_task'
    **kwargs (dict): 其它可选的参数。
""")

add_english_doc('data.operators.agentic_rag.agenticrag_width_qa_generator.WidthQAGVerifyQuestion', """\
An operator that verifies if the generated question can be properly answered.

This operator uses an LLM to attempt answering the generated question and stores
the answer in the llm_answer field for subsequent scoring.

Args:
    llm: language model service instance
    output_question_key (str): question field name, default 'generated_width_task'
    **kwargs (dict): additional user-provided arguments.
""")

add_example('data.operators.agentic_rag.agenticrag_width_qa_generator.WidthQAGVerifyQuestion', """\
```python
from lazyllm.tools.data import agenticrag

op = agenticrag.WidthQAGVerifyQuestion(llm=my_llm)
result = op({
    'generated_width_task': 'What are the capitals of France and UK?',
    'index': 0
})
print(result['llm_answer'])  # LLM's answer to the question
```
""")

# =========================
# WidthQAGFilterByScore
# =========================

add_chinese_doc('data.operators.agentic_rag.agenticrag_width_qa_generator.WidthQAGFilterByScore', """\
根据召回评分过滤广度问题的算子。

该算子对比 golden_answer 和 llm_answer 计算召回评分，
若评分 >= 1 则过滤该样本（表示问题太简单或 LLM 回答太好）；
否则保留数据并清理临时字段。

Args:
    llm: 语言模型服务实例
    **kwargs (dict): 其它可选的参数。
""")

add_english_doc('data.operators.agentic_rag.agenticrag_width_qa_generator.WidthQAGFilterByScore', """\
An operator that filters width questions based on recall score.

This operator compares golden_answer with llm_answer to calculate a recall score.
If score >= 1, the sample is filtered out (indicating the question is too easy
or LLM answered too well); otherwise the data is retained and temporary fields are cleaned.

Args:
    llm: language model service instance
    **kwargs (dict): additional user-provided arguments.
""")

add_example('data.operators.agentic_rag.agenticrag_width_qa_generator.WidthQAGFilterByScore', """\
```python
from lazyllm.tools.data import agenticrag

op = agenticrag.WidthQAGFilterByScore(llm=my_llm)
result = op({
    'original_answer': ['Paris', 'London'],
    'llm_answer': 'Paris is the capital of France and London is the capital of UK',
    'state': 1
})
# Returns data if score < 1, empty list if score >= 1
print(result)
```
""")

# =========================
# qaf1_normalize_texts
# =========================

add_chinese_doc('data.operators.agentic_rag.agenticrag_qaf1_sample_evaluator.qaf1_normalize_texts', """\
规范化预测答案和参考答案文本的函数。

对预测答案和参考答案进行标准化处理，包括：转换为小写、移除标点符号、
移除冠词(a/an/the)、规范化空白字符。规范化后的结果存储在临时字段中，
供后续 F1 分数计算使用。

Args:
    data (dict): 单条数据字典
    prediction_key (str): 预测答案字段名，默认 'refined_answer'
    ground_truth_key (str): 参考答案字段名，默认 'golden_doc_answer'
""")

add_english_doc('data.operators.agentic_rag.agenticrag_qaf1_sample_evaluator.qaf1_normalize_texts', """\
A function that normalizes prediction and ground truth answer texts.

Performs standardization on prediction and ground truth answers, including:
converting to lowercase, removing punctuation, removing articles (a/an/the),
and normalizing whitespace. Normalized results are stored in temporary fields
for subsequent F1 score calculation.

Args:
    data (dict): single data dictionary
    prediction_key (str): prediction answer field name, default 'refined_answer'
    ground_truth_key (str): ground truth answer field name, default 'golden_doc_answer'
""")

add_example('data.operators.agentic_rag.agenticrag_qaf1_sample_evaluator.qaf1_normalize_texts', """\
```python
from lazyllm.tools.data import agenticrag

op = agenticrag.qaf1_normalize_texts(prediction_key='refined_answer', ground_truth_key='golden_doc_answer')
result = op({
    'refined_answer': 'Paris is the capital.',
    'golden_doc_answer': 'The capital is Paris!'
})
print(result['_normalized_prediction'])  # 'paris is capital'
print(result['_normalized_ground_truths'])  # ['capital is paris']
```
""")

# =========================
# qaf1_calculate_score
# =========================

add_chinese_doc('data.operators.agentic_rag.agenticrag_qaf1_sample_evaluator.qaf1_calculate_score', """\
计算问答对的 F1 分数的函数。

基于规范化后的预测答案和参考答案计算 F1 分数（综合考虑精确率和召回率）。
支持多个参考答案，取最高 F1 分数作为最终结果。计算完成后清理临时字段。

Args:
    data (dict): 单条数据字典
    output_key (str): 输出 F1 分数的字段名，默认 'F1Score'
""")

add_english_doc('data.operators.agentic_rag.agenticrag_qaf1_sample_evaluator.qaf1_calculate_score', """\
A function that calculates the F1 score for QA pairs.

Calculates the F1 score (combining precision and recall) based on normalized
prediction and ground truth answers. Supports multiple ground truth answers,
taking the highest F1 score as the final result. Cleans up temporary fields after calculation.

Args:
    data (dict): single data dictionary
    output_key (str): output field name for F1 score, default 'F1Score'
""")

add_example('data.operators.agentic_rag.agenticrag_qaf1_sample_evaluator.qaf1_calculate_score', """\
```python
from lazyllm.tools.data import agenticrag

op = agenticrag.qaf1_calculate_score(output_key='F1Score')
result = op({
    '_normalized_prediction': 'paris is capital',
    '_normalized_ground_truths': ['capital is paris', 'paris capital france']
})
print(result['F1Score'])  # F1 score value between 0.0 and 1.0
```
""")

# cot_ops module docs
add_chinese_doc('data.operators.cot_ops.CoTGenerator', """\
使用大模型为问题生成带思维链（CoT）的推理过程，要求最终答案用 \\\\boxed{{ANSWER}} 包裹。输出写入指定字段。

Args:
    input_key (str): 输入问题字段名，默认 'query'
    output_key (str): 输出 CoT 答案字段名，默认 'cot_answer'
    model: 可选，TrainableModule 或兼容接口；None 时使用默认 Qwen 模型
    user_prompt (str|None): 可选，用户提示前缀；None 时使用默认
    **kwargs: 其它基类参数
""")

add_english_doc('data.operators.cot_ops.CoTGenerator', """\
Use an LLM to generate chain-of-thought reasoning for a question, with final answer wrapped in \\\\boxed{{ANSWER}}. Writes result to the specified output key.

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
res = op(data)  # each item gets 'cot_answer' with CoT and \\\\boxed{{4}}
print(res)
# {'query': 'What is 2+2?', 'cot_answer': '首先，我们需要理解加法的基本概念，即两个或多个数值的总和。在这个问题中，我们需要计算 2 和另一个 2 的和。\n\n第一步，我们识别出第一个数值是 2。\n\n第二步，我们识别出第二个数值也是 2。\n\n第三步，我们将这两个数值相加：2 + 2。\n\n第四步，我们进行计算：2 + 2 = 4。\n\n因此，最终答案是 4，使用规定的格式包裹答案。\n\n最终答案：\\boxed{4}'}
```
""")

add_chinese_doc('data.operators.cot_ops.SelfConsistencyCoTGenerator', """\
对同一问题采样多次 CoT，从 \\\\boxed{{}} 中提取答案并做多数投票，最终保留与多数答案一致的一条 CoT 输出。

Args:
    input_key (str): 输入问题字段名，默认 'query'
    output_key (str): 输出 CoT 答案字段名，默认 'cot_answer'
    num_samples (int): 采样次数，默认 5
    model: 可选；None 时使用默认 Qwen 模型
    user_prompt (str|None): 可选用户提示
    **kwargs: 其它基类参数
""")

add_english_doc('data.operators.cot_ops.SelfConsistencyCoTGenerator', """\
Sample multiple CoT answers for the same question, extract \\\\boxed{{}} answers, take majority vote, and output one CoT that matches the majority answer.

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
从文本中提取 \\\\boxed{{}} 内的数学答案，写入指定输出字段。以 forward 单条方式注册。

Args:
    data (dict): 单条数据字典
    input_key (str): 含答案文本的字段名，默认 'answer'
    output_key (str): 提取结果写入的字段名，默认 'math_answer'
""")

add_english_doc('data.operators.math_ops.math_answer_extractor', """\
Extract the math answer inside \\\\boxed{{}} from text and write to the specified output key. Registered as single-item forward.

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
使用大模型为数学问题生成推理与答案，要求最终结果用 \\\\boxed{{ANSWER}} 包裹。若已有 answer 且未设置 regenerate 则跳过。

Args:
    input_key (str): 问题字段名，默认 'question'
    output_key (str): 答案写入的字段名，默认 'answer'
    regenerate_key (str): 是否强制重新生成的标志字段，默认 'regenerate'
    model: 可选；None 时使用默认 Qwen 模型
    user_prompt (str|None): 可选用户提示
    **kwargs: 其它基类参数
""")

add_english_doc('data.operators.math_ops.MathAnswerGenerator', """\
Use an LLM to generate reasoning and answer for a math question, with final result in \\\\boxed{{ANSWER}}. Skips if answer already exists and regenerate is not set.

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
使用大模型将多条问题融合为一个新问题并生成推理与 \\\\boxed{{}} 答案。需要 list_key 下至少 2 个问题。

Args:
    input_key (str): 融合后问题字段名，默认 'question'
    output_key (str): 推理结果/答案写入的字段名，默认 'answer'
    list_key (str): 问题列表字段名，默认 'question_list'
    model: 可选；None 时使用默认 Qwen 模型
    user_prompt (str|None): 可选用户提示
    **kwargs: 其它基类参数
""")

add_english_doc('data.operators.math_ops.QuestionFusionGenerator', """\
Use an LLM to fuse multiple questions into one and generate reasoning with \\\\boxed{{}} answer. Requires at least 2 questions under list_key.

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

add_chinese_doc('data.operators.codegen_ops.CodeInstructionGenerator', """\
代码生成流水线算子：指令标准化生成器。

从原始对话消息（messages）中抽取用户指令，并将其重写为统一的“代码增强指令”，输出为一条英文描述 + 一个包含完整函数骨架的 Python 代码块。

输出示例结构（默认 input_key='messages', output_key='generated_instruction'):

- messages: 原始多轮对话（保持不变）
- generated_instruction (str): 标准化后的英文指令 + Python 代码块

Args:
    model: LazyLLM 模型对象（必需），会被 share() 后复用。
    prompt_template (str|None): 可选，自定义系统提示词（若提供则替换默认 sys_prompt）。
    input_key (str): 输入对话字段名，默认 'messages'。
    output_key (str): 输出标准化指令字段名，默认 'generated_instruction'。
    **kwargs: 传递给基类算子的其它参数（如 _max_workers、_save_data 等）。
""")

add_english_doc('data.operators.codegen_ops.CodeInstructionGenerator', """\
Code-gen pipeline operator: CodeInstructionGenerator.

Extracts the user instruction from raw messages and rewrites it into a standardized English instruction plus a Python function skeleton code block.

Typical output structure (default input_key='messages', output_key='generated_instruction'):

- messages: original multi-turn messages (unchanged)
- generated_instruction (str): standardized English instruction + Python code block

Args:
    model: a LazyLLM model object (required), shared via share().
    prompt_template (str|None): optional custom system prompt (overrides default).
    input_key (str): input conversation field name, default 'messages'.
    output_key (str): output standardized instruction field name, default 'generated_instruction'.
    **kwargs: extra args forwarded to the base operator (e.g. _max_workers, _save_data).
""")

add_example('data.operators.codegen_ops.CodeInstructionGenerator', r"""
from lazyllm.tools.data.operators.codegen_ops import CodeInstructionGenerator

op = CodeInstructionGenerator(model=model,
                                         input_key='messages',
                                         output_key='generated_instruction')
item = {
    'messages': [
        {'role': 'user', 'content': '写一个 Python 函数，打印 hello'}
    ]
}
res = op(item)
print(res)

# Output Example:
# {
#    'messages': [...],
#    'generated_instruction': "Write a Python function that prints 'hello'.\\n"
#                             "```python\\n"
#                             "def solution():\\n"
#                             "    print('hello')\\n"
#                             "```"
# }
""")

add_chinese_doc('data.operators.codegen_ops.ScriptSynthesizer', """\
代码生成流水线算子：指令到代码生成器。

给定自然语言代码指令（通常是上一阶段生成的 generated_instruction 或精简后的 instruction），生成对应的 Python 源代码文本，并尝试自动去掉 Markdown 代码块外壳，只保留代码本身。

输出示例结构（默认 input_key='instruction', output_key='new_code'):

- instruction: 自然语言代码指令
- new_code (str): 生成的 Python 代码字符串

Args:
    model: LazyLLM 模型对象（必需）。
    prompt_template (str|None): 可选，自定义系统提示词。
    input_key (str): 输入指令字段名，默认 'instruction'。
    output_key (str): 输出代码字段名，默认 'new_code'。
    **kwargs: 传递给基类算子的其它参数。
""")

add_english_doc('data.operators.codegen_ops.ScriptSynthesizer', """\
Code-gen pipeline operator: ScriptSynthesizer.

Given a natural language code instruction (often from the previous generated_instruction or a cleaned instruction field), generates the corresponding Python source code, stripping Markdown code fences when present.

Typical output structure (default input_key='instruction', output_key='new_code'):

- instruction: natural language code instruction
- new_code (str): generated Python code string

Args:
    model: a LazyLLM model object (required).
    prompt_template (str|None): optional custom system prompt.
    input_key (str): input instruction field name, default 'instruction'.
    output_key (str): output code field name, default 'new_code'.
    **kwargs: extra args forwarded to the base operator.
""")

add_example('data.operators.codegen_ops.ScriptSynthesizer', """\
```python
from lazyllm.tools.data.operators.codegen_ops import ScriptSynthesizer

op = ScriptSynthesizer(model=model,
                                    input_key='instruction',
                                    output_key='new_code')
item = {
    'instruction': 'Write a Python function that prints "hello".'
}
res = op(item)
print(res)
# {
#   'instruction': 'Write a Python function that prints "hello".',
#   'new_code': "def solution():\\n    print('hello')"
# }
```
""")

add_chinese_doc('data.operators.codegen_ops.LogicIntegrityAuditor', """\
代码生成流水线算子：代码质量评估器。

对单条 (generated_instruction, generated_code) 样本进行自动代码评审，输出一个质量分数（0–10）与一段文字反馈，默认使用 JSON 格式进行解析。

输出示例结构（默认 input_instruction_key='instruction', input_code_key='new_code'):

- instruction: 标准化指令
- new_code: 生成的代码
- quality_score: 质量得分（int/float，取决于 JsonFormatter 解析）
- feedback: 文字反馈

Args:
    model: LazyLLM 模型对象（必需），会被 JsonFormatter 包装为 JSON 输出。
    prompt_template (str|None): 可选，自定义系统提示词。
    input_instruction_key (str): 输入指令字段名，默认 'instruction'。
    input_code_key (str): 输入代码字段名，默认 'new_code'。
    output_score_key (str): 输出分数字段名，默认 'quality_score'。
    output_feedback_key (str): 输出反馈字段名，默认 'feedback'。
    **kwargs: 传递给基类算子的其它参数。
""")

add_english_doc('data.operators.codegen_ops.LogicIntegrityAuditor', """\
Code-gen pipeline operator: LogicIntegrityAuditor.

Evaluates a single (generated_instruction, generated_code) sample, producing a quality score (0–10) and textual feedback, parsed from a JSON-formatted model response.

Typical output structure (default input_instruction_key='instruction', input_code_key='new_code'):

- instruction: standardized instruction
- new_code: generated code
- quality_score: numeric quality score (int/float depending on JsonFormatter parsing)
- feedback: textual review feedback

Args:
    model: a LazyLLM model object (required), wrapped with JsonFormatter.
    prompt_template (str|None): optional custom system prompt.
    input_instruction_key (str): input instruction field name, default 'instruction'.
    input_code_key (str): input code field name, default 'new_code'.
    output_score_key (str): output score field name, default 'quality_score'.
    output_feedback_key (str): output feedback field name, default 'feedback'.
    **kwargs: extra args forwarded to the base operator.
""")

add_example('data.operators.codegen_ops.LogicIntegrityAuditor', """\
```python
from lazyllm.tools.data.operators.codegen_ops import LogicIntegrityAuditor

op = LogicIntegrityAuditor(model=model)
item = {
    'instruction': "Write a Python function that prints 'hello'.",
    'new_code': "def solution():\\n    print('hello')"
}
res = op(item)
print(res)
# {
#   'instruction': "Write a Python function that prints 'hello'.",
#   'new_code': "def solution():\\n    print('hello')",
#   'quality_score': 8,
#   'feedback': 'Good code. The logic is clear and follows PEP8.'
# }
```
""")

add_chinese_doc('data.operators.codegen_ops.ThresholdSieve', """\
代码生成流水线算子：代码质量分数过滤器。

基于 LogicIntegrityAuditor 的打分结果，对样本进行区间过滤：

- 若样本尚未包含 quality_score/feedback，会先自动调用内部 scorer 进行评估；
- 若得分在 [min_score, max_score] 区间内，则为样本打上标签并保留；
- 否则返回空列表 []，表示此样本在流水线中被过滤掉。

输出示例结构（默认 output_key='quality_score_filter_label'）：

- instruction: ...
- new_code: ...
- quality_score: 8
- feedback: 'Good code. ...'
- quality_score_filter_label: 1  （通过过滤为 1，未通过则样本被丢弃）

Args:
    model: LazyLLM 模型对象（必需），用于内部评估。
    min_score (int): 通过过滤的最小分数（含），默认 7。
    max_score (int): 通过过滤的最大分数（含），默认 10。
    input_instruction_key (str): 输入指令字段名，默认 'instruction'。
    input_code_key (str): 输入代码字段名，默认 'new_code'。
    output_score_key (str): 分数字段名，默认 'quality_score'。
    output_feedback_key (str): 反馈字段名，默认 'feedback'。
    output_key (str): 过滤标签字段名，默认 'quality_score_filter_label'。
    **kwargs: 传递给基类算子的其它参数。
""")

add_english_doc('data.operators.codegen_ops.ThresholdSieve', """\
Code-gen pipeline operator: ThresholdSieve.

Filters samples based on code quality scores produced by LogicIntegrityAuditor:

- If quality_score/feedback are missing, it first calls the internal scorer.
- If the score is within [min_score, max_score], the sample is kept and labeled.
- Otherwise, it returns an empty list [], effectively dropping the sample from the pipeline.

Typical output structure (default output_key='quality_score_filter_label'):

- instruction: ...
- new_code: ...
- quality_score: 8
- feedback: 'Good code. ...'
- quality_score_filter_label: 1  (1 for passed, 0 otherwise; non-passed samples are dropped)

Args:
    model: a LazyLLM model object (required), used by the internal scorer.
    min_score (int): minimum score (inclusive) to pass the filter, default 7.
    max_score (int): maximum score (inclusive) to pass the filter, default 10.
    input_instruction_key (str): input instruction field, default 'instruction'.
    input_code_key (str): input code field, default 'new_code'.
    output_score_key (str): score field name, default 'quality_score'.
    output_feedback_key (str): feedback field name, default 'feedback'.
    output_key (str): filter label field name, default 'quality_score_filter_label'.
    **kwargs: extra args forwarded to the base operator.
""")

add_example('data.operators.codegen_ops.ThresholdSieve', """\
```python
from lazyllm.tools.data.operators.codegen_ops import ThresholdSieve

op = ThresholdSieve(model=model, min_score=7, max_score=10)
item = {
    'instruction': "Write a Python function that prints 'hello'.",
    'new_code': "def solution():\\n    print('hello')"
}
res = op(item)
print(res)
# {
#   'instruction': '...',
#   'new_code': '...',
#   'quality_score': 8,
#   'feedback': 'Good code. The logic is clear and follows PEP8.',
#   'quality_score_filter_label': 1
# }
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
add_chinese_doc('data.operators.filter_op.TargetLanguageFilter', """\
使用 FastText 进行语言识别，仅保留指定语言的文本。

Args:
    input_key (str): 文本字段名，默认 'content'
    target_language (str|list): 目标语言代码，如 'zho_Hans'、'eng_Latn'
    threshold (float): 置信度阈值，默认 0.6
    model_path (str|None): FastText 模型路径
    _concurrency_mode (str): 可选，并发模式
""")

add_english_doc('data.operators.filter_op.TargetLanguageFilter', """\
Filter text by language using FastText. Keeps only texts in the specified language(s).

Args:
    input_key (str): key of the text field, default 'content'
    target_language (str|list): target language code(s), e.g. 'zho_Hans', 'eng_Latn'
    threshold (float): confidence threshold, default 0.6
    model_path (str|None): path to FastText model
    _concurrency_mode (str): optional concurrency mode
""")

add_example('data.operators.filter_op.TargetLanguageFilter', """\
```python
from lazyllm.tools.data import filter

func = filter.TargetLanguageFilter(input_key='content', target_language='zho_Hans', threshold=0.3)
inputs = [{'content': '这是一段中文文本。'}, {'content': 'This is English.'}]
res = func(inputs)
print(res)
# [{'content': '这是一段中文文本。'}]
```
""")

add_chinese_doc('data.operators.filter_op.MinHashDeduplicator', """\
使用 MinHash LSH 去除近似重复文本，批处理时保留首次出现的文本。

Args:
    input_key (str): 文本字段名，默认 'content'
    threshold (float): 相似度阈值，默认 0.85
    num_perm (int): MinHash 排列数，默认 128
    use_n_gram (bool): 是否使用 n-gram，默认 True
    ngram (int): n-gram 长度，默认 5
""")

add_english_doc('data.operators.filter_op.MinHashDeduplicator', """\
Remove near-duplicate texts using MinHash LSH. For batch input, keeps first occurrence of each unique text.

Args:
    input_key (str): key of the text field, default 'content'
    threshold (float): similarity threshold, default 0.85
    num_perm (int): number of MinHash permutations, default 128
    use_n_gram (bool): use n-gram, default True
    ngram (int): n-gram size, default 5
""")

add_example('data.operators.filter_op.MinHashDeduplicator', """\
```python
from lazyllm.tools.data import filter

func = filter.MinHashDeduplicator(input_key='content', threshold=0.85)
inputs = [{'uid': '0', 'content': '这是第一段不同的内容。'}, {'uid': '1', 'content': '这是第一段不同的内容。'}]
res = func(inputs)
print(res)
# [{'uid': '0', 'content': '这是第一段不同的内容。'}]
```
""")

add_chinese_doc('data.operators.filter_op.WordBlocklistFilter', """\
使用 AC 自动机多模式匹配过滤包含敏感词/违禁词超过阈值的文本。

Args:
    input_key (str): 文本字段名，默认 'content'
    blocklist (list|None): 违禁词列表
    blocklist_path (str|None): 违禁词文件路径
    language (str): 语言，'zh' 或 'en'，默认 'zh'
    threshold (int): 允许出现的违禁词最大数量，默认 1
    _concurrency_mode (str): 可选，并发模式
""")

add_english_doc('data.operators.filter_op.WordBlocklistFilter', """\
Filter text containing more than threshold blocked words using Aho-Corasick automaton.

Args:
    input_key (str): key of the text field, default 'content'
    blocklist (list|None): list of blocked words
    blocklist_path (str|None): path to blocklist file
    language (str): language, 'zh' or 'en', default 'zh'
    threshold (int): max allowed occurrences of blocked words, default 1
    _concurrency_mode (str): optional concurrency mode
""")

add_example('data.operators.filter_op.WordBlocklistFilter', """\
```python
from lazyllm.tools.data import filter

func = filter.WordBlocklistFilter(input_key='content', blocklist=['敏感', '违禁'], threshold=0)
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


add_chinese_doc('data.operators.tool_use_ops.ChainedLogicAssembler', """\
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

add_english_doc('data.operators.tool_use_ops.ChainedLogicAssembler', """\
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

add_example('data.operators.tool_use_ops.ChainedLogicAssembler', """\
```python
from lazyllm.tools.data.operators.tool_use_ops import ChainedLogicAssembler

atomic_tasks = [
    {'task': '获取出发地与目的地'},
    {'task': '确认出行日期'},
    {'task': '筛选符合条件的车次'},
]
op = ChainedLogicAssembler(model=model, input_key='atomic_tasks', output_key='sequential_tasks')
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

add_chinese_doc('data.operators.tool_use_ops.TopologyArchitect', """\
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

add_english_doc('data.operators.tool_use_ops.TopologyArchitect', """\
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

add_example('data.operators.tool_use_ops.TopologyArchitect', """\
```python
from lazyllm.tools.data.operators.tool_use_ops import TopologyArchitect

atomic_tasks = [
    {'task': '收集出行需求'},
    {'task': '查询可选车次'},
    {'task': '对比价格与时间'},
    {'task': '完成下单支付'},
]
op = TopologyArchitect(model=model, input_key='atomic_tasks', output_key='para_seq_tasks')
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

add_chinese_doc('data.operators.tool_use_ops.ViabilitySieve', """\
工具调用数据生成算子：组合任务可行性过滤器。

对一组“组合任务”进行可运行性与完备性评审，筛选出被认为合理可行的组合任务列表。

模型内部期望的中间 JSON 结构：

- items: 列表，每项包含 composed_task、is_valid、reason 等字段。

在算子输出中，仅保留 is_valid 为 true 且含有 composed_task 的项；如果模型未按预期输出，则尽量回退返回原 items 或原始 parsed 结果。

Args:
    model: LazyLLM 模型对象（必需）。
    input_composition_key (str): 输入组合任务字段名，默认 'composition_tasks'。
    input_atomic_key (str): 输入原子任务字段名（可选），默认 'atomic_tasks'。
    output_key (str): 输出过滤后组合任务字段名，默认 'filtered_composition_tasks'。
    system_prompt (str|None): 可选系统提示词。
    **kwargs: 传递给基类算子的其它参数。
""")

add_english_doc('data.operators.tool_use_ops.ViabilitySieve', """\
Tool-use data operator: composition task feasibility filter.

Evaluates a list of composed tasks for feasibility and completeness, and filters out invalid ones.

Expected intermediate JSON from the model:

- items: list of dicts with composed_task, is_valid, reason, etc.

On output, only keeps composed_task values where is_valid is true. If the model output does not match the schema, it falls back to returning items or the raw parsed result.

Args:
    model: a LazyLLM model object (required).
    input_composition_key (str): input composition task field name, default 'composition_tasks'.
    input_atomic_key (str): input atomic task field name (optional), default 'atomic_tasks'.
    output_key (str): output filtered composition task field name, default 'filtered_composition_tasks'.
    system_prompt (str|None): optional system prompt.
    **kwargs: extra args passed to the base operator.
""")

add_example('data.operators.tool_use_ops.ViabilitySieve', """\
```python
from lazyllm.tools.data.operators.tool_use_ops import ViabilitySieve

composition_tasks = ['先获取出发地和目的地再筛选车次', '直接随机推荐一个车次']
atomic_tasks = [
    {'task': '获取出发地与目的地'}, {'task': '确认出行日期'}, {'task': '筛选符合条件的车次'}
]
op = ViabilitySieve(model=model,
                           input_composition_key='composition_tasks',
                           input_atomic_key='atomic_tasks',
                           output_key='filtered_composition_tasks')
print(op({'composition_tasks': composition_tasks, 'atomic_tasks': atomic_tasks}))
# {
#   'composition_tasks': [...],
#   'atomic_tasks': [...],
#   'filtered_composition_tasks': ['先获取出发地和目的地再筛选车次', ...]
# }
```
""")

add_chinese_doc('data.operators.tool_use_ops.ProtocolSpecifier', """\
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
    input_composition_key (str): 输入组合任务字段名，默认 'composition_task'。
    input_atomic_key (str): 输入原子任务字段名，默认 'atomic_tasks'。
    output_key (str): 输出函数规格列表字段名，默认 'functions'。
    system_prompt (str|None): 可选系统提示词。
    **kwargs: 传递给基类算子的其它参数。
""")

add_english_doc('data.operators.tool_use_ops.ProtocolSpecifier', """\
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
    input_composition_key (str): input composition task field name, default 'composition_task'.
    input_atomic_key (str): input atomic task field name, default 'atomic_tasks'.
    output_key (str): output function spec list field name, default 'functions'.
    system_prompt (str|None): optional system prompt.
    **kwargs: extra args passed to the base operator.
""")

add_example('data.operators.tool_use_ops.ProtocolSpecifier', """\
```python
from lazyllm.tools.data.operators.tool_use_ops import ProtocolSpecifier

composition_task = '根据用户出发地、目的地和日期查询可选高铁车次并返回候选列表'
atomic_tasks = [
    {'task': '获取出发地与目的地'},
    {'task': '确认出行日期'},
    {'task': '调用车次查询接口并过滤结果'},
]
op = ProtocolSpecifier(model=model,
                       input_composition_key='composition_task',
                       input_atomic_key='atomic_tasks',
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

add_chinese_doc('data.operators.tool_use_ops.DialogueSimulator', """\
工具调用数据生成算子：多轮对话生成器（含 Tool 调用）。

根据组合任务与可用函数列表，生成带有 User / Assistant / Tool 三种角色的多轮对话 JSON，用于构造工具调用训练数据。

输出 JSON 典型结构：

- messages: 列表，每项为：
  - role: 'user' | 'assistant' | 'tool'
  - content: 文本内容
  - name: 工具名（仅 role == 'tool' 时可选）

Args:
    model: LazyLLM 模型对象（必需）。
    input_composition_key (str): 输入组合任务字段名，默认 'composition_task'。
    input_functions_key (str): 输入函数列表字段名，默认 'functions'。
    output_key (str): 输出多轮对话字段名，默认 'conversation'。
    n_turns (int): 期望的轮次数量（提示给模型），默认 6。
    system_prompt (str|None): 可选系统提示词。
    **kwargs: 传递给基类算子的其它参数。
""")

add_english_doc('data.operators.tool_use_ops.DialogueSimulator', """\
Tool-use data operator: multi-turn conversation generator (with tools).

Given a composed task and a list of available functions, generates a multi-turn conversation JSON involving User, Assistant and Tool roles, suitable for tool-calling training data.

Typical JSON structure:

- messages: list of dicts:
  - role: 'user' | 'assistant' | 'tool'
  - content: text content
  - name: tool name (optional, when role == 'tool')

Args:
    model: a LazyLLM model object (required).
    input_composition_key (str): input composition task field name, default 'composition_task'.
    input_functions_key (str): input function list field name, default 'functions'.
    output_key (str): output conversation field name, default 'conversation'.
    n_turns (int): desired number of turns (as a hint to the model), default 6.
    system_prompt (str|None): optional system prompt.
    **kwargs: extra args passed to the base operator.
""")

add_example('data.operators.tool_use_ops.DialogueSimulator', """\
```python
from lazyllm.tools.data.operators.tool_use_ops import DialogueSimulator

composition_task = '根据用户需求查询并推荐合适的高铁车次'
functions = [
    {
        'name': 'query_train_tickets',
        'description': '查询高铁车次',
        'args': [...],
        'returns': {...},
    }
]
op = DialogueSimulator(model=model,
                                    input_composition_key='composition_task',
                                    input_functions_key='functions',
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

add_chinese_doc('data.operators.tool_use_ops.ContextualBeacon', """\
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

add_english_doc('data.operators.tool_use_ops.ContextualBeacon', """\
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

add_example('data.operators.tool_use_ops.ContextualBeacon', r"""
from lazyllm.tools.data.operators.tool_use_ops import ContextualBeacon

op = ContextualBeacon(model=model, input_key='content', output_key='scenario')
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

add_chinese_doc('data.operators.tool_use_ops.ScenarioDiverger', """\
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

add_english_doc('data.operators.tool_use_ops.ScenarioDiverger', """\
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

add_example('data.operators.tool_use_ops.ScenarioDiverger', """\
```python
from lazyllm.tools.data.operators.tool_use_ops import ScenarioDiverger

base = {
    'scene': '用户咨询高铁购票服务',
    'domain': '出行/购票',
    'assistant_goal': '帮助用户完成车次筛选并购票',
}
op = ScenarioDiverger(model=model, input_key='scenario', output_key='expanded_scenarios', n=3)
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

add_chinese_doc('data.operators.tool_use_ops.DecompositionKernel', """\
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

add_english_doc('data.operators.tool_use_ops.DecompositionKernel', """\
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

add_example('data.operators.tool_use_ops.DecompositionKernel', """\
```python
from lazyllm.tools.data.operators.tool_use_ops import DecompositionKernel

scenario = {
    'scene': '用户咨询高铁购票服务',
    'assistant_goal': '帮助用户完成车次筛选并购票',
}
op = DecompositionKernel(model=model, input_key='scenario', output_key='atomic_tasks', n=4)
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

add_chinese_doc('data.operators.text2sql_ops.SQLForge', """\
Text2SQL 数据生成算子：SQL 生成器。

基于数据库 Schema 与样例数据，为给定或全部数据库自动生成可执行的 SQL 语句集合，并标注大致复杂度类型。

主要行为：

- 对每个数据库生成 output_num 条 SQL。
- 内置默认提示词（可自定义 prompt_template），控制难度标签（easy/medium/hard 等）。
- 从模型返回中解析出 ```sql ... ``` 代码块中的 SQL 文本。

Args:
    model: LazyLLM 模型对象（必需），会被 share() 后复用。
    database_manager: 提供数据库 Schema 与样例数据的管理器（必需），需实现：
        - list_databases()
        - get_create_statements_and_insert_statements(db_name)
    output_num (int): 每个数据库生成的 SQL 数量，默认 300。
    prompt_template: 可选，自定义 prompt 构造器对象，需实现 build_prompt(...)。
    system_prompt (str|None): 可选系统提示词，不传则使用内置英文提示。
    **kwargs: 传递给基类 Text2SQLOps/LazyLLMDataBase 的其它参数。
""")

add_english_doc('data.operators.text2sql_ops.SQLForge', """\
Text2SQL data operator: SQLForge.

Generates executable SQL queries for one or multiple databases based on their schema and optional sample data, and labels each query with a rough complexity type.

Behavior:

- Generates output_num SQLs per database.
- Uses a default English system prompt (or a custom prompt_template) to control complexity labels (easy/medium/hard, etc.).
- Parses SQL text from model responses, preferring ```sql ... ``` code blocks.

Args:
    model: a LazyLLM model object (required), shared via share().
    database_manager: database manager (required) implementing:
        - list_databases()
        - get_create_statements_and_insert_statements(db_name)
    output_num (int): number of SQLs to generate per database, default 300.
    prompt_template: optional custom prompt builder with build_prompt(...).
    system_prompt (str|None): optional system prompt, defaults to a built-in English prompt.
    **kwargs: extra args forwarded to the Text2SQLOps/LazyLLMDataBase base class.
""")

add_example('data.operators.text2sql_ops.SQLForge', """\
```python
from lazyllm.tools.data.operators.text2sql_ops import SQLForge

# 假设 database_manager 已封装了你的 SQLite / Postgres 等数据库
op = SQLForge(model=model, database_manager=database_manager, output_num=10)

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

add_chinese_doc('data.operators.text2sql_ops.SQLRuntimeSieve', """\
Text2SQL 数据过滤算子：SQL 可执行性过滤器。

对每条数据中的 SQL 进行简单语法形态过滤（仅保留 SELECT / WITH 开头的查询），并调用 database_manager 进行 EXPLAIN 校验；只保留可在目标库上成功执行的 SQL。

Args:
    database_manager: 提供数据库连接与 explain 能力的管理器（必需），需实现：
        - database_exists(db_id)
        - batch_explain_queries(list[(db_id, sql)])
    **kwargs: 传递给基类算子的其它参数。
""")

add_english_doc('data.operators.text2sql_ops.SQLRuntimeSieve', """\
Text2SQL data operator: SQLRuntimeSieve.

Filters SQL queries by:

1. Keeping only queries that look like SELECT/WITH queries.
2. Calling database_manager to run EXPLAIN (or similar) and keeping only those that execute successfully.

Args:
    database_manager: database manager (required) implementing:
        - database_exists(db_id)
        - batch_explain_queries(list[(db_id, sql)])
    **kwargs: extra args forwarded to the base operator.
""")

add_example('data.operators.text2sql_ops.SQLRuntimeSieve', """\
```python
from lazyllm.tools.data.operators.text2sql_ops import SQLRuntimeSieve

op = SQLRuntimeSieve(database_manager=database_manager)
item = {'db_id': 'db_1', 'SQL': 'SELECT * FROM users;'}
res = op(item)
print(res)  # 若 SQL 可在 db_1 上 explain 成功，则返回原始 dict；否则返回 None
```
""")

add_chinese_doc('data.operators.text2sql_ops.SQLIntentSynthesizer', """\
Text2SQL 数据生成算子：自然语言问题生成器。

基于给定 SQL + 数据库 Schema 以及列注释信息，生成与 SQL 语义对应的自然语言问题，并可附带“外部知识”提示，以支持 Text2SQL 训练。

主要特性：

- 支持多候选问题生成（input_query_num），并通过 embedding 去重/多样性选择。
- 内置输出格式标记：[QUESTION-START]/[QUESTION-END] 与 [EXTERNAL-KNOWLEDGE-START]/[...-END]。

Args:
    model: LazyLLM 文本生成模型（必需）。
    embedding_model: 可选向量模型，用于对候选问题做多样性选择；需支持：
        - generate_embedding_from_input(texts) 或直接可调用(texts)。
    database_manager: 提供 Schema 的管理器（必需），需实现：
        - get_create_statements_and_insert_statements(db_id)
    input_query_num (int): 每条 SQL 生成候选问题的数量，默认 5。
    prompt_template: 可选，自定义 prompt 构造器。
    system_prompt (str|None): 可选系统提示词，默认简要英文助手提示。
    **kwargs: 其它传递给基类算子的参数。
""")

add_english_doc('data.operators.text2sql_ops.SQLIntentSynthesizer', """\
Text2SQL data operator: SQLIntentSynthesizer.

Given a SQL query and database schema (with optional column descriptions), generates a natural language question aligned with the SQL semantics, plus optional external knowledge text.

Key features:

- Generates multiple candidate questions (input_query_num) and selects one using embeddings-based diversity.
- Uses special markers in model output: [QUESTION-START]/[QUESTION-END] and [EXTERNAL-KNOWLEDGE-START]/[...-END].

Args:
    model: text generation model (required).
    embedding_model: optional embedding model, supporting:
        - generate_embedding_from_input(texts) or callable(texts).
    database_manager: schema provider (required) implementing:
        - get_create_statements_and_insert_statements(db_id)
    input_query_num (int): number of question candidates per SQL, default 5.
    prompt_template: optional custom prompt builder.
    system_prompt (str|None): optional system prompt, default simple English helper.
    **kwargs: extra args forwarded to the base operator.
""")

add_example('data.operators.text2sql_ops.SQLIntentSynthesizer', """\
```python
from lazyllm.tools.data.operators.text2sql_ops import SQLIntentSynthesizer

op = SQLIntentSynthesizer(model=model,
                               embedding_model=embedding_model,
                               database_manager=database_manager,
                               input_query_num=5)
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

add_chinese_doc('data.operators.text2sql_ops.TSQLSemanticAuditor', """\
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

add_english_doc('data.operators.text2sql_ops.TSQLSemanticAuditor', """\
Text2SQL data operator: TSQLSemanticAuditor.

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

add_example('data.operators.text2sql_ops.TSQLSemanticAuditor', """\
```python
from lazyllm.tools.data.operators.text2sql_ops import TSQLSemanticAuditor

op = TSQLSemanticAuditor(model=model, database_manager=database_manager)
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

add_chinese_doc('data.operators.text2sql_ops.SQLContextAssembler', """\
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

add_english_doc('data.operators.text2sql_ops.SQLContextAssembler', """\
Text2SQL data operator: SQLContextAssembler.

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

add_example('data.operators.text2sql_ops.SQLContextAssembler', """\
```python
from lazyllm.tools.data.operators.text2sql_ops import SQLContextAssembler

op = SQLContextAssembler(database_manager=database_manager)
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

add_chinese_doc('data.operators.text2sql_ops.SQLReasoningTracer', """\
Text2SQL 数据生成算子：CoT 轨迹生成器。

针对给定 (问题, SQL, 数据库 Schema, 证据) 生成若干条“从问题到 SQL 的链式思考（Chain-of-Thought）”文本，用于训练/分析。

Args:
    model: LazyLLM 模型对象（必需）。
    database_manager: 提供 Schema 的管理器（必需），需实现：
        - get_create_statements_and_insert_statements(db_id)
    prompt_template: 可选，自定义 prompt 构造器。
    output_num (int): 每条样本生成的 CoT 轨迹数量，默认 3（>=1）。
    **kwargs: 其它传递给基类算子的参数。
""")

add_english_doc('data.operators.text2sql_ops.SQLReasoningTracer', """\
Text2SQL data operator: SQLReasoningTracer.

For each (question, SQL, schema, evidence) item, generates multiple chain-of-thought (CoT) reasoning traces from question to SQL.

Args:
    model: a LazyLLM model object (required).
    database_manager: schema provider (required) implementing:
        - get_create_statements_and_insert_statements(db_id)
    prompt_template: optional custom prompt builder.
    output_num (int): number of CoT trajectories per item, default 3 (>=1).
    **kwargs: extra args forwarded to the base operator.
""")

add_example('data.operators.text2sql_ops.SQLReasoningTracer', """\
```python
from lazyllm.tools.data.operators.text2sql_ops import SQLReasoningTracer

op = SQLReasoningTracer(model=model, database_manager=database_manager, output_num=3)
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

add_chinese_doc('data.operators.text2sql_ops.SQLConsensusUnifier', """\
Text2SQL 数据处理算子：CoT 轨迹投票选择器。

对一组 CoT 轨迹（cot_responses）进行 SQL 解析与执行，基于执行结果的一致性与正确性，从中选出“最佳” CoT 及对应 SQL。

行为：

- 从每条 CoT 中解析 SQL（使用与 SQLForge 相同的解析逻辑）。
- 调用 database_manager.batch_execute_queries 执行 SQL，计算结果 signature 与 success。
- 使用投票策略 _vote_select 选择最终 CoT，并将：
  - output_cot_key（默认 'cot_reasoning'）设置为最佳 CoT 文本；
  - 同时覆盖数据中的 'SQL' 字段为最佳 SQL。

Args:
    database_manager: 提供 SQL 执行能力的管理器（必需），需实现：
        - batch_execute_queries(list[(db_id, sql)])
    **kwargs: 其它传递给基类算子的参数。
""")

add_english_doc('data.operators.text2sql_ops.SQLConsensusUnifier', """\
Text2SQL data operator: SQLConsensusUnifier.

Given multiple CoT traces (cot_responses), parses SQL from each, executes them, and selects the best CoT/SQL pair based on execution consistency and success.

Behavior:

- Parses SQL from each CoT using the same logic as SQLForge.
- Calls database_manager.batch_execute_queries to get execution results and signatures.
- Uses a voting strategy (_vote_select) to pick the best candidate, then:
  - sets output_cot_key (default 'cot_reasoning') to the winning CoT,
  - overwrites data['SQL'] with the winning SQL.

Args:
    database_manager: query execution provider (required) implementing:
        - batch_execute_queries(list[(db_id, sql)])
    **kwargs: extra args forwarded to the base operator.
""")

add_example('data.operators.text2sql_ops.SQLConsensusUnifier', """\
```python
from lazyllm.tools.data.operators.text2sql_ops import SQLConsensusUnifier

op = SQLConsensusUnifier(database_manager=database_manager)
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

add_chinese_doc('data.operators.text2sql_ops.SQLSyntaxProfiler', """\
Text2SQL 数据分类算子：SQL 组件难度分类器。

使用 SQL 结构级别的难度评估器（EvalHardness/EvalHardnessLite），根据 SQL 中涉及的组件复杂度对其进行难度打标（easy/medium/hard/extra 等）。

Args:
    difficulty_thresholds (list[int]|None): 难度阈值列表，默认 [2, 4, 6]。
    difficulty_labels (list[str]|None): 难度标签列表，默认 ['easy', 'medium', 'hard', 'extra']。
    **kwargs: 其它传递给基类算子的参数。
""")

add_english_doc('data.operators.text2sql_ops.SQLSyntaxProfiler', """\
Text2SQL data operator: SQLSyntaxProfiler.

Classifies SQL difficulty based on structural components using EvalHardness/EvalHardnessLite, assigning labels such as easy/medium/hard/extra.

Args:
    difficulty_thresholds (list[int]|None): thresholds list, default [2, 4, 6].
    difficulty_labels (list[str]|None): label list, default ['easy', 'medium', 'hard', 'extra'].
    **kwargs: extra args forwarded to the base operator.
""")

add_example('data.operators.text2sql_ops.SQLSyntaxProfiler', """\
```python
from lazyllm.tools.data.operators.text2sql_ops import SQLSyntaxProfiler

op = SQLSyntaxProfiler()
item = {'SQL': 'SELECT count(*) FROM orders WHERE status = \\'paid\\';'}
res = op(item)
print(res)
# {
#   'SQL': 'SELECT count(*) FROM orders WHERE status = \\'paid\\';',
#   'sql_component_difficulty': 'easy'
# }
```
""")

add_chinese_doc('data.operators.text2sql_ops.SQLEffortRanker', """\
Text2SQL 数据分类算子：SQL 执行难度分类器。

基于 SQLContextAssembler 生成的 prompt、多次采样生成 SQL 并与金标 SQL 在数据库上对比执行结果，从“可被模型正确生成的次数”角度对样本执行难度进行分类。

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

add_english_doc('data.operators.text2sql_ops.SQLEffortRanker', """\
Text2SQL data operator: SQLEffortRanker.

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

add_example('data.operators.text2sql_ops.SQLEffortRanker', """\
```python
from lazyllm.tools.data.operators.text2sql_ops import SQLEffortRanker

op = SQLEffortRanker(model=model, database_manager=database_manager, num_generations=15)
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

# LLM based JSON operators docs
add_chinese_doc('data.operators.llm_base_ops.LLMDataJson', """\
基于 LLM 的 JSON 数据处理算子基类。提供结构化输出的基础逻辑，包括自动配置 JsonFormatter、重试机制以及预处理/验证/后处理生命周期。

构造函数参数:\n
- model: LazyLLM 模型实例。
- prompt: 可选，用于引导 LLM 的 Prompt（ChatPrompter 或字符串）。
- max_retries: 最大重试次数，默认 3。
- **kwargs: 其它传递给基类的并发或持久化参数。
""")

add_english_doc('data.operators.llm_base_ops.LLMDataJson', """\
Base class for LLM-based JSON data processing operators. Provides foundational logic for structured output,
including automatic JsonFormatter configuration, retry mechanisms, and a pre/verify/post-processing lifecycle.

Constructor args:\n
- model: a LazyLLM model instance.
- prompt: optional, ChatPrompter or string to guide the LLM.
- max_retries: maximum number of retries, default 3.
- **kwargs: additional concurrency or persistence arguments for the base class.
""")

add_chinese_doc('data.operators.llm_json_ops.FieldExtractor', """\
字段提取器。利用 LLM 根据提供的字段列表从输入文本中提取特定信息。

Args:
    model: LazyLLM 模型实例。
    prompt: 可选，自定义提取 Prompt。
    input_keys: 字段列表，默认为 ['persona', 'text', 'fields']。
    output_key: 结果存储在数据字典中的键名，默认 'structured_data'。
""")

add_english_doc('data.operators.llm_json_ops.FieldExtractor', """\
Field extractor. Uses LLM to extract specific information from input text based on a provided list of fields.

Args:
    model: a LazyLLM model instance.
    prompt: optional custom extraction prompt.
    input_keys: list of input keys, defaults to ['persona', 'text', 'fields'].
    output_key: key name to store results in the data dict, default 'structured_data'.
""")

add_example('data.operators.llm_json_ops.FieldExtractor', """\
```python
from lazyllm import OnlineChatModule
from lazyllm.tools.data.operators.llm_json_ops import FieldExtractor
model = OnlineChatModule(source='sensenova')
op = FieldExtractor(model=model)
inputs = [{
    'text': '张三，28岁，目前在上海',
    'fields': ['name', 'age', 'location']
}]
res = op(inputs)
print(res[0]['structured_data']) # {'name': '张三', 'age': '28', 'location': '上海'}
```
""")

add_chinese_doc('data.operators.llm_json_ops.SchemaExtractor', """\
架构提取器。利用 LLM 根据指定的 Schema（字典或 Pydantic 模型）从文本中提取结构化数据。

Args:
    model: LazyLLM 模型实例。
    prompt: 可选，自定义提取 Prompt。
    input_key: 输入文本的键名，默认 'text'。
    output_key: 结果存储在数据字典中的键名，默认 'structured_data'。
""")

add_english_doc('data.operators.llm_json_ops.SchemaExtractor', """\
Schema extractor. Uses LLM to extract structured data from text according to a specified schema (dict or Pydantic model).

Args:
    model: a LazyLLM model instance.
    prompt: optional custom extraction prompt.
    input_key: key name for input text, default 'text'.
    output_key: key name to store results in the data dict, default 'structured_data'.
""")

add_example('data.operators.llm_json_ops.SchemaExtractor', """\
```python
from lazyllm import OnlineChatModule
from lazyllm.tools.data.operators.llm_json_ops import SchemaExtractor
model = OnlineChatModule(source='sensenova')
op = SchemaExtractor(model=model)
inputs = [{'text': 'Math score is 95', 'schema': {'subject': 'str', 'score': 'int'}}]
res = op(inputs)
print(res[0]['structured_data']) # {'subject': 'Math', 'score': 95}
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

add_chinese_doc('data.embedding.EmbeddingFormatFlagEmbedding', """\
将数据格式化为 FlagEmbedding 训练格式的算子。

该算子将输入的 query、pos（正样本）、neg（负样本）格式化为 FlagEmbedding 框架所需的训练数据格式。
支持添加指令（instruction）字段用于有监督的 Embedding 训练。

Args:
    instruction (str, optional): 指令文本，用于有监督训练场景。默认为 None。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    dict: 包含 query、pos、neg 和可选 prompt 字段的字典。
""")

add_english_doc('data.embedding.EmbeddingFormatFlagEmbedding', """\
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

add_example('data.embedding.EmbeddingFormatFlagEmbedding', """\
```python
from lazyllm.tools.data import embedding

op = embedding.EmbeddingFormatFlagEmbedding(instruction='Represent this sentence for searching relevant passages:')
result = op({'query': 'machine learning', 'pos': ['ML tutorial'], 'neg': ['cooking recipe']})
# Returns: {'query': 'machine learning', 'pos': ['ML tutorial'], 'neg': ['cooking recipe'], 'prompt': 'Represent this sentence for searching relevant passages:'}
```
""")

add_chinese_doc('data.embedding.EmbeddingFormatSentenceTransformers', """\
将数据格式化为 SentenceTransformers 三元组训练格式的算子。

该算子将输入的 query、pos（正样本）、neg（负样本）转换为 SentenceTransformers 框架所需的 anchor-positive-negative 三元组格式。
适用于 MultipleNegativesRankingLoss 等损失函数的训练。

Args:
    **kwargs (dict): 可选的参数，传递给父类。

Returns:
    List[dict]: 包含 anchor、positive、negative 字段的字典列表，每对正负样本生成一个三元组。
""")

add_english_doc('data.embedding.EmbeddingFormatSentenceTransformers', """\
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

add_example('data.embedding.EmbeddingFormatSentenceTransformers', """\
```python
from lazyllm.tools.data import embedding

op = embedding.EmbeddingFormatSentenceTransformers()
result = op({'query': 'machine learning', 'pos': ['ML basics'], 'neg': ['cooking tips']})
# Returns: [{'anchor': 'machine learning', 'positive': 'ML basics', 'negative': 'cooking tips'}]
```
""")

add_chinese_doc('data.embedding.EmbeddingFormatTriplet', """\
将数据格式化为通用三元组格式的算子。

该算子将输入的 query、pos（正样本）、neg（负样本）转换为标准的三元组格式，
字段名为 query、positive、negative。适用于多种 Embedding 训练框架。

Args:
    **kwargs (dict): 可选的参数，传递给父类。

Returns:
    List[dict]: 包含 query、positive、negative 字段的字典列表，每对正负样本生成一个三元组。
""")

add_english_doc('data.embedding.EmbeddingFormatTriplet', """\
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

add_example('data.embedding.EmbeddingFormatTriplet', """\
```python
from lazyllm.tools.data import embedding

op = embedding.EmbeddingFormatTriplet()
result = op({'query': 'deep learning', 'pos': ['neural networks', 'AI'], 'neg': ['history', 'geography']})
# Returns list of triplets combining each positive with each negative
```
""")

add_chinese_doc('data.embedding.EmbeddingTrainTestSplitter', """\
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

add_english_doc('data.embedding.EmbeddingTrainTestSplitter', """\
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

add_example('data.embedding.EmbeddingTrainTestSplitter', """\
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
from lazyllm.tools.data import embedding

data = [{'query': 'machine learning', 'pos': ['ML tutorial', 'deep learning']}, {'query': 'cooking', 'pos': ['recipe']}]
result = embedding.build_embedding_corpus(data, input_pos_key='pos')
# Returns data with '_corpus' field pointing to corpus file containing unique passages
```
""")

add_chinese_doc('data.embedding.EmbeddingInitBM25', """\
初始化 BM25 索引的算子。

该算子基于语料库构建 BM25 索引，用于后续的关键词检索和困难负样本挖掘。
支持中英文分词，使用 jieba 进行中文分词，Stemmer 进行英文词干提取。

Args:
    language (str): 语言类型，'zh' 表示中文，'en' 表示英文，默认为 'zh'。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    List[dict]: 输入数据，每条数据添加了 BM25 索引和相关配置信息。
""")

add_english_doc('data.embedding.EmbeddingInitBM25', """\
An operator that initializes BM25 index.

This operator builds BM25 index based on corpus for subsequent keyword retrieval and hard negative mining.
Supports Chinese and English tokenization, using jieba for Chinese and Stemmer for English stemming.

Args:
    language (str): Language type, 'zh' for Chinese, 'en' for English, defaults to 'zh'.
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    List[dict]: Input data with BM25 index and related configuration added to each item.
""")

add_example('data.embedding.EmbeddingInitBM25', """\
```python
from lazyllm.tools.data import embedding

# First build corpus, then initialize BM25
corpus_op = embedding.build_embedding_corpus(input_pos_key='pos')
bm25_op = embedding.EmbeddingInitBM25(language='zh')
# Returns data with '_bm25' index and tokenizer configuration
```
""")

add_chinese_doc('data.embedding.EmbeddingInitSemantic', """\
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

add_english_doc('data.embedding.EmbeddingInitSemantic', """\
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

add_example('data.embedding.EmbeddingInitSemantic', """\
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
from lazyllm.tools.data import embedding

# After building corpus and initializing BM25
data = {'query': 'machine learning', 'pos': ['ML tutorial'], '_bm25': bm25_index, '_bm25_corpus': corpus}
result = embedding.mine_bm25_negatives(data, num_negatives=5)
# Returns data with 'neg' field containing BM25-mined negative samples
```
""")

add_chinese_doc('data.embedding.mine_random_negatives', """\
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

add_english_doc('data.embedding.mine_random_negatives', """\
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

add_example('data.embedding.mine_random_negatives', """\
```python
from lazyllm.tools.data import embedding

data = {'query': 'machine learning', 'pos': ['ML tutorial'], '_corpus': corpus_path}
result = embedding.mine_random_negatives(data, num_negatives=5, seed=123)
# Returns data with 'neg' field containing randomly selected negative samples
```
""")

add_chinese_doc('data.embedding.EmbeddingMineSemanticNegatives', """\
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

add_english_doc('data.embedding.EmbeddingMineSemanticNegatives', """\
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

add_example('data.embedding.EmbeddingMineSemanticNegatives', """\
```python
from lazyllm.tools.data import embedding

# Assuming embeddings are initialized
semantic_miner = embedding.EmbeddingMineSemanticNegatives(num_negatives=5, embedding_serving=my_embedding_fn)
data = {'query': 'machine learning', 'pos': ['ML tutorial'], '_semantic_embeddings_path': emb_path, '_semantic_corpus': corpus}
result = semantic_miner(data)
# Returns data with 'neg' field containing semantically similar negative samples
```
""")


add_chinese_doc('data.embedding.EmbeddingGenerateQueries', """\
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

add_english_doc('data.embedding.EmbeddingGenerateQueries', """\
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

add_example('data.embedding.EmbeddingGenerateQueries', """\
```python
from lazyllm.tools.data import embedding

# Assuming llm is an LLM service instance
generator = embedding.EmbeddingGenerateQueries(llm=llm, lang='zh')
data = {'_query_prompt': 'Generate queries for: machine learning tutorial'}
result = generator(data)
# Returns data with '_query_response' field containing JSON queries
```
""")

add_chinese_doc('data.embedding.EmbeddingParseQueries', """\
解析生成的查询的算子。

该算子解析 LLM 生成的查询响应，将每条查询展开为独立的数据记录。

Args:
    input_key (str): 输入字段名，默认为 'passage'。
    output_query_key (str): 输出查询字段名，默认为 'query'。
    **kwargs (dict): 其它可选的参数，传递给父类。

Returns:
    List[dict]: 解析后的查询列表，每个查询为一个独立的数据记录。
""")

add_english_doc('data.embedding.EmbeddingParseQueries', """\
An operator that parses generated queries.

This operator parses the query response generated by LLM and expands each query into an independent data record.

Args:
    input_key (str): Input field name, defaults to 'passage'.
    output_query_key (str): Output query field name, defaults to 'query'.
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    List[dict]: List of parsed queries, each query as an independent data record.
""")

add_example('data.embedding.EmbeddingParseQueries', """\
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
    _type: 文件类型 ('pdf', 'html', 'text', 'invalid', 'unsupported')
    _raw_path: 本地文件路径（如果有）
    _url: URL地址（如果是网页）
    _output_path: 预期的Markdown输出路径
    _error: 错误信息（如果有）
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
    _type: File type ('pdf', 'html', 'text', 'invalid', 'unsupported')
    _raw_path: Local file path (if available)
    _url: URL address (if web page)
    _output_path: Expected Markdown output path
    _error: Error message (if any)
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
    _markdown_path: 生成的Markdown文件路径
""")

add_english_doc('data.operators.knowledge_cleaning.file_or_url_to_markdown_converter_api.HTMLToMarkdownConverter', """\
HTML to Markdown converter operator.

This operator uses the trafilatura library to extract content from HTML or XML files and convert to Markdown format.
Supports local HTML files and web URLs, automatically handles page metadata.

Args:
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    dict: Converted data containing the following fields:
    _markdown_path: Path to the generated Markdown file
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
    _markdown_path: 生成的Markdown文件路径
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
    _markdown_path: Path to the generated Markdown file
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
    _text_content: 加载的文本内容
    _load_error: 加载错误信息（如果有）
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
    _text_content: Loaded text content
    _load_error: Loading error message (if any)
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
    _chunks: 分块后的文本列表
    _chunk_error: 分块错误信息（如果有）
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
    _chunks: List of chunked texts
    _chunk_error: Chunking error message (if any)
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
    chunk_path: 保存的JSON文件路径
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
    chunk_path: Path to the saved JSON file
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
    _chunks_data: 分块数据列表
    _chunk_path: 分块文件路径
""")

add_english_doc('data.operators.knowledge_cleaning.kbc_multihop_qa_generator_batch.KBCLoadChunkFile', """\
Chunk file loading operator.

This operator loads JSON or JSONL format chunk files from the specified path.
Supports chunk result files generated from the knowledge base cleaning process.

Args:
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    dict: Data containing chunk data:
    _chunks_data: List of chunk data
    _chunk_path: Chunk file path
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
    _processed_chunks: 预处理后的分块列表
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
    _processed_chunks: List of preprocessed chunks
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
    _info_pairs: 信息对列表，每个包含 premise、intermediate、conclusion 和 related_contexts
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
    _info_pairs: List of information pairs, each containing premise, intermediate, conclusion, and related_contexts
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
    _qa_results: 问答结果列表，每个包含 response 和 info_pair
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
    _qa_results: List of QA results, each containing response and info_pair
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
    _qa_pairs: 解析后的问答对列表
""")

add_english_doc('data.operators.knowledge_cleaning.kbc_multihop_qa_generator_batch.parse_qa_pairs', """\
QA pair parsing function.

This function parses LLM-generated QA responses, extracting valid QA pairs.
Supports multiple response formats (dict, list, string) and merges parsing results with original data.

Args:
    data (dict): Data containing QA results.

Returns:
    dict: Data containing parsed QA pairs:
    _qa_pairs: List of parsed QA pairs
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
    enhanced_chunk_path: 增强后的分块文件路径
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
    enhanced_chunk_path: Path to the enhanced chunk file
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
    _chunks_data: 原始分块数据列表
    _chunk_path: 分块文件路径
""")

add_english_doc('data.operators.knowledge_cleaning.kbc_text_cleaner_batch.KBCLoadRAWChunkFile', """\
Raw chunk file loading operator.

This operator loads JSON or JSONL files containing raw chunks (raw_chunk) from the specified path.
Used in the knowledge base cleaning process to load raw chunk data that needs cleaning.

Args:
    **kwargs (dict): Additional optional arguments passed to the parent class.

Returns:
    dict: Data containing raw chunk data:
    _chunks_data: List of raw chunk data
    _chunk_path: Chunk file path
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
    _cleaned_results: 清洗结果列表，每个包含 response、raw_chunk 和 original_item
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
    _cleaned_results: List of cleaning results, each containing response, raw_chunk, and original_item
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
    _cleaned_chunks: 清洗后的分块列表，每个包含 raw_chunk、cleaned_chunk 和 original_item
""")

add_english_doc('data.operators.knowledge_cleaning.kbc_text_cleaner_batch.extract_cleaned_content', """\
Extract cleaned content function.

This function extracts cleaned text content from LLM cleaning results, handling different response formats.
Supports extracting content between <cleaned_start> and <cleaned_end> tags.

Args:
    data (dict): Data containing cleaning results.

Returns:
    dict: Data containing extracted cleaned content:
    _cleaned_chunks: List of cleaned chunks, each containing raw_chunk, cleaned_chunk, and original_item
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
    cleaned_chunk_path: 清洗后的分块文件路径
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
    cleaned_chunk_path: Path to the cleaned chunk file
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
    _cleaned_response: LLM的清洗响应
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
    _cleaned_response: LLM's cleaning response
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
    _qa_data: 加载的问答数据
    _source_file: 数据来源文件路径（如果从文件加载）
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
    _qa_data: Loaded QA data
    _source_file: Data source file path (if loaded from file)
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
    _is_valid: 数据是否有效
    _error: 错误信息（如果无效）
    _query, _pos, _neg: 标准化后的字段值
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
    _is_valid: Whether data is valid
    _error: Error message (if invalid)
    _query, _pos, _neg: Normalized field values
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
    _is_valid: 数据是否有效
    _error: 错误信息（如果无效）
    _query, _pos, _neg: 标准化后的字段值
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
    _is_valid: Whether data is valid
    _error: Error message (if invalid)
    _query, _pos, _neg: Normalized field values
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

# pt_op
add_chinese_doc('data.operators.pt_op.resolution_filter', """\
按最小/最大宽高过滤图片，保留尺寸在指定范围内的图片路径。   

Args:
    data (dict): 单条数据字典
    image_key (str): 图片路径字段名，默认 'image_path'
    min_width (int): 最小宽度，默认 256
    min_height (int): 最小高度，默认 256
    max_width (int): 最大宽度，默认 4096
    max_height (int): 最大高度，默认 4096
    input_key (str): 可选，覆盖 image_key
""")

add_english_doc('data.operators.pt_op.resolution_filter', """\
Filter images by min/max width and height, keeping only those within the specified resolution range.

Args:
    data (dict): single data dict
    image_key (str): key for image path(s), default 'image_path'
    min_width (int): minimum width, default 256
    min_height (int): minimum height, default 256
    max_width (int): maximum width, default 4096
    max_height (int): maximum height, default 4096
    input_key (str): optional, overrides image_key
""")

add_example('data.operators.pt_op.resolution_filter', """\
```python
from lazyllm.tools.data import pt_mm

op = pt_mm.resolution_filter(min_width=256, min_height=256, max_width=4096, max_height=4096)
res = op([{'image_path': '/path/to/image.jpg'}])
```
""")

add_chinese_doc('data.operators.pt_op.resolution_resize', """\
将图片最长边缩放到不超过 max_side，可原位覆盖或生成新文件。

Args:
    data (dict): 单条数据字典
    image_key (str): 图片路径字段名，默认 'image_path'
    max_side (int): 最长边上限，默认 1024
    inplace (bool): 是否覆盖原文件，默认 True；False 时生成 _resized 后缀新文件
    input_key (str): 可选，覆盖 image_key
""")

add_english_doc('data.operators.pt_op.resolution_resize', """\
Resize image so the longest side does not exceed max_side. Can overwrite in place or save to a new file.

Args:
    data (dict): single data dict
    image_key (str): key for image path(s), default 'image_path'
    max_side (int): max length of longest side, default 1024
    inplace (bool): overwrite original file if True; if False, save with _resized suffix
    input_key (str): optional, overrides image_key
""")

add_example('data.operators.pt_op.resolution_resize', """\
```python
from lazyllm.tools.data import pt_mm

op = pt_mm.resolution_resize(max_side=400, inplace=False)
res = op([{'image_path': '/path/to/image.jpg'}])
# resized file saved as image_resized.jpg in same directory
```
""")

add_chinese_doc('data.operators.pt_op.integrity_check', """\
检查图片文件完整性，过滤损坏或空文件，保留可正常打开的图片路径。

Args:
    data (dict): 单条数据字典
    image_key (str): 图片路径字段名，默认 'image_path'
    input_key (str): 可选，覆盖 image_key
""")

add_english_doc('data.operators.pt_op.integrity_check', """\
Check image file integrity; filter out corrupted or empty files, keep paths of valid images.

Args:
    data (dict): single data dict
    image_key (str): key for image path(s), default 'image_path'
    input_key (str): optional, overrides image_key
""")

add_example('data.operators.pt_op.integrity_check', """\
```python
from lazyllm.tools.data import pt_mm

op = pt_mm.integrity_check()
res = op([{'image_path': '/path/to/image.jpg'}, {'image_path': '/nonexistent.png'}])
# only valid images retained
```
""")

add_chinese_doc('data.operators.pt_op.TextRelevanceFilter', """\
使用 VLM 判断图文相关性，过滤低于阈值的样本。

Args:
    vlm: 视觉语言模型实例
    image_key (str): 图片路径字段名，默认 'image_path'
    text_key (str): 文本字段名，默认 'text'
    threshold (float): 相关性阈值 [0,1]，默认 0.6
    prompt (str): 可选，自定义提示词
""")

add_english_doc('data.operators.pt_op.TextRelevanceFilter', """\
Use VLM to judge image-text relevance; filter samples below the threshold.

Args:
    vlm: vision-language model instance
    image_key (str): key for image path(s), default 'image_path'
    text_key (str): key for text, default 'text'
    threshold (float): relevance threshold [0,1], default 0.6
    prompt (str): optional custom prompt
""")

add_example('data.operators.pt_op.TextRelevanceFilter', """\
```python
from lazyllm.tools.data import pt_mm

vlm = lazyllm.OnlineChatModule(source='sensenova', model='SenseNova-V6-5-Turbo')
op = pt_mm.TextRelevanceFilter(vlm, threshold=0.5)
res = op([{'image_path': '/path/to/image.jpg', 'text': 'a red square'}])
# samples with relevance >= threshold are kept
```
""")

add_chinese_doc('data.operators.pt_op.ImageDedup', """\
基于图片文件哈希去重，保留首次出现的图片，跳过重复项。

Args:
    image_key (str): 图片路径字段名，默认 'image_path'
    hash_method (str): 哈希算法，默认 'md5'
""")

add_english_doc('data.operators.pt_op.ImageDedup', """\
Deduplicate images by file hash; keep first occurrence, skip duplicates.

Args:
    image_key (str): key for image path(s), default 'image_path'
    hash_method (str): hash algorithm, default 'md5'
""")

add_example('data.operators.pt_op.ImageDedup', """\
```python
from lazyllm.tools.data import pt_mm

op = pt_mm.ImageDedup()
batch = [{'image_path': 'a.jpg', 'id': 1}, {'image_path': 'a.jpg', 'id': 2}, {'image_path': 'b.jpg', 'id': 3}]
res = op(batch)
# len(res) == 2, duplicate removed
```
""")

add_chinese_doc('data.operators.pt_op.VQAGenerator', """\
使用 VLM 根据 context 和图片生成视觉问答对（VQA pairs）。

Args:
    vlm: 视觉语言模型实例
    image_key (str): 图片路径字段名，默认 'image_path'
    context_key (str): 上下文字段名，默认 'context'
    num_qa (int): 生成的问答对数量，默认 5
    prompt (str): 可选，自定义提示词
""")

add_english_doc('data.operators.pt_op.VQAGenerator', """\
Use VLM to generate Visual Question Answering (VQA) pairs from context and images.

Args:
    vlm: vision-language model instance
    image_key (str): key for image path(s), default 'image_path'
    context_key (str): key for context, default 'context'
    num_qa (int): number of QA pairs to generate, default 5
    prompt (str): optional custom prompt
""")

add_example('data.operators.pt_op.VQAGenerator', """\
```python
from lazyllm.tools.data import pt_mm

vlm = lazyllm.OnlineChatModule(source='sensenova', model='SenseNova-V6-5-Turbo')
op = pt_mm.VQAGenerator(vlm, num_qa=3)
res = op([{'image_path': '/path/to/image.jpg', 'context': 'A simple image.'}])
# res[0]['qa_pairs'] contains [{'query': '...', 'answer': '...'}, ...]
```
""")

add_chinese_doc('data.operators.pt_op.VQAScorer', """\
使用 VLM 对 VQA 对（query、answer、image_path）进行质量打分，评估图文问答的质量。

Args:
    vlm: 视觉语言模型实例
    image_key (str): 图片路径字段名，默认 'image_path'
    query_key (str): 问题字段名，默认 'query'
    answer_key (str): 答案字段名，默认 'answer'
    prompt (str): 可选，自定义提示词
""")

add_english_doc('data.operators.pt_op.VQAScorer', """\
Use VLM to score VQA pair quality (query, answer, image_path), evaluating how good the visual QA is.

Args:
    vlm: vision-language model instance
    image_key (str): key for image path(s), default 'image_path'
    query_key (str): key for question, default 'query'
    answer_key (str): key for answer, default 'answer'
    prompt (str): optional custom prompt
""")

add_example('data.operators.pt_op.VQAScorer', """\
```python
from lazyllm.tools.data import pt_mm

vlm = lazyllm.OnlineChatModule(source='sensenova', model='SenseNova-V6-5-Turbo')
op = pt_mm.VQAScorer(vlm)
res = op([{
    'image_path': '/path/to/image.jpg',
    'query': 'What color is it?',
    'answer': 'Red',
}])
# res[0]['quality_score'] contains score, relevance, correctness, reason
```
""")

add_chinese_doc('data.operators.pt_op.GraphRetriever', """\
从 context 字段中解析 Markdown 格式的图片链接 `![alt](path)`，提取存在磁盘上的图片路径并写入 img_key。
不修改原始 context；若 context.strip() 为空，则 img_key 为 []，样本仍保留。

Args:
    context_key (str): 文本上下文字段名，默认 'context'
    img_key (str): 图片路径输出字段名，默认 'image_path'
    images_folder (str): 可选，图片根目录，用于解析相对路径
""")

add_english_doc('data.operators.pt_op.GraphRetriever', """\
Parse Markdown-style image links `![alt](path)` from context field, extract existing file paths and write to img_key.
Does not modify source context; if context.strip() is empty, img_key is [] and the sample is kept.

Args:
    context_key (str): key for text context, default 'context'
    img_key (str): key for image path output, default 'image_path'
    images_folder (str): optional root folder for resolving relative paths
""")

add_example('data.operators.pt_op.GraphRetriever', """\
```python
from lazyllm.tools.data import pt_mm

op = pt_mm.GraphRetriever(context_key='context', img_key='img', _save_data=False)
data = {'context': 'Some content ![](/path/to/fig.png)'}
res = op([data])
# res[0]['img'] contains resolved absolute path

# empty context: res[0]['img'] == [], record kept, source context unchanged
empty_res = op([{'context': '   '}])
```
""")

add_chinese_doc('data.operators.pt_op.ContextQualFilter', """\
使用 VLM 或 LLM 评估 context 是否适合生成 QA 对；仅保留 score=1（适合）的样本。

Args:
    llm: 视觉或文本语言模型实例
    context_key (str): 上下文字段名，默认 'context'
    image_key (str): 图片路径字段名，默认 'image_path'
    prompt (str): 可选，自定义提示词
""")

add_english_doc('data.operators.pt_op.ContextQualFilter', """\
Use VLM or LLM to evaluate whether context is suitable for generating QA pairs; keep only samples with score=1 (suitable).

Args:
    llm: vision- or text-language model instance
    context_key (str): key for context, default 'context'
    image_key (str): key for image path(s), default 'image_path'
    prompt (str): optional custom prompt
""")

add_example('data.operators.pt_op.ContextQualFilter', """\
```python
from lazyllm.tools.data import pt

vlm = lazyllm.OnlineChatModule(source='sensenova', model='SenseNova-V6-5-Turbo')
op = pt.ContextQualFilter(vlm)
res = op([{'context': 'Good context for QA.', 'image_path': '/path/to/image.jpg'}])
# only samples with score=1 are kept
```
""")

add_chinese_doc('data.operators.pt_op.Phi4QAGenerator', """\
使用 LLM 将 context（含可选图片）转换为预训练格式的 Phi-4 风格多轮问答对。

Args:
    llm: 视觉或文本语言模型实例
    image_key (str): 图片路径字段名，默认 'image_path'
    context_key (str): 上下文字段名，默认 'context'
    num_qa (int): 生成的问答对数量，默认 5
    prompt (str): 可选，自定义提示词
""")

add_english_doc('data.operators.pt_op.Phi4QAGenerator', """\
Use LLM to convert context (with optional images) into pretraining-format Phi-4 style multi-turn Q&A pairs.

Args:
    llm: vision- or text-language model instance
    image_key (str): key for image path(s), default 'image_path'
    context_key (str): key for context, default 'context'
    num_qa (int): number of QA pairs to generate, default 5
    prompt (str): optional custom prompt
""")

add_example('data.operators.pt_op.Phi4QAGenerator', """\
```python
from lazyllm.tools.data import pt

vlm = lazyllm.OnlineChatModule(source='sensenova', model='SenseNova-V6-5-Turbo')
op = pt.Phi4QAGenerator(vlm, num_qa=2)
res = op([{'context': 'Some context.', 'image_path': '/path/to/image.jpg'}])
# res[0]['qa_pairs'] contains pretraining-format Q&A
```
""")
