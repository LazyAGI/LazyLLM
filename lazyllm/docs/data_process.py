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
