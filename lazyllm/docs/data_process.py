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
