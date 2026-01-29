# Design Philosophy of the Data Processing Module

## Basic Conventions
The basic convention for the data processing module is that the type of the dataset being processed is unified as: `List[dict]`.

- Dataset size: The number of elements in the `List`.
- Element dict: A single piece of data.

## Processing Modes
- In the data processing flow, each operator completes the processing of the entire dataset before passing it to the next operator, until all operators have finished processing to obtain the final result. For example, after the dataset is processed by a filter operator, the filtered dataset is obtained, which is then processed by a scoring operator to obtain the scored dataset, and so on.
- Operators are classified into two types: single data processing operators and full data processing operators.
    - Single data processing operator: Each time it processes a single `dict` data in `List[dict]` and ultimately returns the processed single `dict` data. The framework is primarily responsible for concurrency handling of these operators, applying the operator to each `dict` data in `List[dict]`. Examples include data cleaning operators based on regular expressions, data processing operators based on LLM, etc. For the single data processing mode, the framework automatically identifies the return data type of the operator and handles it accordingly:
        - Returns a dictionary `dict`: Replaces the original single data.
        - Returns a list `List[dict]`: Adds multiple new data entries.
        - Returns `None`: Indicates keeping the passed data reference as is.
        - Returns an empty list `List`: Indicates deleting the data.
    - Full data processing operator: Each time it processes the entire `List[dict]` dataset and ultimately returns the processed entire `List[dict]` dataset. The framework does not handle concurrency for these operators (users need to design it themselves); instead, it calls them sequentially, passing the entire `List[dict]` dataset to the operator for processing. Examples include deduplication operators based on full data, etc.

## Framework Features
1. Different concurrency methods are adopted for different task types to effectively improve performance:
    - Compute-intensive tasks use multi-process concurrency, such as regular expression matching.
    - I/O-intensive tasks use multi-thread concurrency + dynamic task submission (i.e., streaming concurrency processing algorithm: adopting a producer-consumer model to effectively avoid the bucket effect), such as data processing based on LLM.
    - In debug mode, single-thread sequential processing is used for easy debugging.
2. Supports dynamic data storage to avoid data loss caused by task exceptions (dynamic storage during concurrency), and adopts smart storage to avoid performance loss caused by frequent storage.
    - Each operator in the Pipeline automatically generates an independent storage path (in jsonl format) under the configured storage root directory based on the operator name.
    - Smart storage dynamically adjusts the storage frequency based on data volume and processing speed to avoid performance loss from frequent storage.
3. Supports Resume function, allowing tasks to continue from the last interruption point after being interrupted.
4. Supports custom operators (functions/classes) for data processing.
    - Operator registration uses the decorator pattern and maintains IDE code navigation capabilities.
    - Both functions and classes are uniformly registered as class operators, with consistent usage for easy calling.
5. For single data processing mode, there is automatic intent recognition, which automatically identifies the data processing intent based on the data type returned by the operator.
6. Supports both single data processing mode and full data processing mode.
7. Supports progress bar display for task progress.

Core logic of the streaming concurrency processing algorithm:
First, submit a batch of initial tasks to the thread pool up to the maximum concurrency number; then enter a core loop, which waits for and collects the first completed task, immediately produces its result (or exception), and simultaneously takes the next new task from the task iterator and submits it to the thread pool to fill the vacancy; this "complete-produce-replenish" loop continues until the task iterator is exhausted and all submitted tasks are processed.

## Registering Operators

### 1. Simplest Usage

For users new to the framework, you only need to know that the `@data_register` decorator can register operators and automatically provide concurrency, storage, and resume capabilities; other details are handled automatically by the framework. For example, if a user wants to register an operator that converts content to lowercase, implementation is as follows:

```python
from lazyllm.tools.data import data_register # Import registrar

demo = data_register.new_group('demo')  # Create a category for operator grouping

@data_register('data.demo')
def process_lower(data:dict):
    data['content'] = data.get('content', '').lower()
    return data
```

For classes, registration can be quickly achieved through inheritance:
```python
class ProcessLower(demo):
    def forward(self, data:dict):  # Implement processing logic by overriding the forward method
        data['content'] = data.get('content', '').lower()
        return data
```

Usage:
```python
inputs = {'content': 'Hello World'}
m1 = lazyllm.data.demo.process_lower()
m2 = lazyllm.data.demo.ProcessLower()
res1 = m1(inputs)
res2 = m2(inputs)
print('Function result:', res1)
# Function result: [{'content': 'hello world'}]
print('Class result:', res2)
# Class result: [{'content': 'hello world'}]
```


Below is a complete introduction on how to design and register operators.

### 2. Designing Operators

Operators can be functions or classes. For functions:

- The first parameter `data` is a required parameter, and its type is `dict` or `List[dict]`. Note that this parameter is passed lazily.
    - `dict` type indicates: single data (i.e., `dict`) processing mode.
    - `List[dict]` type indicates: full data (i.e., entire dataset `List[dict]`) processing mode.
- The second parameter `input_key` is used to specify the key in `data` to be processed as input. This is an optional parameter. Supports: `None` (default), `str`, or `List[str]` types.
    - `None` indicates that the input Key is handled by the user (i.e., the user does not specify a specific input key and handles it internally in the function).
    - `str` indicates: a single `input_key` in `data` is used as input for processing.
    - `List[str]` indicates: multiple `input_keys` in `data` are used as input for processing.
- The third parameter `output_key` is used to specify the key to store the processed data after processing. This is an optional parameter. Supports: `None` (default), `str`, or `List[str]` types.
    - `None` indicates that the output key is consistent with the input key.
    - `str` indicates that the output is placed in the corresponding key field of `data`.
    - `List[str]` indicates that multiple outputs are placed in multiple key fields of `data`.

Examples are as follows:
```python
# Convert to uppercase, single data processing
def process_uppercase(data:dict, input_key='content'): # Input single data, specify handling 'content' field
    data[input_key] = data.get(input_key, '').upper()  # Extract content of `content` field in data, convert to uppercase and put back to original field
    return data                                        # Return processed dictionary

# Explicitly specify output key
def process_add_suffix(data:dict, input_key='content', output_key='output'):
    data[output_key] = data.get(input_key, '') + '_suffix'
    return data

# Specify multiple keys as input
def process_merge(data:dict, input_key=['key1', 'key2'], output_key='output'):
    data[output_key] = data[input_key[0]] + data[input_key[1]]
    return data

# Full data processing
def process_deduplicate(data:List[dict], input_key='content'):
    seen = set()
    deduplicated_data = []
    for item in data:
        value = item.get(input_key, '')
        if value not in seen:
            seen.add(value)
            deduplicated_data.append(item)
    return deduplicated_data
```

Operators can be classes. Classes need to implement `forward` (single data processing) or `forward_batch_input` (full data processing). Note that you can only choose one. `data` is passed in the `forward` method (this parameter is also passed lazily). Additionally, an `__init__` method can be designed to pass other parameters. Generally, classes are used as operators when shared resources need to be passed, for example: vocabulary filtering operators need to pass vocabulary resources, etc. Examples are as follows:

```python
class WordTableFilter:
    def __init__(self, world_table, input_key='content', **kwargs):
        super().__init__(**kwargs)
        self.world_table = world_table
        self.input_key = input_key

    def forward(self, data: dict):
        content = data.get(self.input_key, '')
        for word in self.world_table:
            if word in content:
                data['filtered'] = True
                return data
        data['filtered'] = False
        return data
```

### 3. Importing Registrar and Registering

The framework provides the registrar `data_register` for registering operators. The registrar mainly provides the following capabilities:

- Decorator for registering operators, supporting both function and class forms of operator registration.
- Endowing concurrency processing capabilities (single data processing operators).
- Endowing dynamic storage and Resume capabilities.
- Endowing progress bar display capabilities.

Registration examples are as follows:
```python
# Import registrar
from lazyllm.tools.data import data_register

demo = data_register.new_group('demo')  # Create a category for operator grouping

# Decorator registers operator, defaults to single data processing operator (default rewrite_func='forward')
@data_register('data.demo')
def process_uppercase(data:dict, input_key='content'):
    ... # Omitted processing logic

# Register as full data processing operator by setting parameter rewrite_func='forward_batch_input'
@data_register('data.demo', rewrite_func='forward_batch_input')
def process_deduplicate(data:List[dict], input_key='content'):
    ... # Omitted processing logic

# Register class operator via inheritance
class WordTableFilter(demo):
    ... # Omitted class implementation logic
```

Set the operator's concurrency mode via the registrar:
```python
@data_register('data.demo', _concurrency_mode='thread')
def process_uppercase(data:dict, input_key='content'):
    ... # omitted processing logic
```

Note that there are three types of concurrency here:

- `thread`: Multi-thread concurrency (using the streaming concurrency processing algorithm mentioned above), suitable for I/O-intensive tasks, such as data processing based on LLM.
- `process`: Multi-process concurrency (default concurrency number calculated based on CPU resources), suitable for compute-intensive tasks, such as regular expression matching, etc.
- `single`: Single-thread sequential processing, suitable for debugging in Debug mode.

## Using Registered Operators for Data Processing

### Data Processing Pipeline Example

Based on the LazyLLM data processing pipeline `pipeline`, you can easily use registered operators for data processing. Examples are as follows:

```python
from lazyllm import pipeline
from lazyllm.tools.data import demo

# Prepare data
data = [
    {'text': 'hello world'},
    {'text': 'hello lazyllm'},
    {'text': 'hello world'},  # Duplicate data
]

# Build data processing pipeline
with pipeline() as ppl:
    ppl.upper = demo.process_uppercase(input_key='text')    # input_key keeps consistent with key in data
    ppl.dedup = demo.process_deduplicate(input_key='text')  # input_key keeps consistent with key in data after previous step processing
    ppl.add_suffix = demo.process_add_suffix(
        input_key='text',
        output_key='text_with_suffix',
    ).set_output('path/to/output')          # Set output result path, export result as jsonl file, and make the result return the absolute path of the export (Note, this is not intermediate storage result, it is final result. Each operator also maintains its own intermediate results.)

# Execute data processing pipeline
result = ppl(data)    # Output is: path/to/output/**.jsonl file
```

### Operator Wrapper Common Hyperparameters Example

The operator wrapper class `LazyLLMDataBase` supports some common hyperparameters for controlling concurrency methods, storage behavior, etc., which are passed directly when initializing the operator:

#### 1. Concurrency Control:
```python
# Provide finer-grained concurrency control
process_add_suffix(
    input_key='text',
    output_key='text_with_suffix',
    _concurrency_mode='thread',  # Concurrency mode: 'thread', 'process', 'single'
    _max_workers=48,             # Max concurrency number
)
```

Note that the precedence for concurrency settings is: parameters provided during operator initialization > parameters provided to the registrar > default values.

#### 2. Storage and Resume Control:
```python
# Control storage behavior
process_uppercase(
    input_key='text',
    _save_data=True,      # Whether to enable intermediate result storage (default is True)
    _ignore_errors=True   # Whether to ignore errors during processing (default is True, errors are logged to explicit error log)
)
```

The framework will generate the following storage path structure based on the configured `data_process_path` (or default `data_pipeline_res` under the working directory):

```bash
-- working_directory
        |-- data_pipeline_res
                |-- process_uppercase              # Operator 1 (Folder name is operator name)
                        |-- process_uppercase_results.jsonl  # Stored intermediate result file
                        |-- process_uppercase_results.jsonl.json # Stored progress file
                        |-- process_uppercase_error.jsonl    # Error log
                |-- process_deduplicate            # Operator 2
                        |-- ...
```
