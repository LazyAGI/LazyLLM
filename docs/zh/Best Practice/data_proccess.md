# 数据处理模块的设计哲学

## 基础约定

数据处理模块的基础约定，被处理的数据集类型统一为：`List[dict]`

- 数据集的大小，即：`List` 中的元素个数；
- 元素dict，即：单条数据。

## 处理模式
- 数据处理流程中，每个算子完成对数据集完整的处理，然后交给下一个算子进行处理，直到所有算子处理完成，得到最终结果。比如，数据集经过过滤算子处理后，得到过滤后的数据集，再经过打分算子处理后，得到打分后的数据集，依次类推；
- 算子分类单条数据处理算子和全量数据处理算子两种；
    - 单条数据处理算子：每次处理 `List[dict]` 中的单条 `dict` 数据，最终返回处理后的单条 `dict` 数据。框架主要负责对这类算子进行并发处理，将这个算子作用于 `List[dict]` 中的每一条 `dict` 数据上。比如，基于正则表达式的数据清洗算子，基于LLM的数据处理算子等；对于单条数据处理模式，框架会自动识别算子返回的数据类型，并根据返回类型进行相应的处理：
        - 返回字典 `dict`：替换原单条数据；
        - 返回列表 `List[dict]`: 新增多条数据;
        - 返回 `None`: 表示沿用传入的data引用；
        - 返回空列表 `List`: 表示删除数据；
    - 全量数据处理算子：每次处理整个 `List[dict]` 数据集，最终返回处理后的整个 `List[dict]` 数据集。框架不对这类算子进行并发处理（需要用户自己设计），直接顺序调用，将整个 `List[dict]` 数据集传入该算子进行处理。比如，基于全量数据的去重算子等；

## 框架特性
1. 不同任务类型采用不同的并发方式，有效提升性能：
    - 计算密集型任务采用多进程并发，比如正则匹配；
    - I/O密集型任务采用多线程并发+动态提交任务（即流式并发处理算法：采用生产者消费者模式，有效避免木桶效应），比如基于LLM的数据处理；
    - Debug模式下，采用单线程顺序处理，方便调试；
2. 支持数据的动态存储，避免任务异常导致数据丢失（并发过程中动态存储），并采用智能存储，避免频繁存储带来的性能损耗；
    - Pipeline 中每个算子会自动在配置的存储根目录下，根据算子名称生成独立的存储路径（jsonl格式）；
    - 智能存储会根据数据量和处理速度，动态调整存储频率，避免频繁存储带来的性能损耗；
3. 支持Resume功能，任务中断后可从上次中断点继续执行；
4. 支持自定义算子（函数/类）进行数据处理。
    - 算子注册采用装饰器模式，且还保持了IDE代码跳转的能力；
    - 函数和类都统一注册为类算子，使用方式一致，方便调用；
5. 对于单条数据处理模式有自动意图识别，根据算子返回数据类型，自动识别数据处理意图：
6. 支持单条数据处理模式和全量数据处理模式；
7. 支持进度条展示任务进度。

流式并发处理算法核心逻辑：
首先向线程池提交一批初始任务至最大并发数；随后进入一个核心循环，该循环会等待并收集第一个完成的任务，将其结果（或异常）立即产出，同时从任务迭代器中取出下一个新任务提交给线程池以填补空缺；此“完成-产出-补充”的循环持续进行，直到任务迭代器耗尽且所有已提交的任务都处理完毕。

## 注册算子

### 1. 最简单用法

对于刚接触该框架的用户，只用知道`@data_register`装饰器可以注册算子，并自动提供并发、存储和resume能力即可，其他细节框架会自动处理。例如用户想注册一个将内容转为小写的算子，只需如下实现：

```python
from lazyllm.tools.data import data_register # 导入注册器

demo = data_register.new_group('demo')  # 创建一个类别，用于算子分组

@data_register('data.demo')
def process_lower(data:dict):
    data['content'] = data.get('content', '').lower()
    return data
```

对于类可通过继承快速实现注册：
```python
class ProcessLower(demo):
    def forward(self, data:dict):  # 通过重写forward方法实现处理逻辑
        data['content'] = data.get('content', '').lower()
        return data
```

使用：
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


下面完整介绍如何设计和注册算子。

### 2. 设计算子

算子可以是函数或者类。对于函数：

- 第一个参数 `data` 必要参数，且类型是 `dict` 或 `List[dict]`，注意该参数是延迟传入的；
    - `dict` 类型表示：单条数据（即 `dict`）处理模式；
    - `List[dict]` 类型表示：全量数据（即整个数据集 `List[dict]`）处理模式；
- 第二个参数 `input_key` 用于指定处理 `data` 中的 key，以作为输入。可选参数。支持：`None`(默认), `str` 或 `List[str]` 类型。
    - `None` 表示输入的Key交给用户自行处理（即用户不指定具体的输入key，在函数内部自行处理）；
    - `str` 表示：`data` 中的单个 `input_key` 作为输入被用作处理；
    - `List[str]` 表示：`data` 中的多个 `input_key` 作为输入被用作处理；
- 第三个参数 `output_key` 用于指定处理 `data` 后存放处理数据的key。可选参数。支持：`None`(默认), `str` 或 `List[str]` 类型。
    - `None` 表示输出的key和输入的key一致；
    - `str` 表示输出被放到对应 `data` 的key字段；
    - `List[str]` 表示多个输出到多个 `data` 的key字段。

示例如下：
```python
# 转换为全大写，单条数据处理
def process_uppercase(data:dict, input_key='content'): # 输入单条数据，指定处理'content'字段
    data[input_key] = data.get(input_key, '').upper()  # 提取data中的`content`字段内容，转为大写后塞回原字段
    return data                                        # 返回处理后的字典

# 明确指定输出key
def process_add_suffix(data:dict, input_key='content', output_key='output'):
    data[output_key] = data.get(input_key, '') + '_suffix'
    return data

# 指定用多个key作为输入
def process_merge(data:dict, input_key=['key1', 'key2'], output_key='output'):
    data[output_key] = data[input_key[0]] + data[input_key[1]]
    return data

# 全量数据处理
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

算子可以是类，类需要实现 `forward`（单条数据处理） 或 `forward_batch_input`(全量数据处理)，注意只能二选一。其中 `data` 在 `forward` 方法中传入（该参数也是延迟传入的）。另外也可以设计 `__init__` 方法用于传入其他参数。一般采用类作为算子时，用于需要传入共用资源的场景，比如：词表过滤算子需要传入词表资源等，示例如下：

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

### 3. 导入注册器并注册

框架提供注册器 `data_register` 用于注册算子。注册器主要提供如下能力：

- 装饰器注册算子，支持函数和类两种形式的算子注册；
- 赋予并发处理能力（单条数据处理算子）；
- 赋予动态存储和Resume能力；
- 赋予进度条展示能力。

注册示例如下：
```python
# 导入注册器
from lazyllm.tools.data import data_register

demo = data_register.new_group('demo')  # 创建一个类别，用于算子分组

# 装饰器注册算子，默认为单条数据处理算子（默认 rewrite_func='forward'）
@data_register('data.demo')
def process_uppercase(data:dict, input_key='content'):
    ... # 省略处理逻辑

# 注册为全量数据处理算子，通过设置参数 rewrite_func='forward_batch_input'
@data_register('data.demo', rewrite_func='forward_batch_input')
def process_deduplicate(data:List[dict], input_key='content'):
    ... # 省略处理逻辑

# 通过继承注册类算子
class WordTableFilter(demo):
    ... # 省略类实现逻辑
```

通过注册器设置算子的并发方式：
```python
@data_register('data.demo', _concurrency_mode='thread')
def process_uppercase(data:dict, input_key='content'):
    ... # 省略处理逻辑
```

注意，这里的并发类型有三种：

- `thread`: 多线程并发（使用上文提到的流式并发处理算法），适用于I/O密集型任务，比如基于LLM的数据处理；
- `process`: 多进程并发（默认根据CPU资源计算并发数），适用于计算密集型任务，比如正则匹配等；
- `single`: 单线程顺序处理，适用于Debug模式下的调试。


## 使用注册的算子进行数据处理

### 数据处理流水线示例

基于 LazyLLM 的数据处理流水线 `pipeline`，可以方便地使用注册的算子进行数据处理。示例如下：

```python
from lazyllm import pipeline
from lazyllm.tools.data import demo

# 准备数据
data = [
    {'text': 'hello world'},
    {'text': 'hello lazyllm'},
    {'text': 'hello world'},  # 重复数据
]

# 构建数据处理流水线
with pipeline() as ppl:
    ppl.upper = demo.process_uppercase(input_key='text')    # input_key 保持和data中key一致
    ppl.dedup = demo.process_deduplicate(input_key='text')  # input_key 保持和上一步处理后data中key一致
    ppl.add_suffix = demo.process_add_suffix(
        input_key='text',
        output_key='text_with_suffix',
    ).set_output('path/to/output')          # 设置输出结果路径，将结果导出为jsonl文件，并让结果返回导出的绝对路径（注意，这里不是中间存储的结果，是最终的结果。每个算子还会额外维护自己的中间结果。）

# 执行数据处理流水线
result = ppl(data)    # 输出是：path/to/output/**.jsonl 文件
```

### 算子包装器公共超参数示例

算子包装器类 `LazyLLMDataBase` 支持一些公共超参数，用于控制并发方式、存储行为等，这些参数在初始化算子时直接传入：

#### 1. 并发控制

```python
# 提供更细粒度的并发控制
process_add_suffix(
    input_key='text',
    output_key='text_with_suffix',
    _concurrency_mode='thread',  # 并发模式: 'thread', 'process', 'single'
    _max_workers=48,             # 最大并发数
)
```

注意，这里的并发优先级顺序为：算子初始化时传入的参数 > 注册器中传入的参数 > 默认值。


#### 2. 存储和Resume控制

```python
# 控制存储行为
process_uppercase(
    input_key='text',
    _save_data=True,      # 是否开启中间结果存储（默认为 True）
    _ignore_errors=True   # 是否忽略处理过程中的错误（默认为 True，错误记录到 explicit error log）
)
```

框架会根据配置的 `data_process_path`（或默认为工作目录下的 `data_pipeline_res`）生成如下结构的存储路径：

```bash
-- working_directory
        |-- data_pipeline_res
                |-- process_uppercase              # 算子1 (文件夹名为算子名)
                        |-- process_uppercase_results.jsonl  # 存储的中间结果文件
                        |-- process_uppercase_results.jsonl.json # 存储的进度文件
                        |-- process_uppercase_error.jsonl    # 错误日志
                |-- process_deduplicate            # 算子2
                        |-- ...
```
