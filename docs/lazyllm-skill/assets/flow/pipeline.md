# Pipeline的用途

Pipeline 是顺次执行的数据流，上一个阶段的输出成为下一个阶段的输入。pipeline 支持函数和仿函数（或仿函数的 type）。

## 参数

- args (list of callables or single callable, default: () ) – 管道的处理阶段。每个元素可以是一个可调用的函数或 LazyLLMFlowsBase.FuncWrap的实例。如果提供了单个列表或元组，则将其解包为管道的阶段。
- post_action (callable, default: None ) – 在管道的最后一个阶段之后执行的可选操作。默认为None。
- auto_capture (bool, default: False ) – 如果为 True，在上下文管理器模式下将自动捕获当前作用域中新定义的变量加入流中。默认为 False。
- kwargs (dict of callables) – 管道的命名处理阶段。每个键值对表示一个命名阶段，其中键是名称，值是可调用的阶段。

## Pipeline基本使用

```python
from lazyllm import pipeline

class Functor(object):
    def __call__(self, x): return x * x

def f1(input): return input + 1
f2 = lambda x: x * 2
f3 = Functor()

assert pipeline(f1, f2, f3, Functor)(1) == 256
```

## 借助lazyllm的注册机制，register注册的函数也可以直接被pipeline使用

```python
import lazyllm
from lazyllm import pipeline, component_register

component_register.new_group('g1')

@component_register('g1')
def test1(input): return input + 1

@component_register('g1')
def test2(input): return input * 3

assert pipeline(lazyllm.g1.test1, lazyllm.g1.test2(launcher=lazyllm.launchers.empty))(1) == 6
```

## 在基本用法之外，还可以使用with语法是代码简洁清晰

```python

from lazyllm import pipeline

class Functor(object):
    def __call__(self, x): return x * x

def f1(input): return input + 1
f2 = lambda x: x * 2
f3 = Functor()

with pipeline() as p:
    p.f1 = f1
    p.f2 = f2
    p.f3 = f3

assert p(1) == 16
```
