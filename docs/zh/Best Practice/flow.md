# 应用搭建的核心：数据流

LazyLLM 中定义了大量的数据流组件，用于让您像搭积木一样，借助 LazyLLM 中提供的工具和组件，来搭建复杂的大模型应用。本节会详细介绍数据流的使用方法。

## 定义和API文档

[](){#use-flow}
数据流的定义和基本使用方法如 [flow][lazyllm.flow.FlowBase] 中所述

## Pipeline

#### 基本使用

Pipeline 是顺次执行的数据流，上一个阶段的输出成为下一个阶段的输入。pipeline 支持函数和仿函数（或仿函数的 type）。一个典型的 pipeline 如下所示:

```python
from lazyllm import pipeline

class Functor(object):
    def __call__(self, x): return x * x

def f1(input): return input + 1
f2 = lambda x: x * 2
f3 = Functor()

assert pipeline(f1, f2, f3, Functor)(1) == 256
```

!!! Note "注意"

    借助 LazyLLM 的注册机制 : [register][lazyllm.common.Register] 注册的函数，也可以直接被 pipeline 使用，下面给出一个例子

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

#### with 语句

除了基本的用法之外，pipeline 还支持一个更为灵活的用法 ``with pipeline() as p`` 来让代码更加的简洁和清晰，示例如下

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

!!! Note "注意"

    ``parallel``, ``diverter``, ``switch``, ``loop`` 等也支持 with 的用法。

#### 参数绑定

[](){#use-bind}

很多时候，我们并不希望一成不变的将上级的输出给到下一级作为输入，某一下游环节可能需要很久之前的某环节的输出，甚至是整个 pipeline 的输入。
在计算图模式的范式下（例如 Dify 和 LlamaIndex），会把函数作为节点，把数据作为边，通过添加边的方式来实现这一行为。
但 LazyLLM 不会让你如此复杂，你仅需要掌握参数绑定，就可以自由的在 pipeline 中从上游向下游传递参数。

假设我们定义了一些函数，本小节会一直使用这些函数，不再重复定义。

```python
def f1(input, input2=0): return input + input2 + 1
def f2(input): return input + 3
def f3(input): return f'f3-{input}'
def f4(in1, in2, in3): return f'get [{in1}], [{in2}], [{in3}]'
```

下面给出一个参数绑定的具体例子：

```python
from lazyllm import pipeline, _0
with pipeline() as p:
    p.f1 = f1
    p.f2 = f2
    p.f3 = f3
    p.f4 = bind(f4, p.input, _0, p.f2)
assert p(1) == 'get [1], [f3-5], [5]'
```

上述例子中， ``bind`` 函数用于参数绑定，它的基本使用方法和 C++ 的 ``std::bind`` 一致，其中 ``_0`` 表示新函数的第0个参数在被绑定的函数的参数表中的位置。
对于上面的案例，整个 pipeline 的输入会作为 f4 的第一个参数（此处我们假设从第一个开始计数），f3 的输出（即新函数的输入）会作为 f4 的第二个参数，f2 的输出会作为 f4 的第三个参数。

!!! Note "注意"

    - 参数绑定仅在一个 pipeline 中生效（注意，当 flow 出现嵌套时，在子 flow 中不生效），仅允许下游函数绑定上游函数的输出作为参数。
    - 使用参数绑定后，平铺的方式传入的参数中，未被 ``_0``, ``_1`` 等 ``placeholder`` 引用的会被丢弃

上面的方式已经足够简单和清晰，如果您仍然觉得 ``bind`` 作为函数不够直观，可以尝试使用如下方式，两种方式没有任何区别：

```python
from lazyllm import pipeline, _0
with pipeline() as p:
    p.f1 = f1
    p.f2 = f2
    p.f3 = f3
    p.f4 = f4 | bind(p.input, _0, p.output("f2"))
assert p(1) == 'get [1], [f3-5], [5]'
```

!!! Note "注意"

    请小心 lambda 函数！如果使用了 lambda 函数，请注意给 lambda 函数加括号，例如 ``(lambda x, y: pass) | bind(1, _0)``

除了 C++ 的 bind 方式之外，作为 Python，我们额外提供了 ``kwargs`` 的参数绑定， ``kwargs`` 和 C++ 的绑定方式可以混合使用，示例如下:

```python
from lazyllm import pipeline, _0
with pipeline() as p:
    p.f1 = f1
    p.f2 = f2
    p.f3 = f3
    p.f4 = f4 | bind(p.input, _0, in3=p.f2)
assert p(1) == 'get [1], [f3-5], [5]'
```

!!! Note "注意"

    通过 ``kwargs`` 绑定的参数的值不能使用 ``_0`` 等

如果 pipeline 的输入比较复杂，可以直接对 ``input`` 做一次简单的解析处理，示例如下:

```python
def f1(input): return dict(a=input[0], b=input[1])
def f2(input): return input['a'] + input['b']
def f3(input, extra): return f'[{input} + {extra}]'

with pipeline() as p1:
    p1.f1 = f1
    with pipeline() as p1.p2:
        p2.f2 = f2
        p2.f3 = f3 | bind(extra=p2.input['b'])
    p1.f3 = f3 | bind(extra=p1.input[0])

assert p1([1, 2]) == '[[3 + 2] + 1]'
```

上面的例子比较复杂，我们逐步来解析。首先输入的 list 经过 ``p1.f1`` 变成 ``dict(a=1, b=2)`` ，则 ``p2`` 的输入也是 ``dict(a=1, b=2)``，经过 ``p2.f2`` 之后输出为 ``3``，
然后 ``p2.f3`` 绑定了 ``p2`` 的输入的 ``['b']``， 即 ``2``, 因此 ``p2.f3`` 的输出是 ``[3 + 2]``, 回到 ``p1.f3``，它绑定了 ``p1`` 的输入的第 ``0`` 个元素，因此最终的输出是 ``[[3 + 2] + 1]``

#### pipeline.bind

当发生 pipeline 的嵌套（或 pipeline 与其他 flow 的嵌套时），我们有时候需要将外层的输入传递到内层中，此时也可以使用 bind，示例如下：

```python
from lazyllm import pipeline, _0
with pipeline() as p1:
    p1.f1 = f1
    p1.f2 = f2
    with pipeline().bind(extra=p1.input[0]) as p1.p2:
        p2.f3 = f3
    p1.p3 = pipeline(f3) | bind(extra=p1.input[1])

assert p1([1, 2]) == '[[3 + 1] + 2]'
```

#### AutoCapture（试验特性）

为了进一步简化代码的复杂性，我们上线了自动捕获 with 块内定义的变量的能力，示例如下：

```python
from lazyllm import pipeline, _0

def f1(input, input2=0): return input + input2 + 1
def f2(input): return input + 3
def f3(input): return f'f3-{input}'
def f4(in1, in2): return f'get [{in1}], [{in2}]'

with pipeline(auto_capture=True) as p:
    p1 = f1
    p2 = f2
    p3 = f3
    p4 = f4 | bind(p.input, _0)

assert p(1) == 'get [1], [f3-5]'
```

!!! Note "注意"

    该能力目前还不是很完善，不推荐大家使用，敬请期待。

## Parallel

Parallel 的所有组件共享输入，并将结果合并输出。 ``parallel`` 的定义方法和 ``pipeline`` 类似，也可以直接在定义 ``parallel`` 时初始化其元素，或在 with 块中初始化其元素。

!!! Note "注意"

    因 ``parallel`` 所有的模块共享输入，因此 ``parallel`` 的输入不支持被参数绑定。

#### 结果后处理

为了进一步简化流程的复杂性，不引入过多的匿名函数，parallel 的结果可以做一个简单的后处理（目前仅支持 ``sum`` 或 ``asdict``），然后传给下一级。下面给出一个例子:

```python
from lazyllm import parallel

def f1(input): return input

with parallel() as p:
    p.f1 = f1
    p.f2 = f1
assert p(1) == (1, 1)

with parallel().asdict as p:
    p.f1 = f1
    p.f2 = f1
assert p(1) == dict(f1=1, f2=1)

with parallel().sum as p:
    p.f1 = f1
    p.f2 = f1
assert p(1) == 2
```

!!! Note "注意"

    如果使用 ``asdict``, 需要为 ``parallel`` 中的元素取名字，返回的 ``dict`` 的 ``key`` 即为元素的名字。

#### 顺序执行

``parallel`` 默认是多线程并行执行的，在一些特殊情况下，可以根据需求改成顺序执行。下面给出一个例子：

```python
from lazyllm import parallel

def f1(input): return input

with parallel.sequential() as p:
    p.f1 = f1
    p.f2 = f1
assert p(1) == (1, 1)
```

!!! Note "注意"

    ``diverter`` 也可以通过 ``.sequential`` 来实现顺序执行

## 小结

本篇着重讲解了 ``pipeline`` 和 ``parallel``，相信您对如何利用 LazyLLM 的 flow 搭建复杂的应用已经有了初步的认识，其他的数据流组件不做过多赘述，您可以参考 [flow][lazyllm.flow.FlowBase] 来获取他们的使用方式。
