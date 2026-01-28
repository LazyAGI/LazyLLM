# Bind用途

通过bind，可以自由的在流程中从上游向下游传递参数。

## Bind的基础用法

用pipelin举例，parallel和diverter操作类似

```python
from lazyllm import pipeline, bind, _0

def f1(input, input2=0):
    return input + input2 + 1

def f2(input):
    return input + 3

def f3(input):
    return f'f3-{input}'

def f4(in1, in2, in3):
    return f'get [{in1}], [{in2}], [{in3}]'

with pipeline() as p:
    p.f1 = f1
    p.f2 = f2
    p.f3 = f3
    # 将 pipeline 的输入、f3 的输出、f2 的输出分别绑定到 f4 的三个参数
    p.f4 = bind(f4, p.input, _0, p.f2)

result = p(1)  # 结果: 'get [1], [f3-5], [5]'
```

## 使用管道操作符

```python
from lazyllm import pipeline, bind, _0

with pipeline() as p:
    p.f1 = f1
    p.f2 = f2
    p.f3 = f3
    # 使用 | 操作符绑定参数
    p.f4 = f4 | bind(p.input, _0, p.output("f2"))

result = p(1)
```

## kwargs 绑定

```python

from lazyllm import pipeline, bind, _0

with pipeline() as p:
    p.f1 = f1
    p.f2 = f2
    p.f3 = f3
    # 使用 kwargs 绑定参数
    p.f4 = f4 | bind(p.input, _0, in3=p.f2)

result = p(1)
```

## 嵌套 pipeline 绑定

```python

from lazyllm import pipeline, bind

with pipeline() as p1:
    p1.f1 = f1
    p1.f2 = f2

    # 将外层输入绑定到内层 pipeline
    with pipeline().bind(extra=p1.input[0]) as p1.p2:
        p2.f3 = f3

    p1.p3 = pipeline(f3) | bind(extra=p1.input[1])

result = p1([1, 2])  # 结果: '[[3 + 1] + 2]'
```

## 复杂数据绑定

```python
from lazyllm import pipeline, bind

def f1(input):
    return dict(a=input[0], b=input[1])

def f2(input):
    return input['a'] + input['b']

def f3(input, extra):
    return f'[{input} + {extra}]'

with pipeline() as p1:
    p1.f1 = f1

    with pipeline() as p1.p2:
        p2.f2 = f2
        # 绑定内层 pipeline 的输入的 ['b'] 字段
        p2.f3 = f3 | bind(extra=p2.input['b'])

    # 绑定外层 pipeline 的输入的第 0 个元素
    p1.f3 = f3 | bind(extra=p1.input[0])

result = p1([1, 2])  # 结果: '[[3 + 2] + 1]'
```
