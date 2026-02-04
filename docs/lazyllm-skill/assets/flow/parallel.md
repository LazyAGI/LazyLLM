# Parallel的作用

Parallel的所有组件共享输入，并将结果合并输出。

## 参数

- args – 基类的可变长度参数列表。
- _scatter (bool, default: False ) – 如果为 True，输入将在项目之间分割。如果为 False，相同的输入将传递给所有项目。默认为 False。
- _concurrent (bool, default: True ) – 如果为 True，操作将使用线程并发执行。如果为 False，操作将顺序执行。默认为 True。
- multiprocessing (bool, default: False ) – 如果为 True，将使用多进程而不是多线程进行并行执行。这可以提供真正的并行性，但会增加进程间通信的开销。默认为 False。
- auto_capture (bool, default: False ) – 如果为 True，在上下文管理器模式下将自动捕获当前作用域中新定义的变量加入流中。默认为 False。
- kwargs – 基类的任意关键字参数。

## 使用parallel进行结果后处理

为了进一步简化流程的复杂性，不引入过多的匿名函数，parallel 的结果可以做一个简单的后处理（目前仅支持 sum 或 asdict），然后传给下一级。下面给出一个例子:

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

## 使用parallel进行顺序执行

parallel 默认是多线程并行执行的，在一些特殊情况下，可以根据需求改成顺序执行。下面给出一个例子：

```python
from lazyllm import parallel

def f1(input): return input

with parallel.sequential() as p:
    p.f1 = f1
    p.f2 = f1
assert p(1) == (1, 1)
```

## 使用property规定返回类型

### astuple property

标记Parellel，使得Parallel每次调用时的返回值由package变为tuple。

### aslist property

标记Parellel，使得Parallel每次调用时的返回值由package变为list。

### sum property

标记Parellel，使得Parallel每次调用时的返回值做一次累加。

### join(self, string)

标记Parellel，使得Parallel每次调用时的返回值通过 string 做一次join。

```python
import lazyllm
test1 = lambda a: a + 1
test2 = lambda a: a * 4
test3 = lambda a: a / 2
ppl = lazyllm.parallel(test1, test2, test3)
>>> ppl(1)
(2, 4, 0.5)
ppl = lazyllm.parallel(a=test1, b=test2, c=test3)
>>> ppl(1)
{2, 4, 0.5}
ppl = lazyllm.parallel(a=test1, b=test2, c=test3).asdict
>>> ppl(2)
{'a': 3, 'b': 8, 'c': 1.0}
ppl = lazyllm.parallel(a=test1, b=test2, c=test3).astuple
>>> ppl(-1)
(0, -4, -0.5)
ppl = lazyllm.parallel(a=test1, b=test2, c=test3).aslist
>>> ppl(0)
[1, 0, 0.0]
ppl = lazyllm.parallel(a=test1, b=test2, c=test3).join('\n')
>>> ppl(1)
'2\n4\n0.5'
```
