# Loop的作用

初始化一个循环流结构，该结构将一系列函数重复应用于输入，直到满足停止条件或达到指定的迭代次数。
Loop结构允许定义一个简单的控制流，其中一系列步骤在循环中应用，可以使用可选的停止条件来根据步骤的输出提前退出循环。

## 参数

- item (callable or list of callables, default: () ) – 将在循环中应用的函数或可调用对象。
- stop_condition (callable, default: None ) – 一个函数，它接受循环中最后一个项目的输出作为输入并返回一个布尔值。如果返回 True，循环将停止。如果为 None，循环将继续直到达到 count。默认为 None。
- count (int, default: maxsize ) – 运行循环的最大迭代次数。默认为 sys.maxsize。
- post_action (callable, default: None ) – 循环结束后调用的函数。默认为 None。
- auto_capture (bool, default: False ) – 如果为 True，在上下文管理器模式下将自动捕获当前作用域中新定义的变量加入流中。默认为 False。
- judge_on_full_input (bool, default: True ) – 如果设置为 True ，则通过 stop_condition 的输入进行条件判断；否则会将输入拆成判定条件和真实的输入两部分，仅对判定条件进行判断。

## loop的基础用法

```python
import lazyllm
loop = lazyllm.loop(lambda x: x * 2, stop_condition=lambda x: x > 10, judge_on_full_input=True)
>>> loop(1)
16
>>> loop(3)
12
with lazyllm.loop(stop_condition=lambda x: x > 10, judge_on_full_input=True) as lp:
...    lp.f1 = lambda x: x + 1
...    lp.f2 = lambda x: x * 2
...
>>> lp(0)
14
```
