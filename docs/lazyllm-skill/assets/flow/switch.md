# Switch的作用

一个根据条件选择并执行流的控制流机制。
Switch类提供了一种根据表达式的值或条件的真实性选择不同流的方法。

## 参数

- args – 可变长度参数列表，交替提供条件和对应的流或函数。可调用条件仅作为谓词执行一次，接收判定输入和关键字参数，并根据返回值的真值进行匹配；不可调用条件则与输入表达式进行比较。
- conversion (callable, default: None ) – 在进行条件匹配之前，对判定表达式 exp 进行转换或预处理的函数。它接收判定输入和关键字参数，并使用统一的 Flow 异常处理。默认为 None。
- post_action (callable, default: None ) – 在执行选定流后要调用的函数。默认为 None。
- judge_on_full_input (bool, default: True ) – 如果设置为 True ， 则通过 switch 的输入进行条件判断，否则会将输入拆成判定条件和真实的输入两部分，仅对判定条件进行判断。

## switch的基础用法

```python
 import lazyllm
 def is_positive(x): return x > 0
...
 def is_negative(x): return x < 0
...
 switch = lazyllm.switch(is_positive, lambda x: 2 * x, is_negative, lambda x : -x, 'default', lambda x : '000', judge_on_full_input=True)

 switch(1)
>>>2
 switch(0)
>>>'000'
 switch(-4)
>>>4

 def is_1(x): return True if x == 1 else False
...
 def is_2(x): return True if x == 2 else False
...
 def is_3(x): return True if x == 3 else False
...
 def t1(x): return 2 * x
...
 def t2(x): return 3 * x
...
 def t3(x): return x
...
 with lazyllm.switch(judge_on_full_input=True) as sw:
...     sw.case[is_1::t1]
...     sw.case(is_2, t2)
...     sw.case[is_3, t3]
...
>>> sw(1)
2
>>> sw(2)
6
>>> sw(3)
3
```
