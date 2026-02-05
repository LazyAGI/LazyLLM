# Diverter的作用

一个流分流器，将输入通过不同的模块以并行方式路由。
Diverter类是一种专门的并行处理形式，其中多个输入分别通过一系列模块并行处理。然后将输出聚合并作为元组返回。

## 参数

- args – 可变长度参数列表，代表并行执行的模块。
- _concurrent (bool, default: True ) – 控制模块是否应并行执行的标志。默认为 True。可用 Diverter.sequential 代替 Diverter 来设置此变量。
- auto_capture (bool, default: False ) – 如果为 True，在上下文管理器模式下将自动捕获当前作用域中新定义的变量加入流中。默认为 False。
- kwargs – 代表额外模块的任意关键字参数，其中键是模块的名称。

## diverter的基本使用

```python
>>> import lazyllm
>>> div = lazyllm.diverter(lambda x: x+1, lambda x: x*2, lambda x: -x)
>>> div(1, 2, 3)
(2, 4, -3)
>>> div = lazyllm.diverter(a=lambda x: x+1, b=lambda x: x*2, c=lambda x: -x).asdict
>>> div(1, 2, 3)
{'a': 2, 'b': 4, 'c': -3}
>>> div(dict(c=3, b=2, a=1))
{'a': 2, 'b': 4, 'c': -3}
```
