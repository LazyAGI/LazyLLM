# Warp的作用

一个流形变器，将单个模块并行应用于多个输入。（只允许一个函数在warp中）
Warp类设计用于将同一个处理模块应用于一组输入。可以有效地将单个模块“形变”到输入上，使每个输入都并行处理。输出被收集并作为元组返回。需要注意的是，这个类不能用于异步任务，如训练和部署。

## 参数

- args – 可变长度参数列表，代表要应用于所有输入的单个模块。
- _scatter (bool) – 是否以分片方式拆分输入，默认 False。
- _concurrent (bool | int, default: True ) – 是否启用并发执行，可设定最大并发数。默认启用并发。
- auto_capture (bool, default: False ) – 如果为 True，在上下文管理器模式下将自动捕获当前作用域中新定义的变量加入流中。默认为 False。
- kwargs – 未来扩展的任意关键字参数。

## Warp的基础用法

```python
import lazyllm
warp = lazyllm.warp(lambda x: x * 2)
>>> warp(1, 2, 3, 4)
(2, 4, 6, 8)
warp = lazyllm.warp(lazyllm.pipeline(lambda x: x * 2, lambda x: f'get {x}'))
>>> warp(1, 2, 3, 4)
('get 2', 'get 4', 'get 6', 'get 8')
```

## 使用package

```python
from lazyllm import package
warp1 = lazyllm.warp(lambda x, y: x * 2 + y)
>>> print(warp1([package(1,2), package(10, 20)]))
(4, 40)
```
