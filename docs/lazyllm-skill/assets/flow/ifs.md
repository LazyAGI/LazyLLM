# IFS的作用

IFS（If-Else Flow Structure）类设计用于根据给定条件的评估有条件地执行两个提供的路径之一（真路径或假路径）。执行选定路径后，可以应用可选的后续操作，并且如果指定，输入可以与输出一起返回。

## 参数

- cond (callable) – 一个接受输入并返回布尔值的可调用对象。它决定执行哪个路径。如果 cond(input) 评估为True，则执行 tpath ；否则，执行 fpath 。
- tpath (callable) – 如果条件为True，则执行的路径。
- fpath (callable) – 如果条件为False，则执行的路径。
- post_action (callable, default: None ) – 执行选定路径后执行的可选可调用对象。可以用于进行清理或进一步处理。默认为None。

## IFS的基础用法

```python
import lazyllm
cond = lambda x: x > 0
tpath = lambda x: x * 2
fpath = lambda x: -x
ifs_flow = lazyllm.ifs(cond, tpath, fpath)
>>> ifs_flow(10)
20
>>> ifs_flow(-5)
5
```
