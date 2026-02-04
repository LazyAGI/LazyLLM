# Graph的作用

一个基于有向无环图（DAG）的复杂流控制结构。
Graph类允许您创建复杂的处理图，其中节点表示处理函数，边表示数据流。它支持拓扑排序来确保正确的执行顺序，并可以处理多输入和多输出的复杂依赖关系。
Graph类特别适用于需要复杂数据流和依赖管理的场景，如机器学习管道、数据处理工作流等。

## 参数

- post_action (callable, default: None ) – 在图执行完成后要调用的函数。默认为 None。
- auto_capture (bool, default: False ) – 是否自动捕获上下文中的变量。默认为 False。
- kwargs – 代表命名节点和对应函数的任意关键字参数。

## graph的基础用法

### end_node

获取图的结束节点。

```python
import lazyllm
with lazyllm.graph() as g:
...     g.process = lambda x: x * 2
end = g.end_node
>>> end.name
'__end__'
```

### start node

获取图的起始节点

```python
import lazyllm
with lazyllm.graph() as g:
...     g.process = lambda x: x * 2
start = g.start_node
>>> start.name
'__start__'
```

### add_const_edge(constant, to_node)

添加一个常量边，将固定值传递给指定节点。
此方法用于将常量值作为输入传递给图中的节点，无需从其他节点获取数据。

参数：
-constant – 要传递的常量值。
-to_node (str or Node) – 目标节点的名称或Node对象。

```python
import lazyllm
with lazyllm.graph() as g:
...     g.add = lambda x, y: x + y
g.add_const_edge(10, 'add')
>>> g._constants
[10]
```

### add_edge(from_node, to_node, formatter=None)

在图中添加一条边，定义节点之间的数据流。
此方法用于定义图中节点之间的连接关系，指定数据如何从一个节点流向另一个节点。

参数：
-from_node (str or Node) – 源节点的名称或Node对象。
-to_node (str or Node) – 目标节点的名称或Node对象。
-formatter (callable, default: None ) – 可选的格式化函数，用于在传递数据时进行转换。默认为 None。

```python
import lazyllm
with lazyllm.graph() as g:
...     g.node1 = lambda x: x * 2
...     g.node2 = lambda x: x + 1
...     g.node3 = lambda x, y: x + y
g.add_edge('__start__', 'node1')
g.add_edge('node1', 'node2')
g.add_edge('node3', '__end__')
g._nodes['node1'].outputs
[<Flow type=Node name=node2>]
def double_input(data):
...     return data * 2
g.add_edge('node1', 'node3', formatter=double_input)
>>> g._nodes['node3'].inputs
{'node1': <function double_input at ...>}
```

### compute_node(sid, node, intermediate_results, futures)

计算单个节点的输出结果。
此方法是图的内部方法，用于执行单个节点的计算，包括获取输入数据、应用格式化函数、调用节点函数等。

参数：
-sid – 会话ID。
-node (Node) – 要计算的节点。
-intermediate_results (dict) – 中间结果存储。
-futures (dict) – 异步任务字典。

```python
import lazyllm
with lazyllm.graph() as g:
...     g.add = lambda x, y: x + y
...     g.multiply = lambda x: x * 2
g.add_edge('__start__', 'add')
g.add_const_edge(5, 'add')
g.add_edge('add', 'multiply')
g.add_edge('multiply', '__end__')
result = g(3)  # x=3, y=5 (常量)
>>> result
16
```

### set_node_arg_name(arg_names)

设置节点的参数名称。
此方法用于为图中的节点设置函数参数的名称，这对于多参数函数的正确调用很重要。

参数：
-arg_names (list) – 参数名称的列表，与节点创建时的顺序对应。

```python
import lazyllm
with lazyllm.graph() as g:
...     g.add = lambda a, b: a + b
...     g.multiply = lambda x, y: x * y
g.set_node_arg_name([['x', 'y'], ['a', 'b']])
>>> g._nodes['add'].arg_names
['x', 'y']
>>> g._nodes['multiply'].arg_names
['a', 'b']
```

### topological_sort()

执行拓扑排序，返回正确的节点执行顺序。
此方法使用Kahn算法对有向无环图进行拓扑排序，确保所有依赖关系都得到满足。

参数：
-List[Node]: 按拓扑顺序排列的节点列表。

```python
import lazyllm
with lazyllm.graph() as g:
...     g.node1 = lambda x: x * 2
...     g.node2 = lambda x: x + 1
...     g.node3 = lambda x, y: x + y
g.add_edge('__start__', 'node1')
g.add_edge('node1', 'node2')
g.add_edge('node1', 'node3')
g.add_edge('node2', 'node3')
g.add_edge('node3', '__end__')
sorted_nodes = g.topological_sort()
>>> [node.name for node in sorted_nodes]
['__start__', 'node1', 'node2', 'node3', '__end__']
g.add_edge('node3', 'node1')
>>> try:
...     g.topological_sort()
... except ValueError as e:
...     print("检测到循环依赖")
检测到循环依赖
```
