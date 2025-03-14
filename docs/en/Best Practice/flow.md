# Core of Application Building: Data Flow

LazyLLM defines a multitude of data flow components that enable you to build complex large model applications using the tools and components provided by LazyLLM, much like building with blocks. This section will provide a detailed introduction to the usage of data flow.

## Definitions and API Documentation

[](){#use-flow}
The definitions and basic usage of data flow are described in [flow][lazyllm.flow.FlowBase].

## Pipeline

#### Basic Usage

A Pipeline is a sequential data flow where the output of one stage becomes the input of the next stage. Pipelines support both functions and functors (or the type of functors). A typical pipeline is as follows:

```python
from lazyllm import pipeline

class Functor(object):
    def __call__(self, x): return x * x

def f1(input): return input + 1
f2 = lambda x: x * 2
f3 = Functor()

assert pipeline(f1, f2, f3, Functor)(1) == 256
```

!!! Note

    Functions registered with LazyLLM's registration mechanism :[register][lazyllm.common.Register] can also be used directly by the pipeline. Below is an example:

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

#### with Statement

In addition to the basic usage, the pipeline also supports a more flexible usage with the ``with pipeline() as p`` statement to make the code more concise and clear. Here is an example:

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

!!! Note

    Components such as ``parallel``, ``diverter``, ``switch``, ``loop``  etc., also support the with statement.

#### Parameter Binding

[](){#use-bind}

Often, we do not want to rigidly pass the output of one stage as the input to the next. Sometimes, a downstream stage may require the output from a much earlier stage or even the input of the entire pipeline.
In computation graph paradigms (like in Dify and LlamaIndex), functions are treated as nodes and data as edges, with behavior implemented by adding edges.
However, LazyLLM simplifies this process, allowing you to achieve this through parameter binding. This enables the free flow of parameters from upstream to downstream within the pipeline.

Assume we have defined some functions, which will be used throughout this section without repeating their definitions.

```python
def f1(input, input2=0): return input + input2 + 1
def f2(input): return input + 3
def f3(input): return f'f3-{input}'
def f4(in1, in2, in3): return f'get [{in1}], [{in2}], [{in3}]'
```

Here is a specific example of parameter binding:

```python
from lazyllm import pipeline, _0
with pipeline() as p:
    p.f1 = f1
    p.f2 = f2
    p.f3 = f3
    p.f4 = bind(f4, p.input, _0, p.f2)
assert p(1) == 'get [1], [f3-5], [5]'
```

In the example above, the ``bind`` function is used for parameter binding. Its basic usage is similar to C++'s ``std::bind``, where ``_0`` indicates the position of the new function's first parameter in the bound function's parameter list.
For the above case,The entire pipeline's input will be used as the first parameter of f4 (assuming we start counting from the first parameter). The output of f3 (i.e., the input to the new function) will be used as the second parameter of f4, and the output of f2 will be used as the third parameter of f4.

!!! Note

    - Parameter binding is effective only within a single pipeline (note that when flows are nested, it does not apply in the subflow). It only allows downstream functions to bind the output of upstream functions as parameters.
    - When using parameter binding, any parameters passed in that are not referenced by ``placeholders`` such as ``_0``, ``_1``, etc., will be discarded.

The above method is already simple and clear enough. If you still find the function ``bind`` not intuitive, you can try the following approach. There is no difference between the two methods:

```python
from lazyllm import pipeline, _0
with pipeline() as p:
    p.f1 = f1
    p.f2 = f2
    p.f3 = f3
    p.f4 = f4 | bind(p.input, _0, p.f2)
assert p(1) == 'get [1], [f3-5], [5]'
```

!!! Note

    Please be careful with lambda functions! If you use a lambda function, make sure to enclose it in parentheses, for example: ``(lambda x, y: pass) | bind(1, _0)``

In addition to the C++ style bind method, as a Python library, we also provide parameter binding using ``kwargs``. You can mix ``kwargs`` with the C++ style binding method. Here's an example:

```python
from lazyllm import pipeline, _0
with pipeline() as p:
    p.f1 = f1
    p.f2 = f2
    p.f3 = f3
    p.f4 = f4 | bind(p.input, _0, in3=p.f2)
assert p(1) == 'get [1], [f3-5], [5]'
```

!!! Note

    The values of parameters bound through ``kwargs`` cannot use ``_0`` and similar placeholders.

If the input to the pipeline is complex, you can directly perform a simple parsing of the ``input``. Here is an example:

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

The example is a bit complex, so let's break it down step by step. First, the input list is processed by  ``p1.f1`` which transforms it into a dictionary: ``dict(a=1, b=2)`` .This dictionary becomes the input for p2. After passing through ``p2.f2``, the output is  ``3``,
Next, ``p2.f3`` is bound to the ``['b']`` value of the ``p2`` input, which is ``2``. Thus, the output of p2.f3 is ``[3 + 2]``. Finally, we return to ``p1.f3``, which is bound to the 0th element of the ``p1`` input. The final output is ``[[3 + 2] + 1]``.

#### pipeline.bind

When nesting pipelines (or pipelines with other flows), sometimes it's necessary to pass the outer layer's input to the inner layer. In such cases, you can use binding. Here's an example:

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

#### AutoCapture (Experimental Feature)

In order to further simplify the complexity of the code, we have introduced the ability to automatically capture variables defined within a with block. Here is an example:

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

!!! Note

    This capability is currently not very mature and is not recommended for use. Stay tuned for updates.

## Parallel

All components of ``parallel`` share the input and merge the results for output. The definition method of ``parallel`` is similar to that of ``pipeline``. You can either initialize its elements directly when defining ``parallel`` or initialize its elements within a with block.

!!! Note

    Since all modules in ``parallel`` share the input, the input to ``parallel`` does not support parameter binding.

#### Result Post-Processing

To further simplify the complexity of the process without introducing too many anonymous functions, the result of parallel can undergo simple post-processing (currently only supporting ``sum`` or ``asdict``) before being passed to the next stage. Here is an example:

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

!!! Note

    If using ``asdict``, you need to name the elements within ``parallel``. The returned ``dict`` will use these names as the ``key``.

#### Sequential Execution

By default, ``parallel`` executes in parallel using multiple threads. In some special cases, you can change it to sequential execution as needed. Here is an example:

```python
from lazyllm import parallel

def f1(input): return input

with parallel.sequential() as p:
    p.f1 = f1
    p.f2 = f1
assert p(1) == (1, 1)
```

!!! Note

    ``diverter`` can also achieve sequential execution through ``.sequential``

## Summary

This article focused on ``pipeline`` and ``parallel``. It is hoped that you now have a basic understanding of how to use LazyLLM's flow to build complex applications. Other data flow components are not discussed in detail here; you can refer to [flow][lazyllm.flow.FlowBase] for their usage.
