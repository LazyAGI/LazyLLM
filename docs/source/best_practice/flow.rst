LazyLLM中的数据流
-----------------

LazyLLM中定义了大量的数据流组件，用于让您像搭积木一样，借助LazyLLM中提供的工具和组件，来搭建复杂的大模型应用。本节会详细介绍数据流的使用方法。

定义和API文档
============
数据流的定义和基本使用方法如 :ref:`apo.flow` 中所述

pipeline
============

基本使用
^^^^^^^^

Pipeline是顺次执行的数据流，上一个阶段的输出成为下一个阶段的输入。pipeline支持函数和仿函数（或仿函数的type）。一个典型的pipeline如下所示:

.. code-block:: python

    from lazyllm import pipeline

    class Functor(object):
        def __call__(self, x): return x * x

    def f1(input): return input + 1
    f2 = lambda x: x * 2
    f3 = Functor()

    assert pipeline(f1, f2, f3, Functor)(1) == 256


.. note::
    借助LazyLLM的注册机制 :ref:`api.components.register` 注册的函数，也可以直接被pipeline使用，下面给出一个例子


.. code-block:: python

    import lazyllm
    from lazyllm import pipeline, component_register

    component_register.new_group('g1')

    @component_register('g1')
    def test1(input): return input + 1

    @component_register('g1')
    def test2(input): return input * 3

    assert pipeline(lazyllm.g1.test1, lazyllm.g1.test2(launcher=lazyllm.launchers.empty))(1) == 6


with语句
^^^^^^^^

除了基本的用法之外，pipeline还支持一个更为灵活的用法 ``with pipeline() as p`` 来让代码更加的简洁和清晰，示例如下

.. code-block:: python

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

.. note::
    ``parallel``, ``diverter`` 等也支持with的用法。

参数绑定
^^^^^^^^

很多时候，我们并不希望一成不变的将上级的输出给到下一级作为输入，某一下游环节可以需要很久之前的某环节的输出，甚至是整个pipeline的输入。
在计算图模式的范式下（例如dify和llamaindex），会把函数作为节点，把数据作为边，通过添加边的方式来实现这一行为。
但LazyLLM不会让你如此复杂，你仅需要掌握参数绑定，就可以自由的在pipeline中从上游向下游传递参数。

假设我们定义了一些函数，本小节会一直使用这些函数，不再重复定义。

.. code-block:: python

    def f1(input, input2=0): return input + input2 + 1
    def f2(input): return input + 3
    def f3(input): return f'f3-{input}'
    def f4(in1, in2, in3): return f'get [{in1}], [{in2}], [{in3}]'

下面给出一个参数绑定的具体例子：

.. code-block:: python

    from lazyllm import pipeline, _0
    with pipeline() as p:
        p.f1 = f1
        p.f2 = f2
        p.f3 = f3
        p.f4 = bind(f4, p.input, _0, p.f2)
    assert p(1) == 'get [1], [test3-5], [5]'

上述例子中， ``bind`` 函数用于参数绑定，它的基本使用方法和C++的 ``std::bind`` 一致，其中 ``_0`` 表示新函数的第0个参数在被绑定的函数的参数表中的位置。
对于上面的案例，整个pipeline的输入会作为f4的第一个参数（此处我们假设从第一个开始计数），f3的输出（即新函数的输入）会作为f4的第二个参数，f2的输出会作为f4的第三个参数。

.. note::

    - 参数绑定仅在一个pipeline中生效，仅允许下游函数绑定上游函数的输出作为参数。
    - 使用参数绑定后，平铺的方式传入的参数中，未被 ``_0``, ``_1``等 ``placeholder`` 引用的会被丢弃

上面的方式已经足够简单和清晰，如果您仍然觉得 ```bind`` 作为函数不够直观，可以尝试使用如下方式，两种方式没有任何区别：

.. code-block:: python

    from lazyllm import pipeline, _0
    with pipeline() as p:
        p.f1 = f1
        p.f2 = f2
        p.f3 = f3
        p.f4 = f4 | bind(p.input, _0, p.f2)
    assert p(1) == 'get [1], [test3-5], [5]'

.. note::

    请小心lambda函数！如果使用了lambda函数，请注意给lambda函数加括号，例如 ``(lambda x, y: pass) | bind(1, _0)``

除了C++的bind方式之外，作为python，我们额外提供了 ``kwargs`` 的参数绑定， ``kwargs``和c++的绑定方式可以混合使用，示例如下:

.. code-block:: python

    from lazyllm import pipeline, _0
    with pipeline() as p:
        p.f1 = f1
        p.f2 = f2
        p.f3 = f3
        p.f4 = f4 | bind(p.input, _0, in3=p.f2)
    assert p(1) == 'get [1], [test3-5], [5]'

.. note::

    通过 ``kwargs`` 绑定的参数的值不能使用 ``_0`` 等

如果pipeline的输入比较复杂，可以直接对 ``input`` 做一次简单的解析处理，示例如下:

.. code-block:: python

    def f1(input): return dict(a=input[0], b=input[1])
    def f2(input): return input['a'] + input['b']
    def f3(input, extro): return f'[{input} + {extro}]'

    with pipeline() as p1:
        p1.f1 = f1
        with pipeline() as p1.p2:
            p2.f2 = f2
            p2.f3 = f3 | bind(extro=p2.input['b'])
        p1.f3 = f3 | bind(extro=p1.input[0])
    
    assert p1([1, 2]) == '[[3 + 2] + 1]'

上面的例子比较复杂，我们逐步来解析。首先输入的list经过 ``p1.f1`` 变成 ``dict(a=1, b=2)`` ，则p2的输入也是 ``dict(a=1, b=2)``，经过 ``p2.f2`` 之后输出为 ``3``，
然后 ``p2.f3`` 绑定了 ``p2`` 的输入的 ``['b']``， 即 ``2``, 因此p2.f3的输出是 ``[3 + 2]``, 回到 ``p1.f3``，它绑定了 ``p1`` 的输入的第 ``0`` 个元素，因此最终的输出是 ``[[3 + 2] + 1]``
