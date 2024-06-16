LazyLLM的顶层核心概念：模块
=========================

Module是LazyLLM中的顶层组件，也是LazyLLM最核心的概念之一。Module具备训练、部署、推理和评测四项关键能力，每个模块可以选择实现其中的部分或者全部的能力，
每项能力都可以由一到多个函数、Component或其他Module组成。本章我们会详细介绍Module的使用方式。

API Reference
------------------------

Module的API文档可以参考 :ref:`api.module`

.. _bestpractice.module.define:

定义一个模块（ ``Module`` ）
------------------------

通过继承
+++++++++++++++

若想定义一个 ``Module`` ，只需要自定义一个类，继承自 ``lazyllm.module.ModuleBase`` 即可。自定义的模块需要实现下列三个方法之一：

1. ``_get_train_tasks``: 定义训练 / 微调任务，返回一个训练 / 微调任务的 ``pipeline`` ，在调用 ``update`` 方法时执行任务
2. ``_get_deploy_tasks``: 定义部署任务，返回一个部署任务的 ``pipeline`` ，在调用 ``start`` 方法时执行部署任务；或者在调用 ``update`` 方法时执行完训练任务后执行部署任务
3. ``forward``: 定义 ``Module`` 的具体执行过程，会被 ``Module.__call__`` 调用。

下面给出一个例子

.. code-block:: python

    >>> import lazyllm

    >>> class MyModule(lazyllm.module.ModuleBase):
    ...    
    ...     def __init__(self, name, return_trace=True):
    ...         super(__class__, self).__init__(return_trace=return_trace)
    ...         self.name = name
    ... 
    ...     def _get_train_tasks(self):
    ...         return lazyllm.pipeline(lambda : print(f'Module {self.name} trained!'))
    ... 
    ...     def _get_deploy_tasks(self):
    ...         return lazyllm.pipeline(lambda : print(f'Module {self.name} deployed!'))
    ... 
    ...     def forward(self, input):
    ...         return f'[Module {self.name} get input: {input}]'
    ... 
    >>> m = MyModule('example')
    >>> m('hello world')
    '[Module example get input: hello world]'
    >>> m.update()
    Module example trained!
    Module example deployed!
    >>> m.start()
    Module example deployed! 
    >>> m.evalset(['hello', 'world'])
    >>> m.update()
    ['[Module example get input: hello]', '[Module example get input: world]']

.. note::
    
    通过调用 ``evalset`` 来设置测试集，所有的 ``Module`` 均可以设置测试集


通过内置的注册器
+++++++++++++++

LazyLLM实现了一个 ``Module`` 的注册器，利用它可以很方便的将函数注册成 ``Module`` 。下面给出一个具体的例子：

.. code-block:: python

    >>> import lazyllm
    >>> lazyllm.module_register.new_group('mymodules')
    >>> @lazyllm.module_register('mymodules')
    ... def m(input):
    ...     return f'module m get input: {input}'
    ... 
    >>> lazyllm.mymodules.m()(1)
    'module m get input: 1'
    >>> m = lazyllm.mymodules.m()
    >>> m.evalset([1, 2, 3])
    >>> m.eval().eval_result
    ['module m get input: 1', 'module m get input: 2', 'module m get input: 3']

Submodules
------------------

Submodules的概念
++++++++++++++++++

与 ``pytorch`` 的 ``Module`` 类似，LazyLLM的 ``Module`` 也有层级的概念，一个 ``Module`` 可以有一个到多个 ``Submodule``。
当使用 ``update`` 函数更新一个  ``Module`` 时，也会对应对其 ``Submodule`` 进行更新，除非显式设置不更新 ``Submodule`` 。
类似的，当使用 ``start`` 函数启动一个  ``Module`` 的部署任务时，也会对应对其 ``Submodule`` 进行部署，除非显式设置不部署 ``Submodule`` 。
下面给出一个例子:

如何构建Submodules
+++++++++++++++++++

您可以通过以下几种方式，让一个 ``Module`` 成为另一个 ``Module`` 的 ``Submodule`` :

1. 作为构造参数传入 ``ActionModule`` 或 ``ServerModule`` 等，下面给出一个例子

.. note::
    
    当flow作为 ``ActionModule`` 或 ``ServerModule`` 的构造参数时，若其中的存在 ``Module`` ，也会变成  ``ActionModule`` 或 ``ServerModule`` 的 ``SubModule``