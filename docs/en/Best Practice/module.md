# Top-Level Core Concept: Module

A Module is the top-level component in LazyLLM and one of its core concepts. A Module possesses four key capabilities: training, deployment, inference, and evaluation. Each Module can choose to implement some or all of these capabilities,
and each capability can be composed of one or more functions, Components, or other Modules. In this chapter, we will provide a detailed introduction to the usage of Modules.

## API Reference

You can refer to the Module's API documentation [module][lazyllm.module.ModuleBase]

Defining a Module (``Module`` )

### By inheriting

To define a ``Module``, you simply need to create a custom class that inherits from ``lazyllm.module.ModuleBase``. The custom module needs to implement at least one of the following three methods:

1. ``_get_train_tasks``: Defines training/fine-tuning tasks, returns a training/fine-tuning task ``pipeline``, and executes the tasks when the ``update`` method is called.
2. ``_get_deploy_tasks``: Defines deployment tasks, returns a deployment task ``pipeline``, and executes deployment tasks when the ``start`` method is called; or after executing training tasks when the ``update`` method is called.
3. ``forward``: Defines the specific execution process of the ``Module``, which will be called by ``Module.__call__.``

Here is an example:

```python
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
<Module type=MyModule name=example>
>>> m.start()
Module example deployed!
<Module type=MyModule name=example>
>>> m.evalset(['hello', 'world'])
>>> m.update().eval_result
Module example trained!
Module example deployed!
['[Module example get input: hello]', '[Module example get input: world]']
```

!!! Note

    The test set is set by calling `evalset`, and there is no need to explicitly override any function. All `Modules` can have a test set.

### Using the Built-in Registry

LazyLLM implements a registry for ``Modules``, which allows you to easily register functions as ``Modules``. Here is a specific example:

```python
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
```

### Submodules

#### Concept of Submodules

Similar to the ``Module`` class in PyTorch, the ``Module`` in LazyLLM also has a hierarchical concept, where a Module can have one or more ``Submodule``.
When using the ``update`` function to update a ``Module``, its ``Submodule`` will also be updated, unless explicitly set not to update the ``Submodule``.
Similarly, when using the ``start`` function to start the deployment task of a ``Module``, its ``Submodule`` will also be deployed, unless explicitly set not to deploy the ``Submodule``.
Here is an example:

#### How to Construct Submodules

You can make one ``Module`` a ``Submodule`` of another ``Module`` in the following ways:

1. Pass it as a constructor argument to ``ActionModule`` or ``ServerModule``, as shown in the example below:

        >>> m1 = MyModule('m1')
        >>> m2 = MyModule('m2')
        >>> am = lazyllm.ActionModule(m1, m2)
        >>> am.submodules
        [<Module type=MyModule name=m1>, <Module type=MyModule name=m2>]
        >>> sm = lazyllm.ServerModule(m1)
        >>> sm.submodules
        [<Module type=MyModule name=m1>]

    !!! Note

        - When a flow is passed as a constructor argument to ``ActionModule`` or ``ServerModule``, any ``Module`` within it will also become a ``Submodule`` of the ``ActionModule`` or ``ServerModule``. Here's an example:

                >>> m1 = MyModule('m1')
                >>> m2 = MyModule('m2')
                >>> m3 = MyModule('m3')
                >>> am = lazyllm.ActionModule(lazyllm.pipeline(m1, lazyllm.parallel(m2, m3)))
                >>> am.submodules
                [<Module type=MyModule name=m1>, <Module type=MyModule name=m2>, <Module type=MyModule name=m3>]
                >>> sm = lazyllm.ServerModule(lazyllm.pipeline(m1, lazyllm.parallel(m2, m3)))
                >>> sm.submodules
                [<Module type=_ServerModuleImpl>]
                >>> sm.submodules[0].submodules
                [<Module type=Action return_trace=False sub-category=Flow type=Pipeline items=[]>
                └- <Flow type=Pipeline items=[]>
                    |- <Module type=MyModule name=m1>
                    └- <Flow type=Parallel items=[]>
                        |- <Module type=MyModule name=m2>
                        └- <Module type=MyModule name=m3>
                ]

        - When directly printing the ``repr`` of a ``Module``, it will display its hierarchical structure, including all its ``Submodules``. Continuing from the previous example:

                >>> sm
                <Module type=Server stream=False return_trace=False>
                └- <Module type=Action return_trace=False sub-category=Flow type=Pipeline items=[]>
                └- <Flow type=Pipeline items=[]>
                    |- <Module type=MyModule name=m1>
                    └- <Flow type=Parallel items=[]>
                        |- <Module type=MyModule name=m2>
                        └- <Module type=MyModule name=m3>

2. Setting another ``Module`` as a member variable in a ``Module`` can make the other ``Module`` become its ``submodule``. Here is an example:

        >>> class MyModule2(lazyllm.module.ModuleBase):
        ...
        ...     def __init__(self, name, return_trace=True):
        ...         super(__class__, self).__init__(return_trace=return_trace)
        ...         self.name = name
        ...         self.m1_1 = MyModule('m1-1')
        ...         self.m1_2 = MyModule('m1-2')
        ...
        >>> m2 = MyModule2('m2')
        >>> m2.submodules
        [<Module type=MyModule name=m1-1>, <Module type=MyModule name=m1-2>]

#### Utilizing Submodules for Joint Application Deployment

When training/fine-tuning or deploying a ``Module``, a depth-first strategy will be used to search for all its ``Submodules`` and deploy them one by one. Here is an example:

```python
>>> class MyModule2(lazyllm.module.ModuleBase):
...
...     def __init__(self, name, return_trace=True):
...         super(__class__, self).__init__(return_trace=return_trace)
...         self.name = name
...         self.m1_1 = MyModule(f'{name} m1-1')
...         self.m1_2 = MyModule(f'{name} m1-2')
...
...     def _get_deploy_tasks(self):
...         return lazyllm.pipeline(lambda : print(f'Module {self.name} deployed!'))
...
...     def __repr__(self):
...         return lazyllm.make_repr('Module', self.__class__, subs=[repr(self.m1_1), repr(self.m1_2)])
...
>>> am = lazyllm.ActionModule(MyModule2('m2-1'), MyModule2('m2-2'))
>>> am
<Module type=Action return_trace=False sub-category=Flow type=Pipeline items=[]>
|- <Module type=MyModule2>
|   |- <Module type=MyModule name=m2-1 m1-1>
|   └- <Module type=MyModule name=m2-1 m1-2>
└- <Module type=MyModule2>
    |- <Module type=MyModule name=m2-2 m1-1>
    └- <Module type=MyModule name=m2-2 m1-2>
>>> am.update()
Module m2-1 m1-1 trained!
Module m2-1 m1-2 trained!
Module m2-2 m1-1 trained!
Module m2-2 m1-2 trained!
Module m2-1 m1-1 deployed!
Module m2-1 m1-2 deployed!
Module m2-1 deployed!
Module m2-2 m1-1 deployed!
Module m2-2 m1-2 deployed!
Module m2-2 deployed!
```

!!! Note

    - It can be seen that when updating the ``ActionModule``, all its ``Submodules`` will be updated together. If there are deployment tasks, they will be executed after all the training/fine-tuning tasks are completed. Since parent modules may depend on submodules, submodules will be deployed first, followed by parent modules.
    - When the ``Redis`` service is configured, the lightweight gateway mechanism provided by LazyLLM can be used to achieve parallel deployment of all services.
