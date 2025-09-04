## Register

::: lazyllm.common.Register
    options:
      heading_level: 3

::: lazyllm.common.registry.LazyDict
    options:
      heading_level: 3
      members: [remove, set_default]

---

::: lazyllm.common.common.ResultCollector
    members: keys, items
    exclude-members:

::: lazyllm.common.common.EnvVarContextManager
    members: 
    exclude-members:

## Bind

::: lazyllm.common.bind
    options:
      heading_level: 3

---

## Package

::: lazyllm.common.package
    options:
      heading_level: 3

---

## Identity

::: lazyllm.common.Identity
    options:
      heading_level: 3

---

## Compilation

::: lazyllm.common.compile_func
    options:
      heading_level: 3

---

## Queue

::: lazyllm.common.FileSystemQueue
    members: [enqueue, dequeue, peek, size, clear, init, get_instance, set_default]
    exclude-members:

::: lazyllm.common.multiprocessing.SpawnProcess
    members: [start]

::: lazyllm.common.queue.SQLiteQueue
    options:
      heading_level: 3

::: lazyllm.common.ReadOnlyWrapper
    members: [set, isNone]
    exclude-members:

::: lazyllm.common.queue.RedisQueue
    members: 
    exclude-members:

---

## Multiprocessing

::: lazyllm.common.ForkProcess
    members: [work, start]
    exclude-members:

---

## Options

::: lazyllm.common.Option
    members: 
    exclude-members:

---

## DynamicDescriptor

::: lazyllm.common.DynamicDescriptor
    members:
    - Impl
    exclude-members:

::: lazyllm.common.CaseInsensitiveDict
    members: 
    exclude-members:

::: lazyllm.common.ProcessPoolExecutor
    members: [submit]
    exclude-members:

---

::: lazyllm.common.ArgsDict
    members: check_and_update, parse_kwargs
    exclude-members:
## Threading

::: lazyllm.common.Thread
    members: [work, get_result]
    exclude-members:

---

## LazyLLMCMD

::: lazyllm.common.LazyLLMCMD
    members: [with_cmd, get_args]
    exclude-members:


::: lazyllm.common.utils.SecurityVisitor
    members: visit_Call, visit_Import, visit_ImportFrom, visit_Attribute
    exclude-members:

::: lazyllm.common.common.Finalizer
    members: 
    exclude-members:

::: lazyllm.common.FlatList.absorb
    members: 
    exclude-members:    