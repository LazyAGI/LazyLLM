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
    members: 
    - keys
    - items
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

## Identity

::: lazyllm.common.Identity
    options:
      heading_level: 3

---

## Compilation

::: lazyllm.common.compile_func
    options:
      heading_level: 3

## Queue

::: lazyllm.common.FileSystemQueue
    members: enqueue, dequeue, peek, size, clear
    exclude-members:

::: lazyllm.common.ReadOnlyWrapper
    members: set, isNone
    exclude-members:

::: lazyllm.common.queue.RedisQueue
    members: 
    exclude-members:

::: lazyllm.common.CaseInsensitiveDict
    members: 
    exclude-members:

::: lazyllm.common.ProcessPoolExecutor
    members: submit
    exclude-members:

## Multiprocessing

::: lazyllm.common.ForkProcess
    members: work, start
    exclude-members:

## Options

::: lazyllm.common.Option
    members: 
    exclude-members:

::: lazyllm.common.multiprocessing.SpawnProcess
    members: start
    exclude-members:

::: lazyllm.common.queue.SQLiteQueue
    options:
      heading_level: 3

## Threading

::: lazyllm.common.Thread
    members: work, get_result
    exclude-members:
    

## LazyLLMCMD

::: lazyllm.common.LazyLLMCMD
    members: with_cmd, get_args
    exclude-members:
    