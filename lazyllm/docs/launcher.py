# flake8: noqa E501
from . import utils
import functools
import lazyllm


add_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm)
add_english_doc = functools.partial(utils.add_english_doc, module=lazyllm)
add_example = functools.partial(utils.add_example, module=lazyllm)


add_chinese_doc('LazyLLMLaunchersBase', '''\
用于统一管理外部进程或分布式作业（训练/推理等）生命周期的启动器抽象基类。不同平台（本地、SLURM、K8s、云资源等）的具体启动器应继承该类并实现核心接口。

Args:
    None.
''')

add_english_doc('LazyLLMLaunchersBase', '''\
An abstract base class that standardizes the lifecycle management of external processes or distributed jobs 
(training/inference, etc.). Concrete launchers for different backends (local, SLURM, K8s, cloud, etc.) should 
inherit and implement the core interfaces.

Args:
    None.
''')

add_chinese_doc('LazyLLMLaunchersBase.makejob', '''\
根据给定命令创建并返回作业/进程句柄。需由子类实现。

Args:
    cmd: 用于创建作业的命令或配置（如字符串、参数列表或作业描述对象）。

Raises:
    NotImplementedError: 基类未实现，子类必须覆盖。
''')

add_english_doc('LazyLLMLaunchersBase.makejob', '''\
Create and return a job/process handle for the given command. Must be implemented by subclasses.

Args:
    cmd: The command or specification to create a job (e.g., string, argv list, or a job spec object).

Raises:
    NotImplementedError: The base class does not implement this method.
''')

add_chinese_doc('LazyLLMLaunchersBase.launch', '''\
启动一个或多个作业，并将其登记到 all_processes[self._id] 中。需由子类实现。

Args:
    *args: 与具体实现相关的位置参数。
    **kw: 与具体实现相关的关键字参数。

Raises:
    NotImplementedError: 基类未实现，子类必须覆盖。
''')

add_english_doc('LazyLLMLaunchersBase.launch', '''\
Launch one or more jobs and register them under all_processes[self._id]. Must be implemented by subclasses.

Args:
    *args: Implementation-specific positional arguments.
    **kw: Implementation-specific keyword arguments.

Raises:
    NotImplementedError: The base class does not implement this method.
''')

add_chinese_doc('LazyLLMLaunchersBase.cleanup', '''\
停止并清理当前启动器登记的所有作业，从 all_processes 中移除相应记录，并在最后阻塞等待作业结束。

Args:
    None.
''')

add_english_doc('LazyLLMLaunchersBase.cleanup', '''\
Stop and clean up all jobs registered under this launcher, remove them from all_processes, and finally wait for termination.

Args:
    None.
''')

add_chinese_doc('LazyLLMLaunchersBase.wait', '''\
阻塞等待当前启动器登记的所有作业结束。

Args:
    None.
''')

add_english_doc('LazyLLMLaunchersBase.wait', '''\
Block until all jobs registered under this launcher finish.

Args:
    None.
''')

add_chinese_doc('LazyLLMLaunchersBase.clone', '''\
深拷贝当前启动器实例并分配新的唯一 _id，返回克隆后的实例。

Args:
    None.

Returns:
    LazyLLMLaunchersBase: 克隆出的启动器实例。
''')

add_english_doc('LazyLLMLaunchersBase.clone', '''\
Deep-copy this launcher, assign a new unique _id, and return the cloned instance.

Args:
    None.

Returns:
    LazyLLMLaunchersBase: The cloned launcher.
''')