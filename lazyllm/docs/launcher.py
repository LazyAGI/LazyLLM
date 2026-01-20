# flake8: noqa E501
from . import utils
import functools
import lazyllm


add_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm.launcher)
add_english_doc = functools.partial(utils.add_english_doc, module=lazyllm.launcher)
add_example = functools.partial(utils.add_example, module=lazyllm.launcher)

# LazyLLMLaunchersBase
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

**Returns:**\n
- LazyLLMLaunchersBase: 克隆出的启动器实例。
''')

add_english_doc('LazyLLMLaunchersBase.clone', '''\
Deep-copy this launcher, assign a new unique _id, and return the cloned instance.

Args:
    None.

**Returns:**\n
- LazyLLMLaunchersBase: The cloned launcher.
''')

# Launcher-EmptyLauncher
add_chinese_doc('EmptyLauncher', '''\
此类是 ``LazyLLMLaunchersBase`` 的子类，作为一个本地的启动器。

Args:
    subprocess (bool): 是否使用子进程来启动。默认为 `False`。
    sync (bool): 是否同步执行作业。默认为 `True`，否则为异步执行。
''')

add_english_doc('EmptyLauncher', '''\
This class is a subclass of ``LazyLLMLaunchersBase`` and serves as a local launcher.

Args:
    subprocess (bool): Whether to use a subprocess to launch. Default is ``False``.
    sync (bool): Whether to execute jobs synchronously. Default is ``True``, otherwise it executes asynchronously.
''')

add_example('EmptyLauncher', '''\
>>> import lazyllm
>>> launcher = lazyllm.launchers.empty()
''')

# Launcher-SlurmLauncher
add_chinese_doc('SlurmLauncher', '''\
此类是 ``LazyLLMLaunchersBase`` 的子类，作为 Slurm 启动器。

具体而言，它提供了启动和配置 Slurm 作业的方法，包括指定分区、节点数量、进程数量、GPU 数量以及超时时间等参数。

Args:
    partition (str): 要使用的 Slurm 分区。默认为 ``None``，此时将使用 ``lazyllm.config['partition']`` 中的默认分区。
                     该配置可通过设置环境变量来生效，如 ``export LAZYLLM_SLURM_PART=a100``。
    nnode (int): 要使用的节点数量。默认为 ``1``。
    nproc (int): 每个节点要使用的进程数量。默认为 ``1``。
    ngpus (int): 每个节点要使用的 GPU 数量。默认为 ``None``，即不使用 GPU。
    timeout (int): 作业的超时时间（以秒为单位）。默认为 ``None``，此时将不设置超时时间。
    sync (bool): 是否同步执行作业。默认为 ``True``，否则为异步执行。
    **kwargs: 额外参数，其中支持：
        - num_can_use_nodes (int): 可使用的最大节点数。默认为 ``5``。
''')

add_english_doc('SlurmLauncher', '''\
This class is a subclass of ``LazyLLMLaunchersBase`` and acts as a Slurm launcher.

Specifically, it provides methods to start and configure Slurm jobs, including specifying parameters such as the partition, number of nodes, number of processes, number of GPUs, and timeout settings.

Args:
    partition (str): The Slurm partition to use. Defaults to ``None``, in which case the default partition in ``lazyllm.config['partition']`` will be used.
                     This configuration can be enabled by setting environment variables, such as ``export LAZYLLM_SLURM_PART=a100``.
    nnode (int): The number of nodes to use. Defaults to ``1``.
    nproc (int): The number of processes per node. Defaults to ``1``.
    ngpus (int): The number of GPUs per node. Defaults to ``None``, meaning no GPUs will be used.
    timeout (int): The timeout for the job in seconds. Defaults to ``None``, in which case no timeout will be set.
    sync (bool): Whether to execute the job synchronously. Defaults to ``True``, otherwise it will be executed asynchronously.
    **kwargs: Extra keyword arguments, including:
        - num_can_use_nodes (int): The maximum number of nodes that can be used. Defaults to ``5``.
''')

add_example('SlurmLauncher', '''\
>>> import lazyllm
>>> launcher = lazyllm.launchers.slurm(partition='partition_name', nnode=1, nproc=1, ngpus=1, sync=False)
''')

# SlurmLauncher methods
add_chinese_doc('SlurmLauncher.makejob', '''\
创建并返回一个 SlurmLauncher.Job 对象。

Args:
    cmd: 要执行的命令字符串。

**Returns:**\n
- SlurmLauncher.Job: 配置好的 Slurm 作业对象。
''')

add_english_doc('SlurmLauncher.makejob', '''\
Creates and returns a SlurmLauncher.Job object.

Args:
    cmd: The command string to execute.

**Returns:**\n
- SlurmLauncher.Job: A configured Slurm job object.
''')

add_chinese_doc('SlurmLauncher.get_idle_nodes', '''\
获取指定分区中当前可用的节点数量，基于可用 GPU 数量。

该方法通过查询 Slurm 队列状态和节点信息，计算每个节点的可用 GPU 数量，并返回一个字典，其中键为节点 IP，值为可用 GPU 数量。

Args:
    partion (str, optional): 要查询的分区名称。默认为 ``None``，此时使用当前启动器的分区。

**Returns:**\n
- dict: 以节点 IP 为键、可用 GPU 数量为值的字典。
''')

add_english_doc('SlurmLauncher.get_idle_nodes', '''\
Obtains the current number of available nodes in the specified partition based on the available number of GPUs.

This method queries the Slurm queue status and node information to calculate the number of available GPUs for each node, and returns a dictionary with node IP as the key and the number of available GPUs as the value.

Args:
    partion (str, optional): The partition name to query. Defaults to ``None``, in which case the current launcher's partition will be used.

**Returns:**\n
- dict: A dictionary with node IP as the key and the number of available GPUs as the value.
''')

add_chinese_doc('SlurmLauncher.launch', '''\
启动 Slurm 作业并管理其执行。

该方法启动指定的 Slurm 作业，并根据同步设置决定是否等待作业完成。如果设置为同步执行，会持续监控作业状态直到完成，然后停止作业。

Args:
    job: 要启动的 SlurmLauncher.Job 对象。

**Returns:**\n
- 作业的返回值。

Raises:
    AssertionError: 如果传入的 job 不是 SlurmLauncher.Job 类型。
''')

add_english_doc('SlurmLauncher.launch', '''\
Launches a Slurm job and manages its execution.

This method starts the specified Slurm job and decides whether to wait for job completion based on the sync setting. If set to synchronous execution, it continuously monitors the job status until completion, then stops the job.

Args:
    job: The SlurmLauncher.Job object to launch.

**Returns:**\n
- The return value of the job.

Raises:
    AssertionError: If the provided job is not a SlurmLauncher.Job type.
''')

# Launcher-ScoLauncher
add_chinese_doc('ScoLauncher', '''\
此类是 ``LazyLLMLaunchersBase`` 的子类，作为SCO (Sensecore)启动器。

具体而言，它提供了启动和配置 SCO 作业的方法，包括指定分区、工作空间名称、框架类型、节点数量、进程数量、GPU 数量以及是否使用 torchrun 等参数。

Args:
    partition (str): 要使用的分区。默认为 ``None``，此时将使用 ``lazyllm.config['partition']`` 中的默认分区。该配置可通过设置环境变量来生效，如 ``export LAZYLLM_SLURM_PART=a100`` 。
    workspace_name (str): SCO 上的工作空间名称。默认为 ``lazyllm.config['sco.workspace']`` 中的配置。该配置可通过设置环境变量来生效，如 ``export LAZYLLM_SCO_WORKSPACE=myspace`` 。
    framework (str): 要使用的框架类型，例如 ``pt`` 代表 PyTorch。默认为 ``pt``。
    nnode  (int): 要使用的节点数量。默认为 ``1``。
    nproc (int): 每个节点要使用的进程数量。默认为 ``1``。
    ngpus: (int): 每个节点要使用的 GPU 数量。默认为 ``1``, 使用1块 GPU。
    torchrun (bool): 是否使用 ``torchrun`` 启动作业。默认为 ``False``。
    sync (bool): 是否同步执行作业。默认为 ``True``，否则为异步执行。
''')

add_english_doc('ScoLauncher', '''\
This class is a subclass of ``LazyLLMLaunchersBase`` and acts as a SCO launcher.

Specifically, it provides methods to start and configure SCO jobs, including specifying parameters such as the partition, workspace name, framework type, number of nodes, number of processes, number of GPUs, and whether to use torchrun or not.

Args:
    partition (str): The Slurm partition to use. Defaults to ``None``, in which case the default partition in ``lazyllm.config['partition']`` will be used. This configuration can be enabled by setting environment variables, such as ``export LAZYLLM_SLURM_PART=a100``.
    workspace_name (str): The workspace name on SCO. Defaults to the configuration in ``lazyllm.config['sco.workspace']``. This configuration can be enabled by setting environment variables, such as ``export LAZYLLM_SCO_WORKSPACE=myspace``.
    framework (str): The framework type to use, for example, ``pt`` for PyTorch. Defaults to ``pt``.
    nnode  (int): The number of nodes to use. Defaults to ``1``.
    nproc (int): The number of processes per node. Defaults to ``1``.
    ngpus (int): The number of GPUs per node. Defaults to ``1``, using 1 GPU.
    torchrun (bool): Whether to start the job with ``torchrun``. Defaults to ``False``.
    sync (bool): Whether to execute the job synchronously. Defaults to ``True``, otherwise it will be executed asynchronously.
''')

add_example('ScoLauncher', '''\
>>> import lazyllm
>>> launcher = lazyllm.launchers.sco(partition='partition_name', nnode=1, nproc=1, ngpus=1, sync=False)
''')

# Launcher-RemoteLauncher
add_chinese_doc('RemoteLauncher', '''\
此类是 ``LazyLLMLaunchersBase`` 的一个子类，它充当了一个远程启动器的代理。它根据配置文件中的 ``lazyllm.config['launcher']`` 条目动态地创建并返回一个对应的启动器实例(例如：``SlurmLauncher`` 或 ``ScoLauncher``)。

Args:
    *args: 位置参数，将传递给动态创建的启动器构造函数。
    sync (bool): 是否同步执行作业。默认为 ``False``。
    **kwargs: 关键字参数，将传递给动态创建的启动器构造函数。

注意事项: 
    - ``RemoteLauncher`` 不是一个直接的启动器，而是根据配置动态创建一个启动器。 
    - 配置文件中的 ``lazyllm.config['launcher']`` 指定一个存在于 ``lazyllm.launchers`` 模块中的启动器类名。该配置可通过设置环境变量 ``LAZYLLM_DEFAULT_LAUNCHER`` 来设置。如：``export LAZYLLM_DEFAULT_LAUNCHER=sco`` , ``export LAZYLLM_DEFAULT_LAUNCHER=slurm`` 。
''')

add_english_doc('RemoteLauncher', '''\
This class is a subclass of ``LazyLLMLaunchersBase`` and acts as a proxy for a remote launcher. It dynamically creates and returns an instance of the corresponding launcher based on the ``lazyllm.config['launcher']`` entry in the configuration file (for example: ``SlurmLauncher`` or ``ScoLauncher``).

Args:
    *args: Positional arguments that will be passed to the constructor of the dynamically created launcher.
    sync (bool): Whether to execute the job synchronously. Defaults to ``False``.
    **kwargs: Keyword arguments that will be passed to the constructor of the dynamically created launcher.

Notes: 
    - ``RemoteLauncher`` is not a direct launcher but dynamically creates a launcher based on the configuration. 
    - The ``lazyllm.config['launcher']`` in the configuration file specifies a launcher class name present in the ``lazyllm.launchers`` module. This configuration can be set by setting the environment variable ``LAZYLLM_DEAULT_LAUNCHER``. For example: ``export LAZYLLM_DEAULT_LAUNCHER=sco``, ``export LAZYLLM_DEAULT_LAUNCHER=slurm``.
''')

add_example('RemoteLauncher', '''\
>>> import lazyllm
>>> launcher = lazyllm.launchers.remote(ngpus=1)
''')

# Job
add_chinese_doc('Job', '''\
通用任务调度执行类。
该类用于封装一个通过启动器（launcher）调度执行的任务，支持命令包装、同步控制、返回值提取、命令固定等功能。

Args:
    cmd (LazyLLMCMD): 要执行的命令对象。
    launcher (Any): 启动器实例，用于实际任务调度执行。
    sync (bool): 是否为同步执行，默认为 True。
''')

add_english_doc('Job', '''\
Generic task scheduling executor.
This class wraps a task that is launched via a launcher, with features like command fixing, output handling, sync control, and return value capturing.

Args:
    cmd (LazyLLMCMD): The command object to be executed.
    launcher (Any): Launcher instance responsible for task dispatching.
    sync (bool): Whether the task should run synchronously. Defaults to True.
''')

add_chinese_doc('Job.get_executable_cmd', '''\
生成最终可执行命令。
如果已缓存固定命令（fixed），则直接返回。否则根据原始命令进行包裹（wrap）并缓存为 `_fixed_cmd`。

Args:
    fixed (bool): 是否使用已固定的命令对象（若已存在）。

**Returns:**\n
- LazyLLMCMD: 可直接执行的命令对象。
''')

add_english_doc('Job.get_executable_cmd', '''\
Generate the final executable command.
If a fixed command already exists, return it. Otherwise, wrap the original command and cache it as `_fixed_cmd`.

Args:
    fixed (bool): Whether to use the cached fixed command.

**Returns:**\n
- LazyLLMCMD: The executable command object.
''')

add_chinese_doc('Job.start', '''\
对外接口：启动作业，并支持失败时的自动重试。
若作业执行失败，会根据 `restart` 参数控制重试次数。

Args:
    restart (int): 重试次数。默认为 3。
    fixed (bool): 是否使用固定后的命令。用于避免多次构建。
''')

add_english_doc('Job.start', '''\
Public interface to start the job with optional retry on failure.
If the job fails, retries execution based on the `restart` parameter.

Args:
    restart (int): Number of times to retry upon failure. Default is 3.
    fixed (bool): Whether to use the fixed version of the command.
''')

add_chinese_doc('Job.restart', '''\
重新启动作业流程。
该函数会先停止已有进程，等待 2 秒后重新启动作业。

Args:
    fixed (bool): 是否使用固定后的命令。
''')

add_english_doc('Job.restart', '''\
Restart the job by first stopping it and then restarting after a short delay.

Args:
    fixed (bool): Whether to reuse the fixed command object.
''')

add_chinese_doc('Job.wait', '''\
挂起当前线程，等待作业执行完成。当前实现为空方法（子类可重写）。
''')

add_english_doc('Job.wait', '''\
Suspend the current thread until the job finishes.
Empty implementation by default; can be overridden in subclasses.
''')

add_chinese_doc('Job.stop', '''\
停止当前作业。
该方法为接口定义，需子类实现，当前抛出 NotImplementedError。
''')

add_english_doc('Job.stop', '''\
Stop the current job.
This method is an interface placeholder and must be implemented by subclasses.
''')

add_chinese_doc('Job.status', '''\
当前作业状态。
该属性为接口定义，需子类实现，当前抛出 NotImplementedError。
''')

add_english_doc('Job.status', '''\
Current job status.
This property is abstract and must be implemented by subclasses.
''')

# K8sLauncher
add_chinese_doc('K8sLauncher', '''\
K8sLauncher是一个基于Kubernetes的部署启动器，用于在Kubernetes集群中部署和管理服务。

Args:
    kube_config_path (str): Kubernetes配置文件路径。
    resource_config_path (str): 资源配置文件路径。
    image (str): 容器镜像。
    volume_configs (list): 卷配置列表。
    svc_type (str): 服务类型，默认为"LoadBalancer"。
    namespace (str): Kubernetes命名空间，默认为"default"。
    gateway_name (str): 网关名称，默认为"lazyllm-gateway"。
    gateway_class_name (str): 网关类名称，默认为"istio"。
    host (str): HTTP主机名，默认为None。
    path (str): HTTP路径，默认为'/generate'。
    gateway_retry (int): 网关重试次数。
''')

add_english_doc('K8sLauncher', '''\
K8sLauncher is a Kubernetes-based deployment launcher for deploying and managing services in a Kubernetes cluster.

Args:
    kube_config_path (str): Path to the Kubernetes configuration file.
    resource_config_path (str): Path to the resource configuration file.
    image (str): Container image.
    volume_configs (list): List of volume configurations.
    svc_type (str): Service type, defaults to "LoadBalancer".
    namespace (str): Kubernetes namespace, defaults to "default".
    gateway_name (str): Gateway name, defaults to "lazyllm-gateway".
    gateway_class_name (str): Gateway class name, defaults to "istio".
    host (str): HTTP hostname, defaults to None.
    path (str): HTTP path, defaults to '/generate'.
    gateway_retry (int): Number of gateway retries.
''')

add_chinese_doc('K8sLauncher.makejob', '''\
创建一个Kubernetes作业实例。

Args:
    cmd (str): 要执行的命令。

**Returns:**\n
- K8sLauncher.Job: 一个新的Kubernetes作业实例。
''')

add_english_doc('K8sLauncher.makejob', '''\
Create a Kubernetes job instance.

Args:
    cmd (str): The command to execute.

**Returns:**\n
- K8sLauncher.Job: A new Kubernetes job instance.
''')

add_chinese_doc('K8sLauncher.launch', '''\
启动一个Kubernetes作业或可调用对象。

Args:
    f (K8sLauncher.Job): 要启动的Kubernetes作业实例。
    *args: 位置参数。
    **kw: 关键字参数。

**Returns:**\n
- Any: 作业的返回值。

Raises:
    RuntimeError: 当提供的不是Deployment对象时抛出。
''')

add_english_doc('K8sLauncher.launch', '''\
Launch a Kubernetes job or callable object.

Args:
    f (K8sLauncher.Job): The Kubernetes job instance to launch.
    *args: Positional arguments.
    **kw: Keyword arguments.

**Returns:**\n
- Any: The return value of the job.

Raises:
    RuntimeError: When the provided object is not a Deployment object.
''')
