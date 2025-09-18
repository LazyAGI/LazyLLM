from lazyllm import launchers
from ..core import ComponentBase


class LazyLLMFinetuneBase(ComponentBase):
    """LazyLLM fine-tuning base component class, inherits from ComponentBase.

Provides base functionality for large language model fine-tuning, supports remote launcher configuration and model path management.

Args:
    base_model (str): Base model path or identifier
    target_path (str): Fine-tuned model output path
    launcher (Launcher, optional): Task launcher, defaults to remote launcher
"""
    __reg_overwrite__ = 'cmd'

    def __init__(self, base_model, target_path, *, launcher=launchers.remote()):  # noqa B008
        super().__init__(launcher=launcher)
        self.base_model = base_model
        self.target_path = target_path
        self.merge_path = None

    def __call__(self, *args, **kw):
        super().__call__(*args, **kw)
        if self.merge_path:
            return self.merge_path
        else:
            return self.target_path


class DummyFinetune(LazyLLMFinetuneBase):
    """DummyFinetune is a subclass of [LazyLLMFinetuneBase][lazyllm.components.LazyLLMFinetuneBase] that serves as a placeholder implementation for fine-tuning.
The class is primarily used for demonstration or testing purposes, as it does not perform any actual fine-tuning logic.

Args:
    base_model: A string specifying the base model name. Defaults to 'base'.
    target_path: A string specifying the target path for fine-tuning outputs. Defaults to 'target'.
    launcher: A launcher instance for executing commands. Defaults to [launchers.remote()][lazyllm.launchers.remote].
    **kw: Additional keyword arguments that are stored for later use.

Returns:
    A string representing a dummy command. The string includes the initial arguments passed during initialization.


Examples:
    >>> from lazyllm.components import DummyFinetune
    >>> from lazyllm import launchers
    >>> # 创建一个 DummyFinetune 实例
    >>> finetuner = DummyFinetune(base_model='example-base', target_path='example-target', launcher=launchers.local(), custom_arg='custom_value')
    >>> # 调用 cmd 方法生成占位命令
    >>> command = finetuner.cmd('--example-arg', key='value')
    >>> print(command)
    ... echo 'dummy finetune!, and init-args is {'custom_arg': 'custom_value'}'
    """
    def __init__(self, base_model='base', target_path='target', *, launcher=launchers.remote(), **kw):  # noqa B008
        super().__init__(base_model, target_path, launcher=launchers.empty)
        self.kw = kw

    def cmd(self, *args, **kw) -> str:
        """The `cmd` method generates a dummy command string for fine-tuning. This method is primarily for testing or demonstration purposes.

Args:
    *args: Positional arguments to be included in the command (not used in this implementation).
    **kw: Keyword arguments to be included in the command (not used in this implementation).

Returns:
    A string representing a dummy command. The string includes the initial arguments (`**kw`) passed during the instance initialization, which are stored in `self.kw`.

Example:
    If the class is initialized with `custom_arg='value'`, calling the `cmd` method will return:
    `"echo 'dummy finetune!, and init-args is {'custom_arg': 'value'}'"`


Examples:
    >>> from lazyllm.components import DummyFinetune
    >>> from lazyllm import launchers
    >>> # 创建一个 DummyFinetune 实例，并传递初始化参数
    >>> finetuner = DummyFinetune(base_model='example-base', target_path='example-target', launcher=launchers.local(), custom_arg='value')
    >>> # 调用 cmd 方法生成占位命令
    >>> command = finetuner.cmd()
    >>> # 打印生成的占位命令
    >>> print(command)
    ... echo 'dummy finetune!, and init-args is {'custom_arg': 'value'}'
    """
        return f'echo \'dummy finetune!, and init-args is {self.kw}\''
