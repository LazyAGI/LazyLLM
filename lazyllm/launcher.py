from lazyllm import LazyLLMRegisterMetaClass


class LazyLLMLauncherBase(object, metaclass=LazyLLMRegisterMetaClass):
    pass

class EmptyLauncher(LazyLLMLauncherBase):
    pass

class SlurmLauncher(LazyLLMLauncherBase):
    pass
    
class ScoLauncher(LazyLLMLauncherBase):
    pass
