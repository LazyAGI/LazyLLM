from lazyllm import LazyLLMRegisterMetaClass


class LazyLLMLauncherBase(object, metaclass=LazyLLMRegisterMetaClass):
    def launch(self, f, *args, **kw):
        raise NotImplementedError

class EmptyLauncher(LazyLLMLauncherBase):
    def launch(self, f, *args, **kw):
        return f(*args, **kw)

class SlurmLauncher(LazyLLMLauncherBase):
    def __init__(self, nproc, ngpus, timeout):
        self.nproc, self.ngpus, self.timeout = nproc, ngpus, timeout
        super(__class__, self).__init__()
    
    def launch(self, f, *args, **kw):
        return super().launch(f, *args, **kw)
    
class ScoLauncher(LazyLLMLauncherBase):
    def __init__(self, nproc, ngpus, timeout):
        self.nproc, self.ngpus, self.timeout = nproc, ngpus, timeout
        super(__class__, self).__init__()

    def launch(self, f, *args, **kw):
        return super().launch(f, *args, **kw)
