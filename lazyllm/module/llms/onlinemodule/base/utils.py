from ....module import ModuleBase
from lazyllm import config


config.add('cache_online_module', bool, False, 'CACHE_ONLINE_MODULE')


class OnlineModuleBase(ModuleBase):
    def __init__(self, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        if config['cache_online_module']:
            self.use_cache()
