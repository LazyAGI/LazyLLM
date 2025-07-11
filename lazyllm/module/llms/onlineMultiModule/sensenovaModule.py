import lazyllm
from lazyllm.module.onlineMultiModule.onlineMultiModuleBase import OnlineMultiModuleBase

class SensenovaModule(OnlineMultiModuleBase):
    def __init__(self, model_series: str, model_name: str, return_trace: bool = False, **kwargs):
        OnlineMultiModuleBase.__init__(self, model_series=model_series,
                                       model_name=model_name or SensenovaTTSModule.MODEL_NAME
                                       or lazyllm.config['sensenova_tts_model_name'],
                                       return_trace=return_trace, **kwargs)

class SensenovaTTSModule(SensenovaModule):
    MODEL_NAME = "nova-tts-1"

    def _forward(self, input: str = None, **kwargs):
        pass
