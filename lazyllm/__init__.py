from .configs import config, Mode
from .common import *  # noqa F403
from .launcher import LazyLLMLaunchersBase
from .flow import LazyLLMFlowsBase, FlowBase, barrier
from .llms import (LazyLLMDataprocBase, LazyLLMFinetuneBase, LazyLLMDeployBase,
                   LazyLLMValidateBase, register as llmregister, Prompter,
                   FastapiApp)
from .module import (ModuleBase, UrlModule, TrainableModule, ActionModule,
                     ServerModule, WebModule, TrialModule, register as moduleregister,
                     OnlineChatModule, OnlineEmbeddingModule)
from .module import Document, Retriever, Rerank
from . import flows
from .client import redis_client
from .rag.tools import (
    Document,
    DocumentImpl,
    DocumentManager,
    DocWebModule
)

pipeline, parallel = flows.pipeline, flows.parallel
diverter, loop = flows.Diverter, flows.Loop
switch, ifs, warp = flows.Switch, flows.IFS, flows.Warp

config.done()


del LazyLLMRegisterMetaClass  # noqa F821
del _get_base_cls_from_registry  # noqa F821
del flows


__all__ = [
    # llms
    'LazyLLMDataprocBase',  #
    'LazyLLMFinetuneBase',        # finetune
    'LazyLLMDeployBase',          # deploy
    'LazyLLMValidateBase',        #
    'llmregister',
    'Prompter',
    'FastapiApp',

    # flow
    'LazyLLMFlowsBase',            # pipeline, parallel
    'FlowBase',
    'barrier',

    # launcher
    'LazyLLMLaunchersBase',        # empty, slurm, sco

    # configs
    'Mode',

    # module
    'ModuleBase',
    'UrlModule',
    'TrainableModule',
    'ActionModule',
    'ServerModule',
    'WebModule',
    'TrialModule',
    'moduleregister',
    'OnlineChatModule',
    'OnlineEmbeddingModule',

    'Document',
    'Retriever',
    'Rerank',

    # client
    'redis_client',

    'Document',
    'DocumentImpl',
    'DocumentManager',
    'DocWebModule'
]

__all__ += common.__all__  # noqa F405
