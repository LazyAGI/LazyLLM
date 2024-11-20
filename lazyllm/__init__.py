# -*- coding: utf-8 -*-

from .configs import config
from .configs import * # noqa F401 of Config
from .common import *  # noqa F403
from .launcher import LazyLLMLaunchersBase
from .flow import *  # noqa F403
from .components import (LazyLLMDataprocBase, LazyLLMFinetuneBase, LazyLLMDeployBase,
                         LazyLLMValidateBase, register as component_register, Prompter,
                         AlpacaPrompter, ChatPrompter, FastapiApp, JsonFormatter, FileFormatter)

from .module import (ModuleBase, ModuleBase as Module, UrlModule, TrainableModule, ActionModule,
                     ServerModule, TrialModule, register as module_register,
                     OnlineChatModule, OnlineEmbeddingModule, AutoModel)
from .client import redis_client
from .hook import LazyLLMHook
from .tools import (Document, Reranker, Retriever, WebModule, ToolManager, FunctionCall,
                    FunctionCallAgent, fc_register, ReactAgent, PlanAndSolveAgent, ReWOOAgent, SentenceSplitter,
                    LLMParser)
from .docs import add_doc

config.done()


del LazyLLMRegisterMetaClass  # noqa F821
del _get_base_cls_from_registry  # noqa F821


__all__ = [
    # components
    'LazyLLMDataprocBase',  #
    'LazyLLMFinetuneBase',        # finetune
    'LazyLLMDeployBase',          # deploy
    'LazyLLMValidateBase',        #
    'component_register',
    'Prompter',
    'AlpacaPrompter',
    'ChatPrompter',
    'FastapiApp',
    'JsonFormatter',
    'FileFormatter',

    # launcher
    'LazyLLMLaunchersBase',        # empty, slurm, sco

    # configs
    'Mode',

    # module
    'ModuleBase',
    'Module',
    'UrlModule',
    'TrainableModule',
    'ActionModule',
    'ServerModule',
    'WebModule',
    'TrialModule',
    'module_register',
    'OnlineChatModule',
    'OnlineEmbeddingModule',
    'AutoModel',

    # client
    'redis_client',

    # hook
    'LazyLLMHook',

    # tools
    'Document',
    'Retriever',
    'Reranker',
    'ToolManager',
    'FunctionCall',
    'FunctionCallAgent',
    'fc_register',
    "LLMParser",
    'ReactAgent',
    'PlanAndSolveAgent',
    'ReWOOAgent',
    'SentenceSplitter',

    # docs
    'add_doc',
]

__all__ += common.__all__  # noqa F405
__all__ += flow.__all__  # noqa F405
