# -*- coding: utf-8 -*-
# flake8: noqa: F401

from .configs import config
from .configs import * # noqa F401 of Config
from .common import *  # noqa F403
from .launcher import LazyLLMLaunchersBase
from .flow import *  # noqa F403
from .components import (LazyLLMDataprocBase, LazyLLMFinetuneBase, LazyLLMDeployBase,
                         LazyLLMValidateBase, register as component_register, Prompter,
                         AlpacaPrompter, ChatPrompter, FastapiApp, JsonFormatter)

from .module import (ModuleBase, UrlModule, TrainableModule, ActionModule,
                     ServerModule, TrialModule, register as module_register,
                     OnlineChatModule, OnlineEmbeddingModule, AutoModel)
from .client import redis_client
from .tools import (Document, Reranker, Retriever, WebModule, ToolManager, FunctionCall,
                    FunctionCallAgent, fc_register, ReactAgent, PlanAndSolveAgent, ReWOOAgent, SentenceSplitter,
                    LLMParser)
from .engine import *  # noqa F403
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
    'module_register',
    'OnlineChatModule',
    'OnlineEmbeddingModule',
    'AutoModel'

    # client
    'redis_client',

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
__all__ += engine.__all__  # noqa F405
