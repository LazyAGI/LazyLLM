# -*- coding: utf-8 -*-

__version__ = '0.7.2'

import importlib
import builtins
from .configs import config, refresh_config, Mode, Config, Namespace as namespace
from .common import *  # noqa F403
from .launcher import LazyLLMLaunchersBase
from .flow import *  # noqa F403
from .components import (LazyLLMDataprocBase, LazyLLMFinetuneBase, LazyLLMDeployBase,
                         LazyLLMValidateBase, register as component_register, Prompter,
                         AlpacaPrompter, ChatPrompter, FastapiApp, JsonFormatter, FileFormatter)

from .module import (ModuleBase, ModuleBase as Module, UrlModule, TrainableModule, ActionModule,
                     ServerModule, TrialModule, register as module_register,
                     OnlineModule, OnlineChatModule, OnlineEmbeddingModule, AutoModel, OnlineMultiModalModule)
from .hook import LazyLLMHook, LazyLLMFuncHook
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .tools import (Document, Reranker, Retriever, WebModule, ToolManager, FunctionCall,
                        FunctionCallAgent, fc_register, ReactAgent, PlanAndSolveAgent, ReWOOAgent, SentenceSplitter,
                        LLMParser)
from .patch import patch_os_env
from .docs import add_doc
config.done()

patch_os_env(lambda key, value: refresh_config(key), refresh_config)

del LazyLLMRegisterMetaClass  # noqa F821
del LazyLLMRegisterMetaABCClass  # noqa F821
del _get_base_cls_from_registry  # noqa F821
del patch_os_env


def __getattr__(name: str):
    if name == 'tools':
        return importlib.import_module('lazyllm.tools')
    elif name in __all__:
        tools = importlib.import_module('lazyllm.tools')
        builtins.globals()[name] = value = getattr(tools, name)
        return value
    raise AttributeError(f"module 'lazyllm' has no attribute '{name}'")


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
    'Config',
    'config',
    'refresh_config',
    'namespace',

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
    'OnlineModule',
    'OnlineChatModule',
    'OnlineEmbeddingModule',
    'OnlineMultiModalModule',
    'AutoModel',

    # hook
    'LazyLLMHook',
    'LazyLLMFuncHook',

    # tools
    'Document',
    'Retriever',
    'Reranker',
    'ToolManager',
    'FunctionCall',
    'FunctionCallAgent',
    'fc_register',
    'LLMParser',
    'ReactAgent',
    'PlanAndSolveAgent',
    'ReWOOAgent',
    'SentenceSplitter',

    # docs
    'add_doc',
]

__all__ += common.__all__  # noqa F405
__all__ += flow.__all__  # noqa F405
