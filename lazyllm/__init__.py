# -*- coding: utf-8 -*-

__version__ = '0.7.5'

import importlib
import builtins
from .configs import config, refresh_config, Mode, Config, Namespace as namespace
from .common import *  # noqa F403
from . import common, flow
from .launcher import LazyLLMLaunchersBase
from .flow import *  # noqa F403
from .components import (LazyLLMDataprocBase, LazyLLMFinetuneBase, LazyLLMDeployBase,
                         LazyLLMValidateBase, register as component_register, Prompter,
                         AlpacaPrompter, ChatPrompter, FastapiApp, JsonFormatter, FileFormatter)

from .module import (ModuleBase, ModuleBase as Module, UrlModule, TrainableModule, ActionModule,
                     ServerModule, TrialModule, register as module_register,
                     OnlineModule, OnlineChatModule, OnlineEmbeddingModule, AutoModel, OnlineMultiModalModule)
from .hook import LazyLLMHook, LazyLLMFuncHook
from .prompt_templates import ActorPrompt, DataPrompt
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .tools import (Document, Reranker, Retriever, WebModule, ToolManager, FunctionCall, SkillManager,
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


_LAZY_SUBMODS = ('tracing', 'tools')


def __getattr__(name: str):
    for submod in _LAZY_SUBMODS:
        if name == submod:
            return importlib.import_module(f'.{submod}', package=__package__)
        mod = importlib.import_module(f'.{submod}', package=__package__)
        if name in getattr(mod, '__all__', ()):
            builtins.globals()[name] = value = getattr(mod, name)
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

    # tracing
    'TracingSetupError',
    'get_trace_context',
    'set_trace_context',

    # tools
    'Document',
    'Retriever',
    'Reranker',
    'ToolManager',
    'SkillManager',
    'FunctionCall',
    'FunctionCallAgent',
    'fc_register',
    'LLMParser',
    'ReactAgent',
    'PlanAndSolveAgent',
    'ReWOOAgent',
    'SentenceSplitter',
    'ActorPrompt',
    'DataPrompt',

    # docs
    'add_doc',
]

__all__ += common.__all__  # noqa F405
__all__ += flow.__all__  # noqa F405
