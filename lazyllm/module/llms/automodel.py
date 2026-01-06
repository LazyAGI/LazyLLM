import os
from typing import Any, Dict, Optional, Union

import lazyllm

from .online_module import OnlineModule
from .trainablemodule import TrainableModule
from .utils import get_target_entry, resolve_model_name, process_trainable_args, process_online_args

lazyllm.config.add('auto_model_config_map_path', str, '', 'AUTO_MODEL_CONFIG_MAP_PATH',
                   description='The default path for automodel config map.')

_DEFAULT_LOCAL_MODEL = 'internlm2-chat-7b'


class AutoModel:

    @staticmethod
    def _build_online_module(config: Dict[str, Any]):
        return OnlineModule(**config)

    @staticmethod
    def _build_trainable_module(use_config: Optional[bool],
                                config: Dict[str, Any],
                                framework: Optional[Union[str, Any]],
                                url: Optional[str]):
        # set LAZYLLM_TRAINABLE_MODULE_CONFIG_MAP_PATH for trainable module to process
        if use_config:
            os.environ['LAZYLLM_TRAINABLE_MODULE_CONFIG_MAP_PATH'] = lazyllm.config['auto_model_config_map_path']
        if framework or url:
            return TrainableModule(**config).deploy_method(url=url, framework=framework)
        return TrainableModule(**config)

    def __new__(cls,
                model: str = '',
                *,
                config_id: Optional[str] = None,
                source: Optional[str] = None,
                type: Optional[str] = None,
                url: Optional[str] = None,
                framework: Optional[Union[str, Any]] = None,
                use_config: bool = True,
                **kwargs: Any):
        module_kwargs = dict(kwargs)

        # check and accomodate user params
        model = model or module_kwargs.pop('base_model', module_kwargs.pop('embed_model_name', None))
        url = url or module_kwargs.pop('base_url', module_kwargs.pop('embed_url', None))
        if not model:
            raise ValueError('`model` is required for AutoModel.')

        deploy_config = module_kwargs.pop('deploy_config', None)
        if deploy_config and not isinstance(deploy_config, dict):
            raise TypeError(f'`deploy_config` should be a dict, got {type(deploy_config).__name__}')

        target_mode, entry = get_target_entry(model, config_id, source, type, 
                                              framework, use_config, deploy_config)

        # build instance of online/trainable module and return
        resolved_model = resolve_model_name(model, entry)
        if target_mode == 'trainable':
            trainable_args = process_trainable_args(model=resolved_model, type=type, use_config=use_config, **module_kwargs)
            return cls._build_trainable_module(use_config=use_config, config=trainable_args,
                                               framework=framework, url=url)
        elif target_mode == 'online':
            online_args = process_online_args(model=resolved_model, source=source, type=type,
                                              url=url, entry=entry, **kwargs)
            return cls._build_online_module(config=online_args)
        else:
            raise Exception(f"Wrong target mode resolved: {target_mode}, only `online` and `trainable` are supported")
