import os
from typing import Any, Dict, Optional, Union

import lazyllm

from .online_module import OnlineModule
from .trainablemodule import TrainableModule
from .utils import load_config_entries, select_config_entry, decide_target_mode

lazyllm.config.add('auto_model_config_map_path', str, '', 'AUTO_MODEL_CONFIG_MAP_PATH',
                   description='The default path for automodel config map.')

_DEFAULT_LOCAL_MODEL = 'internlm2-chat-7b'


class AutoModel:

    @staticmethod
    def _build_online_module(entry: Optional[Dict[str, Any]],
                             module_kwargs: Dict[str, Any],
                             model: Optional[str],
                             source: Optional[str],
                             type: Optional[str],
                             url: Optional[str]):
        entry_overrides = dict(entry or {})
        entry_overrides.pop('deploy_config', None)
        entry_overrides.pop('framework', None)
        online_args = dict(module_kwargs)
        for key, value in entry_overrides.items():
            if value is not None:
                online_args[key] = value
        if model:
            online_args['model'] = model
        if source:
            online_args['source'] = source
        resolved_type = type or online_args.pop('task', None) or online_args.get('type')
        if resolved_type:
            online_args['type'] = resolved_type
        if url:
            online_args['url'] = url
        return OnlineModule(**online_args)

    @staticmethod
    def _build_trainable_module(module_kwargs: Dict[str, Any],
                                model: Optional[str],
                                type: Optional[str],
                                use_config: Optional[bool],
                                framework: Optional[Union[str, Any]],
                                url: Optional[str]):
        config = dict(module_kwargs)
        config['base_model'] = model or config.get('base_model') or _DEFAULT_LOCAL_MODEL
        if framework is not None and 'framework' not in config:
            config['framework'] = framework
        if url is not None and 'url' not in config:
            config['url'] = url
        resolved_type = type or config.pop('task', None) or config.get('type')
        if resolved_type:
            config['type'] = resolved_type
        if use_config:
            os.environ['LAZYLLM_TRAINABLE_MODULE_CONFIG_MAP_PATH'] = lazyllm.config['auto_model_config_map_path']
        return TrainableModule(use_model_map=use_config, **config)

    def __new__(cls,
                model: str = '',
                *,
                source: Optional[str] = None,
                type: Optional[str] = None,
                url: Optional[str] = None,
                framework: Optional[Union[str, Any]] = None,
                port: Optional[Union[int, str]] = None,
                use_config: bool = True,
                **kwargs: Any):
        module_kwargs = dict(kwargs)

        # check and accomodate user params
        model = model or module_kwargs.pop('base_model', module_kwargs.pop('embed_model_name', None))
        url = url or module_kwargs.pop('base_url', module_kwargs.pop('embed_url', None))
        type = type or module_kwargs.pop('task', None)
        if not model:
            raise ValueError('`model` is required for AutoModel.')

        deploy_config = module_kwargs.pop('deploy_config', None)
        if deploy_config and not isinstance(deploy_config, dict):
            raise TypeError(f'`deploy_config` should be a dict, got {type(deploy_config).__name__}')

        # process config entries and distinguish target mode
        entries = load_config_entries(model, use_config)
        online_entry = select_config_entry(entries, 'online', source)
        trainable_entry = select_config_entry(entries, 'trainable', source)

        target_mode = decide_target_mode(source, type, url, framework, port, deploy_config or {},
                                         online_entry is not None, trainable_entry is not None)

        # build instance of online/trainable module and return
        if target_mode == 'trainable':
            module_kwargs.pop('source', None)
            if url:
                module_kwargs['url'] = url
            if framework:
                module_kwargs['framework'] = framework
            if port:
                module_kwargs['port'] = port
            return cls._build_trainable_module(module_kwargs, model, type, use_config, framework, url)
        return cls._build_online_module(online_entry, module_kwargs, model, source, type, url)
