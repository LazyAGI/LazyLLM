from typing import Any, Optional, Union
import lazyllm
from lazyllm import LOG
from .online_module import OnlineModule
from .trainablemodule import TrainableModule
from .utils import get_target_entry, resolve_model_name, process_trainable_args, process_online_args

lazyllm.config.add('auto_model_config_map_path', str, '', 'AUTO_MODEL_CONFIG_MAP_PATH',
                   description='The default path for automodel config map.')


class AutoModel:

    def __new__(cls,
                model: str = '',
                *,
                config_id: Optional[str] = None,
                source: Optional[str] = None,
                type: Optional[str] = None,
                config: Union[str, bool] = True,
                **kwargs: Any):

        # check and accomodate user params
        model = model or kwargs.pop('base_model', kwargs.pop('embed_model_name', None))
        if not model:
            raise ValueError('`model` is required for AutoModel.')
        if kwargs:
            LOG.warning(
                'AutoModel ignores extra kwargs: %s; only kwargs `base_model`/`embed_model_name` are supported.',
                list(kwargs.keys()),
            )

        target_mode, entry = get_target_entry(model, config_id, source, config)

        if entry is not None:
            resolved_model = resolve_model_name(model, entry)
            if target_mode == 'trainable':
                trainable_args = process_trainable_args(model=resolved_model, type=type, source=source,
                                                        use_config=config, entry=entry)
                try:
                    module = TrainableModule(**trainable_args)
                    if module._url or module._impl._get_deploy_tasks.flag:
                        return module
                except Exception as e:
                    LOG.error('Fail to create `TrainableModule`, will try to '
                              f'load model {model} with `OnlineModule`. Since the error: {e}')

            online_args = process_online_args(model=resolved_model, source=source, type=type, entry=entry)
            return OnlineModule(**online_args)
        else:
            try:
                return OnlineModule(model=model, source=source, type=type)
            except KeyError as e:
                LOG.warning('`OnlineModule` creation failed, and will try to '
                            f'load model {model} with local `TrainableModule`. Since the error: {e}')
                return TrainableModule(model, type=type)
