from typing import Any, Optional, Union
import lazyllm
from lazyllm import LOG
from .online_module import OnlineModule
from .trainablemodule import TrainableModule
from .utils import get_candidate_entries, process_trainable_args, process_online_args


class AutoModel:

    def __new__(cls, model: Optional[str] = None, *, config_id: Optional[str] = None, source: Optional[str] = None,  # noqa C901
                type: Optional[str] = None, config: Union[str, bool] = True, **kwargs: Any):
        # check and accomodate user params
        model = model or kwargs.pop('base_model', kwargs.pop('embed_model_name', None))
        chat_models = {k for k in lazyllm.online.chat.keys() if k != 'base'}
        if model in chat_models:
            source, model = model, None

        if not model:
            try:
                return lazyllm.OnlineModule(source=source, type=type)
            except Exception as e:
                raise RuntimeError(f'`model` is not provided in AutoModel, and {e}') from None

        trainable_entry, online_entry = get_candidate_entries(model, config_id, source, config)

        # 1) first: try TrainableModule with trainable config (for directly connecting deployed endpoint)
        if trainable_entry is not None:
            trainable_args = process_trainable_args(
                model=model, type=type, source=source, config=config, entry=trainable_entry
            )
            try:
                module = TrainableModule(**trainable_args)
                if module._url or module._impl._get_deploy_tasks.flag: return module
            except Exception as e:
                LOG.warning('Fail to create `TrainableModule`, will try to '
                            f'load model {model} with `OnlineModule`. Since the error: {e}')

        # 2) second: try OnlineModule with online config if found
        if online_entry is not None:
            online_args = process_online_args(model=model, source=source, type=type, entry=online_entry)
            if online_args: return OnlineModule(**online_args)

        # 3) finally: fallback (no config or config unusable)
        try:
            return OnlineModule(model=model, source=source, type=type)
        except Exception as e:
            LOG.warning('`OnlineModule` creation failed, and will try to '
                        f'load model {model} with local `TrainableModule`. Since the error: {e}')
            return TrainableModule(model, type=type)
