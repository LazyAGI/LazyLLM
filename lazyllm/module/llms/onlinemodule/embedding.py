from typing import Any, Dict

import lazyllm
from lazyllm.components.utils.downloader.model_downloader import LLMType
from .base import OnlineEmbeddingModuleBase
from .base.utils import select_source_with_default_key
from .supplier.doubao import DoubaoEmbed, DoubaoMultimodalEmbed


class __EmbedModuleMeta(type):

    def __instancecheck__(self, __instance: Any) -> bool:
        if isinstance(__instance, OnlineEmbeddingModuleBase):
            return True
        return super().__instancecheck__(__instance)


class OnlineEmbeddingModule(metaclass=__EmbedModuleMeta):

    @staticmethod
    def _encapsulate_parameters(embed_url: str,
                                embed_model_name: str,
                                **kwargs) -> Dict[str, Any]:
        params = {}
        if embed_url is not None:
            params['embed_url'] = embed_url
        if embed_model_name is not None:
            params['embed_model_name'] = embed_model_name
        params.update(kwargs)
        return params

    def __new__(self,
                source: str = None,
                embed_url: str = None,
                embed_model_name: str = None,
                **kwargs):
        params = OnlineEmbeddingModule._encapsulate_parameters(embed_url, embed_model_name, **kwargs)

        if source is None and 'api_key' in kwargs and kwargs['api_key']:
            raise ValueError('No source is given but an api_key is provided.')

        if 'type' in params:
            params.pop('type')
        if kwargs.get('type', 'embed') == 'embed':
            source, default_key = select_source_with_default_key(lazyllm.online.embed,
                                                                 explicit_source=source,
                                                                 type=LLMType.EMBED)
            if default_key and not kwargs.get('api_key'):
                kwargs['api_key'] = default_key
            if source == 'doubao':
                if embed_model_name and embed_model_name.startswith('doubao-embedding-vision'):
                    return DoubaoMultimodalEmbed(**params)
                else:
                    return DoubaoEmbed(**params)
            return getattr(lazyllm.online.embed, source)(**params)
        elif kwargs.get('type') == 'rerank':
            source, default_key = select_source_with_default_key(lazyllm.online.rerank,
                                                                 explicit_source=source,
                                                                 type=LLMType.RERANK)
            if default_key and not kwargs.get('api_key'):
                kwargs['api_key'] = default_key
            return getattr(lazyllm.online.rerank, source)(**params)
        else:
            raise ValueError('Unknown type of online embedding module.')
