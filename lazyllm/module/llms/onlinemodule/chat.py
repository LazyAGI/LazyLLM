import lazyllm
from lazyllm.components.utils.downloader.model_downloader import LLMType
from lazyllm import globals
from typing import Any, Dict, Optional
from lazyllm.common.bind import _MetaBind

from ...servermodule import LLMBase, StaticParams
from lazyllm.module import ModuleBase
from .map_model_type import get_model_type
from .base import OnlineChatModuleBase
from .base.utils import select_source_with_default_key
import threading


class _ChatModuleMeta(_MetaBind):

    def __instancecheck__(self, __instance: Any) -> bool:
        if isinstance(__instance, OnlineChatModuleBase):
            return True
        return super().__instancecheck__(__instance)


globals.config.add('dynamic_llm_source', str, None, 'DYNAMIC_LLM_SOURCE',
                   description='The LLM source to use defined in session scope.')

class OnlineChatModule(ModuleBase, LLMBase, metaclass=_ChatModuleMeta):

    @staticmethod
    def _encapsulate_parameters(base_url: str, model: str, stream: bool, return_trace: bool, **kwargs) -> Dict[str, Any]:
        params = {'stream': stream, 'return_trace': return_trace}
        if base_url is not None:
            params['base_url'] = base_url
        if model is not None:
            params['model'] = model
        params.update(kwargs)
        return params

    def __new__(cls, model: str = None, source: str = None, base_url: str = None, stream: bool = True,
                return_trace: bool = False, skip_auth: bool = False, type: Optional[str] = None,
                api_key: str = None, static_params: Optional[StaticParams] = None, id: Optional[str] = None,
                name: Optional[str] = None, group_id: Optional[str] = None, dynamic_auth: bool = False, **kwargs):
        if model in lazyllm.online.chat and source is None: source, model = model, source
        if dynamic_auth:
            assert source == 'dynamic', 'source should be dynamic for dynamic auth.'
            assert not skip_auth, 'skip_auth should be False for dynamic LLM source.'
        if source == 'dynamic' or (lazyllm.config['dynamic_llm_source'] and not source):
            return super().__new__(cls)

        if source is None and api_key is not None:
            raise ValueError('No source is given but an api_key is provided.')
        source, default_key = select_source_with_default_key(lazyllm.online.chat, source, LLMType.CHAT)
        api_key = api_key or default_key

        if type is None and model:
            type = get_model_type(model)
        if type in ['embed', 'rerank', 'cross_modal_embed']:
            raise AssertionError(f'\'{model}\' should use OnlineEmbeddingModule')
        elif type in ['stt', 'tts', 'sd']:
            raise AssertionError(f'\'{model}\' should use OnlineMultiModalModule')
        params = OnlineChatModule._encapsulate_parameters(base_url, model, stream, return_trace, api_key=api_key,
                                                          skip_auth=skip_auth, type=type.upper() if type else None,
                                                          **kwargs)
        if skip_auth:
            source = source or 'openai'
            if not base_url:
                raise KeyError('base_url must be set for local serving.')

        return getattr(lazyllm.online.chat, source)(**params)

    def __init__(self, model: str = None, source: str = None, base_url: str = None, stream: bool = True,
                 return_trace: bool = False, skip_auth: bool = False, type: Optional[str] = None,
                 api_key: str = None, static_params: Optional[StaticParams] = None, id: Optional[str] = None,
                 name: Optional[str] = None, group_id: Optional[str] = None, dynamic_auth: bool = False, **kwargs):
        assert model is None, 'model should be given in forward method or global config.'
        assert base_url is None, 'base_url should be given in forward method or global config.'
        assert api_key is None or api_key in ('auto', 'dynamic'), 'api_key should be given in forward or globals.config.'

        ModuleBase.__init__(self, id=id, name=name, group_id=group_id, return_trace=return_trace)
        LLMBase.__init__(self, stream=stream, type=type, static_params=static_params)
        self._kwargs = kwargs
        self._skip_auth = skip_auth
        self._api_key = 'dynamic' if (dynamic_auth or api_key == 'auto') else api_key
        self._type = type  # overwrite type to avoid convert None to 'llm'
        self._suppliers: Dict[str, LLMBase] = {}
        self._lock = threading.Lock()

    def _get_supplier(self):
        if (source := globals.config['dynamic_llm_source']) is None:
            raise KeyError('No source is configured for dynamic LLM source.')
        if source not in self._suppliers:
            with self._lock:
                if source not in self._suppliers:
                    self._suppliers[source] = getattr(lazyllm.online.chat, source)(
                        stream=self._stream, type=self._type, static_params=self._static_params,
                        skip_auth=self._skip_auth, api_key=self._api_key,
                        return_trace=self._return_trace, **self._kwargs)
        return self._suppliers[source]

    def forward(self, *args, **kwargs):
        return self._get_supplier().forward(*args, **kwargs)
