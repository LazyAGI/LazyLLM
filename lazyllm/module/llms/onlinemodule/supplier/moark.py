import requests
import lazyllm
from typing import Tuple, List, Dict, Union
from urllib.parse import urljoin
from ..base import OnlineChatModuleBase, OnlineEmbeddingModuleBase, OnlineMultiModalBase
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.components.utils.file_operate import bytes_to_file
from ..fileHandler import FileHandlerBase


class MoarkChat(OnlineChatModuleBase, FileHandlerBase):
    def __init__(self, base_url: str = 'https://api.moark.com/v1/', model: str = 'Qwen3-32B',
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        OnlineChatModuleBase.__init__(self, model_series='MOARK',
                                      api_key=api_key or lazyllm.config['moark_api_key'],
                                      base_url=base_url, model_name=model, stream=stream,
                                      return_trace=return_trace, **kwargs)
        FileHandlerBase.__init__(self)
        if stream:
            self._model_optional_params['stream'] = True

    def _get_system_prompt(self):
        return 'You are an intelligent assistant provided by Moark. You are a helpful assistant.'

    def _validate_api_key(self):
        '''Validate API Key by sending a minimal request'''
        try:
            # Moark validates API key using a minimal chat request
            models_url = urljoin(self._base_url, 'models')
            response = requests.get(models_url, headers=self._header, timeout=10)
            return response.status_code == 200
        except Exception:
            return False


class MoarkEmbedding(OnlineEmbeddingModuleBase):
    def __init__(self, embed_url: str = 'https://api.moark.com/v1/embeddings',
                 embed_model_name: str = 'bge-large-zh-v1.5', api_key: str = None,
                 batch_size: int = 16, **kw):
        super().__init__('MOARK', embed_url, api_key or lazyllm.config['moark_api_key'],
                         embed_model_name, batch_size=batch_size, **kw)


class MoarkReranking(OnlineEmbeddingModuleBase):
    def __init__(self, embed_url: str = 'https://api.moark.com/v1/rerank',
                 embed_model_name: str = 'bge-reranker-v2-m3', api_key: str = None, **kw):
        super().__init__('MOARK', embed_url, api_key or lazyllm.config['moark_api_key'],
                         embed_model_name, **kw)

    def _encapsulated_data(self, query: str, documents: List[str], top_n: int, **kwargs) -> Dict:
        json_data = {
            'model': self._embed_model_name,
            'query': query,
            'documents': documents,
            'top_n': top_n
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)
        return json_data

    def _parse_response(self, response: Dict, input: Union[List, str]) -> List[Tuple]:
        results = response.get('results', [])
        return [(result['index'], result['relevance_score']) for result in results]
