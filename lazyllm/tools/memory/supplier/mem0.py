from ..base import LazyLLMMemoryBase
from functools import lru_cache
from lazyllm import config
from lazyllm.thirdparty import mem0
from lazyllm.module import LLMBase
from typing import Optional, List, Dict, Any


config.add('mem0_api_key', str, '', 'MEM0_API_KEY', description='The API key for Mem0.')

class Mem0Memory():
    def __new__(cls, api_key: Optional[str] = None, topk: int = 10) -> LazyLLMMemoryBase:
        return OnlineMem0Memory(api_key, topk=topk)

class OnlineMem0Memory(LazyLLMMemoryBase):
    @staticmethod
    @lru_cache
    def _get_client(api_key: Optional[str] = None):
        return mem0.MemoryClient(api_key=api_key)

    def __init__(self, api_key: Optional[str] = None, *, topk: int = 10, custom_instruction: Optional[str] = None):
        super().__init__(topk=topk)
        self._client = mem0.MemoryClient(api_key=api_key or config['mem0_api_key'])

    def _add(self, message: List[Dict[str, Any]], user_id: Optional[str] = None, agent_id: Optional[str] = None):
        user_id = f'{user_id}___{"default" if agent_id is None else agent_id}'
        return self._client.add(message, user_id=user_id, version='v2', output_format='v1.1',
                                custom_instructions=self._custom_instruction)

    def _get(self, query: Optional[str] = None, user_id: Optional[str] = None, agent_id: Optional[str] = None):
        user_id = f'{user_id}___{"default" if agent_id is None else agent_id}'
        filters = {'OR': [{'user_id': user_id}]}
        if query:
            memories = self._client.search(query=query, filters=filters, version='v2', output_format='v1.1')
        else:
            memories = self._client.get_all(filters=filters)
        return '\n'.join([str(m['memory']) for m in memories['results']])

class LocalMem0Memory(LazyLLMMemoryBase):
    def __init__(self, data_path: str, llm: Optional[LLMBase] = None, embed: Optional[LLMBase] = None):
        super().__init__()
        raise NotImplementedError('LocalMem0Memory is not implemented')
