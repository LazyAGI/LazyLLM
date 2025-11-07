import imp
from ..base import LazyLLMMemoryBase
from lazyllm import config
from lazyllm.thirdparty import mem0
from lazyllm.module import LLMBase
from typing import Optional, List, Dict, Any


config.add('mem0_api_key', str, '', 'MEM0_API_KEY')

class Mem0Memory():
    def __new__(cls, api_key: Optional[str] = None) -> LazyLLMMemoryBase:
        return OnlineMem0Memory(api_key)
    
class OnlineMem0Memory(LazyLLMMemoryBase):
    def __init__(self, api_key: Optional[str] = None, *, topk: int = 10):
        super().__init__(topk=topk)
        self._client = mem0.Memory()

    def _add(self, message: List[Dict[str, Any]], user_id: Optional[str] = None, agent_id: Optional[str] = None):
        user_id = f'{user_id}@@{"default" if agent_id is None else agent_id}'
        self._client.add(message, user_id=user_id)

    def _get(self, query: Optional[str] = None, user_id: Optional[str] = None, agent_id: Optional[str] = None):
        user_id = f'{user_id}@@{"default" if agent_id is None else agent_id}'
        self._client.search(query=query, user_id=user_id, limit=self._topk)

class LocalMem0Memory(LazyLLMMemoryBase):
    def __init__(self, data_path: str, llm: Optional[LLMBase] = None, embed: Optional[LLMBase] = None):
        super().__init__()
        raise NotImplementedError('LocalMem0Memory is not implemented')
