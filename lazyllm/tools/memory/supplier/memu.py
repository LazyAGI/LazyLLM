from ..base import LazyLLMMemoryBase
from functools import lru_cache
from lazyllm import config
from lazyllm.thirdparty import memu
from lazyllm.module import LLMBase
from typing import Optional, List, Dict, Any


config.add('memu_api_key', str, '', 'MEMU_API_KEY', description='The API key for MemU.')

class MemUMemory():
    def __new__(cls, api_key: Optional[str] = None, topk: int = 10) -> LazyLLMMemoryBase:
        return OnlineMemUMemory(api_key, topk=topk)

class OnlineMemUMemory(LazyLLMMemoryBase):
    @staticmethod
    @lru_cache
    def _get_client(api_key: Optional[str] = None):
        return memu.MemuClient(base_url='https://api.memu.so', api_key=api_key)

    def __init__(self, api_key: Optional[str] = None, *, topk: int = 10):
        super().__init__(topk=topk)
        self._client = self._get_client(api_key or config['memu_api_key'])

    def _add(self, message: List[Dict[str, Any]], user_id: Optional[str] = None, agent_id: Optional[str] = None):
        agent_id = agent_id or 'default'
        self._client.memorize_conversation(
            conversation=message, user_id=user_id, user_name=user_id, agent_id=agent_id, agent_name=agent_id)

    def _get(self, query: Optional[str] = None, user_id: Optional[str] = None, agent_id: Optional[str] = None):
        if query:
            retrieved_memories = self._client.retrieve_related_memory_items(
                query=query, user_id=user_id, agent_id=agent_id, top_k=self._topk)
            if not retrieved_memories or not (mem := getattr(retrieved_memories, 'related_memories', None)): return ''
            return '\n'.join([str(me) for m in mem if (me := getattr(getattr(m, 'memory', {}), 'content', None))])
        else:
            retrieved_memories = self._client.retrieve_default_categories(user_id=user_id, agent_id=agent_id)
            if not retrieved_memories or not (categories := getattr(retrieved_memories, 'categories', None)): return ''
            return '\n'.join([str(summary) for category in categories if (summary := getattr(category, 'summary', ''))])

class LocalMemUMemory(LazyLLMMemoryBase):
    def __init__(self, data_path: str, llm: Optional[LLMBase] = None, embed: Optional[LLMBase] = None):
        super().__init__()
        raise NotImplementedError('LocalMemUMemory is not implemented')
