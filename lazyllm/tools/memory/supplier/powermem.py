from ..base import LazyLLMMemoryBase
from functools import lru_cache
from lazyllm.thirdparty import powermem
from lazyllm.module import LLMBase
from typing import Optional, List, Dict, Any


class PowerMemMemory():
    def __new__(cls, topk: int = 10) -> LazyLLMMemoryBase:
        return OnlinePowerMemMemory(topk=topk)


class OnlinePowerMemMemory(LazyLLMMemoryBase):
    @staticmethod
    @lru_cache
    def _get_client():
        """
        Use powermem.auto_config() to read local .env configuration
        """
        return powermem.Memory(config=powermem.auto_config())

    def __init__(self, *, topk: int = 10, custom_instruction: Optional[str] = None):
        super().__init__(topk=topk)
        # Directly get the client instance, no need to pass the config dictionary
        self._client = self._get_client()
        self._custom_instruction = custom_instruction

    def _add(self, message: List[Dict[str, Any]], user_id: Optional[str] = None, agent_id: Optional[str] = None):
        user_id = f'{user_id}___{"default" if agent_id is None else agent_id}'

        # Convert message list to string
        if isinstance(message, list):
            text_content = "\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in message])
        else:
            text_content = str(message)

        return self._client.add(text_content, user_id=user_id)

    def _get(self, query: Optional[str] = None, user_id: Optional[str] = None, agent_id: Optional[str] = None):
        user_id = f'{user_id}___{"default" if agent_id is None else agent_id}'

        if query:
            # When there is a query, use semantic/hybrid search
            response = self._client.search(query=query, user_id=user_id)
        else:
            # When there is no query, call get_all to retrieve all memories
            response = self._client.get_all(user_id=user_id)

        if isinstance(response, dict):
            memories = response.get('results', [])
        elif isinstance(response, list):
            memories = response
        else:
            memories = []

        # Extract memory content and concatenate
        return '\n'.join([str(m.get('memory', '')) for m in memories])


class LocalPowerMemMemory(LazyLLMMemoryBase):
    def __init__(self, data_path: str, llm: Optional[LLMBase] = None, embed: Optional[LLMBase] = None):
        super().__init__()
        raise NotImplementedError('LocalPowerMemMemory is not implemented')