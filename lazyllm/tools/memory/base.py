from abc import ABC, abstractmethod
from lazyllm import globals
from lazyllm.common import LazyLLMRegisterMetaABCClass
from lazyllm import ChatPrompter
from typing import Union, List, Dict, Any, Optional

class LazyLLMMemoryBase(ABC, metaclass=LazyLLMRegisterMetaABCClass):
    def __init__(self, topk: int = 10):
        self._topk = topk

    def add(self, query: str, output: Optional[str] = None,
            history: Optional[Union[List[List[str]], List[Dict[str, Any]]]] = None,
            user_id: Optional[str] = None, agent_id: Optional[str] = None):
        r = ChatPrompter(history=history, enable_system=False).generate_prompt(query, return_dict=True)['messages']
        if output: r.append(output if isinstance(output, dict) else dict(role='assistant', content=output))
        self._add(r, user_id, agent_id)

    def get(self, query: Optional[str] = None, user_id: Optional[str] = None, agent_id: Optional[str] = None):
        return self._get(query, user_id, agent_id)

    @abstractmethod
    def _add(self, message: List[Dict[str, Any]], user_id: Optional[str] = None, agent_id: Optional[str] = None): pass

    @abstractmethod
    def _get(self, query: Optional[str] = None, user_id: Optional[str] = None, agent_id: Optional[str] = None): pass

    def __call__(self, query: Optional[str] = None):
        return self.get(query, globals.get('user_id'), globals.get('agent_id'))
