import os
from lazyllm.tools.memory.base import LazyLLMMemoryBase
from functools import lru_cache
from lazyllm.thirdparty import powermem
from lazyllm import config
from typing import Optional, List, Dict, Any

config.add('powermem_config_path', str, '', 'POWERMEM_CONFIG_PATH',
           description='Path to powermem config file (.env or JSON). If not set, will use auto_config().')

class PowerMemMemory(object):
    def __new__(cls, topk: int = 10) -> LazyLLMMemoryBase:
        return LocalPowerMemMemory(topk=topk)


class LocalPowerMemMemory(LazyLLMMemoryBase):
    @staticmethod
    @lru_cache
    def _get_client():
        config_path = config['powermem_config_path']
        # load .env from current_dir
        if not config_path:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            default_env = os.path.join(current_dir, '.env')
            if os.path.exists(default_env):
                config_path = default_env
        if config_path:
            if os.path.exists(config_path):
                try:
                    from dotenv import load_dotenv
                    load_dotenv(config_path, override=False)
                    powermemconfig = powermem.config_loader.load_config_from_env()
                    return powermem.Memory(config=powermemconfig)
                except ImportError:
                    pass
                except Exception as e:
                    pass
            # load config_path fail, use auto_config get .env
        return powermem.Memory(config=powermem.auto_config())

    def __init__(self, *, topk: int = 10, custom_instruction: Optional[str] = None):
        super().__init__(topk=topk)
        self._client = self._get_client()
        self._custom_instruction = custom_instruction

    def _add(self, message: List[Dict[str, Any]], user_id: Optional[str] = None, agent_id: Optional[str] = None):
        try:
            user_id = f'{user_id}___{"default" if agent_id is None else agent_id}'
            if isinstance(message, list):
                # Format chat history list into a single string for PowerMem storage
                text_content = '\n'.join([f'{msg.get("role", "unknown")}: {msg.get("content", "")}' for msg in message])
            else:
                text_content = str(message)
            return self._client.add(text_content, user_id=user_id)
        except Exception as e:
            return None

    def _get(self, query: Optional[str] = None, user_id: Optional[str] = None, agent_id: Optional[str] = None):
        try:
            user_id = f'{user_id}___{"default" if agent_id is None else agent_id}'

            if query:
                response = self._client.search(query=query, user_id=user_id)
            else:
                # Fallback to get_all when no specific query is provided to retrieve full context
                response = self._client.get_all(user_id=user_id)

            if isinstance(response, dict):
                memories = response.get('results', [])
            elif isinstance(response, list):
                memories = response
            else:
                memories = []

            return '\n'.join([str(m.get('memory', '')) for m in memories])
        except Exception as e:
            return ""

