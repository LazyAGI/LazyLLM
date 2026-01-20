from .supplier import MemUMemory, Mem0Memory, PowerMemMemory
from .base import LazyLLMMemoryBase
from typing import Optional
import lazyllm


class Memory():
    SUPPLIERS = {'memu': MemUMemory,
                 'mem0': Mem0Memory,
                 'powermem': PowerMemMemory}

    def __new__(cls, source: Optional[str] = None, *args, **kwargs) -> LazyLLMMemoryBase:
        if source is None:
            for source in Memory.SUPPLIERS.keys():
                if lazyllm.config[f'{source}_api_key']: break
            else: raise ValueError(f'No memory supplier found, please set LAZYLLM_{source.upper()}_API_KEY env variable')
        return cls.SUPPLIERS[source](*args, **kwargs)


def memory_hook(query, *inputs, **kw):
    m = Memory()
    output = yield
    m.add(query, output, user_id=globals.get('user_id'), agent_id=globals.get('agent_id'))
