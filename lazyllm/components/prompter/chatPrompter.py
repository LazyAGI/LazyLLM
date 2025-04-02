from typing import List, Union, Optional, Dict
from .builtinPrompt import LazyLLMPrompterBase

class ChatPrompter(LazyLLMPrompterBase):
    def __init__(self, instruction: Union[None, str, Dict[str, str]] = None, extra_keys: Union[None, List[str]] = None,
                 show: bool = False, tools: Optional[List] = None, history: Optional[List[List[str]]] = None):
        super(__class__, self).__init__(show, tools=tools, history=history)
        if isinstance(instruction, dict):
            splice_instruction = instruction.get("system", "") + \
                ChatPrompter.ISA + instruction.get("user", "") + ChatPrompter.ISE
            instruction = splice_instruction
        instruction_template = f'{instruction}\n{{extra_keys}}\n'.replace(
            '{extra_keys}', LazyLLMPrompterBase._get_extro_key_template(extra_keys)) if instruction else ""
        self._init_prompt("{sos}{system}{instruction}{tools}{eos}\n\n{history}\n{soh}\n{user}{input}\n{eoh}{soa}\n",
                          instruction_template)

    @property
    def _split(self): return self._soa if self._soa else None
