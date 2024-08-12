from typing import List, Union, Optional, Dict
from .builtinPrompt import LazyLLMPrompterBase

class ChatPrompter(LazyLLMPrompterBase):
    def __init__(self, instruction: Union[None, str, Dict[str, str]] = None,
                 extro_keys: Union[None, List[str]] = None, show: bool = False, tools: Optional[List] = None):
        super(__class__, self).__init__(show, tools=tools)
        if isinstance(instruction, dict):
            splice_instruction = instruction.get("system", "") + \
                ChatPrompter.ISA + instruction.get("user", "") + ChatPrompter.ISE
            instruction = splice_instruction
        instruction_template = f'{instruction}\n{{extro_keys}}\n'.replace(
            '{extro_keys}', LazyLLMPrompterBase._get_extro_key_template(extro_keys)) if instruction else ""
        self._init_prompt("{sos}{system}{instruction}{tools}{eos}\n\n{history}\n{soh}\n{user}{input}\n{eoh}{soa}\n",
                          instruction_template)

    @property
    def _split(self): return self._soa if self._soa else None
