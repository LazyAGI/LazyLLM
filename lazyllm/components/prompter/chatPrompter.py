from typing import List, Union, Optional, Dict
from .builtinPrompt import LazyLLMPrompterBase

class ChatPrompter(LazyLLMPrompterBase):
    def __init__(self, instruction: Union[None, str, Dict[str, str]] = None, extra_keys: Union[None, List[str]] = None,
                 show: bool = False, tools: Optional[List] = None, skills: Optional[List] = None,
                 history: Optional[List[List[str]]] = None, *, enable_system: bool = True):
        super(__class__, self).__init__(show, tools=tools, skills=skills, history=history, enable_system=enable_system)
        extra_keys_template = LazyLLMPrompterBase._get_extro_key_template(extra_keys)
        if isinstance(instruction, dict):
            splice_instruction = instruction.get('system', '') + \
                ChatPrompter.ISA + instruction.get('user', '') + extra_keys_template + ChatPrompter.ISE
            instruction = splice_instruction
            instruction_template = f'{instruction}\n' if instruction else ''
        else:
            instruction_template = f'{instruction}\n{extra_keys_template}\n' if instruction else ''
        self._init_prompt(
            '{sos}{system}{instruction}{skills}{tools}{eos}\n\n{history}\n{soh}\n{user}{input}\n{eoh}{soa}\n',
            instruction_template)

    @property
    def _split(self): return f'{self._soa}\n' if self._soa else None
