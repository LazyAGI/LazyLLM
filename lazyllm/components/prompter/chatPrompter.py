from typing import List, Union
from .builtinPrompt import LazyLLMPrompterBase

class ChatPrompter(LazyLLMPrompterBase):
    def __init__(self, instruction: Union[None, str] = None,
                 extro_keys: Union[None, List[str]] = None, show: bool = False):
        super(__class__, self).__init__(show)
        instruction_template = f'{instruction}\n{{extro_keys}}\n'.replace(
            '{extro_keys}', LazyLLMPrompterBase._get_extro_key_template(extro_keys))
        self._init_prompt("{sos}{system}{instruction}{tools}{eos}\n\n{history}\n{soh}\n{input}\n{eoh}{soa}\n",
                          instruction_template)

    @property
    def _split(self): return self._soa
