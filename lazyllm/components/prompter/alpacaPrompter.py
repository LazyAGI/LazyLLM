from typing import List, Union
from .builtinPrompt import LazyLLMPrompterBase

class AlpacaPrompter(LazyLLMPrompterBase):
    def __init__(self, instruction: Union[None, str] = None,
                 extro_keys: Union[None, List[str]] = None, show: bool = False):
        super(__class__, self).__init__(show)
        instruction_template = ("Below is an instruction that describes a task, paired with extra messages such as "
                                "input that provides further context if possible. Write a response that "
                                f"appropriately completes the request.\n\n ### Instruction:\n{instruction}"
                                "\n\n" + LazyLLMPrompterBase._get_extro_key_template(extro_keys))
        self._init_prompt("{system}\n{instruction}\n{tools}### Response:\n", instruction_template, "### Response:")

    def _check_values(self, instruction, input, history, tools):
        assert not history, f"Chat history is not supported in {__class__}."
        assert not input, "All keys should in instruction or extro-keys"
