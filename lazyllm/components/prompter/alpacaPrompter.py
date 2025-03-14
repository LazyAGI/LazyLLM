from typing import List, Union, Optional, Dict
from .builtinPrompt import LazyLLMPrompterBase

class AlpacaPrompter(LazyLLMPrompterBase):
    def __init__(self, instruction: Union[None, str, Dict[str, str]] = None, extra_keys: Union[None, List[str]] = None,
                 show: bool = False, tools: Optional[List] = None):
        super(__class__, self).__init__(show, tools=tools)
        if isinstance(instruction, dict):
            splice_struction = instruction.get("system", "") + \
                AlpacaPrompter.ISA + instruction.get("user", "") + AlpacaPrompter.ISE
            instruction = splice_struction
        instruction_template = ("Below is an instruction that describes a task, paired with extra messages such as "
                                "input that provides further context if possible. Write a response that appropriately "
                                f"completes the request.\n\n### Instruction:\n{instruction if instruction else ''}"
                                "\n\n" + LazyLLMPrompterBase._get_extro_key_template(extra_keys))
        self._init_prompt("{system}\n{instruction}\n{tools}\n{user}### Response:\n",
                          instruction_template,
                          "### Response:")

    def _check_values(self, instruction, input, history, tools):
        assert not history, f"Chat history is not supported in {__class__}."
        assert not input, "All keys should in instruction or extra-keys"
