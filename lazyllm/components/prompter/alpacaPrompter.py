from typing import List, Union, Optional, Dict
from .builtinPrompt import LazyLLMPrompterBase

class AlpacaPrompter(LazyLLMPrompterBase):
    """Alpaca格式的Prompter，支持工具调用，不支持历史对话。

Args:
    instruction (Option[str]): 大模型的任务指令，至少带一个可填充的槽位(如 ``{instruction}``)。或者使用字典指定 ``system`` 和 ``user`` 的指令。
    extra_keys (Option[List]): 额外的字段，用户的输入会填充这些字段。
    show (bool): 标志是否打印生成的Prompt，默认为False
    tools (Option[list]): 大模型可以使用的工具集合，默认为None


Examples:
    >>> from lazyllm import AlpacaPrompter
    >>> p = AlpacaPrompter('hello world {instruction}')
    >>> p.generate_prompt('this is my input')
    'You are an AI-Agent developed by LazyLLM.\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\n\\n ### Instruction:\\nhello world this is my input\\n\\n\\n### Response:\\n'
    >>> p.generate_prompt('this is my input', return_dict=True)
    {'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\n\\n ### Instruction:\\nhello world this is my input\\n\\n'}, {'role': 'user', 'content': ''}]}
    >>>
    >>> p = AlpacaPrompter('hello world {instruction}, {input}', extra_keys=['knowledge'])
    >>> p.generate_prompt(dict(instruction='hello world', input='my input', knowledge='lazyllm'))
    'You are an AI-Agent developed by LazyLLM.\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\n\\n ### Instruction:\\nhello world hello world, my input\\n\\nHere are some extra messages you can referred to:\\n\\n### knowledge:\\nlazyllm\\n\\n\\n### Response:\\n'
    >>> p.generate_prompt(dict(instruction='hello world', input='my input', knowledge='lazyllm'), return_dict=True)
    {'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\n\\n ### Instruction:\\nhello world hello world, my input\\n\\nHere are some extra messages you can referred to:\\n\\n### knowledge:\\nlazyllm\\n\\n'}, {'role': 'user', 'content': ''}]}
    >>>
    >>> p = AlpacaPrompter(dict(system="hello world", user="this is user instruction {input}"))
    >>> p.generate_prompt(dict(input="my input"))
    'You are an AI-Agent developed by LazyLLM.\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\n\\n ### Instruction:\\nhello word\\n\\n\\n\\nthis is user instruction my input### Response:\\n'
    >>> p.generate_prompt(dict(input="my input"), return_dict=True)
    {'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\n\\n ### Instruction:\\nhello world'}, {'role': 'user', 'content': 'this is user instruction my input'}]}

    """
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
                          "### Response:\n")

    def _check_values(self, instruction, input, history, tools):
        assert not history, f"Chat history is not supported in {__class__}."
        assert not input, "All keys should in instruction or extra-keys"
