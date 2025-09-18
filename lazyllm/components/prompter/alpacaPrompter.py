from typing import List, Union, Optional, Dict
from .builtinPrompt import LazyLLMPrompterBase

class AlpacaPrompter(LazyLLMPrompterBase):
    """Alpaca-style Prompter, supports tool calls, does not support historical dialogue.


Args:
    instruction (Option[str]): Task instructions for the large model, with at least one fillable slot (e.g. ``{instruction}``). Or use a dictionary to specify the ``system`` and ``user`` instructions.
    extra_keys (Option[List]): Additional fields that will be filled with user input.
    show (bool): Flag indicating whether to print the generated Prompt, default is False.
    tools (Option[list]): Tool-set which is provived for LLMs, default is None.


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
            splice_struction = instruction.get('system', '') + \
                AlpacaPrompter.ISA + instruction.get('user', '') + AlpacaPrompter.ISE
            instruction = splice_struction
        instruction_template = ('Below is an instruction that describes a task, paired with extra messages such as '
                                'input that provides further context if possible. Write a response that appropriately '
                                f'completes the request.\n\n### Instruction:\n{instruction if instruction else ""}'
                                '\n\n' + LazyLLMPrompterBase._get_extro_key_template(extra_keys))
        self._init_prompt('{system}\n{instruction}\n{tools}\n{user}### Response:\n',
                          instruction_template,
                          '### Response:\n')

    def _check_values(self, instruction, input, history, tools):
        assert not history, f'Chat history is not supported in {__class__}.'
        assert not input, 'All keys should in instruction or extra-keys'
