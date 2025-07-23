from typing import List, Union, Optional, Dict
from .builtinPrompt import LazyLLMPrompterBase

class ChatPrompter(LazyLLMPrompterBase):
    """chat prompt, supports tool calls and historical dialogue.

Args:
    instruction (Option[str]): Task instructions for the large model, with 0 to multiple fillable slot, represented by ``{}``. For user instructions, you can pass a dictionary with fields ``user`` and ``system``.
    extra_keys (Option[List]): Additional fields that will be filled with user input.
    show (bool): Flag indicating whether to print the generated Prompt, default is False.


Examples:
    >>> from lazyllm import ChatPrompter
    >>> p = ChatPrompter('hello world')
    >>> p.generate_prompt('this is my input')
    'You are an AI-Agent developed by LazyLLM.hello world\\n\\n\\n\\n\\n\\nthis is my input\\n\\n'
    >>> p.generate_prompt('this is my input', return_dict=True)
    {'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\nhello world\\n\\n'}, {'role': 'user', 'content': 'this is my input'}]}
    >>>
    >>> p = ChatPrompter('hello world {instruction}', extra_keys=['knowledge'])
    >>> p.generate_prompt(dict(instruction='this is my ins', input='this is my inp', knowledge='LazyLLM-Knowledge'))
    'You are an AI-Agent developed by LazyLLM.hello world this is my ins\\nHere are some extra messages you can referred to:\\n\\n### knowledge:\\nLazyLLM-Knowledge\\n\\n\\n\\n\\n\\n\\nthis is my inp\\n\\n'
    >>> p.generate_prompt(dict(instruction='this is my ins', input='this is my inp', knowledge='LazyLLM-Knowledge'), return_dict=True)
    {'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\nhello world this is my ins\\nHere are some extra messages you can referred to:\\n\\n### knowledge:\\nLazyLLM-Knowledge\\n\\n\\n'}, {'role': 'user', 'content': 'this is my inp'}]}
    >>> p.generate_prompt(dict(instruction='this is my ins', input='this is my inp', knowledge='LazyLLM-Knowledge'), history=[['s1', 'e1'], ['s2', 'e2']])
    'You are an AI-Agent developed by LazyLLM.hello world this is my ins\\nHere are some extra messages you can referred to:\\n\\n### knowledge:\\nLazyLLM-Knowledge\\n\\n\\n\\n\\ns1e1s2e2\\n\\nthis is my inp\\n\\n'
    >>>
    >>> p = ChatPrompter(dict(system="hello world", user="this is user instruction {input} "))
    >>> p.generate_prompt(dict(input="my input", query="this is user query"))
    'You are an AI-Agent developed by LazyLLM.hello world\\n\\n\\n\\nthis is user instruction my input this is user query\\n\\n'
    >>> p.generate_prompt(dict(input="my input", query="this is user query"), return_dict=True)
    {'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\nhello world'}, {'role': 'user', 'content': 'this is user instruction my input this is user query'}]}
    """
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
    def _split(self): return f'{self._soa}\n' if self._soa else None
