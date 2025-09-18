from typing import List, Union, Optional, Dict
from .builtinPrompt import LazyLLMPrompterBase

class ChatPrompter(LazyLLMPrompterBase):
    """Prompt constructor for multi-turn dialogue, inherits from `LazyLLMPrompterBase`.

Supports tool calling, conversation history, and customizable instruction templates. Accepts instructions as either plain string or dict with separate `system` and `user` components, automatically merging them into a unified prompt template. Also supports injecting extra user-defined fields.

Args:
    instruction (Option[str | Dict[str, str]]): The prompt instruction template. Can be a string or a dict with `system` and `user` keys. If a dict is given, the components will be merged using special delimiters.
    extra_keys (Option[List[str]]): A list of additional keys that will be filled by user input to enrich the prompt context.
    show (bool): Whether to print the generated prompt. Default is False.
    tools (Option[List]): A list of tools available to the model for function-calling tasks. Default is None.
    history (Option[List[List[str]]]): Dialogue history in the format [[user, assistant], ...]. Used to provide conversational memory. Default is None.


Examples:
    >>> from lazyllm import ChatPrompter
    
    - Simple instruction string
    >>> p = ChatPrompter('hello world')
    >>> p.generate_prompt('this is my input')
    'You are an AI-Agent developed by LazyLLM.hello world\\nthis is my input\\n'
    
    >>> p.generate_prompt('this is my input', return_dict=True)
    {'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\nhello world'}, {'role': 'user', 'content': 'this is my input'}]}
    
    - Using extra_keys
    >>> p = ChatPrompter('hello world {instruction}', extra_keys=['knowledge'])
    >>> p.generate_prompt({
    ...     'instruction': 'this is my ins',
    ...     'input': 'this is my inp',
    ...     'knowledge': 'LazyLLM-Knowledge'
    ... })
    'You are an AI-Agent developed by LazyLLM.hello world this is my ins\\nHere are some extra messages you can referred to:\\n\\n### knowledge:\\nLazyLLM-Knowledge\\nthis is my inp\\n'
    
    - With conversation history
    >>> p.generate_prompt({
    ...     'instruction': 'this is my ins',
    ...     'input': 'this is my inp',
    ...     'knowledge': 'LazyLLM-Knowledge'
    ... }, history=[['s1', 'e1'], ['s2', 'e2']])
    'You are an AI-Agent developed by LazyLLM.hello world this is my ins\\nHere are some extra messages you can referred to:\\n\\n### knowledge:\\nLazyLLM-Knowledge\\ns1|e1\\ns2|e2\\nthis is my inp\\n'
    
    - Using dict format for system/user instructions
    >>> p = ChatPrompter(dict(system="hello world", user="this is user instruction {input}"))
    >>> p.generate_prompt({'input': "my input", 'query': "this is user query"})
    'You are an AI-Agent developed by LazyLLM.hello world\\nthis is user instruction my input this is user query\\n'
    
    >>> p.generate_prompt({'input': "my input", 'query': "this is user query"}, return_dict=True)
    {'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\nhello world'}, {'role': 'user', 'content': 'this is user instruction my input this is user query'}]}
    """
    def __init__(self, instruction: Union[None, str, Dict[str, str]] = None, extra_keys: Union[None, List[str]] = None,
                 show: bool = False, tools: Optional[List] = None, history: Optional[List[List[str]]] = None):
        super(__class__, self).__init__(show, tools=tools, history=history)
        if isinstance(instruction, dict):
            splice_instruction = instruction.get('system', '') + \
                ChatPrompter.ISA + instruction.get('user', '') + ChatPrompter.ISE
            instruction = splice_instruction
        instruction_template = f'{instruction}\n{{extra_keys}}\n'.replace(
            '{extra_keys}', LazyLLMPrompterBase._get_extro_key_template(extra_keys)) if instruction else ''
        self._init_prompt('{sos}{system}{instruction}{tools}{eos}\n\n{history}\n{soh}\n{user}{input}\n{eoh}{soa}\n',
                          instruction_template)

    @property
    def _split(self): return f'{self._soa}\n' if self._soa else None
