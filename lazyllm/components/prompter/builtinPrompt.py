from typing import Dict, Union, Any, List, Callable, Optional
from ...common import LazyLLMRegisterMetaClass
from lazyllm import LOG
import json5 as json
from functools import reduce
import copy
import re

class LazyLLMPrompterBase(metaclass=LazyLLMRegisterMetaClass):
    """The base class of Prompter. A custom Prompter needs to inherit from this base class and set the Prompt template and the Instruction template using the `_init_prompt` function provided by the base class, as well as the string used to capture results. Refer to  [prompt](/Best%20Practice/prompt) for further understanding of the design philosophy and usage of Prompts.

Both the Prompt template and the Instruction template use ``{}`` to indicate the fields to be filled in. The fields that can be included in the Prompt are `system`, `history`, `tools`, `user` etc., while the fields that can be included in the instruction_template are `instruction` and `extra_keys`. If the ``instruction`` field is a string, it is considered as a system instruction; if it is a dictionary, it can only contain the keys ``user`` and ``system``. ``user`` represents the user input instruction, which is placed before the user input in the prompt, and ``system`` represents the system instruction, which is placed after the system prompt in the prompt.
``instruction`` is passed in by the application developer, and the ``instruction`` can also contain ``{}`` to define fillable fields, making it convenient for users to input additional information.


Examples:
    >>> from lazyllm.components.prompter import PrompterBase
    >>> class MyPrompter(PrompterBase):
    ...     def __init__(self, instruction = None, extra_keys = None, show = False):
    ...         super(__class__, self).__init__(show)
    ...         instruction_template = f'{instruction}\\n{{extra_keys}}\\n'.replace('{extra_keys}', PrompterBase._get_extro_key_template(extra_keys))
    ...         self._init_prompt("<system>{system}</system>\\n</instruction>{instruction}</instruction>{history}\\n{input}\\n, ## Response::", instruction_template, '## Response::')
    ... 
    >>> p = MyPrompter('ins {instruction}')
    >>> p.generate_prompt('hello')
    '<system>You are an AI-Agent developed by LazyLLM.</system>\\n</instruction>ins hello\\n\\n</instruction>\\n\\n, ## Response::'
    >>> p.generate_prompt('hello world', return_dict=True)
    {'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\nins hello world\\n\\n'}, {'role': 'user', 'content': ''}]}
    """
    ISA = "<!lazyllm-spliter!>"
    ISE = "</!lazyllm-spliter!>"

    def __init__(self, show=False, tools=None, history=None):
        self._set_model_configs(system='You are an AI-Agent developed by LazyLLM.', sos='',
                                soh='', soa='', eos='', eoh='', eoa='')
        self._show = show
        self._tools = tools
        self._pre_hook = None
        self._history = history or []

    def _init_prompt(self, template: str, instruction_template: str, split: Union[None, str] = None):
        self._template = template
        self._instruction_template = instruction_template
        if split:
            assert not hasattr(self, '_split')
            self._split = split

    @staticmethod
    def _get_extro_key_template(extra_keys, prefix='Here are some extra messages you can referred to:\n\n'):
        if extra_keys:
            if isinstance(extra_keys, str): extra_keys = [extra_keys]
            assert isinstance(extra_keys, (tuple, list)), 'Only str, tuple[str], list[str] are supported'
            return prefix + ''.join([f"### {k}:\n{{{k}}}\n\n" for k in extra_keys])
        return ''

    def _handle_tool_call_instruction(self):
        tool_dict = {}
        for key in ["tool_start_token", "tool_args_token", "tool_end_token"]:
            if getattr(self, f"_{key}", None) and key in self._instruction_template:
                tool_dict[key] = getattr(self, f"_{key}")
        return reduce(lambda s, kv: s.replace(f"{{{kv[0]}}}", kv[1]), tool_dict.items(), self._instruction_template)

    def _set_model_configs(self, system: str = None, sos: Union[None, str] = None, soh: Union[None, str] = None,
                           soa: Union[None, str] = None, eos: Union[None, str] = None,
                           eoh: Union[None, str] = None, eoa: Union[None, str] = None,
                           soe: Union[None, str] = None, eoe: Union[None, str] = None,
                           separator: Union[None, str] = None, plugin: Union[None, str] = None,
                           interpreter: Union[None, str] = None, stop_words: Union[None, List[str]] = None,
                           tool_start_token: Union[None, str] = None, tool_end_token: Union[None, str] = None,
                           tool_args_token: Union[None, str] = None):

        local = locals()
        for name in ['system', 'sos', 'soh', 'soa', 'eos', 'eoh', 'eoa', 'soe', 'eoe', 'tool_start_token',
                     'tool_end_token', 'tool_args_token']:
            if local[name] is not None: setattr(self, f'_{name}', local[name])

        if getattr(self, "_instruction_template", None):
            self._instruction_template = self._handle_tool_call_instruction()

    def _get_tools(self, tools, *, return_dict):
        if self._tools:
            assert tools is None
            tools = self._tools

        return tools if return_dict else '### Function-call Tools. \n\n' + json.dumps(tools) + '\n\n' if tools else ''

    def _get_histories(self, history, *, return_dict):  # noqa: C901
        if not self._history and not history: return ''
        if return_dict:
            content = []
            for item in self._history + (history or []):
                if isinstance(item, list):
                    assert len(item) <= 2, "history item length cannot be greater than 2"
                    if len(item) > 0: content.append({"role": "user", "content": item[0]})
                    if len(item) > 1: content.append({"role": "assistant", "content": item[1]})
                elif isinstance(item, dict):
                    content.append(item)
                else:
                    LOG.error(f"history: {history}")
                    raise ValueError("history must be a list of list or dict")
            return content
        else:
            ret = ''.join([f'{self._soh}{h}{self._eoh}{self._soa}{a}{self._eoa}' for h, a in self._history])
            if not history: return ret
            if isinstance(history[0], list):
                return ret + ''.join([f'{self._soh}{h}{self._eoh}{self._soa}{a}{self._eoa}' for h, a in history])
            elif isinstance(history[0], dict):
                for item in history:
                    if item['role'] == "user":
                        ret += f'{self._soh}{item["content"]}{self._eoh}'
                    elif item['role'] == "assistant":
                        ret += f'{self._soa}'
                        ret += f'{item.get("content", "")}'
                        for idx in range(len(item.get('tool_calls', []))):
                            tool = item['tool_calls'][idx]['function']
                            if getattr(self, "_tool_args_token", None):
                                tool = tool['name'] + self._tool_args_token + \
                                    json.dumps(tool['arguments'], ensure_ascii=False)
                            ret += (f'{getattr(self, "_tool_start_token", "")}' + '\n'
                                    f'{tool}'
                                    f'{getattr(self, "_tool_end_token", "")}' + '\n')
                        ret += f'{self._eoa}'
                    elif item['role'] == "tool":
                        try:
                            content = json.loads(item['content'].strip())
                        except Exception:
                            content = item['content']
                        ret += f'{getattr(self, "_soe", "")}{content}{getattr(self, "_eoe", "")}'

                return ret
            else:
                raise NotImplementedError('Cannot transform json history to {type(history[0])} now')

    def _get_instruction_and_input(self, input):
        prompt_keys = list(set(re.findall(r'\{(\w+)\}', self._instruction_template)))
        if isinstance(input, (str, int)):
            if len(prompt_keys) == 1:
                return self._instruction_template.format(**{prompt_keys[0]: input}), ''
            else:
                assert len(prompt_keys) == 0
                return self._instruction_template, input
        assert isinstance(input, dict), f'expected types are str, int and dict, bug get {type(input)}(`{input})`'
        kwargs = {k: input.pop(k) for k in prompt_keys}
        assert len(input) <= 1, f"Unexpected keys found in input: {list(input.keys())}"
        return (reduce(lambda s, kv: s.replace(f"{{{kv[0]}}}", kv[1]),
                       kwargs.items(),
                       self._instruction_template)
                if len(kwargs) > 0 else self._instruction_template,
                list(input.values())[0] if input else "")

    def _check_values(self, instruction, input, history, tools): pass

    # Used for TrainableModule(local deployed)
    def _generate_prompt_impl(self, instruction, input, user, history, tools, label):
        is_tool = False
        if isinstance(input, dict):
            input = input.get('content', '')
            is_tool = input.get('role') == 'tool'
        elif isinstance(input, list):
            is_tool = any(item.get('role') == 'tool' for item in input)
            input = "\n".join([item.get('content', '') for item in input])
        params = dict(system=self._system, instruction=instruction, input=input, user=user, history=history, tools=tools,
                      sos=self._sos, eos=self._eos, soh=self._soh, eoh=self._eoh, soa=self._soa, eoa=self._eoa)
        if is_tool:
            params['soh'] = getattr(self, "_soe", self._soh)
            params['eoh'] = getattr(self, "_eoe", self._eoh)
        return self._template.format(**params) + (label if label else '')

    # Used for OnlineChatModule
    def _generate_prompt_dict_impl(self, instruction, input, user, history, tools, label):
        if not history: history = []
        if isinstance(input, str):
            history.append({"role": "user", "content": input})
        elif isinstance(input, dict):
            history.append(input)
        elif isinstance(input, list) and all(isinstance(ele, dict) for ele in input):
            history.extend(input)
        elif isinstance(input, tuple) and len(input) == 1:
            # Note tuple size 1 with one single string is not expected
            history.append({"role": "user", "content": input[0]})
        else:
            raise TypeError("input must be a string or a dict")

        if user:
            history[-1]["content"] = user + history[-1]['content']

        history.insert(0, {"role": "system",
                           "content": self._system + "\n" + instruction if instruction else self._system})

        return dict(messages=history, tools=tools) if tools else dict(messages=history)

    def pre_hook(self, func: Optional[Callable] = None):
        self._pre_hook = func
        return self

    def _split_instruction(self, instruction: str):
        system_instruction = instruction
        user_instruction = ""
        if LazyLLMPrompterBase.ISA in instruction and LazyLLMPrompterBase.ISE in instruction:
            # The instruction includes system prompts and/or user prompts
            pattern = re.compile(r"%s(.*)%s" % (LazyLLMPrompterBase.ISA, LazyLLMPrompterBase.ISE), re.DOTALL)
            ret = re.split(pattern, instruction)
            system_instruction = ret[0]
            user_instruction = ret[1]

        return system_instruction, user_instruction

    def generate_prompt(self, input: Union[str, List, Dict[str, str], None] = None,
                        history: List[Union[List[str], Dict[str, Any]]] = None,
                        tools: Union[List[Dict[str, Any]], None] = None,
                        label: Union[str, None] = None,
                        *, show: bool = False, return_dict: bool = False) -> Union[str, Dict]:
        """
Generate a corresponding Prompt based on user input.

Args:
    input (Option[str | Dict]): The input from the prompter, if it's a dict, it will be filled into the slots of the instruction; if it's a str, it will be used as input.
    history (Option[List[List | Dict]]): Historical conversation, can be ``[[u, s], [u, s]]`` or in openai's history format, defaults to None.
    tools (Option[List[Dict]]): A collection of tools that can be used, used when the large model performs FunctionCall, defaults to None.
    label (Option[str]): Label, used during fine-tuning or training, defaults to None.
    show (bool): Flag indicating whether to print the generated Prompt, defaults to False.
    return_dict (bool): Flag indicating whether to return a dict, generally set to True when using ``OnlineChatModule``. If returning a dict, only the ``instruction`` will be filled. Defaults to False.
"""
        input = copy.deepcopy(input)
        if self._pre_hook:
            input, history, tools, label = self._pre_hook(input, history, tools, label)
        instruction, input = self._get_instruction_and_input(input)
        history = self._get_histories(history, return_dict=return_dict)
        tools = self._get_tools(tools, return_dict=return_dict)
        self._check_values(instruction, input, history, tools)
        instruction, user_instruction = self._split_instruction(instruction)
        func = self._generate_prompt_dict_impl if return_dict else self._generate_prompt_impl
        result = func(instruction, input, user_instruction, history, tools, label)
        if self._show or show: LOG.info(result)
        return result

    def get_response(self, output: str, input: Union[str, None] = None) -> str:
        """Used to truncate the Prompt, keeping only valuable output.

Args:
        output (str): The output of the large model.
        input (Option[str]): The input of the large model. If this parameter is specified, any part of the output that includes the input will be completely truncated. Defaults to None.
"""
        if input and output.startswith(input):
            return output[len(input):]
        return output if getattr(self, "_split", None) is None else output.split(self._split)[-1]

class EmptyPrompter(LazyLLMPrompterBase):

    def generate_prompt(self, input, history=None, tools=None, label=None, show=False):
        if self._show or show: LOG.info(input)
        return input
