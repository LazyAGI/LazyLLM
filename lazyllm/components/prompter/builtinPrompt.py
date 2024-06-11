from typing import Dict, Union, Any, List
from ...common import LazyLLMRegisterMetaClass
from lazyllm import LOG
import json
import re

class LazyLLMPrompterBase(metaclass=LazyLLMRegisterMetaClass):
    def __init__(self, show=False, tools=None):
        self._set_model_configs(system='You are an AI-Agent developed by LazyLLM.', sos='<|start_system|>',
                                soh='<|Human|>:', soa='<|Assistant|>:', eos='<|end_system|>', eoh='', eoa='')
        self._show = show
        self._tools = tools

    def _init_prompt(self, template: str, instruction_template: str, split: Union[None, str] = None):
        self._template = template
        self._instruction_template = instruction_template
        if split:
            assert not hasattr(self, '_split')
            self._split = split

    @staticmethod
    def _get_extro_key_template(extro_keys, prefix='Here are some extra messages you can referred to:\n\n'):
        if extro_keys:
            return prefix + ''.join([f"### {k}:\n{{{k}}}\n\n" for k in extro_keys])
        return ''

    def _set_model_configs(self, system: str = None, sos: Union[None, str] = None, soh: Union[None, str] = None,
                           soa: Union[None, str] = None, eos: Union[None, str] = None,
                           eoh: Union[None, str] = None, eoa: Union[None, str] = None):
        local = locals()
        for name in ['system', 'sos', 'soh', 'soa', 'eos', 'eoh', 'eoa']:
            if local[name] is not None: setattr(self, f'_{name}', local[name])

    def _get_tools(self, tools, *, return_dict):
        if self._tools:
            assert tools is None
            tools = self.tools

        return tools if return_dict else '### Function-call Tools. \n\n' + json.dumps(tools) + '\n\n' if tools else ''

    def _get_histories(self, history, *, return_dict):
        if history is None or len(history) == 0: return ''
        if return_dict:
            content = []
            for item in history:
                if isinstance(item, list):
                    assert len(item) <= 2, "history item length cannot be greater than 2"
                    if len(item) > 0: content.append({"role": "user", "content": item[0]})
                    if len(item) > 1: content.append({"role": "assistant", "content": item[1]})
                elif isinstance(item, dict):
                    content.append(item)
                else:
                    raise ValueError("history must be a list of list or dict")
            return content
        else:
            if not isinstance(history[0], list):
                raise NotImplementedError('Cannot transform json history to list now')
            return ''.join([f'{self._soh}{h}{self._eoh}{self._soa}{a}{self._eoa}' for h, a in history])

    def _get_instruction_and_input(self, input):
        prompt_keys = list(set(re.findall(r'\{(\w+)\}', self._instruction_template)))
        if isinstance(input, str):
            if len(prompt_keys) == 1:
                return self._instruction_template.format(**{prompt_keys[0]: input}), ''
            else:
                assert len(prompt_keys) == 0
                return self._instruction_template, input
        assert isinstance(input, dict)
        kwargs = {k: input.pop(k) for k in prompt_keys}
        assert len(input) <= 1, f'Unexpected keys found in input: {list(input.keys())}'
        return (self._instruction_template.format(**kwargs) if len(kwargs) > 0 else self._instruction_template,
                list(input.values())[0] if input else '')

    def _check_values(self, instruction, input, history, tools): pass

    # Used for TrainableModule(local deployed)
    def _generate_prompt_impl(self, instruction, input, history, tools, label):
        params = dict(system=self._system, instruction=instruction, input=input, history=history, tools=tools,
                      sos=self._sos, eos=self._eos, soh=self._soh, eoh=self._eoh, soa=self._soa, eoa=self._eoa)
        return self._template.format(**params) + (label if label else '')

    # Used for OnlineChatModule
    def _generate_prompt_dict_impl(self, instruction, input, history, tools, label):
        if not history: history = []
        if isinstance(input, str):
            history.append({"role": "user", "content": input})
        elif isinstance(input, dict):
            history.append(input)
        else:
            raise TypeError("input must be a string or a dict")

        history.insert(0, {"role": "system",
                           "content": self._system + "\n" + instruction if instruction else self._system})

        return dict(messages=history, tools=tools) if tools else dict(messages=history)

    def generate_prompt(self, input: Union[str, Dict[str, str], None] = None,
                        history: List[Union[List[str], Dict[str, Any]]] = None,
                        tools: Union[List[Dict[str, Any]], None] = None,
                        label: Union[str, None] = None,
                        *, show: bool = False, return_dict: bool = False) -> str:
        instruction, input = self._get_instruction_and_input(input)
        history = self._get_histories(history, return_dict=return_dict)
        tools = self._get_tools(tools, return_dict=return_dict)
        self._check_values(instruction, input, history, tools)
        func = self._generate_prompt_dict_impl if return_dict else self._generate_prompt_impl
        result = func(instruction, input, history, tools, label)
        if self._show or show: LOG.info(result)
        return result

    def get_response(self, output: str, input: Union[str, None] = None) -> str:
        if input and output.startswith(input):
            return output[len(input):]
        return output if getattr(self, '_split', None) is None else output.split(self._split)[-1]

class EmptyPrompter(LazyLLMPrompterBase):
    def generate_prompt(self, input, history=None, tools=None, label=None, show=False):
        if self._show or show: LOG.info(input)
        return input
