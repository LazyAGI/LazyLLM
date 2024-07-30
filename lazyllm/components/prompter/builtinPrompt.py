from typing import Dict, Union, Any, List, Callable, Optional
from ...common import LazyLLMRegisterMetaClass
from lazyllm import LOG
import json5 as json
from functools import reduce
import copy
import re
import uuid

class LazyLLMPrompterBase(metaclass=LazyLLMRegisterMetaClass):
    ISA = "<!lazyllm-spliter!>"
    ISE = "</!lazyllm-spliter!>"

    def __init__(self, show=False, tools=None):
        self._set_model_configs(system='You are an AI-Agent developed by LazyLLM.', sos='',
                                soh='', soa='', eos='', eoh='', eoa='')
        self._show = show
        self._tools = tools
        self._pre_hook = None

    def _init_prompt(self, template: str, instruction_template: str, split: Union[None, str] = None,
                     tool_delimiter: str = "<|tool_calls|>"):
        self._template = template
        self._instruction_template = instruction_template
        if split:
            assert not hasattr(self, '_split')
            self._split = split
        self._tool_delimiter = tool_delimiter

    @staticmethod
    def _get_extro_key_template(extro_keys, prefix='Here are some extra messages you can referred to:\n\n'):
        if extro_keys:
            if isinstance(extro_keys, str): extro_keys = [extro_keys]
            assert isinstance(extro_keys, (tuple, list)), 'Only str, tuple[str], list[str] are supported'
            return prefix + ''.join([f"### {k}:\n{{{k}}}\n\n" for k in extro_keys])
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
                    LOG.error(f"history: {history}")
                    raise ValueError("history must be a list of list or dict")
            return content
        else:
            if isinstance(history[0], list):
                return ''.join([f'{self._soh}{h}{self._eoh}{self._soa}{a}{self._eoa}' for h, a in history])
            elif isinstance(history[0], dict):
                ret = ""
                for item in history:
                    if item['role'] == "user":
                        ret += f'{self._soh}{item["content"]}{self._eoh}'
                    elif item['role'] == "assistant":
                        ret += f'{self._soa}'
                        if len(item.get("content", "")) > 0:
                            ret += f'{item["content"]}'
                        if len(item.get("tool_calls", "")) > 0:
                            for idx in range(len(item['tool_calls'])):
                                tool = item['tool_calls'][idx]['function']
                                if getattr(self, "_tool_args_token", None):
                                    tool = tool['name'] + self._tool_args_token + tool['arguments']
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
            pattern = re.compile(r"%s(.*)%s" % (LazyLLMPrompterBase.ISA, LazyLLMPrompterBase.ISE))
            ret = re.split(pattern, instruction)
            system_instruction = ret[0]
            user_instruction = ret[1]

        return system_instruction, user_instruction

    def generate_prompt(self, input: Union[str, List, Dict[str, str], None] = None,
                        history: List[Union[List[str], Dict[str, Any]]] = None,
                        tools: Union[List[Dict[str, Any]], None] = None,
                        label: Union[str, None] = None,
                        *, show: bool = False, return_dict: bool = False) -> Union[str, Dict]:
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

    def get_response(self, output: str, input: Union[str, None] = None) -> str:  # noqa: C901
        def preprocess_output(output: str, input: Union[str, None]) -> str:
            if input and output.startswith(input):
                output = output[len(input):]
            output = output if getattr(self, "_split", None) is None else output.split(self._split)[-1]
            return output

        def parse_arguments_with_args_token(output: str) -> (str, dict):
            items = output.split(self._tool_args_token)
            if len(items) < 2:
                return "", {}
            func_name = items[0].strip()
            arguments = (items[1].strip().split(self._tool_end_token)[0] if getattr(self, "_tool_end_token", None)
                         else items[1].strip())
            return func_name, arguments

        def parse_arguments_without_args_token(output: str) -> (str, dict):
            items = output.split(self._tool_end_token)[0] if getattr(self, "_tool_end_token", None) else output
            func_name = ""
            arguments = {}
            try:
                items = json.loads(items.strip())
                func_name = items.get('name', '')
                arguments = items.get("parameters", items.get("arguments", {}))
            except Exception:
                LOG.error(f"tool calls info {items} parse error")

            return func_name, arguments

        def parse_arguments_with_tools(output: Dict[str, Any], tools: List[str],
                                       tool_calls: List[Dict[str, str]]) -> bool:
            func_name = ''
            arguments = {}
            is_tc = False
            if output.get('name', '') in tools:
                is_tc = True
                func_name = output.get('name', '')
                arguments = output.get("parameters", output.get("arguments", {}))
                tool_calls.append({'name': func_name, 'arguments': arguments})
            return is_tc

        def parse_tool_start_token(output: str) -> (str, List[Dict]):
            tool_calls = []
            segs = output.split(self._tool_start_token)
            content = segs[0]
            for seg in segs[1:]:
                func_name, arguments = parse_arguments_with_args_token(seg.strip())\
                    if getattr(self, "_tool_args_token", None)\
                    else parse_arguments_without_args_token(seg.strip())
                if func_name:
                    tool_calls.append({"name": func_name, "arguments": arguments})

            return content, tool_calls

        def parse_tools(output: str) -> (str, List[Dict]):
            tool_calls = []
            tools = {tool['function']['name'] for tool in self._tools}
            lines = output.strip().split("\n")
            content = []
            is_tool_call = False
            for idx, line in enumerate(lines):
                if line.startswith("{") and idx > 0:
                    func_name = lines[idx - 1].strip()
                    if func_name in tools:
                        is_tool_call = True
                        if func_name == content[-1].strip():
                            content.pop()
                        arguments = "\n".join(lines[idx:]).strip()
                        tool_calls.append({'name': func_name, "arguments": arguments})
                if "{" in line and 'name' in line:
                    try:
                        items = json.loads(line.strip())
                        if isinstance(items, list):
                            for item in items:
                                is_tool_call = parse_arguments_with_tools(item, tools, tool_calls)
                        elif isinstance(items, dict):
                            is_tool_call = parse_arguments_with_tools(items, tools, tool_calls)
                    except Exception:
                        LOG.error(f"tool calls info {line} parse error")
                if not is_tool_call:
                    content.append(line)
            content = "\n".join(content) if len(content) > 0 else ''
            return content, tool_calls

        def extract_tool_calls(output: str) -> (str, List[Dict]):
            tool_calls = []
            content = ''
            if getattr(self, "_tool_start_token", None) and self._tool_start_token in output:
                content, tool_calls = parse_tool_start_token(output)
            elif self._tools:
                content, tool_calls = parse_tools(output)
            else:
                content = output

            return content, tool_calls

        def build_response(content: str, tool_calls: List[Dict[str, str]]) -> str:
            tc = [{'id': str(uuid.uuid4().hex), 'type': 'function', 'function': tool_call} for tool_call in tool_calls]
            if content and tc:
                return self._tool_delimiter.join([content, json.dumps(tc, ensure_ascii=False)])
            elif not content and tc:
                return self._tool_delimiter + json.dumps(tc, ensure_ascii=False)
            else:
                return content

        output = preprocess_output(output, input)
        LOG.info(f"output: {output}")
        content, tool_calls = extract_tool_calls(output)
        return build_response(content, tool_calls)

class EmptyPrompter(LazyLLMPrompterBase):

    def generate_prompt(self, input, history=None, tools=None, label=None, show=False):
        if self._show or show: LOG.info(input)
        return input
