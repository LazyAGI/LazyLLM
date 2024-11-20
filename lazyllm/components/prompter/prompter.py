import re
import json
import collections
from lazyllm import LOG

templates = dict(
    # Template used by Alpaca-LoRA.
    alpaca={
        "prompt": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",  # noqa E501
        "response_split": "### Response:",
    },
    # Template used by internLM
    puyu={
        "prompt": "<bos><|System|>:You are an AI assistant whose name is InternLM (书生·浦语).\n- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.<eosys>\n<|Human|>:{instruction}<eoh>\n<|Assistant|>:please tell me what to do。ി\n<|Human|>:{input}<eoh>\n<|Assistant|>:", # noqa E501
        "response_split": "<|Assistant|>:",
    }
)

class Prompter(object):
    def __init__(self, prompt=None, response_split=None, *, chat_prompt=None,
                 history_symbol='llm_chat_history', eoa=None, eoh=None, show=False):
        self._prompt, self._response_split = prompt, response_split
        self._chat_prompt = chat_prompt
        self._history_symbol, self._eoa, self._eoh = history_symbol, eoa, eoh
        self._show = show
        self._prompt_keys = list(set(re.findall(r'\{(\w+)\}', self._prompt))) if prompt else []
        if chat_prompt is not None:
            chat_keys = set(re.findall(r'\{(\w+)\}', self._chat_prompt))
            assert set(self._prompt_keys).issubset(chat_keys)
            assert chat_keys - set(self._prompt_keys) == set([self._history_symbol])
            self.use_history = True
        else:
            self.use_history = history_symbol in self._prompt_keys
            if self.use_history:
                self._prompt_keys.pop(self._prompt_keys.index(history_symbol))
                self._chat_prompt = self._prompt

    @classmethod
    def from_dict(cls, prompt, *, show=False):
        assert isinstance(prompt, dict)
        return cls(**prompt, show=show)

    @classmethod
    def from_template(cls, template_name, *, show=False):
        return cls.from_dict(templates[template_name], show=show)

    @classmethod
    def from_file(cls, fname, *, show=False):
        with open(fname) as fp:
            return cls.from_dict(json.load(fp), show=show)

    @classmethod
    def empty(cls):
        return cls()

    def is_empty(self):
        return self._prompt is None

    def generate_prompt(self, input, history=None, tools=None, label=None, show=False):
        if not self.is_empty():
            assert tools is None
            # datasets.formatting.formatting.LazyDict is used in transformers
            if not isinstance(input, collections.abc.Mapping):
                assert len(self._prompt_keys) == 1, (
                    f'invalid prompt `{self._prompt}` for <{type(input)}> input `{input}`')
                input = {self._prompt_keys[0]: input}
            try:
                if self.use_history and isinstance(history, list) and len(history) > 0:
                    assert isinstance(history[0], list), 'history must be list of list'
                    input[self._history_symbol] = self._eoa.join([self._eoh.join(h) for h in history])
                    input = self._chat_prompt.format(**input)
                else:
                    if self.use_history: input[self._history_symbol] = ''
                    input = self._prompt.format(**input)
            except Exception:
                raise RuntimeError(f'Generate prompt failed, and prompt is {self._prompt}; chat-prompt'
                                   f' is {self._chat_prompt}; input is {input}; history is {history}')
            if label: input += label
        if self._show or show: LOG.info(input)
        return input

    def get_response(self, response, input=None):
        if input and response.startswith(input):
            return response[len(input):]
        return response if self._response_split is None else response.split(self._response_split)[-1]
