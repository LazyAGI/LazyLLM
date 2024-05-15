import re
import json

templates = dict()

class Prompter(object):
    def __init__(self, prompt=None, response_split=None, history_prompt=None,
                 history_symbol='llm_chat_history', eoa=None, eoh=None):
        self._prompt, self._response_split = prompt, response_split
        self._history_symbol, self._eoa, self._eoh = history_symbol, eoa, eoh
        self._prompt_keys = list(set(re.findall(r'\{(\w+)\}', self._prompt))) if prompt else []
        if history_prompt is not None:
            hp_keys = set(re.findall(r'\{(\w+)\}', self._prompt))
            assert set(self._prompt_keys).issubset(hp_keys)
            assert hp_keys - set(self._prompt_keys) == set([self._history_symbol])
            self.use_history = True
        else:
            self.use_history = history_symbol in self._prompt_keys
            if self.use_history:
                self._prompt_keys.pop(self._prompt_keys.index(history_symbol))
                self._history_prompt = self._prompt

    @classmethod
    def from_json(cls, prompt):
        return cls(prompt.get('prompt', None), prompt.get('response_split', None),
                   prompt.get('history_prompt', None), prompt.get('history_symbol', None),
                   prompt.get('eoa', None), prompt.get('eoh', None))

    @classmethod
    def from_template(cls, template_name):
        return cls.from_json(templates[template_name])

    @classmethod
    def from_file(cls, fname):
        with open(fname) as fp:
            return cls.from_json(json.load(fp))

    @classmethod
    def empty(cls):
        return cls()

    def is_empty(self):
        return self._prompt is None

    def generate_prompt(self, input, history=None):
        if not self.is_empty():
            if not isinstance(input, dict):
                assert len(self._prompt_keys) == 1, f'invalid prompt `{self._prompt}` for input `{input}`'
                input = {self._prompt_keys[0]: input}

            if self.use_history and isinstance(history, list) and len(history) > 0:
                assert isinstance(history[0], list), 'history must be list of list'
                input[self._history_symbol] = self._eoa.join([self._eoh.join(h) for h in history])
                input = self._history_prompt.format(**input)
            else:
                if self.use_history: input[self._history_symbol] = ''
                input = self._prompt.format(**input)
        return input

    def get_response(self, response, input=None):
        if input and response.startswith(input):
            return response[len(input):]
        return response if self._response_split is None else response.split(self._response_split)[-1]

    def verbose(self, flag):
        self._verbose = flag
