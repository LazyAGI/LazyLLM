import re
import json
import collections
from lazyllm import LOG

templates = dict(
    # Template used by Alpaca-LoRA.
    alpaca={
        'prompt': 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n',  # noqa E501
        'response_split': '### Response:',
    },
    # Template used by internLM
    puyu={
        'prompt': '<bos><|System|>:You are an AI assistant whose name is InternLM (书生·浦语).\n- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.<eosys>\n<|Human|>:{instruction}<eoh>\n<|Assistant|>:please tell me what to do。ി\n<|Human|>:{input}<eoh>\n<|Assistant|>:', # noqa E501
        'response_split': '<|Assistant|>:',
    }
)

class Prompter(object):
    """Prompt generator class for LLM input formatting. Supports template-based prompting, history injection, and response extraction.

This class allows prompts to be defined via string templates, loaded from dicts, files, or predefined names.
It supports history-aware formatting for multi-turn conversations and adapts to both mapping and string input types.

Args:
    prompt (Optional[str]): Prompt template string with format placeholders.
    response_split (Optional[str]): Optional delimiter to split model response and extract useful output.
    chat_prompt (Optional[str]): Chat template string, must contain a history placeholder.
    history_symbol (str): Name of the placeholder for historical messages, default is 'llm_chat_history'.
    eoa (Optional[str]): Delimiter between assistant/user in history items.
    eoh (Optional[str]): Delimiter between user-assistant pairs.
    show (bool): Whether to print the final prompt when generating. Default is False.


Examples:
    >>> from lazyllm import Prompter
    
    >>> p = Prompter(prompt="Answer the following: {question}")
    >>> p.generate_prompt("What is AI?")
    'Answer the following: What is AI?'
    
    >>> p.generate_prompt({"question": "Define machine learning"})
    'Answer the following: Define machine learning'
    
    >>> p = Prompter(
    ...     prompt="Instruction: {instruction}",
    ...     chat_prompt="Instruction: {instruction}\\nHistory:\\n{llm_chat_history}",
    ...     history_symbol="llm_chat_history",
    ...     eoa="</s>",
    ...     eoh="|"
    ... )
    >>> p.generate_prompt(
    ...     input={"instruction": "Translate this."},
    ...     history=[["hello", "你好"], ["how are you", "你好吗"]]
    ... )
    'Instruction: Translate this.\\nHistory:\\nhello|你好</s>how are you|你好吗'
    
    >>> prompt_conf = {
    ...     "prompt": "Task: {task}",
    ...     "response_split": "---"
    ... }
    >>> p = Prompter.from_dict(prompt_conf)
    >>> p.generate_prompt("Summarize this article.")
    'Task: Summarize this article.'
    
    >>> full_output = "Task: Summarize this article.---This is the summary."
    >>> p.get_response(full_output)
    'This is the summary.'
    """
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
        """Initializes a Prompter instance from a prompt configuration dictionary.

Args:
    prompt (Dict): A dictionary containing prompt-related configuration. Must include 'prompt' key.
    show (bool): Whether to display the generated prompt. Defaults to False.

**Returns:**

- Prompter: An initialized Prompter instance.
"""
        assert isinstance(prompt, dict)
        return cls(**prompt, show=show)

    @classmethod
    def from_template(cls, template_name, *, show=False):
        """Loads prompt configuration from a template name and initializes a Prompter instance.

Args:
    template_name (str): Name of the template. Must exist in the `templates` dictionary.
    show (bool): Whether to display the generated prompt. Defaults to False.

**Returns:**

- Prompter: An initialized Prompter instance.
"""
        return cls.from_dict(templates[template_name], show=show)

    @classmethod
    def from_file(cls, fname, *, show=False):
        """Loads prompt configuration from a JSON file and initializes a Prompter instance.

Args:
    fname (str): Path to the JSON configuration file.
    show (bool): Whether to display the generated prompt. Defaults to False.

Returns:
    Prompter: An initialized Prompter instance.
"""
        with open(fname) as fp:
            return cls.from_dict(json.load(fp), show=show)

    @classmethod
    def empty(cls):
        """Creates an empty Prompter instance.

Returns:
    Prompter: A Prompter instance without any prompt configuration.
"""
        return cls()

    def _is_empty(self):
        return self._prompt is None

    def generate_prompt(self, input, history=None, tools=None, label=None, show=False):
        """Generates a formatted prompt string based on input and optional conversation history.

Args:
    input (Union[str, Dict]): User input. Can be a single string or a dictionary with multiple fields.
    history (Optional[List[List[str]]]): Multi-turn dialogue history, e.g., [['u1', 'a1'], ['u2', 'a2']].
    tools (Optional[Any]): Not supported. Must be None.
    label (Optional[str]): Optional label to append to the prompt, commonly used for training.
    show (bool): Whether to print the generated prompt. Defaults to False.

Returns:
    str: The final formatted prompt string.
"""
        if not self._is_empty():
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
        """Extracts the actual model answer from the full response returned by an LLM.

Args:
    response (str): The full raw output from the model.
    input (Optional[str]): If the response starts with the input, that part will be removed.

Returns:
    str: The cleaned model response.
"""
        if input and response.startswith(input):
            return response[len(input):]
        return response if self._response_split is None else response.split(self._response_split)[-1]
