import copy
from typing import List, Dict, Union, Optional

from lazyllm.components.prompter import PrompterBase
from lazyllm.components.formatter import FormatterBase
from lazyllm.components.formatter.formatterbase import (encode_query_with_filepaths,
                                                        decode_query_with_filepaths,
                                                        LAZYLLM_QUERY_PREFIX)

from ..module import ModuleBase

class OnlineMultiModuleBase(ModuleBase):
    def __init__(self, model_series: str, model_name: str = None, return_trace: bool = False, **kwargs):
        super().__init__(return_trace=return_trace)
        self._model_series = model_series
        self._model_name = model_name

    @property
    def series(self):
        return self._model_series

    @property
    def type(self):
        return "LLM"

    @property
    def stream(self):
        return self._stream

    @stream.setter
    def stream(self, v: Union[bool, Dict[str, str]]):
        self._stream = v

    def share(self, prompt: PrompterBase = None, format: FormatterBase = None, stream: Optional[bool] = None,
              history: List[List[str]] = None, copy_static_params: bool = False):
        new = copy.copy(self)
        new._hooks = set()
        new._set_mid()
        if prompt is not None: new.prompt(prompt, history=history)
        if format is not None: new.formatter(format)
        if stream is not None: new.stream = stream
        if copy_static_params:
            new._static_params = copy.deepcopy(self._static_params)
        return new

    def _format_input_files(self, files: List[str]):
        return files

    def _format_output_files(self, output: List[str]):
        return output

    def _forward(self, input: Union[Dict, str] = None, files: List[str] = [], **kwargs):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method")

    def forward(self, input: Union[Dict, str] = None, *, lazyllm_files=None, **kw):
        files = lazyllm_files
        if isinstance(input, str) and input.startswith(LAZYLLM_QUERY_PREFIX):
            input = decode_query_with_filepaths(input)
            input, files = input['query'], input['files']
        files = self._format_input_files(files)
        output, output_files = self._forward(input=input, files=files, **kw)
        output_files = self._format_output_files(output_files)
        return encode_query_with_filepaths(query=output, files=output_files)
