import copy
from typing import List, Dict, Union

from lazyllm.components.formatter.formatterbase import (encode_query_with_filepaths,
                                                        decode_query_with_filepaths,
                                                        LAZYLLM_QUERY_PREFIX)
import lazyllm
from ...module import ModuleBase


class OnlineMultiModalBase(ModuleBase):
    def __init__(self, model_series: str, model_name: str = None, return_trace: bool = False, **kwargs):
        super().__init__(return_trace=return_trace)
        self._model_series = model_series
        self._model_name = model_name
        self._validate_model_config()

    def _validate_model_config(self):
        """Validate model configuration"""
        if not self._model_series:
            raise ValueError("model_series cannot be empty")
        if not self._model_name:
            lazyllm.LOG.warning(f"model_name not specified for {self._model_series}")

    @property
    def series(self):
        return self._model_series

    @property
    def type(self):
        return "MultiModal"

    def share(self):
        """Create a shared instance of the module"""
        new = copy.copy(self)
        return new

    def _format_input_files(self, files: List[str]):
        """Default input file formatting - to be overridden by subclasses"""
        return files

    def _format_output_files(self, output_files: List[str]):
        """Default output file formatting - to be overridden by subclasses"""
        return output_files

    def _forward(self, input: Union[Dict, str] = None, files: List[str] = None, **kwargs):
        """Forward method to be implemented by subclasses"""
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method")

    def forward(self, input: Union[Dict, str] = None, *, lazyllm_files=None, **kw):
        """Main forward method with file handling"""
        try:
            files = lazyllm_files

            # Handle encoded query with files
            if isinstance(input, str) and input.startswith(LAZYLLM_QUERY_PREFIX):
                decoded_input = decode_query_with_filepaths(input)
                input, files = decoded_input['query'], decoded_input['files']

            # Format input files (default implementation, can be overridden)
            files = self._format_input_files(files)

            # Call the concrete implementation
            output, output_files = self._forward(input=input, files=files, **kw)

            # Format output files (default implementation, can be overridden)
            output_files = self._format_output_files(output_files)

            # Return encoded result
            return output or encode_query_with_filepaths(query=output, files=output_files)

        except Exception as e:
            lazyllm.LOG.error(f"Error in {self.__class__.__name__}.forward: {str(e)}")
            raise

    def __repr__(self):
        return lazyllm.make_repr('Module', 'OnlineMultiModal',
                                 series=self._model_series,
                                 name=self._model_name,
                                 return_trace=self._return_trace)
