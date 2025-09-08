import copy
from typing import List, Dict, Union
import lazyllm
from ....servermodule import LLMBase
from .utils import OnlineModuleBase


class OnlineMultiModalBase(OnlineModuleBase, LLMBase):
    def __init__(self, model_series: str, model_name: str = None, return_trace: bool = False, **kwargs):
        super().__init__(return_trace=return_trace)
        self._model_series = model_series
        self._model_name = model_name
        self._validate_model_config()

    def _validate_model_config(self):
        '''Validate model configuration'''
        if not self._model_series:
            raise ValueError('model_series cannot be empty')
        if not self._model_name:
            lazyllm.LOG.warning(f'model_name not specified for {self._model_series}')

    @property
    def series(self):
        return self._model_series

    @property
    def type(self):
        return 'MultiModal'

    def share(self):
        '''Create a shared instance of the module'''
        new = copy.copy(self)
        return new

    def _forward(self, input: Union[Dict, str] = None, files: List[str] = None, **kwargs):
        '''Forward method to be implemented by subclasses'''
        raise NotImplementedError(f'Subclass {self.__class__.__name__} must implement this method')

    def forward(self, input: Union[Dict, str] = None, *, lazyllm_files=None, **kwargs):
        '''Main forward method with file handling'''
        try:
            input, files = self._get_files(input, lazyllm_files)
            call_params = {'input': input, **kwargs}
            if files: call_params['files'] = files
            return self._forward(**call_params)

        except Exception as e:
            lazyllm.LOG.error(f'Error in {self.__class__.__name__}.forward: {str(e)}')
            raise

    def __repr__(self):
        return lazyllm.make_repr('Module', 'OnlineMultiModalModule',
                                 series=self._model_series,
                                 name=self._model_name,
                                 return_trace=self._return_trace)
