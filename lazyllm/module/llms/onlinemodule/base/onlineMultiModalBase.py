from typing import List, Dict, Union, Optional
import lazyllm
from ....servermodule import LLMBase
from .utils import OnlineModuleBase
from ..map_model_type import get_model_type


class OnlineMultiModalBase(OnlineModuleBase, LLMBase):
    def __init__(self, model_series: str, model_name: str = None, return_trace: bool = False,
                 api_key: Optional[Union[str, List[str]]] = None, base_url: str = None, **kwargs):
        super().__init__(api_key=api_key, return_trace=return_trace)
        self._model_series = model_series
        self._model_name = model_name
        self._base_url = base_url
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

    def _forward(self, input: Union[Dict, str] = None, files: List[str] = None, **kwargs):
        '''Forward method to be implemented by subclasses'''
        raise NotImplementedError(f'Subclass {self.__class__.__name__} must implement this method')

    def forward(self, input: Union[Dict, str] = None, *, lazyllm_files=None, 
                url: str = None, model: str = None, **kwargs):
        '''Main forward method with file handling'''
        try:
            input, files = self._get_files(input, lazyllm_files)
            runtime_url = url or kwargs.pop('base_url', None) or self._base_url
            runtime_model = model or kwargs.pop('model_name', None) or self._model_name
            if get_model_type(runtime_model) not in ('sd', 'stt', 'tts'):
                raise ValueError(f"Model type must be 'sd', 'stt' or 'tts', got model {runtime_model}")
            
            call_params = {'input': input, **kwargs}
            if files: call_params['files'] = files
            return self._forward(**call_params, model=runtime_model, url=runtime_url)

        except Exception as e:
            lazyllm.LOG.error(f'Error in {self.__class__.__name__}.forward: {str(e)}')
            raise

    def __repr__(self):
        return lazyllm.make_repr('Module', 'OnlineMultiModalModule',
                                 series=self._model_series,
                                 name=self._model_name,
                                 return_trace=self._return_trace)
