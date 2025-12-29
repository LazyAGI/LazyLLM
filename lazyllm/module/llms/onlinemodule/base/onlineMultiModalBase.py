import copy
from contextlib import contextmanager
from typing import List, Dict, Union, Optional
import lazyllm
from ....servermodule import LLMBase
from .utils import OnlineModuleBase


class OnlineMultiModalBase(OnlineModuleBase, LLMBase):
    def __init__(self, model_series: str, model_name: str = None, return_trace: bool = False,
                 api_key: Optional[Union[str, List[str]]] = None, **kwargs):
        super().__init__(api_key=api_key, return_trace=return_trace)
        self._model_series = model_series
        self._model_name = model_name
        self._base_url = kwargs.get('base_url')
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

    def _set_base_url(self, base_url: Optional[str]):
        self._base_url = base_url

    def share(self, *, model: Optional[str] = None):
        '''Create a shared instance of the module'''
        new = copy.copy(self)
        if model is not None:
            new._model_name = model
        return new

    @contextmanager
    def _override_endpoint(self, *, model: Optional[str] = None, base_url: Optional[str] = None):
        old_model = self._model_name
        old_base_url = getattr(self, '_base_url', None)
        if model is not None:
            self._model_name = model
        if base_url is not None:
            self._set_base_url(base_url)
        try:
            yield
        finally:
            self._model_name = old_model
            if base_url is not None:
                self._set_base_url(old_base_url)

    def _forward(self, input: Union[Dict, str] = None, files: List[str] = None, **kwargs):
        '''Forward method to be implemented by subclasses'''
        raise NotImplementedError(f'Subclass {self.__class__.__name__} must implement this method')

    def forward(self, input: Union[Dict, str] = None, *, lazyllm_files=None, **kwargs):
        '''Main forward method with file handling'''
        try:
            input, files = self._get_files(input, lazyllm_files)
            call_params = {'input': input, **kwargs}
            if files: call_params['files'] = files
            runtime_base_url = kwargs.pop('base_url', None)
            runtime_model = kwargs.pop('model', None)
            with self._override_endpoint(model=runtime_model, base_url=runtime_base_url):
                return self._forward(**call_params)

        except Exception as e:
            lazyllm.LOG.error(f'Error in {self.__class__.__name__}.forward: {str(e)}')
            raise

    def __repr__(self):
        return lazyllm.make_repr('Module', 'OnlineMultiModalModule',
                                 series=self._model_series,
                                 name=self._model_name,
                                 return_trace=self._return_trace)
