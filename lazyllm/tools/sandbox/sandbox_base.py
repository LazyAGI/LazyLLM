import os
from typing import List, Optional
from lazyllm.module.module import ModuleBase

class SandboxBase(ModuleBase):
    SUPPORTED_LANGUAGES: List[str] = []

    def __init__(self, output_dir_path: Optional[str] = None, return_trace: bool = True):
        super().__init__(return_trace=return_trace)
        self._output_dir_path = output_dir_path or os.getcwd()

    def _is_available(self) -> None:
        raise NotImplementedError('Subclasses must implement this method')

    def _execute(self, code: str, language: str = 'python', input_files: Optional[List[str]] = None,
                 output_files: Optional[List[str]] = None) -> str:
        raise NotImplementedError('Subclasses must implement this method')

    def forward(self, code: str, language: str = 'python', input_files: Optional[List[str]] = None,
                output_files: Optional[List[str]] = None) -> str:
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f'Language {language} not supported by {self.__class__.__name__}')
        return self._execute(code, language, input_files, output_files)
