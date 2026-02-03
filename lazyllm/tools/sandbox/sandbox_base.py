import os
from typing import List, Optional
from lazyllm.module.module import ModuleBase

class SandboxBase(ModuleBase):
    SUPPORTED_LANGUAGES: List[str] = []

    def __init__(self, output_dir_path: Optional[str] = None, return_trace: bool = True):
        super().__init__(return_trace=return_trace)
        if not output_dir_path:
            output_dir_path = os.path.join(os.getcwd(), 'lazyllm_sandbox_output_files')
        os.makedirs(output_dir_path, exist_ok=True)
        self._output_dir_path = output_dir_path

    def _is_available(self) -> None:
        raise NotImplementedError('Subclasses must implement this method')

    def _execute(self, code: str, input_files: Optional[List[str]] = None,
                 output_files: Optional[List[str]] = None) -> str:
        raise NotImplementedError('Subclasses must implement this method')

    def forward(self, code: str, input_files: Optional[List[str]] = None, output_files: Optional[List[str]] = None):
        return self._execute(code, input_files, output_files)
