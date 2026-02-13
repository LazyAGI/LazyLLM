import os
from typing import Generator, List, Optional, Tuple
from lazyllm import config
from lazyllm.module.module import ModuleBase
from lazyllm.common.registry import LazyLLMRegisterMetaClass

config.add('sandbox_type', str, 'dummy', 'SANDBOX_TYPE')


class LazyLLMSandboxBase(ModuleBase, metaclass=LazyLLMRegisterMetaClass):
    SUPPORTED_LANGUAGES: List[str] = []

    def __init__(self, output_dir_path: Optional[str] = None, return_trace: bool = True,
                 project_dir: Optional[str] = None):
        super().__init__(return_trace=return_trace)
        self._output_dir_path = output_dir_path or os.getcwd()
        self._project_dir = project_dir
        if self._project_dir and not os.path.isdir(self._project_dir):
            raise FileNotFoundError(f'Project directory not found: {self._project_dir}')
        self._check_available()

    # -- Abstract methods (subclasses must implement) --------------------------

    def _check_available(self) -> None:
        raise NotImplementedError

    def _create_context(self) -> dict:
        '''Create execution context. See class docstring for context structure.'''
        raise NotImplementedError

    def _execute(self, code: str, language: str, context: dict,
                 output_files: Optional[List[str]] = None) -> dict:
        '''Run code in sandbox. File staging and output collection are handled by forward().'''
        raise NotImplementedError

    def _process_input_files(self, input_files: List[str], context: dict) -> None:
        '''Stage input files into context for execution.'''
        raise NotImplementedError

    def _process_output_files(self, result: dict, output_files: List[str], context: dict) -> List[str]:
        '''Collect output files from result/context, save to _output_dir_path, return saved paths.'''
        raise NotImplementedError

    def _process_project_dir(self, context: dict) -> None:
        '''Stage project .py files into context. Use _collect_project_py_files() for iteration.'''
        raise NotImplementedError

    def _cleanup_context(self, context: dict) -> None:
        '''Clean up resources from _create_context. Called in finally block. Default is no-op.'''
        pass

    # -- Utilities -------------------------------------------------------------

    def _validate_input_files(self, input_files: Optional[List[str]]) -> Optional[List[str]]:
        '''Validate existence and return absolute paths. Called automatically in forward().'''
        if not input_files:
            return input_files
        validated = []
        for f in input_files:
            abs_path = os.path.abspath(f)
            if not os.path.isfile(abs_path):
                raise FileNotFoundError(f'Input file not found: {f}')
            validated.append(abs_path)
        return validated

    def _collect_project_py_files(self) -> Generator[Tuple[str, str], None, None]:
        '''Yield (abs_path, rel_path) for each .py file in _project_dir.
        rel_path is relative to _project_dir, not cwd.'''
        if not self._project_dir:
            return
        abs_dir = os.path.abspath(self._project_dir)
        for root, _, files in os.walk(abs_dir):
            for name in files:
                if name.endswith('.py'):
                    abs_path = os.path.join(root, name)
                    yield abs_path, os.path.relpath(abs_path, abs_dir)

    def _ensure_output_dir(self) -> None:
        os.makedirs(self._output_dir_path, exist_ok=True)

    # -- Template method -------------------------------------------------------

    def forward(self, code: str, language: str = 'python', input_files: Optional[List[str]] = None,
                output_files: Optional[List[str]] = None) -> str:
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f'Language {language} not supported by {self.__class__.__name__}')
        input_files = self._validate_input_files(input_files)

        context = self._create_context()
        try:
            if self._project_dir:
                self._process_project_dir(context)
            if input_files:
                self._process_input_files(input_files, context)

            result = self._execute(code, language, context, output_files)

            if output_files and isinstance(result, dict):
                result['output_files'] = self._process_output_files(result, output_files, context)
            return result
        finally:
            self._cleanup_context(context)
