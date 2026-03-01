import os
import re
from dataclasses import asdict, dataclass, field
from typing import Generator, List, Optional, Tuple
from lazyllm import config
from lazyllm.module.module import ModuleBase
from lazyllm.common.registry import LazyLLMRegisterMetaClass

config.add('sandbox_type', str, 'dummy', 'SANDBOX_TYPE')

SANDBOX_TOOL_RESULT_PREFIX = 'LAZYLLM_TOOL_RESULT:'


def create_sandbox(sandbox_type=None, **kwargs):
    import lazyllm
    sandbox_type = sandbox_type or config['sandbox_type']
    try:
        return lazyllm.sandbox[sandbox_type](**kwargs)
    except KeyError as e:
        raise ValueError(
            f'Sandbox type {sandbox_type!r} not found, '
            f'supported sandbox types: {list(lazyllm.sandbox.keys())}'
        ) from e


@dataclass
class _SandboxResult:
    success: bool
    stdout: str = ''
    stderr: str = ''
    returncode: int = 0
    output_files: List[str] = field(default_factory=list)
    error_message: str = ''

    def to_dict(self) -> dict:
        return asdict(self)


class LazyLLMSandboxBase(ModuleBase, metaclass=LazyLLMRegisterMetaClass):
    SUPPORTED_LANGUAGES: List[str] = []

    def __init__(self, output_dir_path: Optional[str] = None, return_trace: bool = True,
                 project_dir: Optional[str] = None, return_sandbox_result: bool = False):
        super().__init__(return_trace=return_trace)
        self._output_dir_path = output_dir_path or os.getcwd()
        self._project_dir = project_dir
        if self._project_dir and not os.path.isdir(self._project_dir):
            raise FileNotFoundError(f'Project directory not found: {self._project_dir}')
        self._available_checked = False
        self._return_sandbox_result = return_sandbox_result

    def _check_available(self) -> None:
        raise NotImplementedError

    def _create_context(self) -> dict:
        raise NotImplementedError

    def _execute(self, code: str, language: str, context: dict,
                 output_files: Optional[List[str]] = None) -> _SandboxResult:
        raise NotImplementedError

    def _process_input_files(self, input_files: List[str], context: dict) -> None:
        raise NotImplementedError

    def _process_output_files(self, result: _SandboxResult, output_files: List[str], context: dict) -> List[str]:
        raise NotImplementedError

    def _process_project_dir(self, context: dict) -> None:
        raise NotImplementedError

    def _cleanup_context(self, context: dict) -> None:
        pass

    def _validate_input_files(self, input_files: Optional[List[str]]) -> Optional[List[str]]:
        if not input_files:
            return input_files
        for f in input_files:
            if not os.path.isfile(f):
                raise FileNotFoundError(f'Input file not found: {f}')

    def _collect_project_py_files(self) -> Generator[Tuple[str, str], None, None]:
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

    def forward(self, code: str, language: str = 'python', input_files: Optional[List[str]] = None,
                output_files: Optional[List[str]] = None) -> dict:
        if not self._available_checked:
            self._check_available()
            self._available_checked = True

        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f'Language {language} not supported by {self.__class__.__name__}')
        self._validate_input_files(input_files)

        context = self._create_context()
        try:
            if self._project_dir:
                self._process_project_dir(context)
            if input_files:
                self._process_input_files(input_files, context)

            result = self._execute(code, language, context, output_files)

            if output_files and result.success:
                result.output_files = self._process_output_files(result, output_files, context)

            if self._return_sandbox_result:
                return result.to_dict()
            else:
                if result.success:
                    match = re.search(rf'^{SANDBOX_TOOL_RESULT_PREFIX}(.*)', result.stdout, re.MULTILINE)
                    return match.group(1) if match else result.stdout
                return result.stderr
        finally:
            self._cleanup_context(context)
