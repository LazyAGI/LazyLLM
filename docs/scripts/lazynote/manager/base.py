import inspect
import textwrap
from abc import ABC, abstractmethod
from typing import Optional, List

import libcst as cst
from pydantic import BaseModel, Field

from lazynote.parser import BaseParser
from lazynote.schema import MemberType, get_member_type
from lazynote.editor import BaseEditor
from enum import Enum
import importlib
import pkgutil
import asyncio
import traceback

class DocstringMode(str, Enum):
    """Enumeration for different modes of handling docstrings."""
    TRANSLATE = "translate"
    POLISH = "polish"
    CLEAR = "clear"
    FILL = "fill"

class BaseManager(BaseModel, ABC):
    """
    Executor for modifying module docstrings. Currently supports module-level or file-level modifications.

    Subclasses need to override the `gen_docstring` method to generate custom docstrings.

    Attributes:
        parser (Optional[BaseParser]): The parser used to parse the module. Defaults to an instance of BaseParser.
        pattern (DocstringMode): The mode for handling docstrings.
        skip_on_error (bool): Whether to skip errors or raise them. Defaults to False.
    """
    parser: Optional[BaseParser] = Field(default_factory=BaseParser)
    pattern: DocstringMode
    skip_on_error: bool = False  # Add class attribute

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.parser is None:
            self.parser = BaseParser(skip_modules=kwargs.get('skip_modules', []))
        self.skip_on_error = kwargs.get('skip_on_error', self.skip_on_error)

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def gen_docstring(self, old_docstring: Optional[str], node_code: str) -> str:
        """
        Generate a new docstring. Subclasses must implement this method to provide custom logic.

        Args:
            old_docstring (Optional[str]): The old docstring.
            node_code (str): The code of the node.

        Returns:
            str: The new docstring.
        """
        pass

    @staticmethod
    def is_defined_in_module(member: object, module: object) -> bool:
        """
        Check if a member is defined in a given module.

        Args:
            member (object): The member to check.
            module (object): The module to check against.

        Returns:
            bool: True if the member is defined in the module, False otherwise.
        """
        if hasattr(member, '__module__'):
            return member.__module__ == module.__name__
        elif isinstance(member, property):
            return member.fget.__module__ == module.__name__
        elif isinstance(member, staticmethod):
            return member.__func__.__module__ == module.__name__
        elif isinstance(member, classmethod):
            return member.__func__.__module__ == module.__name__
        elif hasattr(member, '__wrapped__'):
            return member.__wrapped__.__module__ == module.__name__
        return False

    def _handle_error(self, error_message: str, error: Exception) -> None:
        """
        Handle errors according to the skip_on_error flag.

        Args:
            error_message (str): The error message to display.
            error (Exception): The exception that was raised.
        """
        if self.skip_on_error:
            print(f"{error_message}: {error}")
            traceback.print_exc()
        else:
            raise error

    def _write_code_to_file(self, module: object, code: str) -> None:
        """
        Write modified code back to the module file.

        Args:
            module (object): The module to write to.
            code (str): The modified code.
        """
        module_file_path = inspect.getfile(module)
        with open(module_file_path, 'w', encoding='utf-8') as file:
            file.write(code)

    async def _sem_task(self, task, modname: str, semaphore: asyncio.Semaphore) -> None:
        """
        Run a task with a semaphore to limit concurrency.

        Args:
            task: The task to run.
            modname (str): The module name associated with the task.
            semaphore (asyncio.Semaphore): The semaphore to limit concurrency.
        """
        async with semaphore:
            try:
                await task
            except Exception as e:
                self._handle_error(f"Skipping {modname} due to import error", e)

    def traverse(self, obj: object, skip_modules: Optional[List[str]] = None) -> None:
        """
        Traverse through the package or module to process docstrings.

        Args:
            obj (object): The object to traverse.
            skip_modules (Optional[List[str]]): List of modules to skip.
        """
        if skip_modules is None:
            skip_modules = []

        if get_member_type(obj) == MemberType.PACKAGE:
            for _, modname, ispkg in pkgutil.walk_packages(obj.__path__, obj.__name__ + "."):
                if any(modname.startswith(skip_mod) for skip_mod in skip_modules):
                    continue
                if ispkg:
                    continue

                try:
                    submodule = importlib.import_module(modname)
                    self.parser.parse(submodule, self)
                except Exception as e:
                    self._handle_error(f"Skipping {modname} due to import error", e)

        elif get_member_type(obj) == MemberType.MODULE:
            try:
                self.parser.parse(obj, self)
            except Exception as e:
                self._handle_error(f"Skipping {obj.__name__} due to import error", e)

    async def atraverse(self, obj: object, skip_modules: Optional[List[str]] = None, max_concurrency: int = 10) -> None:
        """
        Asynchronously traverse through the package or module to process docstrings.

        Args:
            obj (object): The object to traverse.
            skip_modules (Optional[List[str]]): List of modules to skip.
            max_concurrency (int): Maximum number of concurrent tasks.
        """
        if skip_modules is None:
            skip_modules = []

        semaphore = asyncio.Semaphore(max_concurrency)
        loop = asyncio.get_event_loop()

        if get_member_type(obj) == MemberType.PACKAGE:
            tasks = []
            for _, modname, ispkg in pkgutil.walk_packages(obj.__path__, obj.__name__ + "."):
                if any(modname.startswith(skip_mod) for skip_mod in skip_modules):
                    continue
                if ispkg:
                    continue

                try:
                    submodule = importlib.import_module(modname)
                    task = loop.run_in_executor(None, self.parser.parse, submodule, self)
                    tasks.append(self._sem_task(task, modname, semaphore))
                except Exception as e:
                    self._handle_error(f"Skipping {modname} due to import error", e)
            await asyncio.gather(*tasks)

        elif get_member_type(obj) == MemberType.MODULE:
            try:
                task = loop.run_in_executor(None, self.parser.parse, obj, self)
                await self._sem_task(task, obj.__name__, semaphore)
            except Exception as e:
                self._handle_error(f"Skipping {obj.__name__} due to import error", e)

    def modify_docstring(self, module: object) -> Optional[str]:
        """
        Modify the docstring of a given module.

        Args:
            module (object): The module to modify.

        Returns:
            Optional[str]: The modified code, or None if an error occurred.
        """
        try:
            source_code = inspect.getsource(module)
            source_code = textwrap.dedent(source_code)
            tree = cst.parse_module(source_code)
            transformer = BaseEditor(
                gen_docstring=self.gen_docstring, pattern=self.pattern, module=module)
            modified_tree = tree.visit(transformer)
            self._write_code_to_file(module, modified_tree.code)
            return modified_tree.code
        except Exception as e:
            self._handle_error(f"Skipping module {module.__name__} due to error", e)
            return None
