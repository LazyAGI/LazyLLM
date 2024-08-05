from typing import Optional
from lazynote.manager.base import BaseManager

class CustomManager(BaseManager):
    def gen_docstring(self, old_docstring: Optional[str], pattern: str, node_code: str) -> str:
        """
        Custom logic to generate a new docstring.

        Args:
            old_docstring (Optional[str]): The old docstring.
            pattern (str): The pattern string to be added.
            node_code (str): The node code.

        Returns:
            str: The new docstring.
        """
        if old_docstring:
            return f"{old_docstring}\n\n{pattern}"
        else:
            return f"{pattern}"
