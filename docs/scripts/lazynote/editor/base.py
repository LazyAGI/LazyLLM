import inspect
from typing import Callable, Optional, Any, Dict, Set

import libcst as cst
import libcst.matchers as m


class BaseEditor(cst.CSTTransformer):
    """
    A tool for transforming code text and generating new code text.
    """

    def __init__(self, gen_docstring: Callable[[Optional[str], str], str], pattern: str, module: Any) -> None:
        """
        Initializes the BaseEditor.

        Args:
            gen_docstring (Callable[[Optional[str], str], str]): A function to generate docstrings.
            pattern (str): A pattern to match.
            module (Any): The module to be transformed.
        """
        self.gen_docstring = gen_docstring
        self.pattern = pattern
        self.module = module
        self.module_dict = self.create_module_dict(module)
        self.current_class: Optional[str] = ''

    def create_module_dict(self, module: Any) -> Dict[str, Any]:
        """
        Creates a dictionary of module members.

        Args:
            module (Any): The module to inspect.

        Returns:
            Dict[str, Any]: A dictionary of module members.
        """
        module_dict = {}
        seen_objects: Set[Any] = set()
        for name, obj in inspect.getmembers(module):
            module_dict[name] = obj
            if inspect.isclass(obj):
                self.add_class_members_to_dict(module_dict, obj, name, seen_objects)
        return module_dict

    def add_class_members_to_dict(
            self, module_dict: Dict[str, Any], cls: Any, parent_name: str, seen_objects: Set[Any]) -> None:
        """
        Adds class members to the module dictionary.

        Args:
            module_dict (Dict[str, Any]): The module dictionary.
            cls (Any): The class to inspect.
            parent_name (str): The parent name of the class.
            seen_objects (Set[Any]): A set of seen objects to avoid infinite recursion.
        """
        if cls in seen_objects:
            return
        seen_objects.add(cls)
        for name, obj in inspect.getmembers(cls):
            full_name = f"{parent_name}.{name}"
            module_dict[full_name] = obj
            if inspect.isclass(obj):
                self.add_class_members_to_dict(module_dict, obj, full_name, seen_objects)

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        """
        Called when leaving a FunctionDef node.

        Args:
            original_node (cst.FunctionDef): The original FunctionDef node.
            updated_node (cst.FunctionDef): The updated FunctionDef node.

        Returns:
            cst.FunctionDef: The updated FunctionDef node with a new docstring.
        """
        full_name = (
            f"{self.current_class}.{original_node.name.value}"
            if self.current_class else original_node.name.value
        )
        obj = self._get_obj_by_name(full_name)
        docstring = obj.__doc__ if obj else None
        return self._update_node_with_new_docstring(original_node, updated_node, docstring)

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        """
        Called when visiting a ClassDef node.

        Args:
            node (cst.ClassDef): The ClassDef node.
        """
        self.current_class = f'{self.current_class}.{node.name.value}'.lstrip('.')

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        """
        Called when leaving a ClassDef node.

        Args:
            original_node (cst.ClassDef): The original ClassDef node.
            updated_node (cst.ClassDef): The updated ClassDef node.

        Returns:
            cst.ClassDef: The updated ClassDef node with a new docstring.
        """
        self.current_class = self.current_class[:(lambda x: 0 if x < 0 else x)(self.current_class.rfind('.'))]
        obj = self._get_obj_by_name(original_node.name.value)
        docstring = obj.__doc__ if obj else None
        return self._update_node_with_new_docstring(original_node, updated_node, docstring)

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        """
        Called when leaving a Module node.

        Args:
            original_node (cst.Module): The original Module node.
            updated_node (cst.Module): The updated Module node.

        Returns:
            cst.Module: The updated Module node with a new docstring.
        """
        return updated_node

    def _get_obj_by_name(self, name: str) -> Optional[Any]:
        """
        Gets an object by its name from the module dictionary.

        Args:
            name (str): The name of the object.

        Returns:
            Optional[Any]: The object if found, otherwise None.
        """
        return self.module_dict.get(name, None)

    def _update_node_with_new_docstring(
            self, original_node: cst.CSTNode, updated_node: cst.CSTNode, docstring: Optional[str]) -> cst.CSTNode:
        """
        Updates a node with a new docstring.

        Args:
            original_node (cst.CSTNode): The original node.
            updated_node (cst.CSTNode): The updated node.
            docstring (Optional[str]): The new docstring.

        Returns:
            cst.CSTNode: The updated node with the new docstring.
        """
        node_code = cst.Module([]).code_for_node(original_node)
        old_docstring = docstring
        new_body = []

        if isinstance(updated_node.body, tuple):
            body = updated_node.body
        else:
            body = getattr(updated_node.body, 'body', [])

        # Extract existing docstring if present and build new body without it
        for stmt in body:
            if m.matches(stmt, m.SimpleStatementLine(body=[m.Expr(m.SimpleString())])):
                old_docstring = cst.ensure_type(stmt.body[0].value, cst.SimpleString).value.strip('\"\'')
            else:
                new_body.append(stmt)

        new_docstring = self.gen_docstring(old_docstring, node_code)

        # Create a new docstring node if new_docstring is provided
        new_docstring_node = (
            cst.SimpleStatementLine([cst.Expr(cst.SimpleString(f'"""{new_docstring}"""'))]) if new_docstring else None
        )

        if new_docstring_node:
            # Check if the function body is a SimpleStatementSuite (single-line function)
            if isinstance(updated_node.body, cst.SimpleStatementSuite):
                # Create a new IndentedBlock containing the original function body statements
                new_body = cst.IndentedBlock(
                    body=[
                        new_docstring_node,
                        cst.SimpleStatementLine(
                            body=[
                                cst.Expr(
                                    value=updated_node.body.body[0]
                                )
                            ]
                        )
                    ]
                )

                # Replace the original function body with the new IndentedBlock
                return updated_node.with_changes(body=new_body)
            else:
                new_body.insert(0, new_docstring_node)

        # Update the body with the new list of statements
        try:
            if isinstance(updated_node.body, tuple):
                updated_body = tuple(new_body)
            else:
                updated_body = updated_node.body.with_changes(body=new_body)
        except Exception as e:
            print(f"Error updating body with new statements: {new_body}")
            print(f"Error message: {e}")
            raise

        return updated_node.with_changes(body=updated_body)
