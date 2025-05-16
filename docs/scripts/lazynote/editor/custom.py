from typing import Callable, Optional, Any, Dict, Set
from lazyllm import LOG
import libcst as cst
import libcst.matchers as m
from .base import BaseEditor


class CustomEditor(BaseEditor):
    """
    A custom tool for transforming code text and generating new code text.
    """
    def __init__(self, gen_class_docstring: Callable[[Optional[str], str], str],  gen_docstring: Callable[[Optional[str], str], str], pattern: str, module: Any) -> None:
        super().__init__(gen_docstring, pattern, module)
        self.current_class_doc_dict: Dict[str, str] = {}
        self.gen_class_docstring = gen_class_docstring

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        func_name = f"{self.current_class}.{original_node.name.value}".lstrip('.')
        if func_name in self.current_class_doc_dict:
            docstring = self.current_class_doc_dict[func_name]
        else: 
            obj = self._get_obj_by_name(func_name)
            docstring = obj.__doc__ if obj else None
        return self._update_node_with_new_docstring(original_node, updated_node, docstring)

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        if f"{self.current_class}.{node.name.value}" in self.current_class_doc_dict:
            self.current_class = f'{self.current_class}.{node.name.value}'.lstrip('.')
            return
        
        obj = self._get_obj_by_name(node.name.value)
        docstring = obj.__doc__ if obj else None
        node_code = cst.Module([]).code_for_node(node)
        res = self.gen_class_docstring(docstring, node_code)
        self.current_class_doc_dict = {}
        for name, docstring in res.items():
            self.current_class_doc_dict[f"{self.current_class}.{name}".lstrip('.')] = docstring
        self.current_class = f'{self.current_class}.{node.name.value}'.lstrip('.')

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        new_docstring = self.current_class_doc_dict.get(self.current_class, None)
        updated_node = self._update_node_with_new_docstring(original_node, updated_node, new_docstring)
        self.current_class = self.current_class[:(lambda x: 0 if x < 0 else x)(self.current_class.rfind('.'))]
        return updated_node

  