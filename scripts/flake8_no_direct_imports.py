import ast
import os
import importlib.util
from pathlib import Path

module_path = Path('lazyllm/thirdparty/modules.py').resolve()

spec = importlib.util.spec_from_file_location('modules', module_path)
modules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modules)

modules = [m if isinstance(m, str) else m[0] for m in modules.modules if (m != 'os' and m[0] != 'os')]


EXCLUDE_PATHS = [
    'tests/',
    'lazyllm/components/finetune/alpaca-lora/',
    'lazyllm/components/finetune/collie/',
    'lazyllm/engine/',
    'lazyllm/components/deploy/relay/',
]


class NoDirectImportChecker:
    name = 'flake8-no-direct-imports'
    version = '0.1.0'

    def __init__(self, tree, filename):
        self.tree = tree
        self.filename = filename

    def _excluded(self) -> bool:
        normalized = self.filename.replace(os.sep, '/')
        return any(normalized.startswith(excl) or f'/{excl}' in normalized for excl in EXCLUDE_PATHS)

    def _is_thirdparty_annotation(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Constant) and isinstance(node.value, str): return False
        if isinstance(node, ast.Name): return node.id in modules

        if isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name): return root.id in modules

        if isinstance(node, ast.Subscript):
            return self._is_thirdparty_annotation(node.value) or self._is_thirdparty_annotation(node.slice)
        if isinstance(node, ast.Tuple):
            return any(self._is_thirdparty_annotation(e) for e in node.elts)

        return False

    def run(self):  # noqa C901
        if self._excluded(): return
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in modules:
                        yield (node.lineno, node.col_offset, f'NID001 Do not import {alias.name} directly. '
                               f'Use `from thirdparty import {alias.name}` instead.', type(self))
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in modules:
                    if not node.module.startswith('thirdparty'):
                        yield (node.lineno, node.col_offset,
                               f'NID002 Do not import {[alias.name for alias in node.names]}'
                               f'from {node.module}. Use `from thirdparty import {node.module}` '
                               f'and access via `{node.module}.xxx`.', type(self))

        for node in ast.walk(self.tree):
            if isinstance(node, ast.AnnAssign):
                if node.annotation and self._is_thirdparty_annotation(node.annotation):
                    yield (node.lineno, node.col_offset, 'NID003 Thirdparty types in annotations must be quoted',
                           type(self))
            elif isinstance(node, ast.FunctionDef):
                args = node.args.args + node.args.kwonlyargs
                if node.args.vararg:
                    args.append(node.args.vararg)
                if node.args.kwarg:
                    args.append(node.args.kwarg)

                for arg in args:
                    if arg.annotation and self._is_thirdparty_annotation(arg.annotation):
                        yield (arg.lineno, arg.col_offset, 'NID004 Thirdparty types in function '
                               'annotations must be quoted', type(self))

                if node.returns and self._is_thirdparty_annotation(node.returns):
                    yield node.lineno, node.col_offset, 'NID005 Thirdparty return type must be quoted', type(self)
