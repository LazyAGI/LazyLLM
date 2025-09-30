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

    def run(self):
        normalized = self.filename.replace(os.sep, '/')
        for excl in EXCLUDE_PATHS:
            if normalized.startswith(excl) or f'/{excl}' in normalized:
                return

        for node in self.tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in modules:
                        yield (
                            node.lineno,
                            node.col_offset,
                            f'NID001 Do not import {alias.name} directly. '
                            f'Use `from thirdparty import {alias.name}` instead.',
                            type(self),
                        )

            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in modules:
                    if not node.module.startswith('thirdparty'):
                        yield (
                            node.lineno,
                            node.col_offset,
                            f'NID002 Do not import {[alias.name for alias in node.names]} from {node.module}. '
                            f'Use `from thirdparty import {node.module}` and use `{node.module}.xxx` instead.',
                            type(self),
                        )
