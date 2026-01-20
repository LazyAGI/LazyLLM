from os import PathLike, makedirs
from os.path import expanduser, expandvars, isfile, join, normpath
from typing import Union, Dict, Callable, Any, Optional
import re
import os
from contextlib import contextmanager
import cloudpickle
import ast
import pickle
import base64
import argparse

def check_path(
    path: Union[str, PathLike],
    exist: bool = True,
    file: bool = True,
    parents: bool = True,
) -> str:
    '''
    Check path and return corrected path.
    '''
    # normalize and expand a path
    path = normpath(expandvars(expanduser(path)))
    if exist and file and not isfile(path):
        raise FileNotFoundError(path)
    else:
        if file:
            dir_path = normpath(join(path, '..'))
        else:
            dir_path = path
        if parents:
            makedirs(dir_path, exist_ok=True)
    return path

class SecurityVisitor(ast.NodeVisitor):  # noqa C901
    '''
    AST-based security analyzer to detect unsafe operations in Python code.

    IMPORTANT: Method names within this class (e.g., `visit_Call`, `visit_Import`) **should not**
    be renamed to lowercase. These method names are part of the `NodeVisitor` pattern from the `ast`
    module and must remain consistant with this naming convention to function correctly.
    '''

    # **Dangerous built-in functions**
    DANGEROUS_BUILTINS = {'exec', 'eval', 'open', 'compile', 'getattr',
                          'setattr', '__import__', 'globals', 'locals', 'vars'}

    # **Dangerous os operations**
    DANGEROUS_OS_CALLS = {'system', 'popen', 'remove', 'rmdir', 'unlink', 'rename'}

    # **Dangerous sys operations**
    DANGEROUS_SYS_CALLS = {'exit', 'modules'}

    # **Dangerous modules**
    DANGEROUS_MODULES = {'pickle', 'subprocess', 'socket', 'shutil', 'requests', 'inspect', 'tempfile'}

    def visit_Call(self, node):  # noqa C901
        '''Check function calls'''
        # Direct calls to dangerous built-in functions
        if isinstance(node.func, ast.Name) and node.func.id in self.DANGEROUS_BUILTINS:
            raise ValueError(f'⚠️ Detected dangerous function call: {node.func.id}')

        # Check for __import__ calls with string arguments
        if isinstance(node.func, ast.Name) and node.func.id == '__import__':
            if node.args and isinstance(node.args[0], ast.Str):
                module_name = node.args[0].s
                if module_name in self.DANGEROUS_MODULES:
                    raise ValueError(f'⚠️ Detected dangerous module import via __import__: {module_name}')

        # Check for indirect __import__ calls (function calls that might return __import__)
        if isinstance(node.func, ast.Call):
            # Check if this is a call to a function that might return __import__
            if isinstance(node.func.func, ast.Name):
                func_name = node.func.func.id
                if func_name in ['get_import', 'import_func']:  # Common patterns
                    raise ValueError(f'⚠️ Detected suspicious function call that might return __import__: {func_name}')

        # Check for attribute access that might lead to __import__
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'ImportHelper':
                if node.func.attr == 'get_import':
                    raise ValueError('⚠️ Detected suspicious method call: ImportHelper.get_import')

        # os / sys related calls
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == 'os' and node.func.attr in self.DANGEROUS_OS_CALLS:
                raise ValueError(f'⚠️ Detected dangerous os call: os.{node.func.attr}')
            if node.func.value.id == 'sys' and node.func.attr in self.DANGEROUS_SYS_CALLS:
                raise ValueError(f'⚠️ Detected dangerous sys call: sys.{node.func.attr}')

        self.generic_visit(node)

    def visit_Import(self, node):
        '''Check import statements'''
        for alias in node.names:
            if alias.name in self.DANGEROUS_MODULES:
                raise ValueError(f'⚠️ Detected dangerous module import: {alias.name}')

    def visit_ImportFrom(self, node):
        '''Check from ... import statements'''
        if node.module in self.DANGEROUS_MODULES:
            raise ValueError(f'⚠️ Detected dangerous module import: {node.module}')

    def visit_Attribute(self, node):
        '''Check os.environ and tempfile usage'''
        if isinstance(node.value, ast.Name):
            if node.value.id == 'os' and node.attr == 'environ':
                raise ValueError('⚠️ Detected dangerous access: os.environ')
            if node.value.id == 'tempfile':
                raise ValueError(f'⚠️ Detected dangerous usage of tempfile: tempfile.{node.attr}')

        self.generic_visit(node)

    def visit_Lambda(self, node):
        '''Check lambda functions that might return __import__'''
        # Check if lambda body returns __import__
        if isinstance(node.body, ast.Name) and node.body.id == '__import__':
            raise ValueError('⚠️ Detected lambda function returning __import__')
        self.generic_visit(node)

    def visit_ListComp(self, node):
        '''Check list comprehensions that might contain __import__'''
        # Check if the expression in list comprehension is __import__
        if isinstance(node.elt, ast.Name) and node.elt.id == '__import__':
            raise ValueError('⚠️ Detected list comprehension containing __import__')
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        '''Check function definitions that might return __import__'''
        # Check if function returns __import__
        for stmt in node.body:
            if isinstance(stmt, ast.Return):
                if isinstance(stmt.value, ast.Name) and stmt.value.id == '__import__':
                    raise ValueError(f'⚠️ Detected function {node.name} returning __import__')
        self.generic_visit(node)

def compile_func(func_code: str, global_env: Optional[Dict[str, Any]] = None) -> Callable:
    fname = re.search(r'def\s+(\w+)\s*\(', func_code).group(1)
    module = ast.parse(func_code)
    SecurityVisitor().visit(module)
    func = compile(module, filename='<ast>', mode='exec')
    local_dict = {}
    exec(func, global_env if global_env is not None else local_dict, local_dict)
    return local_dict.pop(fname)

def obj2str(obj: Any) -> str:
    return base64.b64encode(pickle.dumps(obj)).decode('utf-8')

def str2obj(data: str) -> Any:
    return None if data is None else pickle.loads(base64.b64decode(data.encode('utf-8')))

def str2bool(v: str) -> bool:
    ''' Boolean type converter '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'on'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'off'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def dump_obj(f):
    @contextmanager
    def env_helper():
        os.environ['LAZYLLM_ON_CLOUDPICKLE'] = 'ON'
        yield
        os.environ['LAZYLLM_ON_CLOUDPICKLE'] = 'OFF'

    with env_helper():
        return None if f is None else base64.b64encode(cloudpickle.dumps(f)).decode('utf-8')

def load_obj(f):
    return cloudpickle.loads(base64.b64decode(f.encode('utf-8')))
