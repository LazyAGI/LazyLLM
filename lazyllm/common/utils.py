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
    # normalize and expand a path
    path = normpath(expandvars(expanduser(path)))
    if exist and file and not isfile(path):
        raise FileNotFoundError(path)
    else:
        if file:
            dir_path = normpath(join(path, ".."))
        else:
            dir_path = path
        if parents:
            makedirs(dir_path, exist_ok=True)
    return path

class SecurityVisitor(ast.NodeVisitor):

    # **Dangerous built-in functions**
    DANGEROUS_BUILTINS = {"exec", "eval", "open", "compile", "getattr", "setattr"}

    # **Dangerous os operations**
    DANGEROUS_OS_CALLS = {"system", "popen", "remove", "rmdir", "unlink", "rename"}

    # **Dangerous sys operations**
    DANGEROUS_SYS_CALLS = {"exit", "modules"}

    # **Dangerous modules**
    DANGEROUS_MODULES = {"pickle", "subprocess", "socket", "shutil", "requests", "inspect", "tempfile"}

    def visit_Call(self, node):
        # Direct calls to dangerous built-in functions
        if isinstance(node.func, ast.Name) and node.func.id in self.DANGEROUS_BUILTINS:
            raise ValueError(f"⚠️ Detected dangerous function call: {node.func.id}")

        # os / sys related calls
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "os" and node.func.attr in self.DANGEROUS_OS_CALLS:
                raise ValueError(f"⚠️ Detected dangerous os call: os.{node.func.attr}")
            if node.func.value.id == "sys" and node.func.attr in self.DANGEROUS_SYS_CALLS:
                raise ValueError(f"⚠️ Detected dangerous sys call: sys.{node.func.attr}")

        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name in self.DANGEROUS_MODULES:
                raise ValueError(f"⚠️ Detected dangerous module import: {alias.name}")

    def visit_ImportFrom(self, node):
        if node.module in self.DANGEROUS_MODULES:
            raise ValueError(f"⚠️ Detected dangerous module import: {node.module}")

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            if node.value.id == "os" and node.attr == "environ":
                raise ValueError("⚠️ Detected dangerous access: os.environ")
            if node.value.id == "tempfile":
                raise ValueError(f"⚠️ Detected dangerous usage of tempfile: tempfile.{node.attr}")

        self.generic_visit(node)

def compile_func(func_code: str, global_env: Optional[Dict[str, Any]] = None) -> Callable:
    """
将一段 python 函数字符串编译成一个可执行函数并返回。

Args:
    func_code (str): 包含 python 函数代码的字符串
    global_env (str): 在 python 函数中用到的包和全局变量


Examples:

    from lazyllm.common import compile_func
    code_str = 'def Identity(v): return v'
    identity = compile_func(code_str)
    assert identity('hello') == 'hello'
    """
    fname = re.search(r'def\s+(\w+)\s*\(', func_code).group(1)
    module = ast.parse(func_code)
    SecurityVisitor().visit(module)
    func = compile(module, filename="<ast>", mode="exec")
    local_dict = {}
    exec(func, global_env if global_env is not None else local_dict, local_dict)
    return local_dict.pop(fname)

def obj2str(obj: Any) -> str:
    return base64.b64encode(pickle.dumps(obj)).decode('utf-8')

def str2obj(data: str) -> Any:
    return None if data is None else pickle.loads(base64.b64decode(data.encode('utf-8')))

def str2bool(v: str) -> bool:
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
