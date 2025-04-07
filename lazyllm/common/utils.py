from os import PathLike, makedirs
from os.path import expanduser, expandvars, isfile, join, normpath
from typing import Union, Dict, Callable, Any, Optional
import re
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
    """
    Check path and return corrected path.
    """
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


def compile_func(func_code: str, global_env: Optional[Dict[str, Any]] = None) -> Callable:
    fname = re.search(r'def\s+(\w+)\s*\(', func_code).group(1)
    module = ast.parse(func_code)
    func = compile(module, filename="<ast>", mode="exec")
    local_dict = {}
    exec(func, global_env, local_dict)
    return local_dict[fname]

def obj2str(obj: Any) -> str:
    return base64.b64encode(pickle.dumps(obj)).decode('utf-8')

def str2obj(data: str) -> Any:
    return None if data is None else pickle.loads(base64.b64decode(data.encode('utf-8')))

def str2bool(v: str) -> bool:
    """ Boolean type converter """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'on'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'off'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
