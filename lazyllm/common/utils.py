from os import PathLike, makedirs
from os.path import expanduser, expandvars, isfile, join, normpath
from typing import Union
import re
import ast

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


def compile_code(code):
    fname = re.search(r'def\s+(\w+)\s*\(', code).group(1)
    module = ast.parse(code)
    code = compile(module, filename="<ast>", mode="exec")
    local_dict = {}
    exec(code, {}, local_dict)
    return local_dict[fname]
