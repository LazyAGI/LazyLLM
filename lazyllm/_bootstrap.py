import importlib
import toml
from pathlib import Path
from typing import List
from lazyllm.common import LOG

def _check_package_installed(package_name: str | List[str]) -> bool:
    if isinstance(package_name, list):
        for name in package_name:
            if importlib.util.find_spec(name) is None:
                return False
    else:
        if importlib.util.find_spec(package_name) is None:
            return False
    return True

def _load_toml_dep_group(group_name: str) -> List[str]:
    toml_file_path = Path(__file__).resolve().parents[2] / 'pyproject.toml'
    try:
        with open(toml_file_path, 'r') as f:
            return toml.load(f)['tool']['poetry']['extras'][group_name]
    except FileNotFoundError:
        LOG.error('pyproject.toml missing. Cannot extract required dependencies.')

def _check_dependency_by_group(group_name: str):
    if globals().get('_DEPS_INSTALLED_' + group_name, False):
        return

    missing_pack = []
    for name in _load_toml_dep_group(group_name):
        if not _check_package_installed(name):
            missing_pack.append(name)
    if len(missing_pack) > 0:
        LOG.error(f'Missing package(s): {missing_pack}\nYou can install them by:\n    lazyllm install {group_name}')
        exit(1)
    else:
        globals()['_DEPS_INSTALLED_' + group_name] = True
