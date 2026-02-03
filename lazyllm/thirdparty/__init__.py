import importlib
import toml
import re
import os
import lazyllm
from typing import List, Any
from lazyllm.common import LOG
from lazyllm.configs import config
from .modules import modules
from pathlib import Path
from functools import lru_cache

package_name_map = {
    'huggingface_hub': 'huggingface-hub',
    'jwt': 'PyJWT',
    'rank_bm25': 'rank-bm25',
    'faiss': 'faiss-cpu',
    'flash_attn': 'flash-attn',
    'sklearn': 'scikit-learn',
    'volcenginesdkarkruntime': 'volcengine-python-sdk[ark]',
    'opensearchpy': 'opensearch-py',
    'memu': 'memu-py',
    'mem0': 'mem0ai',
    'pptx': 'python-pptx',
    'docx': 'python-docx',
    'bs4': 'beautifulsoup4',
    'Stemmer': 'pystemmer',
    'psycopg2': 'psycopg2-binary',
    'yaml': 'pyyaml',
}

package_name_map_reverse = {v: k for k, v in package_name_map.items()}
module_names = [m[0] if isinstance(m, list) else m for m in modules]

requirements = {}

def get_pip_install_cmd(packages_to_install: List[str]):
    assert len(packages_to_install) > 0
    if len(requirements) == 0:
        prepare_requirements_dict()
    install_parts = []
    for name in packages_to_install:
        if name in package_name_map:
            name = package_name_map[name]
        install_parts.append(name + requirements.get(name, ''))
    if len(install_parts) > 0:
        return 'pip install ' + ', '.join(install_parts)
    return None

def split_package_version(s: str, pattern: re.Pattern):
    m = pattern.match(s)
    if not m:
        raise ValueError(f'Invalid package version format: {s}')
    name = m.group('name')
    version = m.group('version') or ''
    return name, version

def load_toml_dict() -> dict[str, Any]:
    toml_file_path = Path(__file__).resolve().parents[2] / 'pyproject.toml'
    if not toml_file_path.exists():
        toml_file_path = os.path.join(lazyllm.__path__[0], 'pyproject.toml')

    try:
        with open(toml_file_path, 'r') as f:
            return toml.load(f)
    except FileNotFoundError:
        LOG.error('pyproject.toml is missing. Please reinstall LazyLLM.')
        raise FileNotFoundError('pyproject.toml is missing. Please reinstall LazyLLM.')

def prepare_requirements_dict():
    toml_config = load_toml_dict()

    pattern = re.compile(r'''
        ^\s*
        (?P<name>[A-Za-z0-9_.-]+)
        \s*
        (?P<version>(==|<=|>=|<|>).*?)?
        \s*$
    ''', re.VERBOSE)

    required_dependencies = toml_config['project']['dependencies']
    for dep in required_dependencies:
        name, version = split_package_version(dep, pattern)
        requirements[name] = version

    optional_dependencies = toml_config['tool']['poetry']['dependencies']
    for name, spec in optional_dependencies.items():
        version = spec.get('version', '')
        if version == '*':
            version = ''
        requirements[name] = version

class PackageWrapper(object):
    def __init__(self, key, *sub_package, package=None, register_patches=None) -> None:
        self._Wrapper__key = key
        self._Wrapper__package = package
        self._Wrapper__sub_packages = sorted(sub_package, reverse=True)
        self._Wrapper__patches = []
        self._Wrapper__lib = None
        if register_patches: self.register_patches(register_patches)

    def register_patches(self, patch_func):
        if isinstance(patch_func, list):
            self._Wrapper__patches.extend(patch_func)
        else:
            self._Wrapper__patches.append(patch_func)

    def __getattribute__(self, __name):
        if __name in ('_Wrapper__key', '_Wrapper__package', '_Wrapper__patches',
                      '_Wrapper__lib', '_Wrapper__sub_packages', 'register_patches'):
            return super(__class__, self).__getattribute__(__name)
        matched_subpackages = []
        for sub_package in self._Wrapper__sub_packages:
            if __name == sub_package.split('.')[0]:
                matched_subpackages.append(sub_package[len(__name) + 1:])
        if matched_subpackages:
            return PackageWrapper(f'{self._Wrapper__key}.{__name}', *matched_subpackages,
                                  register_patches=self._Wrapper__patches)
        if self._Wrapper__lib is None:
            try:
                self._Wrapper__lib = importlib.import_module(self._Wrapper__key, package=self._Wrapper__package)
                for patch_func in self._Wrapper__patches: patch_func()
            except ImportError:
                pip_cmd = get_pip_install_cmd([self._Wrapper__key])
                err_msg = f'Cannot import module `{self._Wrapper__key}`, please install it by `{pip_cmd}`'
                raise ImportError(err_msg) from None
        return getattr(self._Wrapper__lib, __name)

    def __setattr__(self, __name, __value):
        if __name in ('_Wrapper__key', '_Wrapper__package', '_Wrapper__patches',
                      '_Wrapper__lib', '_Wrapper__sub_packages'):
            return super(__class__, self).__setattr__(__name, __value)
        setattr(importlib.import_module(
            self._Wrapper__key, package=self._Wrapper__package), __name, __value)

for m in modules:
    if isinstance(m, str):
        vars()[m] = PackageWrapper(m)
    else:
        vars()[m[0]] = PackageWrapper(m[0], *m[1:])

def check_packages(names):
    assert isinstance(names, list)
    missing_pack = []
    for name in names:
        if not check_package_installed(name):
            missing_pack.append(name)
    if len(missing_pack) > 0:
        cmd = get_pip_install_cmd(missing_pack)
        LOG.warning(f'Some packages are not found, please install it by \'{cmd}\'')

def check_package_installed(package_name: str | List[str]) -> bool:
    if isinstance(package_name, list):
        for name in package_name:
            if importlib.util.find_spec(name) is None:
                return False
    else:
        if importlib.util.find_spec(package_name) is None:
            return False
    return True

@lru_cache
def load_toml_dep_group(group_name: str) -> List[str]:
    try:
        toml_config = load_toml_dict()
        return toml_config['tool']['poetry']['extras'][group_name]
    except KeyError:
        LOG.error(f'Group {group_name} not found in pyproject.toml.')
        raise KeyError(f'''Group {group_name} not found in pyproject.toml.
You cloud report issue to https://github.com/LazyAGI/LazyLLM in case specific deps group needed.''')

@lru_cache
def check_dependency_by_group(group_name: str):
    missing_pack = []
    for name in load_toml_dep_group(group_name):
        real_name = package_name_map_reverse.get(name, name)
        if not (config['init_doc'] and real_name in module_names or check_package_installed(real_name)):
            missing_pack.append(name)
    if len(missing_pack) > 0:
        msg = f'Missing package(s): {missing_pack}\nYou can install them by:\n    lazyllm install {group_name}'
        LOG.error(msg)
        raise ImportError(msg)
    else:
        return True
