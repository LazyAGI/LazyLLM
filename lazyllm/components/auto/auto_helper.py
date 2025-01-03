import re
import pkg_resources
import functools
from packaging import version

from .dependencies.modelsconfig import models_config
from lazyllm import LOG

def model_map(name):
    match = re.search(r'(\d+)[bB]', name)
    size = int(match.group(1)) if match else 0
    return 'LLAMA_7B' if size <= 7 else 'LLAMA_20B' if size <= 20 else 'LLAMA_70B' if size <= 70 else 'LLAMA_100B'

def get_model_name(name_or_path):
    return name_or_path.split('/')[-1]

def get_configs(name):
    if name in models_config: return models_config[name]
    return models_config.get(name.split('-')[0], dict())

def compare_versions(version1, version2):
    v1 = version.parse(version1)
    v2 = version.parse(version2)
    if v1 > v2:
        return 1
    elif v1 < v2:
        return -1
    else:
        return 0

@functools.lru_cache
def check_requirements(requirements):
    packages = [line.strip() for line in requirements.split('\n') if line.strip()]

    not_installed = []
    for package in packages:
        parts = package.split('==') if '==' in package else package.split('>=') if '>=' in package else [package]
        try:
            try:
                installed = pkg_resources.get_distribution(parts[0])
            except pkg_resources.DistributionNotFound:
                installed = pkg_resources.get_distribution('lazyllm-' + parts[0])
            if len(parts) > 1:
                # if parts[1] not in installed.version:
                if compare_versions(installed.version, parts[1]) == -1:
                    not_installed.append(f"{package} (Installed: {installed.version}, Required: {parts[1]})")
        except pkg_resources.DistributionNotFound:
            not_installed.append(f"Required: {package}")
    if len(not_installed) != 0:
        LOG.warning(f"Because of missing packages, the model may not run. The required packages are: {not_installed}")
    return len(not_installed) == 0
