import re
import pkg_resources

from .dependencies.modelsconfig import models_config
from .dependencies.requirements import requirements

def model_map(name):
    match = re.search(r'(\d+)[bB]', name)
    size = int(match.group(1)) if match else 0
    return 'LLAMA_7B' if size <= 7 else 'LLAMA_20B' if size <= 20 else 'LLAMA_100B'

def get_model_name(name_or_path):
    return name_or_path.split('/')[-1]

def get_configs(name):
    if name in models_config: return models_config[name]
    return models_config.get(name.split('-')[0], dict())

def check_requirements(frame):
    packages = [line.strip() for line in requirements[frame].split('\n') if line.strip()]

    not_installed = []
    for package in packages:
        parts = package.split('==') if '==' in package else package.split('>=') if '>=' in package else [package]
        try:
            installed = pkg_resources.get_distribution(parts[0])
            if len(parts) > 1:
                if parts[1] not in installed.version:
                    not_installed.append(f"{package} (Installed: {installed.version}, Required: {parts[1]})")
        except pkg_resources.DistributionNotFound:
            not_installed.append(f"Required: {package}")
    return len(not_installed) == 0
