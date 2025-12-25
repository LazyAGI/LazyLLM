import importlib.metadata
from typing import List
from lazyllm.common import LOG

def check_package_installed(package_name: str | List[str]) -> bool:
    try:
        if isinstance(package_name, list):
            for name in package_name:
                importlib.metadata.version(name)
        else:
            importlib.metadata.version(package_name)
        return True
    except:
        return False

def warn_missing_packages(package_name: List[str]):
    missing_pack = []
    for name in package_name:
        if not check_package_installed(name):
            missing_pack.append(name)
    if len(missing_pack) > 0:
        LOG.error(f"Missing packages: {missing_pack}")
        exit(1)
