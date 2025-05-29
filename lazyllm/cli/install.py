import sys
import subprocess
import toml
import requests
import platform
import os
import argparse
import importlib.metadata

from collections import OrderedDict

_PYPROJECT_CACHE = None
PYPROJECT_TOML_URL = "https://raw.githubusercontent.com/LazyAGI/LazyLLM/main/pyproject.toml"
UNSUPPORTED_ON_DARWIN_WIN = [
    'full', 'standard', 'fintune-all', 'alpaca-lora', 'colie', 'llama-factory', 'deploy-all', 'vllm',
    'lmdeploy', 'lightllm', 'infinity'
]

def load_pyproject_from_lazyllm_path():
    try:
        import lazyllm
        lazyllm_path = lazyllm.__path__[0]  # Get the path of the lazyllm package
        pyproject_path = os.path.join(lazyllm_path, 'pyproject.toml')
        if os.path.exists(pyproject_path):
            with open(pyproject_path, 'r') as f:
                return toml.load(f)
        else:
            return None
    except (FileNotFoundError, toml.TomlDecodeError):
        print("Could not find or parse pyproject.toml in lazyllm path.")
        return None

def load_local_pyproject():
    try:
        with open('pyproject.toml', 'r') as f:
            return toml.load(f)
    except (FileNotFoundError, toml.TomlDecodeError):
        print("Could not find or parse the local pyproject.toml file.")
        sys.exit(1)

def load_remote_pyproject():
    try:
        response = requests.get(PYPROJECT_TOML_URL)
        response.raise_for_status()
        return toml.loads(response.text)
    except (requests.RequestException, toml.TomlDecodeError) as e:
        print(f"Failed to download or parse remote pyproject.toml file: {e}")
        sys.exit(1)

def load_pyproject():
    global _PYPROJECT_CACHE
    if _PYPROJECT_CACHE is not None:
        return _PYPROJECT_CACHE
    for loader in (load_pyproject_from_lazyllm_path, load_local_pyproject, load_remote_pyproject):
        cfg = loader()
        if cfg:
            _PYPROJECT_CACHE = cfg
            return cfg
    sys.exit(1)

def load_extras():
    config = load_pyproject()
    try:
        return config['tool']['poetry']['extras']
    except KeyError:
        print("No 'extras' information found in the pyproject.toml file.")
        sys.exit(1)

def load_dependencies():
    config = load_pyproject()
    try:
        return config['tool']['poetry']['dependencies']
    except KeyError:
        print("No 'dependencies' information found in the pyproject.toml file.")
        sys.exit(1)

def load_extras_descriptions():
    config = load_pyproject()
    try:
        return config['tool']['lazyllm']['extras_descriptions']
    except KeyError:
        print("No 'extras_descriptions' information found in the pyproject.toml file.")
        sys.exit(1)

def install_packages(packages):
    if isinstance(packages, str):
        packages = [packages]
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)
    except subprocess.CalledProcessError as e:
        print(f"安装失败: {e}")
        sys.exit(1)

def parse_caret_to_tilde_version(version):
    if version.startswith("^"):
        version_parts = version[1:].split(".")
        if len(version_parts) > 1:
            return f"~={version_parts[0]}.{version_parts[1]}"
        else:
            return f"~={version_parts[0]}"
    return version

def process_package(package_name_with_version, dependencies):
    if '==' in package_name_with_version:
        package_name, _ = package_name_with_version.split('==', 1)
        package_name = package_name.strip()
    else:
        package_name = package_name_with_version
    if package_name in dependencies:
        version_spec = dependencies[package_name]
        if isinstance(version_spec, dict):
            version_spec = version_spec.get('version', '')
        elif isinstance(version_spec, str):
            version_spec = version_spec.strip()
        if version_spec == '*' or version_spec == '':
            return package_name
        elif version_spec.startswith("^"):
            version_spec = parse_caret_to_tilde_version(version_spec)
        return f"{package_name}{version_spec}"
    else:
        print(f"Error: Package '{package_name}' is not listed in the 'dependencies' section of pyproject.toml.")
        sys.exit(1)

def install_multiple_packages(package_names_with_versions):
    dependencies = load_dependencies()
    packages_to_install = []
    for package in package_names_with_versions:
        package_with_version = process_package(package, dependencies)
        packages_to_install.append(package_with_version)
    install_packages(packages_to_install)

def install(commands):  # noqa C901
    extras_desc = load_extras_descriptions()
    epilog_lines = ["Supported extras groups:"]
    for name, desc in extras_desc.items():
        epilog_lines.append(f"  {name:<15}  {desc}")
    epilog = "\n".join(epilog_lines)

    parser = argparse.ArgumentParser(
        prog="lazyllm install",
        description="Install one or more extras groups or individual packages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog
    )
    parser.add_argument(
        "items",
        nargs="+",
        metavar="ITEM",
        help="Extras group(s) or package name(s) to install"
    )
    args = parser.parse_args(commands)
    items = args.items

    if platform.system() in ["Darwin", "Windows"] and \
       any(i in UNSUPPORTED_ON_DARWIN_WIN for i in items):
        print("Extras for finetune/local inference are not supported on macOS/Windows.")
        sys.exit(1)

    extras = load_extras()        # dict of extras
    deps = load_dependencies()  # dict of dependencies
    to_install = OrderedDict()

    for cmd in items:
        if cmd in extras:
            for pkg in extras[cmd]:
                spec = process_package(pkg, deps)
                to_install[spec] = None
        else:
            spec = process_package(cmd, deps)
            to_install[spec] = None

    if not to_install:
        print("No packages to install, please check your command.")
        sys.exit(1)

    pkgs = list(to_install.keys())
    filtered_pkgs = [p for p in pkgs if not p.startswith("flash-attn")]

    if filtered_pkgs:
        install_packages(filtered_pkgs)

    extra_pkgs = set()

    for p in pkgs:
        if p.startswith("flash-attn"):
            try:
                tc_ver = importlib.metadata.version("torch")
            except importlib.metadata.PackageNotFoundError:
                pass
            else:
                if tc_ver == "2.5.1":
                    extra_pkgs.add("flash-attn==2.7.0.post2")
                else:
                    extra_pkgs.add(p)

    if extra_pkgs:
        install_packages(list(extra_pkgs))
