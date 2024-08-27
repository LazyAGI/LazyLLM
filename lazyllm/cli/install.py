import sys
import subprocess
import toml
import requests
import platform
import os
import lazyllm

PYPROJECT_TOML_URL = "https://raw.githubusercontent.com/LazyAGI/LazyLLM/main/pyproject.toml"

def load_pyproject_from_lazyllm_path():
    try:
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
    config = load_pyproject_from_lazyllm_path()
    if config is not None:
        return config
    config = load_local_pyproject()
    if config is not None:
        return config
    return load_remote_pyproject()

def load_packages():
    config = load_pyproject()
    try:
        return config['tool']['poetry']['extras']
    except KeyError:
        print("No 'extras' information found in the pyproject.toml file.")
        sys.exit(1)

def install_packages(packages):
    if isinstance(packages, str):
        packages = [packages]
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)
    except subprocess.CalledProcessError as e:
        print(f"安装失败: {e}")
        sys.exit(1)

def install_additional_packages(package_list):
    for package in package_list:
        install_packages(package)

def install_full():
    packages = load_packages()
    install_packages(packages['full'])
    additional_packages = [
        "git+https://github.com/hiyouga/LLaMA-Factory.git@9dcff3a#egg=llamafactory",
        "git+https://github.com/ModelTC/lightllm@e6452fd#egg=lightllm",
        "flash-attn>=2.5.8"
    ]
    install_additional_packages(additional_packages)

def install_standard():
    packages = load_packages()
    install_packages(packages['standard'])
    additional_packages = [
        "git+https://github.com/hiyouga/LLaMA-Factory.git@9dcff3a#egg=llamafactory"
    ]
    install_additional_packages(additional_packages)

def install_single_package(package_name):
    install_packages(package_name)

def main():
    if len(sys.argv) < 3 or sys.argv[1] != "install":
        print("Usage: lazyllm install [full|standard|package_name]")
        sys.exit(1)

    command = sys.argv[2]

    if platform.system() == "Darwin":
        if command in ["full", "standard"]:
            print("Installation of 'full' or 'standard' packages is not supported on macOS.")
            sys.exit(1)

    if command == "full":
        install_full()
    elif command == "standard":
        install_standard()
    else:
        install_single_package(command)

if __name__ == "__main__":
    main()
