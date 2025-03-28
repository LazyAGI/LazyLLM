import sys
import subprocess
import toml
import requests
import platform
import os

PYPROJECT_TOML_URL = "https://raw.githubusercontent.com/LazyAGI/LazyLLM/main/pyproject.toml"

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

def load_dependencies():
    config = load_pyproject()
    try:
        return config['tool']['poetry']['dependencies']
    except KeyError:
        print("No 'dependencies' information found in the pyproject.toml file.")
        sys.exit(1)

def install_packages(packages):
    if isinstance(packages, str):
        packages = [packages]
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)
    except subprocess.CalledProcessError as e:
        print(f"安装失败: {e}")
        sys.exit(1)

def install_full():
    packages = load_packages()
    install_multiple_packages(packages['full'])
    install_packages(["flash-attn==2.7.0.post2", "transformers==4.46.1"])

def install_standard():
    packages = load_packages()
    install_multiple_packages(packages['standard'])
    install_packages("transformers==4.46.1")

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

def install(commands):
    if not commands:
        print("Usage: lazyllm install [full|standard|package_name]")
        sys.exit(1)

    if platform.system() == "Darwin":
        if any(command == "full" or command == "standard" for command in commands):
            print("Installation of 'full' or 'standard' packages is not supported on macOS.")
            sys.exit(1)

    if len(commands) == 1:
        command = commands[0]
        if command == "full":
            install_full()
        elif command == "standard":
            install_standard()
        else:
            install_multiple_packages([command])
    else:
        install_multiple_packages(commands)
