import sys
import subprocess
import toml

def load_packages():
    with open('pyproject.toml', 'r') as f:
        config = toml.load(f)
    return config['tool']['poetry']['extras']

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

    if command == "full":
        install_full()
    elif command == "standard":
        install_standard()
    else:
        install_single_package(command)

if __name__ == "__main__":
    main()
