import toml
import re
import os

def load_toml():
    with open('../pyproject.toml', 'r') as f:
        config = toml.load(f)
    all_dependencies = config['tool']['poetry']['dependencies']
    light = {}
    full = {}
    for package_name in all_dependencies:
        if isinstance(all_dependencies[package_name], dict):
            full[package_name] = all_dependencies[package_name].get('version', '')
        elif isinstance(all_dependencies[package_name], str):
            version = all_dependencies[package_name]
            full[package_name] = version
            light[package_name] = version
    del full['python']
    del light['python']
    return full, light

def parse_requirement(line):
    match = re.match(r'^([a-zA-Z0-9_-]+)([><=~!^*]+.*)?$', line.strip())
    if match:
        package_name = match.group(1)
        version = match.group(2) if match.group(2) else '*'
        return package_name, version
    return None, None

def compare_versions(version_spec, req_version):
    if version_spec.startswith('^') and req_version == '*':
        return True
    return version_spec == req_version

def check_requirements(level, requirements_file):
    mismatched = []
    missing = []
    with open(requirements_file, 'r') as file:
        requirements = file.readlines()

    req_dict = {}
    for req_line in requirements:
        req_name, req_version = parse_requirement(req_line)
        if req_name:
            req_dict[req_name] = req_version

    for package_name, version_spec in level.items():
        if package_name in req_dict:
            if not compare_versions(version_spec, req_dict[package_name]):
                mismatched.append(
                    f"{package_name}: toml version {version_spec} does not match "
                    f"requirements {req_dict[package_name]}"
                )
        else:
            missing.append(f"{package_name} is missing from requirements")

    if missing or mismatched:
        if missing:
            print("Missing packages:")
            for msg in missing:
                print(msg)
        if mismatched:
            print("Mismatched packages:")
            for msg in mismatched:
                print(msg)
        raise ValueError("There are missing or mismatched packages.")
    else:
        print("All packages matched successfully.")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    full, light = load_toml()
    print('=' * 30, 'full', '=' * 30)
    check_requirements(full, '../requirements.full.txt')
    print('=' * 30, 'light', '=' * 30)
    check_requirements(light, '../requirements.txt')
