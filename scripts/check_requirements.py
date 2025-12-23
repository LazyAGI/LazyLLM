import toml
import re
import os

pattern = re.compile(r'''
    ^\s*
    (?P<name>[A-Za-z0-9_.-]+)
    \s*
    (?P<version>(==|<=|>=|<|>).*?)?
    \s*$
''', re.VERBOSE)

def split_package_version(s: str):
    m = pattern.match(s)
    if not m:
        raise ValueError(f'Invalid package version format: {s}')
    name = m.group('name')
    version = m.group('version') or ''
    return name, version

def load_toml():
    with open('../pyproject.toml', 'r') as f:
        config = toml.load(f)
    required_dependencies = config['project']['dependencies']
    deps_dict = {}
    for dep in required_dependencies:
        name, version = split_package_version(dep)
        deps_dict[name] = version

    return deps_dict

def version_match_or_not(source_version, target_version):
    if target_version == '':
        return True
    return source_version == target_version

def check_requirements(requirements_from_toml, requirements_file):
    mismatched = []
    missing = []
    with open(requirements_file, 'r') as file:
        requirements_from_txt = file.readlines()

    req_dict = {}
    for req_line in requirements_from_txt:
        name, version = split_package_version(req_line)
        req_dict[name] = version

    for package_name, version_spec in requirements_from_toml.items():
        if package_name in req_dict:
            if not version_match_or_not(version_spec, req_dict[package_name]):
                mismatched.append(
                    f'{package_name:<25} toml version: {version_spec:<25}'
                    f'requirements version: \'{req_dict[package_name]}\' does not match'
                )
        else:
            missing.append(f'{package_name} is missing from requirements')

    if missing or mismatched:
        if missing:
            print('Missing packages:')
            for msg in missing:
                print(msg)
        if mismatched:
            print('Mismatched packages:')
            for msg in mismatched:
                print(msg)
        raise ValueError('There are missing or mismatched packages.')
    else:
        print('All packages matched successfully.')


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print('=' * 30, 'light: requirements.txt', '=' * 30)
    check_requirements(load_toml(), '../requirements.txt')
