'''
Replace [tool.poetry.dependencies] to [project.optional-dependencies]
For poetry install ; poetry lock
'''
from pathlib import Path
import tomlkit

def expand_caret(version: str) -> str:
    '''
    Convert ^x.y.z into >=x.y.z,<upper-bound
    according to PEP 440 compatible semantics.
    '''
    if not version.startswith('^'):
        return version

    raw = version[1:]
    expanded = f'>={raw},<'
    parts = raw.split('.')

    add_one = False
    for i in range(len(parts)):
        try:
            n = int(parts[i])
            if not add_one and n != 0:
                add_one = True
                n += 1
            parts[i] = str(n)
        except ValueError:
            continue

    if not add_one:
        raise ValueError(f'Invalid version string: {version}')

    return expanded + '.'.join(parts)


def normalize_version(version: str) -> str:
    version = version.strip()

    if version == '*':
        return ''
    elif version.startswith('^'):
        return expand_caret(version)

    return version


def main():
    path = Path('pyproject.toml')
    if not path.exists():
        raise FileNotFoundError('pyproject.toml not found')

    doc = tomlkit.parse(path.read_text())

    project = doc.get('project')
    if project is None:
        project = tomlkit.table()
        doc['project'] = project
    project['requires-python'] = '>=3.10,<3.13'

    poetry_optional_deps = doc['tool']['poetry']['dependencies']

    proj_optional_deps = []

    for name, spec in poetry_optional_deps.items():
        version = normalize_version(spec['version'])
        proj_optional_deps.append(f'{name}{version}')

    project = doc['project']
    project['optional-dependencies'] = tomlkit.table()

    arr = tomlkit.array()
    arr.multiline(True)
    for dep in sorted(proj_optional_deps):
        arr.append(dep)

    project = doc['project']
    project['optional-dependencies'] = tomlkit.table()
    project['optional-dependencies']['optional'] = arr

    # remove poetry sections
    doc['tool'].pop('poetry', None)

    path.write_text(tomlkit.dumps(doc))
    print(f'Migrated {len(proj_optional_deps)} deps with caret expanded to PEP 621 compatible ranges')


if __name__ == '__main__':
    main()
