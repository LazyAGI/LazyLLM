from pathlib import Path

import pytest
import toml
from packaging.requirements import Requirement

from lazyllm.cli.install import process_package


@pytest.mark.parametrize('version,inside,outside', [
    ('^0.0.1', ['0.0.1'], ['0.0.0', '0.0.2', '0.1.0']),
    ('^0.0.11', ['0.0.11'], ['0.0.10', '0.0.12', '0.1.0']),
    ('^0.2.4', ['0.2.4', '0.2.9'], ['0.2.3', '0.3.0', '1.0.0']),
    ('^3.0.1', ['3.0.1', '3.9.0'], ['3.0.0', '4.0.0']),
    ('^0', ['0.0.0', '0.9.0'], ['1.0.0']),
    ('^0.0', ['0.0.0', '0.0.9'], ['0.1.0']),
    ('^1', ['1.0.0', '1.9.0'], ['0.9.0', '2.0.0']),
    ('^1.2.3rc1', ['1.2.3rc1', '1.2.3', '1.9.0'], ['1.2.3b1', '2.0.0']),
])
def test_caret_package_boundaries(version, inside, outside):
    requirement = Requirement(process_package('example', {'example': {'version': version}}))
    for value in inside:
        assert value in requirement.specifier
    for value in outside:
        assert value not in requirement.specifier


def test_declared_package_minimums():
    project = Path(__file__).resolve().parents[2] / 'pyproject.toml'
    dependencies = toml.load(project)['tool']['poetry']['dependencies']
    for name, value in dependencies.items():
        version = value.get('version', '') if isinstance(value, dict) else value
        if version.startswith('^'):
            requirement = Requirement(process_package(name, dependencies))
            assert version[1:] in requirement.specifier


@pytest.mark.parametrize('version', ['==1.2.3', '>=1.0,<2.0', '~=1.2'])
def test_non_caret_unchanged(version):
    assert process_package('example', {'example': version}) == 'example' + version
