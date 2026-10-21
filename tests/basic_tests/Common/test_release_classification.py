import os
from pathlib import Path
import subprocess

import pytest
import yaml


def workflow():
    source = Path(__file__).resolve().parents[3] / '.github/workflows/publish_release.yml'
    return yaml.safe_load(source.read_text(encoding='utf-8'))


@pytest.mark.parametrize('tag,expected', [
    ('1.4.0a0', True), ('1.4.0alpha1', True), ('1.4.0b1', True),
    ('1.4.0rc1', True), ('1.4.0.dev1', True),
    ('1.4.0', False), ('1.4.0.post1', False), ('0.7.5', False),
])
def test_actual_release_classification_step(tmp_path, tag, expected):
    jobs = workflow()['jobs']
    step = next((step for step in jobs['check-exists']['steps'] if step.get('id') == 'release_kind'), None)
    if step is None:
        release = next(step for step in jobs['check-release']['steps'] if step.get('id') == 'create_release')
        assert release['with']['prerelease'] == expected
        return
    output = tmp_path / 'output'
    env = dict(os.environ, RELEASE_VERSION=tag, GITHUB_OUTPUT=str(output), PIP_NO_INDEX='1',
               PIP_DISABLE_PIP_VERSION_CHECK='1')
    subprocess.run(['bash', '-e', '-c', step['run']], env=env, check=True, capture_output=True)
    assert output.read_text(encoding='utf-8').strip() == f'is_prerelease={str(expected).lower()}'


def test_release_input_uses_classification_job_output():
    jobs = workflow()['jobs']
    assert jobs['check-exists']['outputs']['is_prerelease'] == '${{ steps.release_kind.outputs.is_prerelease }}'
    assert 'check-exists' in jobs['check-release']['needs']
    step = next(step for step in jobs['check-release']['steps'] if step.get('id') == 'create_release')
    assert step['with']['prerelease'] == '${{ needs.check-exists.outputs.is_prerelease }}'
