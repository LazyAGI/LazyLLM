import pytest

from lazyllm.tools.fs.client import _FSRouter


@pytest.mark.parametrize('path', ['C:/skills/visible/SKILL.md', r'C:\skills\visible\SKILL.md'])
def test_windows_drive_paths_are_local(path):
    assert _FSRouter()._parse(path) == ('file', None, path)


def test_cloud_path_retains_its_provider():
    assert _FSRouter()._parse('feishu@space:/folder') == ('feishu', 'space', '/folder')
