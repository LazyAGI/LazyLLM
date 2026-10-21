import base64
from pathlib import Path

import pytest

from lazyllm.components.utils.file_operate import bytes_to_file, infer_file_extension

magic = pytest.importorskip('magic')
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a8WQAAAAASUVORK5CYII=')
GIF = base64.b64decode('R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==')


@pytest.mark.parametrize('data,mime,suffix', [(PNG, 'image/png', '.png'), (GIF, 'image/gif', '.gif')])
def test_magic_extension_and_real_file(tmp_path, data, mime, suffix):
    assert magic.from_buffer(data, mime=True) == mime
    assert infer_file_extension(data) == suffix
    output = Path(bytes_to_file(data, str(tmp_path)))
    assert output.suffix == suffix
    assert output.read_bytes() == data


def test_unknown_magic_extension(tmp_path):
    data = bytes(256)
    assert magic.from_buffer(data, mime=True) == 'application/octet-stream'
    assert infer_file_extension(data) == '.bin'
    assert Path(bytes_to_file(data, str(tmp_path))).suffix == '.bin'


def test_magic_list_conversion(tmp_path):
    outputs = bytes_to_file([PNG, GIF], str(tmp_path))
    assert [Path(value).suffix for value in outputs] == ['.png', '.gif']
    assert [Path(value).read_bytes() for value in outputs] == [PNG, GIF]
