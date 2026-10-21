import base64

import pytest

from lazyllm.components.utils.file_operate import _audio_to_base64, _image_to_base64, ocr_to_base64


@pytest.mark.parametrize('suffix,convert,mime,data', [
    ('png', _image_to_base64, 'image/png', b'\x89PNG\r\n\x1a\n'),
    ('PNG', _image_to_base64, 'image/png', b'\x89PNG\r\n\x1a\n'),
    ('jpg', _image_to_base64, 'image/jpeg', b'\xff\xd8\xff\xd9'),
    ('JpG', _image_to_base64, 'image/jpeg', b'\xff\xd8\xff\xd9'),
    ('wav', _audio_to_base64, 'audio/wav', b'RIFF\x00\x00\x00\x00WAVE'),
    ('WAV', _audio_to_base64, 'audio/wav', b'RIFF\x00\x00\x00\x00WAVE'),
    ('pdf', ocr_to_base64, 'application/pdf', b'%PDF-1.4'),
    ('PDF', ocr_to_base64, 'application/pdf', b'%PDF-1.4'),
])
def test_case_insensitive_file_mime(tmp_path, suffix, convert, mime, data):
    path = tmp_path / ('sample.' + suffix)
    path.write_bytes(data)
    encoded, actual_mime = convert(str(path))
    assert base64.b64decode(encoded) == data
    assert actual_mime == mime
