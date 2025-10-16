import os
import base64
import datetime
import tempfile
import re
from pathlib import Path
from typing import Optional, Tuple, Union

from lazyllm import LOG, config

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

MIME_TYPE = {
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'jfif': 'image/jpeg',
    'jpe': 'image/jpeg',
    'png': 'image/png',
    'apng': 'image/png',
    'gif': 'image/gif',
    'bmp': 'image/bmp',
    'dib': 'image/bmp',
    'tif': 'image/tiff',
    'tiff': 'image/tiff',
    'webp': 'image/webp',
    'ico': 'image/x-icon',
    'icns': 'image/icns',
    'mp3': 'audio/mpeg',
    'wav': 'audio/wav',
    'flac': 'audio/flac',
    'aac': 'audio/aac',
    'ogg': 'audio/ogg',
    'm4a': 'audio/mp4',
    'wma': 'audio/x-ms-wma',
    'mp4': 'video/mp4',
    'avi': 'video/avi',
    'mov': 'video/mov',
    'pdf': 'application/pdf',
    'txt': 'text/plain',
    'html': 'text/html',
    'json': 'application/json',
    'xml': 'application/xml'
}

# Create reverse mapping for efficient MIME type to extension lookup
# if multiple extensions have the same MIME type, keep the first one
MIME_TO_EXT = {v: k for k, v in reversed(MIME_TYPE.items())}

IMAGE_MIME_TYPE = {k: v for k, v in MIME_TYPE.items() if v.startswith('image/')}
AUDIO_MIME_TYPE = {k: v for k, v in MIME_TYPE.items() if v.startswith('audio/')}
OCR_MIME_TYPE = {k: v for k, v in MIME_TYPE.items() if k in ['pdf', 'jpg', 'jpeg', 'png']}

def _delete_old_files(directory):
    now = datetime.datetime.now()
    for root, dirs, files in os.walk(directory):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                creation_time = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
                if (now - creation_time).days > 1:
                    os.remove(file_path)
                    LOG.info(f'Deleted: {file_path}')
            except Exception as e:
                LOG.error(f'Error deleting file {file_path}: {e}')
        for name in dirs:
            dir_path = os.path.join(root, name)
            try:
                creation_time = datetime.datetime.fromtimestamp(os.path.getctime(dir_path))
                if (now - creation_time).days > 1:
                    os.rmdir(dir_path)
                    LOG.info(f'Deleted: {dir_path}')
            except Exception as e:
                LOG.error(f'Error deleting directory {dir_path}: {e}')

def _is_base64_with_mime(input_str: str):
    pattern = r'^data:([^;]+);base64,(.+)$'
    if isinstance(input_str, str) and re.match(pattern, input_str):
        return True
    return False

def _split_base64_with_mime(input_str: str):
    '''
    Split base64 string with MIME type

    Args:
        input_str: String in format 'data:{mime_type};base64,{base64_str}'

    Returns:
        Tuple of (base64_str, mime_type) or (input_str, None) if invalid format
    '''
    pattern = r'^data:([^;]+);base64,(.+)$'
    if match := re.match(pattern, input_str):
        return match.group(2), match.group(1)
    return input_str, None


def _file_to_base64(file_path: str, mime_types: dict) -> Optional[Tuple[str, Optional[str]]]:
    '''
    Convert file to base64 string with MIME type

    Args:
        file_path: Path to the file
        mime_types: Dictionary of supported MIME types

    Returns:
        Tuple of (base64_str, mime_type) or None if error
    '''
    try:
        with open(file_path, 'rb') as f:
            file_base64 = base64.b64encode(f.read()).decode('utf-8')
            ext = Path(file_path).suffix.lstrip('.')
            mime = mime_types.get(ext)
            return file_base64, mime
    except Exception as e:
        LOG.error(f'Error encoding file {file_path} to base64: {e}')
        return None


def _image_to_base64(file_path: str) -> Optional[Tuple[str, Optional[str]]]:
    return _file_to_base64(file_path, IMAGE_MIME_TYPE)


def _audio_to_base64(file_path: str) -> Optional[Tuple[str, Optional[str]]]:
    return _file_to_base64(file_path, AUDIO_MIME_TYPE)

def ocr_to_base64(file_path: str) -> Optional[Tuple[str, Optional[str]]]:
    return _file_to_base64(file_path, OCR_MIME_TYPE)


def _base64_to_file(base64_str: Union[str, list[str]], target_dir: Optional[str] = None) -> Union[str, list[str]]:
    '''
    Convert base64 string to file

    Args:
        base64_str: Base64 data URL string or list of base64 strings
        target_dir: Optional target directory

    Returns:
        Path to the created file

    Raises:
        ValueError: If base64 format is invalid or MIME type is unsupported
    '''
    if isinstance(base64_str, list):
        return [_base64_to_file(item, target_dir) for item in base64_str]
    base64_data, mime_type = _split_base64_with_mime(base64_str)

    if mime_type is None:
        raise ValueError('Invalid base64 format')

    if suffix := MIME_TO_EXT.get(mime_type):
        suffix = f'.{suffix}'
    else:
        raise ValueError(f'Unsupported MIME type: {mime_type}')

    target_dir = target_dir if target_dir and os.path.isdir(target_dir) else config['temp_dir']
    os.makedirs(target_dir, exist_ok=True)

    file_path = tempfile.NamedTemporaryFile(prefix='base64_to_file_', suffix=suffix, dir=target_dir, delete=False).name

    with open(file_path, 'wb') as f:
        f.write(base64.b64decode(base64_data))

    os.chmod(file_path, 0o644)
    return file_path

def infer_file_extension(data: bytes) -> str:
    if MAGIC_AVAILABLE:
        try:
            file_type = magic.from_buffer(data, mime=True)
            return MIME_TO_EXT.get(file_type, '.bin')
        except Exception:
            LOG.warning('Magic detection failed, using simple magic detection')
    return simple_magic_detection(data)

def simple_magic_detection(data: bytes) -> str:
    if len(data) < 4: return '.bin'
    if data.startswith(b'\x89PNG\r\n\x1a\n'): return '.png'
    if data.startswith(b'\xff\xd8'): return '.jpg'
    if data.startswith((b'GIF87a', b'GIF89a')): return '.gif'
    if data.startswith(b'RIFF') and len(data) >= 12 and data[8:12] == b'WAVE': return '.wav'
    if (data.startswith(b'ID3') or data.startswith(b'\xff\xfb') or data.startswith(b'\xff\xf3')): return '.mp3'
    if data.startswith(b'RIFF') and len(data) >= 12 and data[8:12] == b'WEBP': return '.webp'
    return '.bin'

def bytes_to_file(bytes_str: Union[bytes, list[bytes]], target_dir: Optional[str] = None) -> Union[str, list[str]]:
    '''
    Convert byte string to file
    '''
    assert isinstance(bytes_str, (bytes, list)), 'bytes_str must be a bytes or list of bytes'
    if isinstance(bytes_str, list):
        return [bytes_to_file(item, target_dir) for item in bytes_str]
    elif isinstance(bytes_str, bytes):
        output_dir = target_dir if target_dir and os.path.isdir(target_dir) else config['temp_dir']
        os.makedirs(output_dir, exist_ok=True)
        file_extension = infer_file_extension(bytes_str)
        temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix=file_extension, delete=False, dir=output_dir)
        temp_file.write(bytes_str)
        temp_file.close()

        os.chmod(temp_file.name, 0o644)
        return temp_file.name
