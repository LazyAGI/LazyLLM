import os
import base64
import datetime
import tempfile
import re
from pathlib import Path
from typing import Optional, Tuple

from lazyllm import LOG

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
}

# Create reverse mapping for efficient MIME type to extension lookup
MIME_TO_EXT = {v: k for k, v in MIME_TYPE.items()}

IMAGE_MIME_TYPE = {k: v for k, v in MIME_TYPE.items() if v.startswith('image/')}
AUDIO_MIME_TYPE = {k: v for k, v in MIME_TYPE.items() if v.startswith('audio/')}

def delete_old_files(directory):
    now = datetime.datetime.now()
    for root, dirs, files in os.walk(directory):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                creation_time = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
                if (now - creation_time).days > 1:
                    os.remove(file_path)
                    LOG.info(f"Deleted: {file_path}")
            except Exception as e:
                LOG.error(f"Error deleting file {file_path}: {e}")
        for name in dirs:
            dir_path = os.path.join(root, name)
            try:
                creation_time = datetime.datetime.fromtimestamp(os.path.getctime(dir_path))
                if (now - creation_time).days > 1:
                    os.rmdir(dir_path)
                    LOG.info(f"Deleted: {dir_path}")
            except Exception as e:
                LOG.error(f"Error deleting directory {dir_path}: {e}")

def is_base64_with_mime(input_str: str):
    pattern = r'^data:([^;]+);base64,(.+)$'
    if isinstance(input_str, str) and re.match(pattern, input_str):
        return True
    return False

def split_base64_with_mime(input_str: str):
    """
    Split base64 string with MIME type

    Args:
        input_str: String in format 'data:{mime_type};base64,{base64_str}'

    Returns:
        Tuple of (base64_str, mime_type) or (input_str, None) if invalid format
    """
    pattern = r'^data:([^;]+);base64,(.+)$'
    if match := re.match(pattern, input_str):
        return match.group(2), match.group(1)
    return input_str, None


def file_to_base64(file_path: str, mime_types: dict) -> Optional[Tuple[str, Optional[str]]]:
    """
    Convert file to base64 string with MIME type

    Args:
        file_path: Path to the file
        mime_types: Dictionary of supported MIME types

    Returns:
        Tuple of (base64_str, mime_type) or None if error
    """
    try:
        with open(file_path, 'rb') as f:
            file_base64 = base64.b64encode(f.read()).decode('utf-8')
            ext = Path(file_path).suffix.lstrip('.')
            mime = mime_types.get(ext)
            return file_base64, mime
    except Exception as e:
        LOG.error(f"Error encoding file {file_path} to base64: {e}")
        return None


def image_to_base64(file_path: str) -> Optional[Tuple[str, Optional[str]]]:
    return file_to_base64(file_path, IMAGE_MIME_TYPE)


def audio_to_base64(file_path: str) -> Optional[Tuple[str, Optional[str]]]:
    return file_to_base64(file_path, AUDIO_MIME_TYPE)


def base64_to_file(base64_str: str, target_dir: Optional[str] = None) -> str:
    """
    Convert base64 string to file

    Args:
        base64_str: Base64 data URL string
        target_dir: Optional target directory

    Returns:
        Path to the created file

    Raises:
        ValueError: If base64 format is invalid or MIME type is unsupported
    """
    base64_data, mime_type = split_base64_with_mime(base64_str)

    if mime_type is None:
        raise ValueError("Invalid base64 format")

    if suffix := MIME_TO_EXT.get(mime_type):
        suffix = f'.{suffix}'
    else:
        raise ValueError(f"Unsupported MIME type: {mime_type}")

    target_dir = target_dir if target_dir and os.path.isdir(target_dir) else None

    file_path = tempfile.NamedTemporaryFile(prefix="base64_to_file_", suffix=suffix, dir=target_dir, delete=False).name

    with open(file_path, 'wb') as f:
        f.write(base64.b64decode(base64_data))

    os.chmod(file_path, 0o644)
    return file_path
