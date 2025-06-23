import os
import base64
import datetime
import tempfile
import re

from lazyllm import LOG

IMAGE_MIME_TYPE = {
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
    'icns': 'image/icns'
}
AUDIO_MIME_TYPE = {
    'mp3': 'audio/mpeg',
    'wav': 'audio/wav',
    'flac': 'audio/flac',
    'aac': 'audio/aac',
    'ogg': 'audio/ogg',
    'm4a': 'audio/mp4',
    'wma': 'audio/x-ms-wma',
}

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
    if isinstance(input_str, str) and input_str.startswith('data:') and ';base64,' in input_str:
        return True
    return False

def split_base64_with_mime(input_str: str):
    """
    Split base64 string with MIME type

    Args:
        input_str: String in format 'data:{mime_type};base64,{base64_str}'

    Returns:
        tuple: (base64_str, mime_type) or (input_str, None)
    """
    # Use regex to match all parts at once
    pattern = r'^data:([^;]+);base64,(.+)$'
    match = re.match(pattern, input_str)

    if match:
        mime_type = match.group(1)
        base64_str = match.group(2)
        return base64_str, mime_type
    return input_str, None

def image_to_base64(directory):
    try:
        with open(directory, 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
            ext = directory.split(".")[-1]
            mime = IMAGE_MIME_TYPE.get(ext)
        return image_base64, mime
    except Exception as e:
        LOG.error(f"Error in base64 encode {directory}: {e}")

def base64_to_image(base64_str: str):
    base64_data, mime_type = split_base64_with_mime(base64_str)

    if mime_type is None:
        raise ValueError("Invalid base64 format")

    suffix = None
    for ext, mime in IMAGE_MIME_TYPE.items():
        if mime == mime_type:
            suffix = f'.{ext}'
            break
    if suffix is None:
        raise ValueError(f"Unsupported image MIME type: {mime_type}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_data))
        string = temp_file.name
    return string

def audio_to_base64(directory):
    try:
        with open(directory, 'rb') as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')
            ext = directory.split(".")[-1]
            mime = AUDIO_MIME_TYPE.get(ext)
        return audio_base64, mime
    except Exception as e:
        LOG.error(f"Error in base64 encode {directory}: {e}")

def base64_to_audio(base64_str: str):
    base64_data, mime_type = split_base64_with_mime(base64_str)

    if mime_type is None:
        raise ValueError("Invalid base64 format")

    suffix = None
    for ext, mime in AUDIO_MIME_TYPE.items():
        if mime == mime_type:
            suffix = f'.{ext}'
            break
    if suffix is None:
        raise ValueError(f"Unsupported audio MIME type: {mime_type}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_data))
        string = temp_file.name
    return string
