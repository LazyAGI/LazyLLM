import os
import base64
import datetime

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
    'wav': 'audio/wav',
    'mp3': 'audio/mpeg',
    'ogg': 'audio/ogg',
    'flac': 'audio/flac',
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

def image_to_base64(directory):
    try:
        with open(directory, 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
            ext = directory.split(".")[-1]
            mime = IMAGE_MIME_TYPE.get(ext)
        return image_base64, mime
    except Exception as e:
        LOG.error(f"Error in base64 encode {directory}: {e}")

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
    # 提取MIME类型和base64编码部分
    mime_type = base64_str.split(';')[0].split(':')[1]
    base64_str = base64_str.split(',')[1]
    # 根据MIME类型确定文件后缀
    suffix = None
    for ext, mime in AUDIO_MIME_TYPE.items():
        if mime == mime_type:
            suffix = f'.{ext}'
            break
    if suffix is None:
        raise ValueError(f"Unsupported audio MIME type: {mime_type}")

    # 创建临时文件
    import tempfile
    import base64
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_str))
        string = temp_file.name
    return string
