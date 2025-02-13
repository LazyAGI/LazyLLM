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
