import os
import glob
import shutil
import logging
from pathlib import Path
from itertools import chain

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DEST_DIRS = {
    'zh': Path('docs/zh/Tutorial'),
    'en': Path('docs/en/Tutorial'),
}
ASSETS_SRC = Path('Tutorial/assets')


def _ensure_dest_dirs():
    for dest in DEST_DIRS.values():
        dest.mkdir(parents=True, exist_ok=True)


def _link_media_dirs():
    '''
    Create symbolic links in docs/{zh|en}/Tutorial pointing to the actual
    {videos, images} directories found under Tutorial/rag/notebook/chapter*/.
    Each link will be named after the basename of the original directory.
    '''
    patterns = [
        'Tutorial/rag/notebook/chapter*/*_videos',
        'Tutorial/rag/notebook/chapter*/*_images',
    ]
    paths = list(chain.from_iterable(glob.glob(p, recursive=True) for p in patterns))

    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            continue

        name = path.name
        target = path.resolve()

        for dest in DEST_DIRS.values():
            link_path = dest / name
            try:
                if link_path.exists() or link_path.is_symlink():
                    if link_path.is_symlink() or link_path.is_file():
                        link_path.unlink()
                    elif link_path.is_dir():
                        shutil.rmtree(link_path)
                os.symlink(target, link_path)
                logger.info(f'Created symlink: {link_path} → {target}')
            except Exception as e:
                logger.error(f'Failed to create symlink: {link_path} → {target}\nReason: {e}')


def _copy_assets():
    if not ASSETS_SRC.exists():
        return

    for dest_root in DEST_DIRS.values():
        dest = dest_root / 'assets'
        try:
            if dest.exists():
                if dest.is_symlink() or dest.is_file():
                    dest.unlink()
                elif dest.is_dir():
                    shutil.rmtree(dest)
            shutil.copytree(ASSETS_SRC, dest)
            logger.info(f'Copied assets directory to {dest}')
        except Exception as e:
            logger.error(f'Failed to copy assets directory to {dest}\nReason: {e}')


def link_assets():
    _ensure_dest_dirs()
    _link_media_dirs()
    _copy_assets()


if __name__ == '__main__':
    link_assets()
