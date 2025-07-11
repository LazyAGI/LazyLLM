import os
import glob
import shutil
from pathlib import Path
from itertools import chain


def link_assets():
    """
    Create symbolic links in docs/zh/Tutorial pointing to the actual
    {videos, images} directories found under Tutorial/rag/notebook/chapter*/.
    Each link will be named after the basename of the original directory.
    """
    dest_dir = Path("docs/zh/Tutorial")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Manually expand the {videos,images} part
    patterns = [
        "Tutorial/rag/notebook/chapter*/*_videos",
        "Tutorial/rag/notebook/chapter*/*_images",
    ]
    paths = list(chain.from_iterable(glob.glob(p, recursive=True) for p in patterns))

    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            continue

        name = path.name
        target = path.resolve()
        link_path = dest_dir / name

        try:
            if link_path.exists():
                if link_path.is_symlink() or link_path.is_file():
                    link_path.unlink()
                elif link_path.is_dir():
                    shutil.rmtree(link_path)
            os.symlink(target, link_path)
            print(f"✅ Linked {link_path} → {target}")
        except Exception as e:
            print(f"❌ Failed to link {link_path} → {target}: {e}")


if __name__ == "__main__":
    link_assets()
