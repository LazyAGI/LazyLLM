import nbformat
import re
import os
import glob
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Regex pattern to replace: <source src="./xxx"> → <source src="../xxx">
VIDEO_SRC_PATTERN = re.compile(r'(<source\s+src=")\./([^"]+)(")')
script_dir = os.path.dirname(os.path.abspath(__file__))


ZH_OUTPUT_DIR = Path(script_dir) / 'zh' / 'Tutorial'
EN_OUTPUT_DIR = Path(script_dir) / 'en' / 'Tutorial'


def convert_ipynb_to_md(ipynb_path: str):
    '''
    Convert a Jupyter notebook to Markdown, replacing video src paths.

    Args:
        ipynb_path (str): Path to the input .ipynb file.
    '''
    ipynb_path = Path(ipynb_path)
    notebook_name = ipynb_path.stem
    output_dir = EN_OUTPUT_DIR if notebook_name.endswith('.en') else ZH_OUTPUT_DIR
    notebook_name = notebook_name.replace('.en', '')
    md_path = output_dir / f'{notebook_name}.md'
    os.makedirs(md_path.parent, exist_ok=True)

    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    md_lines = []

    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            text = cell.source
            # Replace video source path from ./ to ../
            text = VIDEO_SRC_PATTERN.sub(r'\1../\2\3', text)
            md_lines.append(text)
        elif cell.cell_type == 'code':
            if cell.get('source'):
                md_lines.append(f'```python\n{cell.source}\n```')

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(md_lines))

    # Try to get relative path for logging; fallback to filename if error occurs
    try:
        relative_md_path = md_path.relative_to(Path.cwd())
    except ValueError:
        relative_md_path = md_path.name

    logger.info(f'✅ Converted: {ipynb_path.name} → {relative_md_path}')


if __name__ == '__main__':
    # Batch convert all notebooks matching the pattern
    input_files = glob.glob('Tutorial/rag/notebook/chapter*/*.ipynb', recursive=True)

    for ipynb_file in input_files:
        convert_ipynb_to_md(ipynb_file)
