import nbformat
import re
import os
import glob
import logging
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Regex pattern to replace: <source src="./xxx"> → <source src="../xxx">
VIDEO_SRC_PATTERN = re.compile(r'(<source\s+src=")\./([^"]+)(")')
script_dir = os.path.dirname(os.path.abspath(__file__))


ZH_OUTPUT_DIR = Path(script_dir) / 'zh' / 'Tutorial'
EN_OUTPUT_DIR = Path(script_dir) / 'en' / 'Tutorial'
FROM_DATA_TO_LLM_SRC = Path('Tutorial/from-data-to-llm/docs')
FROM_DATA_TO_LLM_ZH_DEST = ZH_OUTPUT_DIR / 'from-data-to-llm'
FROM_DATA_TO_LLM_EN_DEST = EN_OUTPUT_DIR / 'from-data-to-llm'
FROM_DATA_TO_LLM_EN_PLACEHOLDER = '''# From Data to LLM

The English version of this tutorial is being prepared and will be published later.

For now, you can read the Chinese version here:
[From Data to LLM Chinese tutorial](/zh-cn/latest/Tutorial/from-data-to-llm/).
'''
FROM_DATA_TO_LLM_IFRAME_PATTERN = re.compile(
    r'(<iframe\s+src=")(?:\.\./)?assets/course_map\.html(")'
)
FROM_DATA_TO_LLM_IFRAME_BLOCK_PATTERN = re.compile(
    r'<iframe[^>]+course_map\.html[^>]*>\s*</iframe>'
)
FROM_DATA_TO_LLM_ZH_INTRO = '''# 从数据到大模型

本课程旨在为开发者和算法工程师提供一套从数据到模型再到应用的 LLM 全栈实战指南。
课程核心聚焦于数据工程，并将其深度融入到模型训练的每一个环节——从底座预训练
(Pre-training) 到指令微调 (SFT)，再到人类价值观对齐 (RLHF/GRPO)。

课程不仅涵盖了纯文本、多模态、Embedding 等多维度的技术原理，更引入了系统工程视角，
详解分布式训练、高效部署与模型合规。特别值得一提的是，本课程贯穿了 LazyLLM
全流程实战与 Agent（智能体）的双重应用：既教授如何构建具备 Agent 能力的模型，
也演示如何利用 Agent 自动化流水线来清洗和合成高质量数据，助力企业构建闭环的
"数据飞轮"。
'''


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


def _replace_directory(src: Path, dest: Path):
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        elif dest.is_dir():
            shutil.rmtree(dest)
    shutil.copytree(src, dest)


def _fix_from_data_to_llm_paths(dest: Path):
    for md_path in dest.rglob('*.md'):
        text = md_path.read_text(encoding='utf-8')
        iframe_src = 'assets/course_map.html' if md_path == dest / 'index.md' else '../assets/course_map.html'
        fixed_text = FROM_DATA_TO_LLM_IFRAME_PATTERN.sub(rf'\1{iframe_src}\2', text)
        if fixed_text != text:
            md_path.write_text(fixed_text, encoding='utf-8')


def _split_from_data_to_llm_entry_pages(dest: Path):
    index_path = dest / 'index.md'
    outline_path = dest / 'outline.md'

    index_path.write_text(FROM_DATA_TO_LLM_ZH_INTRO, encoding='utf-8')

    if outline_path.exists():
        outline_text = outline_path.read_text(encoding='utf-8')
        outline_heading = '## 课程大纲'
        outline_start = outline_text.find(outline_heading)
        if outline_start != -1:
            iframe_match = FROM_DATA_TO_LLM_IFRAME_BLOCK_PATTERN.search(outline_text[:outline_start])
            iframe_block = f'{iframe_match.group(0)}\n\n' if iframe_match else ''
            # Keep the LazyLLM outline page focused on the interactive course map.
            outline_path.write_text(iframe_block, encoding='utf-8')


def copy_from_data_to_llm_docs():
    if not FROM_DATA_TO_LLM_SRC.exists():
        logger.warning(f'Skipped From Data to LLM docs, source not found: {FROM_DATA_TO_LLM_SRC}')
        return

    _replace_directory(FROM_DATA_TO_LLM_SRC, FROM_DATA_TO_LLM_ZH_DEST)
    _split_from_data_to_llm_entry_pages(FROM_DATA_TO_LLM_ZH_DEST)
    _fix_from_data_to_llm_paths(FROM_DATA_TO_LLM_ZH_DEST)
    logger.info(f'✅ Copied: {FROM_DATA_TO_LLM_SRC} → {FROM_DATA_TO_LLM_ZH_DEST}')

    FROM_DATA_TO_LLM_EN_DEST.mkdir(parents=True, exist_ok=True)
    en_index = FROM_DATA_TO_LLM_EN_DEST / 'index.md'
    en_index.write_text(FROM_DATA_TO_LLM_EN_PLACEHOLDER, encoding='utf-8')
    logger.info(f'✅ Created English placeholder: {en_index}')


if __name__ == '__main__':
    # Batch convert all notebooks matching the pattern
    input_files = glob.glob('Tutorial/rag/notebook/chapter*/*.ipynb', recursive=True)

    for ipynb_file in input_files:
        convert_ipynb_to_md(ipynb_file)

    copy_from_data_to_llm_docs()
