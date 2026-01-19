from pathlib import Path
import tempfile
from lazyllm.thirdparty import fsspec, docx2txt
from typing import Callable, Optional, List, Dict
from lazyllm import LOG, pipeline, _0
from lazyllm.common import bind
from lxml import etree
from docx import Document as Docx
from docx.styles.style import ParagraphStyle
from docx.table import Table
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.text import paragraph
import copy
import subprocess
import os
import re

from ..doc_node import DocNode
from .readerBase import LazyLLMReaderBase, get_default_fs, is_default_fs

def doc_to_docx(doc_path: str) -> str:
    fname = str(Path(doc_path).stem)
    dir_path = str(Path(doc_path).parent)
    docx_path = os.path.join(os.path.dirname(doc_path), f'{fname}.docx')

    if os.path.exists(docx_path):
        return docx_path

    cmd = f'soffice --headless --convert-to docx "{doc_path}" --outdir "{dir_path}"'
    try:
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        LOG.info(f'Command output: {result.stdout}')
    except subprocess.CalledProcessError as e:
        LOG.error(f'Command failed with error code {e.returncode}: {e.stderr}')
    except Exception as e:
        LOG.error(f'Unexpected error: {e}')

    if not os.path.exists(docx_path):
        LOG.error(f'> !!! File conversion failed {doc_path} ==> {docx_path}')
        return None
    else:
        return Path(docx_path)

class DocxReader(LazyLLMReaderBase):
    def __init__(self, extra_info: Optional[Dict] = None, post_func: Optional[Callable] = None,
                 extract_process: Optional[Callable] = None, return_trace: bool = True):
        super().__init__(return_trace=return_trace)
        self.extra_info = extra_info
        self.extract_process = extract_process or self._default_extract
        self.post_func = post_func or self._default_post

    def _load_data(self, file: Path, fs: Optional['fsspec.AbstractFileSystem'] = None,
                   **kwargs) -> List[DocNode]:
        try:
            return self._enhanced_load(file, fs, **kwargs)

        except Exception:
            if fs:
                with fs.open(file) as f:
                    text = docx2txt.process(f)
            else:
                text = docx2txt.process(file)

            return [DocNode(text=text)]

    def _enhanced_load(self, file: Path, fs: Optional['fsspec.AbstractFileSystem'] = None, **kwargs) -> List[DocNode]:
        with pipeline() as p:
            p.f1 = self._read_file
            p.f2 = bind(self.extract_process, file, _0)
            p.f3 = bind(self.post_func, _0, **kwargs)

        nodes = p(file, fs)

        return nodes

    def _read_file(self, file: Path, fs: Optional['fsspec.AbstractFileSystem'] = None) -> Docx:
        fs = fs or get_default_fs()

        try:
            file_size = fs.size(file)
            if file_size == 0:
                return []

        except Exception as e:
            LOG.error(f'Fail to load file for {file}: {e}')
            return []

        if file.name.endswith('.doc'):
            if not is_default_fs(fs):
                with tempfile.NamedTemporaryFile(suffix='.doc', delete=False) as tmp_file:
                    temp_docx_path = tmp_file.name

                try:
                    with fs.open(file, 'rb') as remote_file:
                        tmp_file.write(remote_file.read())

                    converted_file = doc_to_docx(temp_docx_path)
                    if converted_file is None:
                        LOG.error(f'Failed to convert .doc file: {file}')
                        return None

                    file = Path(converted_file)
                    fs = get_default_fs()
                finally:
                    os.unlink(temp_docx_path)

            else:
                file = doc_to_docx(file)
                if file is None:
                    LOG.error(f'Failed to convert .doc file: {file}')
                    return None

        temp_path = None

        try:
            if is_default_fs(fs):
                doc = Docx(docx=str(file))
            else:
                with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp_file:
                    temp_path = tmp_file.name

                with fs.open(file, 'rb') as remote_file:
                    tmp_file.write(remote_file.read())

                doc = Docx(docx=temp_path)
            return doc

        except Exception as e:
            LOG.error(f'[ERROR] file--{file.name}--Wrong file, failed to read file: {e}')
            raise

    def _default_extract(self, file, doc) -> List[DocNode]:  # noqa: C901
        base_metadata = {'file_name': file.name}
        if self.extra_info is not None:
            base_metadata.update(self.extra_info)

        doc_list = []

        paragraphs = list(doc.paragraphs)
        tables = list(doc.tables)
        paragraph_idx = 0
        table_idx = 0

        elements = list(doc.element.body)
        for element in elements:
            if element.tag.endswith('tbl'):
                if table_idx < len(tables):
                    table = tables[table_idx]
                    table_idx += 1

                    table_node = self._process_table(table, base_metadata, self.extra_info)
                    doc_list.append(table_node)

            elif element.tag.endswith('p'):
                if paragraph_idx >= len(paragraphs):
                    continue

                para = paragraphs[paragraph_idx]
                paragraph_idx += 1

                has_image = False
                for run in para.runs:
                    has_drawing = run.element.xpath('.//*[local-name()="drawing"]')
                    has_imagedata = run.element.xpath('.//*[local-name()="imagedata"]')
                    if has_drawing or has_imagedata:
                        has_image = True
                        break

                    if has_image:
                        image_nodes = self._extract_images_from_paragraph(
                            para, doc, base_metadata, self.extra_info
                        )
                        if image_nodes:
                            doc_list.extend(image_nodes)

                        if self._content_clean(element):
                            continue

                    math_text = self._extract_math_from_element(element)
                    if math_text:
                        math_node = self._process_math(math_text, base_metadata, self.extra_info)
                        doc_list.append(math_node)
                        if self._content_clean(element):
                            continue

                    if self._content_clean(element):
                        continue

                    content = element.text.replace('\u3000', ' ').strip('\n') if element.text else ''

                    para_node = self._process_paragraph(para, content, base_metadata, self.extra_info)
                    doc_list.append(para_node)

        return doc_list

    def _default_post(self, doc_list, **kwargs) -> List[DocNode]:
        for index, node in enumerate(doc_list):
            node.metadata['index'] = index

        for node in doc_list:
            node.excluded_embed_metadata_keys = ['style_dict', 'type', 'index', 'text_level', 'lines']
            node.excluded_llm_metadata_keys = ['style_dict', 'type', 'index', 'text_level', 'lines']

        return doc_list

    def _content_clean(self, element) -> bool:
        content = element.text.replace('\u3000', ' ').strip('\n') if element.text else ''
        content_clean = content.replace(' ', '').replace('\n', '')

        return content_clean is None

    def _get_aligned_type(self, para: paragraph) -> str:
        aligned_type = {
            WD_ALIGN_PARAGRAPH.LEFT: 'left',
            WD_ALIGN_PARAGRAPH.CENTER: 'center',
            WD_ALIGN_PARAGRAPH.RIGHT: 'right',
            WD_ALIGN_PARAGRAPH.JUSTIFY: 'both_ends',
            WD_ALIGN_PARAGRAPH.DISTRIBUTE: 'distribute',
            None: '',
        }

        try:
            alignment = para.alignment
        except Exception:
            try:
                alignment = para._p.get(qn('w:align'))  # noqa: SLF001
            except Exception:
                alignment = None

        return aligned_type.get(alignment, '')

    def _get_style_info(self, style: ParagraphStyle) -> dict:
        try:
            font = style.font
            style_dict = {
                'style_name': style.name,
                'style_type': style.type,
                'font_name': font.name if font.name else None,
                'font_bold': bool(font.bold) if font.bold is not None else False,
                'font_size': font.size.pt if font.size else None,
            }
        except Exception:
            style_dict = {
                'style_name': style.name if style else '',
                'style_type': style.type if style else None,
                'font_name': None,
                'font_bold': False,
                'font_size': None,
            }
        return style_dict

    def _extract_images_from_paragraph(self, para, doc: Docx, base_metadata: dict,
                                       extra_info: Optional[Dict]) -> List[DocNode]:
        image_nodes = []

        for run in para.runs:
            has_drawing = run.element.xpath('.//*[local-name()="drawing"]')
            has_imagedata = run.element.xpath('.//*[local-name()="imagedata"]')

            if not (has_drawing or has_imagedata):
                continue

            for shape in run._element.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}blip'):  # noqa: SLF001, E501
                embed_id = shape.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                if not embed_id:
                    continue

                try:
                    image_text = ''

                    metadata = copy.deepcopy(base_metadata)
                    metadata['type'] = 'image'

                    image_nodes.append(DocNode(
                        text=image_text,
                        metadata=metadata,
                        global_metadata=extra_info
                    ))
                except Exception as e:
                    LOG.error(f'[Docx Reader] Failed to extract image: {e}')

        return image_nodes

    def _extract_math_from_element(self, element) -> Optional[str]:
        math_nodes = element.xpath('.//*[local-name()="oMath"] | .//*[local-name()="oMathPara"]')
        if not math_nodes:
            return None

        try:
            math = math_nodes[0]
            return self._math_to_text(math)
        except Exception as e:
            LOG.error(f'[Docx Reader] Failed to extract math: {e}')
            return None

    def _math_to_text(self, math_node) -> str:
        texts = []

        def recurse(n):
            tag = etree.QName(n).localname
            if tag == 't':
                texts.append(n.text if n.text else '')
            else:
                for child in n:
                    recurse(child)

        try:
            recurse(math_node)
            return ''.join(texts)
        except Exception as e:
            LOG.error(f'[Docx Reader] Failed to convert math to text: {e}')
            return ''

    def _table_to_markdown(self, table: Table) -> str:
        if not table.rows:
            return '\n[empty table]\n'

        try:
            col_size = len(table.rows[0].cells)
            md_lines = []

            for row_idx, row in enumerate(table.rows):
                cells = []
                for i in range(min(col_size, len(row.cells))):
                    cell = row.cells[i]
                    text = getattr(cell, 'text', '') or ''.join(
                        p.text for p in getattr(cell, 'paragraphs', [])
                    )
                    cells.append(text.replace('\n', ' ').replace('\r', ' ').strip())

                cells.extend([''] * (col_size - len(cells)))

                md_lines.append('| ' + ' | '.join(cells) + ' |')

                if row_idx == 0:
                    md_lines.append('|' + '|'.join([' --- '] * col_size) + '|')

            return '\n' + '\n'.join(md_lines) + '\n'

        except Exception:
            return '\n[Table parse failed]\n'

    def _process_table(self, table: Table, base_metadata: dict,
                       extra_info: Optional[Dict] = None) -> DocNode:
        metadata = copy.deepcopy(base_metadata)
        metadata['type'] = 'table'

        table_md = self._table_to_markdown(table)
        table_text = table_md

        return DocNode(text=table_text, metadata=metadata, global_metadata=extra_info)

    def _check_run_bold(self, para) -> bool:
        try:
            runs = [run for run in para.runs if run.text.strip()]
            return bool(runs and all(run.font.bold for run in runs))
        except Exception:
            return False

    def _process_paragraph(self, para, content: str, base_metadata: dict,
                           extra_info: Optional[Dict]) -> DocNode:
        metadata = copy.deepcopy(base_metadata)
        style_dict = self._get_style_info(para.style)

        if not style_dict.get('font_bold') and self._check_run_bold(para):
            style_dict['font_bold'] = True

        aligned_type = self._get_aligned_type(para)
        style_dict.update({'aligned_type': aligned_type})

        number_title_pattern = r'^(\d{1,2}(?:\.\d{1,2})+)\s*([^\d].*)$'
        if style_dict.get('font_bold') and re.match(number_title_pattern, content):
            content = f'**{content}**'

        metadata['style_dict'] = style_dict
        metadata['type'] = 'text'

        return DocNode(text=content, metadata=metadata, global_metadata=extra_info)

    def _process_math(self, math_text: str, base_metadata: dict,
                      extra_info: Optional[Dict]) -> DocNode:
        metadata = copy.deepcopy(base_metadata)
        metadata['type'] = 'equation'
        return DocNode(text=math_text, metadata=metadata, global_metadata=extra_info)
