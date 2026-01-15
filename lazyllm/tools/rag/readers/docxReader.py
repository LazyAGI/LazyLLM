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
import uuid
import subprocess
import os
import re

from ..doc_node import DocNode
from .readerBase import LazyLLMReaderBase, get_default_fs, is_default_fs
from ..parser import CaptionFootnoteParser, TYPE_CONFIG

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
    def __init__(self, save_image: bool = False, extra_info: Optional[Dict] = None,
                 extract_process: Optional[Callable] = None, return_trace: bool = True,
                 post_process: Optional[Callable] = None, enhanced: bool = False
                 ):
        super().__init__(return_trace=return_trace)
        self.save_image = save_image
        self.extra_info = extra_info
        self.enhanced = enhanced
        self.extract_process = extract_process or self._default_extract
        self.post_process = post_process or self._default_post

    def _load_data(self, file: Path, fs: Optional['fsspec.AbstractFileSystem'] = None,
                   **kwargs) -> List[DocNode]:
        if not self.enhanced:
            if fs:
                with fs.open(file) as f:
                    text = docx2txt.process(f)
            else:
                text = docx2txt.process(file)

            return [DocNode(text=text)]

        return self._enhanced_load(file, fs, **kwargs)

    def _enhanced_load(self, file: Path, fs: Optional['fsspec.AbstractFileSystem'] = None, **kwargs) -> List[DocNode]:
        with pipeline() as p:
            p.f1 = self._read_file
            p.f2 = bind(self.extract_process, file, _0)
            p.f3 = bind(self.post_process, _0, **kwargs)

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

                    style_name = para.style.name if para.style else ''
                    if self._is_toc_or_index(content, style_name):
                        continue

                    title_level = self._detect_title_level(para)
                    if title_level is not None:
                        title_node = self._process_title(
                            para, content, base_metadata, title_level, self.extra_info
                        )
                        doc_list.append(title_node)
                        continue

                    para_node = self._process_paragraph(para, content, base_metadata, self.extra_info)
                    doc_list.append(para_node)

        return doc_list

    def _default_post(self, doc_list, **kwargs) -> List[DocNode]:
        for index, node in enumerate(doc_list):
            node.metadata['index'] = index

        parsed_nodes = CaptionFootnoteParser(save_image=self.save_image)(doc_list)
        parsed_nodes = self._split_lines(parsed_nodes)
        parser = kwargs.get('parser', None)
        if parser is not None:
            LOG.info(f'[DocxReader._load_data] Calling parser with {len(parsed_nodes)} nodes, kwargs: {kwargs}')
            parsed_nodes = parser(parsed_nodes, **kwargs)

        for node in parsed_nodes:
            node.excluded_embed_metadata_keys = ['style_dict', 'type', 'index', 'text_level', 'lines']
            node.excluded_llm_metadata_keys = ['style_dict', 'type', 'index', 'text_level', 'lines']

        return parsed_nodes

    def _content_clean(self, element) -> bool:
        content = element.text.replace('\u3000', ' ').strip('\n') if element.text else ''
        content_clean = content.replace(' ', '').replace('\n', '')

        return content_clean is None

    def _split_lines(self, nodes: List[DocNode]) -> List[DocNode]:
        split_pattern = re.compile(r'[。\n]')
        special_type = ['table', 'image', 'equation']

        for node in nodes:
            text = getattr(node, 'text', '') or ''
            if not text.strip():
                continue

            existing_lines = node.metadata.get('lines') or []
            cleaned_existing = [
                {
                    'content': item.get('content', ''),
                    'type': item.get('type', node.metadata.get('type', 'text')),
                }
                for item in existing_lines
            ]

            if node.metadata.get('type') in special_type:
                lines = cleaned_existing + [
                    {'content': node.text, 'type': node.metadata.get('type')}
                ]
                node.metadata['lines'] = lines
                continue

            segments = []
            start = 0
            for match in split_pattern.finditer(text):
                end = match.start()
                segment = text[start:end].strip()
                if segment:
                    segments.append(segment)
                start = match.end()
            tail = text[start:].strip()
            if tail:
                segments.append(tail)

            merged_segments = []
            for segment in segments:
                if len(segment) < 20 and merged_segments:
                    merged_segments[-1] = merged_segments[-1] + segment
                else:
                    merged_segments.append(segment)

            if merged_segments:
                lines = cleaned_existing + [
                    {'content': seg, 'type': node.metadata.get('type', 'text')}
                    for seg in merged_segments
                ]
                node.metadata['lines'] = lines

        return nodes

    # TODO: modify private function beblow
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

    def _table_to_markdown(self, table: Table) -> str:  # noqa: C901
        '''将表格转换为 Markdown 格式'''
        try:
            md = '\n'
            try:
                col_size = len(list(table.columns))
            except Exception:
                # 如果获取列数失败，尝试从第一行获取
                if len(table.rows) > 0:
                    try:
                        col_size = len(table.rows[0].cells)
                    except Exception:
                        col_size = 1
                else:
                    col_size = 1

            _cells = 1

            for idx, row in enumerate(table.rows):
                # 添加表头分隔线
                if idx == 1:
                    md += '|'
                    for _ in range(col_size):
                        md += '---|'
                    md += '\n'

                md += '|'
                try:
                    __cells = 0
                    for cell in row.cells:
                        __cells += 1
                        _cells = __cells if __cells > _cells else _cells
                        t = ''
                        try:
                            for cell_ele in cell.paragraphs:
                                t += cell_ele.text.replace('\r', '').replace('\n', '')
                        except Exception:
                            # 如果无法访问段落，使用备用方法
                            try:
                                t = str(cell.text).replace('\r', '').replace('\n', '')
                            except Exception:
                                t = '[单元格解析失败]'
                        md += f' {t} |'
                except Exception:
                    # 备用方案：尝试简单的行遍历
                    try:
                        md = '\n'
                        for row_idx, row in enumerate(table.rows):
                            if row_idx == 1:
                                # 添加表头分隔线
                                md += '|'
                                for _ in range(col_size):
                                    md += '---|'
                                md += '\n'
                            md += '|'
                            for cell in row.cells:
                                try:
                                    cell_text = ''
                                    for para in cell.paragraphs:
                                        cell_text += para.text.replace('\r', '').replace('\n', '')
                                    md += f' {cell_text} |'
                                except Exception:
                                    md += ' [单元格错误] |'
                            md += '\n'
                        md += '\n'
                        return md
                    except Exception:
                        md += 'ERROR_PARSING_TABLE'
                    return md
                md += '\n'
            md += '\n'
            return md
        except Exception as e:
            LOG.error(f'Error converting table to markdown: {e}')
            return '\n[表格解析失败]\n'

    def _extract_outline_level(self, xml_str: str) -> Optional[int]:
        '''从 XML 字符串中提取大纲级别'''
        match = re.search(r'<w:outlineLvl[^>]*w:val="(\d+)"', xml_str)
        if match:
            return int(match.group(1))
        return None

    def _detect_title_level(self, para) -> Optional[int]:  # noqa: C901
        '''
        检测段落是否为标题及其级别

        检测策略（按优先级）：
        1. 检查段落的大纲级别（outline level）
        2. 检查段落样式的大纲级别
        3. 检查样式名称是否包含标题关键词
        4. 检查内容格式（数字编号、中文编号、章节格式、括号编号、字母编号等）
        5. 检查字体加粗且居中对齐

        Returns:
            标题级别（1-9），如果不是标题则返回 None
        '''
        if not para.text or para.text.strip() == '':
            return None

        # 策略1: 检查段落的大纲级别
        paragraph_xml = para._p.xml  # noqa: SLF001
        level = self._extract_outline_level(paragraph_xml)
        if level is not None:
            return level + 1  # Word 中级别从 0 开始，我们转换为从 1 开始

        # 策略2: 检查段落样式的大纲级别
        style = para.style
        while style is not None:
            level = self._extract_outline_level(style.element.xml)
            if level is not None:
                return level + 1
            style = style.base_style

        # 策略3: 检查样式名称是否包含标题关键词
        style_name = para.style.name.lower() if para.style else ''

        # 排除表标题、图标题、公式标题相关的样式名称（避免误识别为文档标题）
        # 使用全局配置 TYPE_CONFIG 来检查
        for _, config in TYPE_CONFIG.items():
            if any(keyword in style_name for keyword in config['style_keywords']):
                return None  # 如果是 caption 样式，不识别为文档标题

        title_patterns = [
            (r'heading\s*(\d+)', 1),  # Heading 1, Heading 2, etc.
            (r'标题\s*(\d+)', 1),      # 标题 1, 标题 2, etc.
            (r'标题', 1),              # 标题
            (r'heading', 1),           # heading
        ]

        for pattern, _ in title_patterns:
            match = re.search(pattern, style_name)
            if match:
                if match.groups():
                    try:
                        return int(match.group(1))
                    except (ValueError, IndexError):
                        return 1
                return 1

        # 策略4: 检查内容格式（多种编号格式）
        content = para.text.strip()

        # 模式1: 数字编号格式（如 2.2.2 xxx、1.1 xxx、3.1.3蒸发量 等）
        # 匹配格式：数字.数字.数字...（每个数字部分都是1-2位整数）
        number_title_pattern = r'^(\d{1,2}(?:\.\d{1,2})+)\s*([^\d].*)$'
        match = re.match(number_title_pattern, content)
        if match:
            number_part = match.group(1)
            level = number_part.count('.') + 1
            # 限制级别在合理范围内（1-9）
            return max(1, min(level, 9))

        # 模式2: 中文数字编号格式（如 一、xxx、二、xxx）
        cn_number_pattern = r'^[一二三四五六七八九十百]+[、\.．]\s*(.+)$'
        match = re.match(cn_number_pattern, content)
        if match:
            return 1

        # 模式4: 章节格式（如 第一章 xxx、第一节 xxx）
        cn_num = r'[一二三四五六七八九十百千万零壹贰叁肆伍陆柒捌玖拾佰仟]'
        # 第X篇/卷/章 -> 1级
        if re.match(rf'^第\s*{cn_num}+\s*[篇卷章]\s*(.+)$', content):
            return 1
        # 第X节 -> 2级
        if re.match(rf'^第\s*{cn_num}+\s*节\s*(.+)$', content):
            return 2
        # 第X条 -> 3级
        if re.match(rf'^第\s*{cn_num}+\s*条\s*(.+)$', content):
            return 3

        # 模式5: 字母编号格式（如 A. xxx、a. xxx、A.1.2 xxx 等）
        letter_pattern = r'^([A-Za-z](?:\.[A-Za-z0-9]+)*)[\.、]\s+(.+)$'
        match = re.match(letter_pattern, content)
        if match:
            letter_part = match.group(1)
            # 计算级别：点的数量 + 1（如 A.1.2 有2个点，级别为3）
            level = letter_part.count('.') + 1
            # 限制级别在合理范围内（1-9）
            return max(1, min(level, 9))

        # 策略5: 检查字体加粗且居中对齐（作为 Level 1 标题的兜底策略）
        try:
            # 检查样式加粗
            is_bold = para.style.font.bold
            if not is_bold:
                # 检查所有 Run 是否加粗（允许忽略空白 Run）
                runs = [run for run in para.runs if run.text.strip()]
                is_bold = runs and all(run.font.bold for run in runs)

            if is_bold and self._get_aligned_type(para) == 'center':
                return 1
        except Exception:
            pass

        return None

    def _get_style_info(self, style: ParagraphStyle) -> dict:
        '''获取段落样式信息'''
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

    def _is_toc_or_index(self, content: str, style_name: str) -> bool:
        '''判断是否为目录或索引'''
        # 检查内容
        if re.fullmatch(r'目\s*录', content.strip()):
            return True

        # 检查样式名称
        style_lower = style_name.lower()
        if 'toc' in style_lower or '目录' in style_lower or 'index' in style_lower:
            return True

        return False

    def _extract_images_from_paragraph(self, para, doc: Docx, base_metadata: dict,
                                       extra_info: Optional[Dict]) -> List[DocNode]:
        '''从段落中提取图片，返回图片节点列表'''
        image_nodes = []

        for run in para.runs:
            # 检查是否包含图片
            has_drawing = run.element.xpath('.//*[local-name()="drawing"]')
            has_imagedata = run.element.xpath('.//*[local-name()="imagedata"]')

            if not (has_drawing or has_imagedata):
                continue

            # 提取图片数据
            image_filename = None
            for shape in run._element.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}blip'):  # noqa: SLF001, E501
                embed_id = shape.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                if not embed_id:
                    continue

                try:
                    image_part = doc.part.related_parts[embed_id]

                    if self.save_image:
                        # 保存图片文件
                        image_data = image_part.blob
                        # 生成图片文件名
                        original_filename = os.path.basename(image_part.partname)
                        file_extension = os.path.splitext(original_filename)[1] or '.png'
                        image_filename = f'{uuid.uuid4()}{file_extension}'

                        # 保存图片
                        try:
                            os.makedirs(self.image_save_path, exist_ok=True)
                        except Exception as e:
                            LOG.error(f'[Docx Reader] Failed to create directory: {self.image_save_path}: {e}')
                            continue

                        image_save_path = os.path.join(self.image_save_path, image_filename)
                        with open(image_save_path, 'wb') as img_file:
                            img_file.write(image_data)

                        LOG.info(f'图片保存到: {image_save_path}')
                    else:
                        # 不保存图片，但生成一个占位符文件名
                        original_filename = os.path.basename(image_part.partname)
                        file_extension = os.path.splitext(original_filename)[1] or '.png'
                        image_filename = f'image_{uuid.uuid4()}{file_extension}'

                    # 构建图片文本（只有在 save_image=True 时才生成 markdown 格式）
                    if self.save_image:
                        image_text = f'![]({image_filename})'
                    else:
                        # 如果 save_image=False，使用空文本或占位符
                        image_text = ''

                    # 创建图片节点
                    metadata = copy.deepcopy(base_metadata)
                    metadata['type'] = 'image'
                    if image_filename:
                        metadata['image_path'] = image_filename

                    image_nodes.append(DocNode(
                        text=image_text,
                        metadata=metadata,
                        global_metadata=extra_info
                    ))
                except Exception as e:
                    LOG.error(f'[Docx Reader] Failed to extract image: {e}')

        return image_nodes

    def _extract_math_from_element(self, element) -> Optional[str]:
        '''从元素中提取数学公式'''
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
        '''将数学公式节点转换为文本'''
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

    def _process_table(self, table: Table, base_metadata: dict, extra_info: Optional[Dict] = None) -> DocNode:
        '''处理表格元素'''
        metadata = copy.deepcopy(base_metadata)
        metadata['type'] = 'table'

        # 构建表格文本
        table_md = self._table_to_markdown(table)
        table_text = table_md

        return DocNode(text=table_text, metadata=metadata, global_metadata=extra_info)

    def _check_run_bold(self, para) -> bool:
        '''检查段落中的 Runs 是否全为粗体'''
        try:
            runs = [run for run in para.runs if run.text.strip()]
            return bool(runs and all(run.font.bold for run in runs))
        except Exception:
            return False

    def _process_title(self, para, content: str, base_metadata: dict, level: int, extra_info: Optional[Dict]) -> DocNode:
        '''处理标题段落，返回标题节点'''
        metadata = copy.deepcopy(base_metadata)
        style_dict = self._get_style_info(para.style)

        # 检查 Run 级别的加粗
        if not style_dict.get('font_bold') and self._check_run_bold(para):
            style_dict['font_bold'] = True

        aligned_type = self._get_aligned_type(para)
        style_dict.update({'aligned_type': aligned_type})

        metadata['style_dict'] = style_dict
        metadata['type'] = 'text'
        metadata['text_level'] = level

        return DocNode(text=content, metadata=metadata, global_metadata=extra_info)

    def _process_paragraph(self, para, content: str, base_metadata: dict, extra_info: Optional[Dict]) -> DocNode:
        '''处理普通段落，返回段落节点'''
        metadata = copy.deepcopy(base_metadata)
        style_dict = self._get_style_info(para.style)

        # 检查 Run 级别的加粗
        if not style_dict.get('font_bold') and self._check_run_bold(para):
            style_dict['font_bold'] = True

        aligned_type = self._get_aligned_type(para)
        style_dict.update({'aligned_type': aligned_type})

        # 仅当文本是编号标题格式且加粗时，才添加 Markdown 加粗标记
        number_title_pattern = r'^(\d{1,2}(?:\.\d{1,2})+)\s*([^\d].*)$'
        if style_dict.get('font_bold') and re.match(number_title_pattern, content):
            content = f'**{content}**'

        metadata['style_dict'] = style_dict
        metadata['type'] = 'text'

        return DocNode(text=content, metadata=metadata, global_metadata=extra_info)

    def _process_toc(self, para, content: str, base_metadata: dict, extra_info: Optional[Dict]) -> DocNode:
        '''处理目录节点，返回目录节点'''
        metadata = copy.deepcopy(base_metadata)
        style_dict = self._get_style_info(para.style)
        aligned_type = self._get_aligned_type(para)
        style_dict.update({'aligned_type': aligned_type})

        metadata['style_dict'] = style_dict
        metadata['type'] = 'toc'

        return DocNode(text=content, metadata=metadata, global_metadata=extra_info)

    def _process_math(self, math_text: str, base_metadata: dict, extra_info: Optional[Dict]) -> DocNode:
        '''处理数学公式，返回公式节点'''
        metadata = copy.deepcopy(base_metadata)
        metadata['type'] = 'equation'
        return DocNode(text=math_text, metadata=metadata, global_metadata=extra_info)
