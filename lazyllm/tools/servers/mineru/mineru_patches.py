import copy
from mineru.backend.pipeline import (  # noqa: NID002
    pipeline_middle_json_mkcontent,
)
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import (  # noqa: NID002
    merge_para_with_text as pipeline_merge_para_with_text,
    get_title_level,
)
from mineru.backend.vlm import (  # noqa: NID002
    vlm_middle_json_mkcontent,
)
from mineru.backend.vlm.vlm_middle_json_mkcontent import (  # noqa: NID002
    merge_para_with_text as vlm_merge_para_with_text,
)
from mineru.utils.enum_class import (  # noqa: NID002
    BlockType,
    ContentType,
)


# patches to mineru (to output bbox)
def _parse_line_spans(para_block, page_idx):
    lines_metas = []
    if 'lines' in para_block:
        for line_info in para_block['lines']:
            spans = line_info.get('spans')
            if not spans:
                continue
            for span in spans:
                line_meta = copy.deepcopy(span)
                line_meta.pop('score', None)
                cross_page = line_meta.pop('cross_page', None)
                line_meta['page'] = page_idx + 1 if cross_page is True else page_idx
                lines_metas.append(line_meta)
    return lines_metas


# patches to pipeline

def pipeline_make_blocks_to_content_list(para_block, img_buket_path, page_idx, page_size):  # noqa: C901
    para_type = para_block['type']
    para_content = {}
    if para_type in [
        BlockType.TEXT,
        BlockType.LIST,
        BlockType.INDEX,
    ]:
        para_content = {
            'type': ContentType.TEXT,
            'text': pipeline_merge_para_with_text(para_block),
            'lines': _parse_line_spans(para_block, page_idx)
        }
    elif para_type == BlockType.DISCARDED:
        para_content = {
            'type': para_type,
            'text': pipeline_merge_para_with_text(para_block),
            'lines': _parse_line_spans(para_block, page_idx)
        }
    elif para_type == BlockType.TITLE:
        para_content = {
            'type': ContentType.TEXT,
            'text': pipeline_merge_para_with_text(para_block),
            'lines': _parse_line_spans(para_block, page_idx)
        }
        title_level = get_title_level(para_block)
        if title_level != 0:
            para_content['text_level'] = title_level
    elif para_type == BlockType.INTERLINE_EQUATION:
        if len(para_block['lines']) == 0 or len(para_block['lines'][0]['spans']) == 0:
            return None
        para_content = {
            'type': ContentType.EQUATION,
            'img_path': f"{img_buket_path}/{para_block['lines'][0]['spans'][0].get('image_path', '')}",
            'lines': _parse_line_spans(para_block, page_idx)
        }
        if para_block['lines'][0]['spans'][0].get('content', ''):
            para_content['text'] = pipeline_merge_para_with_text(para_block)
            para_content['text_format'] = 'latex'
    elif para_type == BlockType.IMAGE:
        image_lines_metas = []
        para_content = {'type': ContentType.IMAGE, 'img_path': '',
                        BlockType.IMAGE_CAPTION: [], BlockType.IMAGE_FOOTNOTE: []}
        for block in para_block['blocks']:
            image_lines_metas.extend(_parse_line_spans(block, page_idx))
            if block['type'] == BlockType.IMAGE_BODY:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.IMAGE:
                            if span.get('image_path', ''):
                                para_content['img_path'] = f"{img_buket_path}/{span['image_path']}"
            if block['type'] == BlockType.IMAGE_CAPTION:
                para_content[BlockType.IMAGE_CAPTION].append(pipeline_merge_para_with_text(block))
            if block['type'] == BlockType.IMAGE_FOOTNOTE:
                para_content[BlockType.IMAGE_FOOTNOTE].append(pipeline_merge_para_with_text(block))
        para_content['lines'] = image_lines_metas
    elif para_type == BlockType.TABLE:
        table_lines_metas = []
        para_content = {'type': ContentType.TABLE, 'img_path': '',
                        BlockType.TABLE_CAPTION: [], BlockType.TABLE_FOOTNOTE: []}
        for block in para_block['blocks']:
            table_lines_metas.extend(_parse_line_spans(block, page_idx))
            if block['type'] == BlockType.TABLE_BODY:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.TABLE:
                            if span.get('html', ''):
                                para_content[BlockType.TABLE_BODY] = f"{span['html']}"

                            if span.get('image_path', ''):
                                para_content['img_path'] = f"{img_buket_path}/{span['image_path']}"

            if block['type'] == BlockType.TABLE_CAPTION:
                para_content[BlockType.TABLE_CAPTION].append(pipeline_merge_para_with_text(block))
            if block['type'] == BlockType.TABLE_FOOTNOTE:
                para_content[BlockType.TABLE_FOOTNOTE].append(pipeline_merge_para_with_text(block))
        para_content['lines'] = table_lines_metas

    page_width, page_height = page_size
    para_bbox = para_block.get('bbox')
    if para_bbox:
        para_content['bbox'] = para_bbox
    else:
        para_content['bbox'] = [0, 0, 0, 0]

    para_content['page_idx'] = page_idx
    para_content['page_width'] = page_width
    para_content['page_height'] = page_height

    return para_content


pipeline_middle_json_mkcontent.make_blocks_to_content_list = pipeline_make_blocks_to_content_list


# # patches to vlm

def vlm_make_blocks_to_content_list(para_block, img_buket_path, page_idx, page_size):  # noqa: C901
    para_type = para_block['type']
    para_content = {}
    if para_type in [
        BlockType.TEXT,
        BlockType.REF_TEXT,
        BlockType.PHONETIC,
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.PAGE_NUMBER,
        BlockType.ASIDE_TEXT,
        BlockType.PAGE_FOOTNOTE,
    ]:
        para_content = {
            'type': para_type,
            'text': vlm_merge_para_with_text(para_block),
            'lines': _parse_line_spans(para_block, page_idx)
        }
    elif para_type == BlockType.LIST:
        lines = []
        para_content = {
            'type': para_type,
            'sub_type': para_block.get('sub_type', ''),
            'list_items': [],
        }
        for block in para_block['blocks']:
            item_text = vlm_merge_para_with_text(block)
            if item_text.strip():
                para_content['list_items'].append(item_text)
                lines.extend(_parse_line_spans(block, page_idx))
        para_content['lines'] = lines
    elif para_type == BlockType.TITLE:
        title_level = get_title_level(para_block)
        para_content = {
            'type': ContentType.TEXT,
            'text': vlm_merge_para_with_text(para_block),
            'lines': _parse_line_spans(para_block, page_idx)
        }
        if title_level != 0:
            para_content['text_level'] = title_level
    elif para_type == BlockType.INTERLINE_EQUATION:
        para_content = {
            'type': ContentType.EQUATION,
            'text': vlm_merge_para_with_text(para_block),
            'text_format': 'latex',
            'lines': _parse_line_spans(para_block, page_idx)
        }
    elif para_type == BlockType.IMAGE:
        image_lines_metas = []
        para_content = {'type': ContentType.IMAGE, 'img_path': '', BlockType.IMAGE_CAPTION: [],
                        BlockType.IMAGE_FOOTNOTE: []}
        for block in para_block['blocks']:
            image_lines_metas.extend(_parse_line_spans(block, page_idx))
            if block['type'] == BlockType.IMAGE_BODY:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.IMAGE:
                            if span.get('image_path', ''):
                                para_content['img_path'] = f"{img_buket_path}/{span['image_path']}"
            if block['type'] == BlockType.IMAGE_CAPTION:
                para_content[BlockType.IMAGE_CAPTION].append(vlm_merge_para_with_text(block))
            if block['type'] == BlockType.IMAGE_FOOTNOTE:
                para_content[BlockType.IMAGE_FOOTNOTE].append(vlm_merge_para_with_text(block))
        para_content['lines'] = image_lines_metas
    elif para_type == BlockType.TABLE:
        table_lines_metas = []
        para_content = {'type': ContentType.TABLE, 'img_path': '',
                        BlockType.TABLE_CAPTION: [], BlockType.TABLE_FOOTNOTE: []}
        for block in para_block['blocks']:
            table_lines_metas.extend(_parse_line_spans(block, page_idx))
            if block['type'] == BlockType.TABLE_BODY:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.TABLE:

                            if span.get('html', ''):
                                para_content[BlockType.TABLE_BODY] = f"{span['html']}"

                            if span.get('image_path', ''):
                                para_content['img_path'] = f"{img_buket_path}/{span['image_path']}"

            if block['type'] == BlockType.TABLE_CAPTION:
                para_content[BlockType.TABLE_CAPTION].append(vlm_merge_para_with_text(block))
            if block['type'] == BlockType.TABLE_FOOTNOTE:
                para_content[BlockType.TABLE_FOOTNOTE].append(vlm_merge_para_with_text(block))
        para_content['lines'] = table_lines_metas
    elif para_type == BlockType.CODE:
        code_lines_metas = []
        para_content = {'type': BlockType.CODE, 'sub_type': para_block['sub_type'], BlockType.CODE_CAPTION: []}
        for block in para_block['blocks']:
            code_lines_metas.extend(_parse_line_spans(block, page_idx))
            if block['type'] == BlockType.CODE_BODY:
                para_content[BlockType.CODE_BODY] = vlm_merge_para_with_text(block)
                if para_block['sub_type'] == BlockType.CODE:
                    para_content['guess_lang'] = para_block['guess_lang']
            if block['type'] == BlockType.CODE_CAPTION:
                para_content[BlockType.CODE_CAPTION].append(vlm_merge_para_with_text(block))
        para_content['lines'] = code_lines_metas
    page_width, page_height = page_size
    para_bbox = para_block.get('bbox')
    if para_bbox:
        para_content['bbox'] = para_bbox
    else:
        para_content['bbox'] = [0, 0, 0, 0]
    para_content['page_idx'] = page_idx
    para_content['page_width'] = page_width
    para_content['page_height'] = page_height
    return para_content

vlm_middle_json_mkcontent.make_blocks_to_content_list = vlm_make_blocks_to_content_list
