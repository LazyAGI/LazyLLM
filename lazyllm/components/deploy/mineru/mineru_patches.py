import copy
from mineru.backend.pipeline import pipeline_middle_json_mkcontent
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import merge_para_with_text as pipeline_merge_para_with_text
from mineru.backend.vlm import vlm_middle_json_mkcontent
from mineru.backend.vlm.vlm_middle_json_mkcontent import merge_para_with_text as vlm_merge_para_with_text
from mineru.utils.enum_class import BlockType, ContentType

# patches to mineru (to output bbox)

def _parse_line_spans(para_block, page_idx):
    lines_metas = []
    if 'lines' in para_block:
        for line_info in para_block['lines']:
            if not line_info['spans']:
                continue
            line_meta = copy.deepcopy(line_info['spans'][0])
            line_meta.pop('score', None)
            cross_page = line_meta.pop('cross_page', None)
            line_meta['page'] = page_idx + 1 if cross_page is True else page_idx
            lines_metas.append(line_meta)
    return lines_metas


# patches to pipeline

def pipeline_make_blocks_to_content_list(para_block, img_buket_path, page_idx):  # noqa: C901
    para_type = para_block['type']
    para_content = {}
    if para_type in [BlockType.TEXT, BlockType.LIST, BlockType.INDEX]:
        para_content = {
            'type': ContentType.TEXT,
            'text': pipeline_merge_para_with_text(para_block),
            'lines': _parse_line_spans(para_block, page_idx)
        }
    elif para_type == BlockType.TITLE:
        para_content = {
            'type': ContentType.TEXT,
            'text': pipeline_merge_para_with_text(para_block),
            'lines': _parse_line_spans(para_block, page_idx)
        }
        title_level = pipeline_middle_json_mkcontent.get_title_level(para_block)
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
        para_content = {
            'type': ContentType.IMAGE,
            'img_path': '',
            BlockType.IMAGE_CAPTION: [],
            BlockType.IMAGE_FOOTNOTE: []
        }
        for block in para_block['blocks']:
            image_lines_metas.extend(_parse_line_spans(block, page_idx))
            if block['type'] == BlockType.IMAGE_BODY:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.IMAGE:
                            if span.get('image_path', ''):
                                para_content['img_path'] = f"{img_buket_path}/{span['image_path']}"
            if block['type'] == BlockType.IMAGE_CAPTION:
                para_content[BlockType.IMAGE_CAPTION].append(
                    pipeline_merge_para_with_text(block))
            if block['type'] == BlockType.IMAGE_FOOTNOTE:
                para_content[BlockType.IMAGE_FOOTNOTE].append(
                    pipeline_merge_para_with_text(block))
        para_content['lines'] = image_lines_metas
    elif para_type == BlockType.TABLE:
        para_content = {
            'type': ContentType.TABLE,
            'img_path': '',
            BlockType.TABLE_CAPTION: [],
            BlockType.TABLE_FOOTNOTE: []
        }
        table_lines_metas = []
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
                para_content[BlockType.TABLE_CAPTION].append(
                    pipeline_merge_para_with_text(block))
            if block['type'] == BlockType.TABLE_FOOTNOTE:
                para_content[BlockType.TABLE_FOOTNOTE].append(
                    pipeline_merge_para_with_text(block))
        para_content['lines'] = table_lines_metas

    para_content['page_idx'] = page_idx
    para_content['bbox'] = para_block['bbox']
    return para_content


pipeline_middle_json_mkcontent.make_blocks_to_content_list = pipeline_make_blocks_to_content_list


# patches to vlm

def vlm_make_blocks_to_content_list(para_block, img_buket_path, page_idx):  # noqa: C901
    para_type = para_block['type']
    para_content = {}
    if para_type in [BlockType.TEXT, BlockType.LIST, BlockType.INDEX]:
        para_content = {
            'type': ContentType.TEXT,
            'text': vlm_merge_para_with_text(para_block),
            'lines': _parse_line_spans(para_block, page_idx)
        }
    elif para_type == BlockType.TITLE:
        title_level = vlm_middle_json_mkcontent.get_title_level(para_block)
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
        para_content = {
            'type': ContentType.IMAGE,
            'img_path': '',
            BlockType.IMAGE_CAPTION: [],
            BlockType.IMAGE_FOOTNOTE: []
        }
        for block in para_block['blocks']:
            image_lines_metas.extend(_parse_line_spans(block, page_idx))
            if block['type'] == BlockType.IMAGE_BODY:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.IMAGE:
                            if span.get('image_path', ''):
                                para_content['img_path'] = f"{img_buket_path}/{span['image_path']}"
            if block['type'] == BlockType.IMAGE_CAPTION:
                para_content[BlockType.IMAGE_CAPTION].append(
                    vlm_merge_para_with_text(block))
            if block['type'] == BlockType.IMAGE_FOOTNOTE:
                para_content[BlockType.IMAGE_FOOTNOTE].append(
                    vlm_merge_para_with_text(block))
        para_content['lines'] = image_lines_metas
    elif para_type == BlockType.TABLE:
        table_lines_metas = []
        para_content = {
            'type': ContentType.TABLE,
            'img_path': '',
            BlockType.TABLE_CAPTION: [],
            BlockType.TABLE_FOOTNOTE: []
        }
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
                para_content[BlockType.TABLE_CAPTION].append(
                    vlm_merge_para_with_text(block))
            if block['type'] == BlockType.TABLE_FOOTNOTE:
                para_content[BlockType.TABLE_FOOTNOTE].append(
                    vlm_merge_para_with_text(block))
        para_content['lines'] = table_lines_metas

    para_content['page_idx'] = page_idx
    para_content['bbox'] = para_block['bbox']
    return para_content

vlm_middle_json_mkcontent.make_blocks_to_content_list = vlm_make_blocks_to_content_list
