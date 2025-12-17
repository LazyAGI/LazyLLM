import base64
import requests
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Callable, Union

from lazyllm import LOG
from ..doc_node import DocNode
from .readerBase import LazyLLMReaderBase


class PaddleOCRPDFReader(LazyLLMReaderBase):
    def __init__(self, url: str = None, api_key: str = None,
                 callback: Optional[Callable[[List[dict], Path, dict], List[DocNode]]] = None,
                 format_block_content: bool = True,
                 use_layout_detection: bool = True,
                 use_chart_recognition: bool = True,
                 split_doc: bool = True,
                 drop_types: List[str] = None,
                 post_func: Optional[Callable] = None,
                 return_trace: bool = True,
                 images_dir: str = None):
        super().__init__(return_trace=return_trace)
        if not url and not api_key:
            raise ValueError('Either url or api_key must be provided')
        if url:
            self._url = url.rstrip('/') + '/layout-parsing'
        else:
            self._url = 'https://k4q3k6o0l1hbx6jc.aistudio-app.com/layout-parsing'
        if api_key:
            self._headers = {
                'Authorization': f'token {api_key}',
                'Content-Type': 'application/json'
            }
        else:
            self._headers = {'Content-Type': 'application/json'}
        self._format_block_content = format_block_content
        self._use_layout_detection = use_layout_detection
        self._use_chart_recognition = use_chart_recognition
        self._split_doc = split_doc
        self._post_func = post_func
        if images_dir:
            self._images_dir = Path(images_dir)
            self._images_dir.mkdir(exist_ok=True)
        else:
            self._images_dir = None
        self._drop_types = (
            list(drop_types)
            if drop_types is not None
            else ['aside_text', 'header', 'footer', 'number', 'header_image', 'seal']
        )

    def _load_data(self, file: Path, extra_info: Optional[Dict] = None,
                   use_cache: bool = True, **kwargs) -> List[DocNode]:
        try:
            if isinstance(file, str):
                file = Path(file)
            elements = self._parse_pdf_elements(file)
            docs = self._build_nodes(elements, file, extra_info)

            if self._post_func:
                docs = self._post_func(docs)
                assert isinstance(docs, list), f'Expected list, got {type(docs)}, please check your post function'
                for node in docs:
                    assert isinstance(node, DocNode), f'Expected DocNode, got {type(node)}, \
                        please check your post function'
                    node.global_metadata = extra_info
            if not docs:
                LOG.warning(f'[PaddleOCRPDFReader] No elements found in PDF: {file}')
            return docs
        except Exception as e:
            LOG.error(f'[PaddleOCRPDFReader] Error loading data from {file}: {e}')
            return []

    def _parse_pdf_elements(self, pdf_path: Path, **kwargs) -> Union[List[dict], List[str]]:    # noqa: C901
        if not pdf_path.exists():
            raise FileNotFoundError(f'File not found: {pdf_path}')
        try:
            with open(pdf_path, 'rb') as f:
                file_bytes = f.read()
                file_data = base64.b64encode(file_bytes).decode('ascii')

            payload = {
                'file': file_data,
                'fileType': 0 if str(pdf_path).endswith('.pdf') else 1,  # 0 for PDF, 1 for image
                'formatBlockContent': self._format_block_content,
                'useLayoutDetection': self._use_layout_detection,
                'useChartRecognition': self._use_chart_recognition,
                'prettifyMarkdown': True,
                **kwargs
            }

            response = requests.post(self._url, json=payload, headers=self._headers, timeout=600)
            response.raise_for_status()
            if response.status_code != 200:
                LOG.error(f'[PaddleOCRPDFReader] POST failed: {response.text}')
                return []

            res = response.json()
            if not isinstance(res, dict) or not res.get('result'):
                LOG.error(f'[PaddleOCRPDFReader] Invalid response: {res}')
                return []

            layout_parsing_results = res['result'].get('layoutParsingResults', [])
            if not layout_parsing_results:
                LOG.warning(f'[PaddleOCRPDFReader] No layout parsing results found in PDF: {pdf_path}')
                return []

            # parse result to md or json
            all_elements = []
            md_content = []
            if self._images_dir:
                img_save_dir = self._images_dir / str(uuid.uuid4())
                img_save_dir.mkdir(exist_ok=True)
            else:
                img_save_dir = None

            for page_idx, page_result in enumerate(layout_parsing_results):
                parsing_res_list = page_result.get('prunedResult', {}).get('parsing_res_list', [])

                # get markdown info
                markdown_info = page_result.get('markdown', {})
                md_text = markdown_info.get('text', '')
                md_images = markdown_info.get('images', {})

                # save images to images dir and update markdown image path
                if self._images_dir:
                    self._save_images(md_images, page_idx, img_save_dir)
                    for img_path, img_save_path in md_images.items():
                        md_text = md_text.replace(img_path, img_save_path)

                md_content.append(md_text)
                if not self._split_doc:
                    continue
                # extract content blocks
                page_elements = self._extract_content_blocks(parsing_res_list, page_idx, md_images)
                all_elements.extend(page_elements)

            return all_elements if self._split_doc else md_content

        except requests.exceptions.RequestException as e:
            LOG.error(f'[PaddleOCRPDFReader] POST failed: {e}, detail: \
                      {e.response.text if hasattr(e, "response") and e.response is not None else ""}')
            return []
        except Exception as e:
            LOG.exception(e)
            LOG.error(f'[PaddleOCRPDFReader] Error parsing PDF: {e}')
            return []

    def _extract_content_blocks(self, parsing_res_list: List[dict], page_idx: int, img_map: dict) -> List[dict]:
        '''Extract content blocks from PaddleOCR parsing results and convert to standard format.'''
        blocks = []
        for item in parsing_res_list:
            if item.get('block_label') in self._drop_types:
                continue

            block = {}
            block['bbox'] = item.get('block_bbox', [])
            block['type'] = item.get('block_label', 'text')
            block['page'] = page_idx
            block['content'] = item.get('block_content', '')

            if block['type'] in ['paragraph_title', 'doc_title']:
                block['text_level'] = block['content'].count('#')
                block['content'] = block['content'].replace('#', '').strip()
            elif block['type'] == 'image':
                image_path = []
                for img_name, img_obj in img_map.items():
                    if img_obj.startswith(('http://', 'https://')) or len(img_obj) <= 92:
                        block['content'] = block['content'].replace(img_name, img_obj)
                        image_path.append(img_obj)
                block['image_path'] = image_path
            elif block['type'] == 'ocr':
                block['type'] = 'page'

            blocks.append(block)
        return blocks

    def _build_nodes(self, elements: Union[List[dict], List[str]], file: Path,
                     extra_info: Optional[Dict] = None) -> List[DocNode]:
        docs = []
        if self._split_doc:
            for e in elements:
                metadata = {'file_name': file.name}
                metadata.update({k: v for k, v in e.items() if k != 'content'})
                node = DocNode(text=e.get('content', ''), metadata=metadata, global_metadata=extra_info)
                node.excluded_embed_metadata_keys = [k for k in metadata.keys() if k not in ['file_name']]
                node.excluded_llm_metadata_keys = [k for k in metadata.keys() if k not in ['file_name']]
                docs.append(node)
        else:
            text_chunks = elements if isinstance(elements, list) and elements and isinstance(elements[0], str) else []
            metadata = {'file_name': file.name}
            node = DocNode(text='\n'.join(text_chunks), metadata=metadata, global_metadata=extra_info)
            node.excluded_embed_metadata_keys = [k for k in metadata.keys() if k not in ['file_name']]
            node.excluded_llm_metadata_keys = [k for k in metadata.keys() if k not in ['file_name']]
            docs.append(node)
        return docs

    def _save_images(self, md_images: Dict[str, str], page_idx: int, img_save_dir: Path) -> None:
        for img_path, img_base64 in md_images.items():
            img_filename = Path(img_path).name
            img_filename = f'{page_idx}_{img_filename}'
            img_save_path = img_save_dir / img_filename

            try:
                # Determine if it's a URL or base64 encoded
                if isinstance(img_base64, str):
                    if img_base64.startswith(('http://', 'https://')):
                        # Download the image
                        response = requests.get(img_base64, timeout=10)
                        response.raise_for_status()
                        img_save_path.write_bytes(response.content)
                    # Check if it's a data URI format
                    elif img_base64.startswith('data:'):
                        # Extract base64 part (after the comma)
                        base64_data = img_base64.split(',', 1)[1] if ',' in img_base64 else img_base64
                        img_save_path.write_bytes(base64.b64decode(base64_data))
                    else:
                        # Pure base64 encoded
                        img_save_path.write_bytes(base64.b64decode(img_base64))
                else:
                    # If not a string, try to decode directly
                    img_save_path.write_bytes(base64.b64decode(img_base64))

                md_images[img_path] = str(img_save_path)
            except Exception as e:
                LOG.warning(f'[PaddleOCRPDFReader] Failed to save image {img_path}: {e}')
                md_images[img_path] = md_images[img_path]
                continue
