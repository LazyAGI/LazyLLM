import os
import json
import subprocess
import platform
import shutil
import hashlib
import uuid
import copy
from pathlib import Path
from glob import glob
from fastapi import UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional, Union
import tempfile, shutil, atexit
from mineru.cli.common import aio_do_parse, read_fn, pdf_suffixes, image_suffixes

from lazyllm import ServerModule, LOG
from lazyllm import FastapiApp as app

################################# patches to mineru ###############################################

def parse_line_spans(para_block, page_idx):
    lines_metas = []
    page = page_idx
    if 'lines' in para_block:
        for line_info in para_block['lines']:
            if not line_info['spans']:
                continue
            line_meta = copy.deepcopy(line_info['spans'][0])
            line_meta.pop('score', None)
            if_cross_page = line_meta.pop('cross_page', None)
            line_meta['page'] = page + 1 if if_cross_page == True else page
            lines_metas.append(line_meta)
    return lines_metas

# patches to pipeline
from mineru.backend.pipeline import pipeline_middle_json_mkcontent
from mineru.utils.enum_class import BlockType, ContentType
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import merge_para_with_text

def make_blocks_to_content_list(para_block, img_buket_path, page_idx):
    para_type = para_block['type']
    para_content = {}
    if para_type in [BlockType.TEXT, BlockType.LIST, BlockType.INDEX]:
        para_content = {
            'type': ContentType.TEXT,
            'text': merge_para_with_text(para_block),
            'lines': parse_line_spans(para_block, page_idx)
        }
    elif para_type == BlockType.TITLE:
        para_content = {
            'type': ContentType.TEXT,
            'text': merge_para_with_text(para_block),
            'lines': parse_line_spans(para_block, page_idx)
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
            'lines': parse_line_spans(para_block, page_idx)
        }
        if para_block['lines'][0]['spans'][0].get('content', ''):
            para_content['text'] = merge_para_with_text(para_block)
            para_content['text_format'] = 'latex'
    elif para_type == BlockType.IMAGE:
        image_lines_metas = []
        para_content = {'type': ContentType.IMAGE, 'img_path': '', BlockType.IMAGE_CAPTION: [], BlockType.IMAGE_FOOTNOTE: []}
        for block in para_block['blocks']:
            image_lines_metas.extend(parse_line_spans(block, page_idx))
            if block['type'] == BlockType.IMAGE_BODY:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.IMAGE:
                            if span.get('image_path', ''):
                                para_content['img_path'] = f"{img_buket_path}/{span['image_path']}"
            if block['type'] == BlockType.IMAGE_CAPTION:
                para_content[BlockType.IMAGE_CAPTION].append(merge_para_with_text(block))
            if block['type'] == BlockType.IMAGE_FOOTNOTE:
                para_content[BlockType.IMAGE_FOOTNOTE].append(merge_para_with_text(block))
        para_content['lines'] = image_lines_metas
    elif para_type == BlockType.TABLE:
        para_content = {'type': ContentType.TABLE, 'img_path': '', BlockType.TABLE_CAPTION: [], BlockType.TABLE_FOOTNOTE: []}
        table_lines_metas = []
        for block in para_block['blocks']:
            table_lines_metas.extend(parse_line_spans(block, page_idx))
            if block['type'] == BlockType.TABLE_BODY:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.TABLE:
                            if span.get('html', ''):
                                para_content[BlockType.TABLE_BODY] = f"{span['html']}"

                            if span.get('image_path', ''):
                                para_content['img_path'] = f"{img_buket_path}/{span['image_path']}"

            if block['type'] == BlockType.TABLE_CAPTION:
                para_content[BlockType.TABLE_CAPTION].append(merge_para_with_text(block))
            if block['type'] == BlockType.TABLE_FOOTNOTE:
                para_content[BlockType.TABLE_FOOTNOTE].append(merge_para_with_text(block))
        para_content['lines'] = table_lines_metas
        
    para_content['page_idx'] = page_idx
    para_content['bbox'] = para_block['bbox']
    return para_content

pipeline_middle_json_mkcontent.make_blocks_to_content_list = make_blocks_to_content_list


# patches to vlm
from mineru.backend.vlm import vlm_middle_json_mkcontent
from mineru.utils.enum_class import MakeMode, BlockType, ContentType
from mineru.backend.vlm.vlm_middle_json_mkcontent import merge_para_with_text

def make_blocks_to_content_list(para_block, img_buket_path, page_idx):
    para_type = para_block['type']
    para_content = {}
    if para_type in [BlockType.TEXT, BlockType.LIST, BlockType.INDEX]:
        para_content = {
            'type': ContentType.TEXT,
            'text': merge_para_with_text(para_block),
            'lines': parse_line_spans(para_block, page_idx)
        }
    elif para_type == BlockType.TITLE:
        title_level = vlm_middle_json_mkcontent.get_title_level(para_block)
        para_content = {
            'type': ContentType.TEXT,
            'text': merge_para_with_text(para_block),
            'lines': parse_line_spans(para_block, page_idx)
        }
        if title_level != 0:
            para_content['text_level'] = title_level
    elif para_type == BlockType.INTERLINE_EQUATION:
        para_content = {
            'type': ContentType.EQUATION,
            'text': merge_para_with_text(para_block),
            'text_format': 'latex',
            'lines': parse_line_spans(para_block, page_idx)
        }
    elif para_type == BlockType.IMAGE:
        image_lines_metas = []
        para_content = {'type': ContentType.IMAGE, 'img_path': '', BlockType.IMAGE_CAPTION: [], BlockType.IMAGE_FOOTNOTE: []}
        for block in para_block['blocks']:
            image_lines_metas.extend(parse_line_spans(block, page_idx))
            if block['type'] == BlockType.IMAGE_BODY:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.IMAGE:
                            if span.get('image_path', ''):
                                para_content['img_path'] = f"{img_buket_path}/{span['image_path']}"
            if block['type'] == BlockType.IMAGE_CAPTION:
                para_content[BlockType.IMAGE_CAPTION].append(merge_para_with_text(block))
            if block['type'] == BlockType.IMAGE_FOOTNOTE:
                para_content[BlockType.IMAGE_FOOTNOTE].append(merge_para_with_text(block))
        para_content['lines'] = image_lines_metas
    elif para_type == BlockType.TABLE:
        table_lines_metas = []
        para_content = {'type': ContentType.TABLE, 'img_path': '', BlockType.TABLE_CAPTION: [], BlockType.TABLE_FOOTNOTE: []}
        for block in para_block['blocks']:
            table_lines_metas.extend(parse_line_spans(block, page_idx))
            if block['type'] == BlockType.TABLE_BODY:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.TABLE:

                            if span.get('html', ''):
                                para_content[BlockType.TABLE_BODY] = f"{span['html']}"

                            if span.get('image_path', ''):
                                para_content['img_path'] = f"{img_buket_path}/{span['image_path']}"

            if block['type'] == BlockType.TABLE_CAPTION:
                para_content[BlockType.TABLE_CAPTION].append(merge_para_with_text(block))
            if block['type'] == BlockType.TABLE_FOOTNOTE:
                para_content[BlockType.TABLE_FOOTNOTE].append(merge_para_with_text(block))
        para_content['lines'] = table_lines_metas

    para_content['page_idx'] = page_idx
    para_content['bbox'] = para_block['bbox']
    return para_content

vlm_middle_json_mkcontent.make_blocks_to_content_list = make_blocks_to_content_list
#############################################################################################

os.environ['MINERU_MODEL_SOURCE'] = "modelscope"


def check_libreoffice():
    system = platform.system()
    
    if system != "Linux":
        LOG.warning(f"[MINERU SERVER] The current system type only supports PDF parsing: {system}")
        return False
    
    libreoffice_installed = False
    commands = ['libreoffice', 'soffice']
    
    for cmd in commands:
        try:
            result = subprocess.run([cmd, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip().split('\n')[0]
                LOG.info(f"[MINERU SERVER] LibreOffice is installed: {version}")
                libreoffice_installed = True
                break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    if not libreoffice_installed:
        LOG.warning("[MINERU SERVER] LibreOffice is not installed, only PDF is supported")
        return False

    try:
        output = subprocess.check_output(['fc-list', ':lang=zh'], encoding='utf-8')
        if not output.strip(): 
            LOG.warning("[MINERU SERVER] No Chinese fonts were detected, the converted document may not display Chinese content properly.")
            LOG.warning("[MINERU SERVER] It is recommended to install Chinese fonts: sudo apt install fonts-noto-cjk")
    except Exception as e:
        LOG.error(f"[MINERU SERVER] Font check failed")

    return True


class MineruServer:
    def __init__(self, 
                 cache_dir: str = None,                       
                 image_save_dir: str = None, 
                 default_backend: str = "vlm-sglang-engine", 
                 default_lang_list: List[str] = ["ch_server"],
                 default_parse_method: str = "auto",
                 default_formula_enable: bool = True,
                 default_table_enable: bool = True,
                 default_return_md: bool = False,
                 default_return_content_list: bool = True,
                 mem_fraction_static: float = 0.5,
                 ):
        self._default_backend = default_backend
        self._cache_dir = cache_dir
        self._image_save_dir = image_save_dir
        self._default_lang_list = default_lang_list
        self._default_parse_method = default_parse_method
        self._default_formula_enable = default_formula_enable
        self._default_table_enable = default_table_enable
        self._default_return_md = default_return_md
        self._default_return_content_list = default_return_content_list
        self._mem_fraction_static = mem_fraction_static
        self._supported_office_types = ['.pptx', '.ppt', '.docx', '.doc'] if check_libreoffice() else []
        LOG.info(f"[MINERU SERVER] Supported office types: {self._supported_office_types}")
        self._middle_file_dir = tempfile.mkdtemp()
        atexit.register(lambda: shutil.rmtree(self._middle_file_dir, ignore_errors=True))
        try:
            for path in [self._cache_dir, self._image_save_dir]:
                if path:
                    os.makedirs(path, exist_ok=True)
        except Exception as e:
            raise Exception(f"Failed to create directory: {e}")
    
    @app.post("/api/v1/pdf_parse")
    async def parse_pdf(self, 
                        files: List[str] = Form([]),
                        upload_files: List[UploadFile] = File([]),
                        use_cache: bool = Form(False, description="Whether to use cache, chache_dir should be set if use_cache is True"), 
                        lang_list: List[str] = Form(["ch_server"], description="Language list, supports 'ch_server' or 'ch'"), 
                        backend: str = Form("pipeline", description="Parsing mode, supports 'pipeline' or 'vlm-sglang-engine'"),
                        parse_method: str = Form("auto"),
                        formula_enable: bool = Form(True, description="Whether to enable formula parsing"),
                        table_enable: bool = Form(True, description="Whether to enable table parsing"),
                        return_md: bool = Form(False, description="Whether to return markdown content"),
                        return_content_list: bool = Form(True, description="Whether to return content list")):
        LOG.info(f"[MINERU SERVER] GOT INITIAL FILES: {files}")
        if files and upload_files:
            raise HTTPException(status_code=400, detail="Either provide only 'files' or only 'upload_files'!")
        for file in files:
            if not os.path.isfile(file):
                raise HTTPException(status_code=400, detail=f"File Not Found: {file}")
            
        unique_id = str(uuid.uuid4())
        
        if upload_files:
            files = await self._resolve_upload_files(upload_files, unique_id)
        
        for file in files:
            if Path(file).suffix.lower() not in self._supported_office_types + [".pdf"]:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {Path(file).suffix}")
        
        backend = backend or self._default_backend
        lang_list = lang_list or self._default_lang_list
        parse_method = parse_method or self._default_parse_method
        formula_enable = formula_enable or self._default_formula_enable
        table_enable = table_enable or self._default_table_enable
        return_md = return_md or self._default_return_md
        
        LOG.info(f"[MINERU SERVER] GOT FILE {[Path(file).stem for file in files]} --- BACKEND: {backend}")
        results = {file:{} for file in files}
        if use_cache and not self._cache_dir:
            LOG.warning("[MINERU SERVER] CACHE_DIR is not set, the Cache will not be used!")
        
        if use_cache and self._cache_dir:
            cache_complete, results = self._check_cache(files, results, backend, return_md, return_content_list)
            if cache_complete:
                LOG.info(f"[MINERU SERVER] RETURN RESULTS FROM CACHE: {files}")
                results = [results[file] for file in files]
                return JSONResponse(status_code=200, content={"result": results, "unique_id": unique_id})


        try:
            unique_dir = os.path.join(self._middle_file_dir, unique_id)
            os.makedirs(unique_dir, exist_ok=True)
            
            files_to_process = []
            for file in files:
                need_process = False
                if return_md and not results[file].get('md_content'):
                    need_process = True
                if return_content_list and not results[file].get('content_list'):
                    need_process = True
                if need_process:
                    files_to_process.append(file)

            pdf_file_names = []
            pdf_bytes_list = []

            for file in files_to_process:
                if not os.path.isfile(file):
                    return JSONResponse(
                        status_code=400,
                        content={"error": f"File Not Found: {file}"}
                    )
                pdf_file_name, pdf_byte = self._load_files(Path(file), unique_dir)
                pdf_file_names.append(pdf_file_name)
                pdf_bytes_list.append(pdf_byte)

            actual_lang_list = lang_list
            if len(actual_lang_list) != len(pdf_file_names):
                actual_lang_list = [actual_lang_list[0] if actual_lang_list else "ch"] * len(pdf_file_names)

            await aio_do_parse(
                output_dir=unique_dir,
                pdf_file_names=pdf_file_names,
                pdf_bytes_list=pdf_bytes_list,
                p_lang_list=actual_lang_list,
                backend=backend,
                parse_method=parse_method,
                formula_enable=formula_enable,
                table_enable=table_enable,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_dump_md=True,
                f_dump_middle_json=False,
                f_dump_model_output=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=True,
                mem_fraction_static=self._mem_fraction_static
            )
            
            for pdf_name, pdf_path in zip(pdf_file_names, files_to_process):
                if backend.startswith("pipeline"):
                    parse_dir = os.path.join(unique_dir, pdf_name, parse_method)
                else:
                    parse_dir = os.path.join(unique_dir, pdf_name, "vlm")

                if os.path.exists(parse_dir):
                    hash_id = self._file_sha256(pdf_path)
                    md_content = self._read_parse_result(".md", pdf_name, parse_dir)
                    content_list = self._read_parse_result("_content_list.json", pdf_name, parse_dir)
                    
                    if return_md:
                        if md_content:
                            results[pdf_path]["md_content"] = md_content
                    if return_content_list:
                        if content_list:
                            results[pdf_path]["content_list"] = content_list
                    if self._cache_dir:
                        self._cache_parse_result(hash_id, results[pdf_path], mode=backend)
                        
                    if self._image_save_dir:
                        source_dir = Path(f"{parse_dir}/images/")
                        target_dir = Path(self._image_save_dir)
                        for jpg_file in source_dir.glob("*.jpg"):
                            shutil.move(str(jpg_file), str(target_dir / jpg_file.name))
                
            shutil.rmtree(unique_dir)
            results = [results[file] for file in files]
            LOG.info(f"[MINERU SERVER] RETURN RESULTS: {files}")
            return JSONResponse(
                status_code=200,
                content={
                    'result': results,
                    'unique_id': unique_id
                }
            )
        except Exception as e:
            LOG.error(f"[MINERU SERVER] Parse Failed: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to process file: {str(e)}"}
            )
    
    async def _resolve_upload_files(self, upload_files: List[UploadFile], unique_id: str) -> List[str]:
        """将上传的 UploadFile 保存为临时文件路径列表"""
        if not upload_files:
            return []

        temp_upload_dir = os.path.join(self._middle_file_dir, f"upload_{unique_id}")
        os.makedirs(temp_upload_dir, exist_ok=True)
        file_paths = []
        for upload_file in upload_files:
            content = await upload_file.read()
            temp_file_path = os.path.join(temp_upload_dir, upload_file.filename)
            with open(temp_file_path, "wb") as f:
                f.write(content)
            file_paths.append(temp_file_path)
        return file_paths
    
    def _check_cache(self, files, results, backend, return_md, return_content_list):
        """检查缓存是否存在，如果存在则直接返回"""
        if not self._cache_dir:
            return False, results
        
        cache_complete = True

        for file in files:
            hash_id = self._file_sha256(file)
            result = {}

            if return_content_list:
                json_path = os.path.join(self._cache_dir, backend, f"{hash_id}_content_list.json")
                if os.path.isfile(json_path):
                    with open(json_path, "r") as f:
                        result["content_list"] = json.load(f)
                else:
                    cache_complete = False

            if return_md:
                md_path = os.path.join(self._cache_dir, backend, f"{hash_id}.md")
                if os.path.isfile(md_path):
                    with open(md_path, "r", encoding="utf-8") as f:
                        result["md_content"] = f.read()
                else:
                    cache_complete = False

            results[file].update(result)

        if not (return_md or return_content_list):
            cache_complete = False

        return cache_complete, results

    def _read_parse_result(self, file_suffix_identifier: str, pdf_name: str, parse_dir: str) -> Optional[Union[str, dict]]:
        """从结果文件中读取解析结果"""
        result_file_path = os.path.join(parse_dir, f"{pdf_name}{file_suffix_identifier}")
        if os.path.exists(result_file_path):
            try:
                if file_suffix_identifier == ".md":
                    with open(result_file_path, "r", encoding="utf-8") as fp:
                        return fp.read()
                elif file_suffix_identifier == "_content_list.json":
                    with open(result_file_path, "r", encoding="utf-8") as fp:
                        return json.load(fp)
            except Exception as e:
                LOG.error(f"[MINERU SERVER] Failed to read result file {result_file_path}: {e}")
                return None
        return None
    
    def _cache_parse_result(self, hash_id: str, result: dict, mode: str):
        """缓解析结果到文件"""
        try:
            cache_subdir = os.path.join(self._cache_dir, mode)
            os.makedirs(cache_subdir, exist_ok=True)
            
            md_content = result.get('md_content', None)
            if md_content:
                cache_path = os.path.join(cache_subdir, f"{hash_id}.md")
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(md_content)
                
            content_list = result.get('content_list', None)
            if content_list:
                cache_path = os.path.join(cache_subdir, f"{hash_id}_content_list.json")
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(content_list, f, ensure_ascii=False, indent=4)
                
        except Exception as e:
            LOG.error(f"Failed to cache data for {hash_id}: {e}")
    
    def _load_files(self, file_path: str, unique_dir: str):
        suffix = file_path.suffix.lower()
        if suffix in pdf_suffixes + image_suffixes + self._supported_office_types:
            if suffix in self._supported_office_types:
                self._convert_file_to_pdf(file_path, unique_dir)
                output_path = os.path.join(unique_dir, file_path.name.replace(suffix, '.pdf'))
                file_path = Path(output_path)
            try:
                pdf_bytes = read_fn(file_path)
                return (file_path.stem, pdf_bytes)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"File Not Found: {file_path}")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_path.suffix}")
    
    def _convert_file_to_pdf(self, input_path, output_dir):
        """Convert a single document (ppt, doc, etc.) to PDF."""
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"The input file {input_path} does not exist.")

        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            'libreoffice',
            '--headless',
            '--norestore',
            '--invisible',
            '--convert-to', 'pdf',
            '--outdir', str(output_dir),
            str(input_path)
        ]

        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if process.returncode != 0:
            raise Exception(f"LibreOffice convert failed: {process.stderr.decode()}")
        
    def _file_sha256(self, file_path: str) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


class MineruServerModule(ServerModule):
    def __init__(self,                  
                 cache_dir: str = None,                              
                 image_save_dir: str = None,                          
                 default_backend: str = "vlm-sglang-engine", 
                 default_lang_list: List[str] = ["ch_server"],
                 default_parse_method: str = "auto",
                 default_formula_enable: bool = True,
                 default_table_enable: bool = True,
                 default_return_md: bool = False,
                 default_return_content_list: bool = True,
                 *args, **kwargs):
        mineru_server = MineruServer(cache_dir=cache_dir, 
                                     image_save_dir=image_save_dir, 
                                     default_backend=default_backend, 
                                     default_lang_list=default_lang_list, 
                                     default_parse_method=default_parse_method, 
                                     default_formula_enable=default_formula_enable, 
                                     default_table_enable=default_table_enable, 
                                     default_return_md=default_return_md, 
                                     default_return_content_list=default_return_content_list)
        super().__init__(mineru_server, *args, **kwargs)

    