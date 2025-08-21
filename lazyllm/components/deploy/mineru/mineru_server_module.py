import os
import json
import subprocess
import platform
import shutil
import hashlib
import uuid
import tempfile
import atexit
from pathlib import Path
from fastapi import UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional, Union

from lazyllm import ServerModule, LOG
from lazyllm import FastapiApp as app

from mineru.cli.common import aio_do_parse, read_fn, pdf_suffixes, image_suffixes
from . import mineru_patches  # noqa: F401


def _check_libreoffice():
    system = platform.system()

    if system != 'Linux':
        LOG.warning(f'[MINERU SERVER] The current system type only supports PDF parsing: {system}')
        return False

    libreoffice_installed = False
    commands = ['libreoffice', 'soffice']

    for cmd in commands:
        try:
            result = subprocess.run([cmd, '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip().split('\n')[0]
                LOG.info(f'[MINERU SERVER] LibreOffice is installed: {version}')
                libreoffice_installed = True
                break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    if not libreoffice_installed:
        LOG.warning('[MINERU SERVER] LibreOffice is not installed, only PDF is supported')
        return False

    try:
        output = subprocess.check_output(['fc-list', ':lang=zh'], encoding='utf-8')
        if not output.strip():
            LOG.warning('[MINERU SERVER] No Chinese fonts were detected, \
                        the converted document may not display Chinese content properly. \
                        It is recommended to install Chinese fonts: sudo apt install fonts-noto-cjk')
    except Exception:
        LOG.error('[MINERU SERVER] Font check failed')

    return True


class MineruServerBase:
    def __init__(self, cache_dir: str = None, image_save_dir: str = None,
                 default_backend: str = 'pipeline', default_lang: str = 'ch_server',
                 default_parse_method: str = 'auto', default_formula_enable: bool = True,
                 default_table_enable: bool = True, default_return_md: bool = False,
                 default_return_content_list: bool = True, mem_fraction_static: float = 0.8):
        if default_backend not in ['pipeline', 'vlm-sglang-engine', 'vlm-transformers']:
            raise ValueError(f'Invalid backend: {default_backend}, \
                             only support pipeline, vlm-sglang-engine, vlm-transformers')
        if default_lang not in ['ch', 'ch_server', 'ch_lite', 'en']:
            raise ValueError(f'Invalid language: {default_lang}, \
                             only support ch, ch_server, ch_lite, en')
        self._default_backend = default_backend
        self._cache_dir = cache_dir
        if image_save_dir:
            self._image_save_dir = os.path.join(image_save_dir, 'images')
        else:
            self._image_save_dir = None
        self._default_lang = default_lang
        self._default_parse_method = default_parse_method
        self._default_formula_enable = default_formula_enable
        self._default_table_enable = default_table_enable
        self._default_return_md = default_return_md
        self._default_return_content_list = default_return_content_list
        self._mem_fraction_static = mem_fraction_static
        self._supported_office_types = ['.pptx', '.ppt', '.docx', '.doc'] if _check_libreoffice() else []
        LOG.info(f'[MINERU SERVER] Supported office types: {self._supported_office_types}')
        self._middle_file_dir = tempfile.mkdtemp()
        atexit.register(lambda: shutil.rmtree(self._middle_file_dir, ignore_errors=True))
        try:
            for path in [self._cache_dir, self._image_save_dir]:
                if path:
                    os.makedirs(path, exist_ok=True)
        except Exception as e:
            raise Exception(f'Failed to create directory: {e}')

    @app.post('/api/v1/pdf_parse')
    async def parse_pdf(self,   # noqa: C901
                        files: List[str] = Form([]),  # noqa B008
                        upload_files: List[UploadFile] = File([]),  # noqa B008
                        use_cache: bool = Form(False, description='if True, chache_dir should be set'),  # noqa B008
                        lang: str = Form('ch_server',  # noqa B008
                                         description='only use for pipeline,ch|ch_server|ch_lite|en'),
                        backend: str = Form(None, description='Parsing mode, vlm-sglang-engine|pipeline'),  # noqa B008
                        parse_method: str = Form('auto'),  # noqa B008
                        formula_enable: bool = Form(None, description='Whether to enable formula parsing'),  # noqa B008
                        table_enable: bool = Form(None, description='Whether to enable table parsing'),  # noqa B008
                        return_md: bool = Form(None, description='Whether to return markdown content'),  # noqa B008
                        return_content_list: bool = Form(None, description='Whether to return content list')):  # noqa B008
        if files and upload_files:
            raise HTTPException(status_code=400, detail='Either provide only \'files\' or only \'upload_files\'!')
        for file in files:
            if not os.path.isfile(file):
                raise HTTPException(status_code=400, detail=f'File Not Found: {file}')

        if lang and lang not in ['ch', 'ch_server', 'ch_lite', 'en']:
            raise HTTPException(status_code=400, detail=f'Invalid language: {lang}, \
                                only support ch, ch_server, ch_lite, en')

        if backend and backend not in ['pipeline', 'vlm-sglang-engine', 'vlm-transformers']:
            raise HTTPException(status_code=400, detail=f'Invalid backend: {backend}, \
                                only support pipeline, vlm-sglang-engine, vlm-transformers')

        unique_id = str(uuid.uuid4())
        unique_dir = os.path.join(self._middle_file_dir, unique_id)
        os.makedirs(unique_dir, exist_ok=True)

        if upload_files:
            files = await self._resolve_upload_files(upload_files, unique_dir)

        for file in files:
            if Path(file).suffix.lower() not in self._supported_office_types + ['.pdf']:
                raise HTTPException(status_code=400, detail=f'Unsupported file type: {Path(file).suffix}')

        backend = backend or self._default_backend
        lang = lang or self._default_lang
        parse_method = parse_method or self._default_parse_method
        formula_enable = formula_enable if formula_enable is not None else self._default_formula_enable
        table_enable = table_enable if table_enable is not None else self._default_table_enable
        return_md = return_md if return_md is not None else self._default_return_md
        return_content_list = return_content_list if return_content_list is not None \
            else self._default_return_content_list

        LOG.info(f'[MINERU SERVER] GOT FILE {[Path(file).stem for file in files]} --- BACKEND: {backend}')

        try:
            results = {file: {} for file in files}
            if use_cache and not self._cache_dir:
                LOG.warning('[MINERU SERVER] CACHE_DIR is not set, the Cache will not be used!')

            files_to_process = files
            if use_cache and self._cache_dir:
                results, files_to_process = self._check_cache(files, results, backend,
                                                              return_md, return_content_list,
                                                              table_enable, formula_enable)
                if not files_to_process:
                    LOG.info(f'[MINERU SERVER] RETURN RESULTS FROM CACHE: {files}')
                    results = [results[file] for file in files]
                    return JSONResponse(status_code=200, content={'result': results, 'unique_id': unique_id})

            mineru_results = await self._run_mineru(files_to_process, unique_dir, backend, lang,
                                                    parse_method, formula_enable, table_enable,
                                                    return_md, return_content_list)
            results.update(mineru_results)
            results = [results[file] for file in files]
            LOG.info(f'[MINERU SERVER] RETURN RESULTS: {files}')
            return JSONResponse(status_code=200,
                                content={'result': results, 'unique_id': unique_id})
        except Exception as e:
            LOG.error(f'[MINERU SERVER] Parse Failed: {str(e)}')
            return JSONResponse(status_code=500,
                                content={'error': f'Failed to process file: {str(e)}'})
        finally:
            shutil.rmtree(unique_dir)

    async def _run_mineru(self, files_to_process, unique_dir, backend, lang,  # noqa: C901
                          parse_method, formula_enable, table_enable,
                          return_md, return_content_list):
        results = {file: {} for file in files_to_process}

        pdf_file_names = []
        pdf_bytes_list = []

        for file in files_to_process:
            pdf_file_name, pdf_byte = self._load_files(Path(file), unique_dir)
            pdf_file_names.append(pdf_file_name)
            pdf_bytes_list.append(pdf_byte)

        lang_list = [lang] * len(pdf_bytes_list)

        params = dict(output_dir=unique_dir, pdf_file_names=pdf_file_names,
                      pdf_bytes_list=pdf_bytes_list, p_lang_list=lang_list, backend=backend,
                      parse_method=parse_method, formula_enable=formula_enable,
                      table_enable=table_enable, f_draw_layout_bbox=False, f_draw_span_bbox=False,
                      f_dump_md=True, f_dump_middle_json=False, f_dump_model_output=False,
                      f_dump_orig_pdf=False, f_dump_content_list=True)
        if backend == 'vlm-sglang-engine':
            params['mem_fraction_static'] = self._mem_fraction_static

        await aio_do_parse(**params)

        for pdf_name, pdf_path in zip(pdf_file_names, files_to_process):
            # Directory output by mineru
            if backend.startswith('pipeline'):
                parse_dir = os.path.join(unique_dir, pdf_name, parse_method)
            else:
                parse_dir = os.path.join(unique_dir, pdf_name, 'vlm')

            if os.path.exists(parse_dir):
                hash_id = self._file_sha256(pdf_path)
                md_content = self._read_parse_result('.md', pdf_name, parse_dir)
                content_list = self._read_parse_result('_content_list.json', pdf_name, parse_dir)

                if return_md:
                    if md_content:
                        results[pdf_path]['md_content'] = md_content
                if return_content_list:
                    if content_list:
                        results[pdf_path]['content_list'] = content_list
                if self._cache_dir:
                    self._cache_parse_result(hash_id, results[pdf_path], mode=backend,
                                             table_enable=table_enable,
                                             formula_enable=formula_enable)

                if self._image_save_dir:
                    source_dir = Path(f'{parse_dir}/images/')
                    target_dir = Path(self._image_save_dir)
                    for jpg_file in source_dir.glob('*.jpg'):
                        shutil.move(str(jpg_file), str(target_dir / jpg_file.name))

        return results

    async def _resolve_upload_files(self, upload_files: List[UploadFile], unique_dir: str) -> List[str]:
        if not upload_files:
            return []

        temp_upload_dir = os.path.join(self._middle_file_dir, f'{unique_dir}/upload')
        os.makedirs(temp_upload_dir, exist_ok=True)
        file_paths = []
        for upload_file in upload_files:
            content = await upload_file.read()
            temp_file_path = os.path.join(temp_upload_dir, upload_file.filename)
            with open(temp_file_path, 'wb') as f:
                f.write(content)
            file_paths.append(temp_file_path)
        return file_paths

    def _get_func_suffix(self, table_enable, formula_enable):
        if table_enable and formula_enable:
            return '_a'
        elif table_enable:
            return '_t'
        elif formula_enable:
            return '_f'
        else:
            return '_n'

    def _check_cache(self, files, results, backend, return_md, return_content_list,
                     table_enable, formula_enable):
        if not self._cache_dir:
            return results, files

        func_suffix = self._get_func_suffix(table_enable, formula_enable)
        func_suffix_map = {'_a': ['_a'],
                           '_t': ['_t', '_a'],
                           '_f': ['_f', '_a'],
                           '_n': ['_n', '_a', '_t', '_f']}
        func_suffix_list = func_suffix_map[func_suffix]

        uncached_files = []

        for file in files:
            file_hash = self._file_sha256(file)
            valid_hash_ids = [file_hash + func_suffix for func_suffix in func_suffix_list]
            result = {}

            file_content_list_found = False
            file_md_found = False

            if return_content_list:
                for valid_hash in valid_hash_ids:
                    json_path = os.path.join(self._cache_dir, backend, f'{valid_hash}_content_list.json')
                    if os.path.isfile(json_path):
                        with open(json_path, 'r', encoding='utf-8') as f:
                            result['content_list'] = json.load(f)
                        file_content_list_found = True
                        break

            if return_md:
                for valid_hash in valid_hash_ids:
                    md_path = os.path.join(self._cache_dir, backend, f'{valid_hash}.md')
                    if os.path.isfile(md_path):
                        with open(md_path, 'r', encoding='utf-8') as f:
                            result['md_content'] = f.read()
                        file_md_found = True
                        break

            results[file].update(result)

            file_cache_complete = True
            if return_content_list and not file_content_list_found:
                file_cache_complete = False
            if return_md and not file_md_found:
                file_cache_complete = False

            if not file_cache_complete:
                uncached_files.append(file)

        return results, uncached_files

    def _read_parse_result(self, file_suffix_identifier: str,
                           pdf_name: str, parse_dir: str) -> Optional[Union[str, dict]]:
        result_file_path = os.path.join(parse_dir, f'{pdf_name}{file_suffix_identifier}')
        if os.path.exists(result_file_path):
            try:
                if file_suffix_identifier == '.md':
                    with open(result_file_path, 'r', encoding='utf-8') as fp:
                        return fp.read()
                elif file_suffix_identifier == '_content_list.json':
                    with open(result_file_path, 'r', encoding='utf-8') as fp:
                        return json.load(fp)
            except Exception:
                LOG.error(f'[MINERU SERVER] Failed to read result file {result_file_path}')
                return None
        return None

    def _cache_parse_result(self, hash_id: str, result: dict, mode: str,
                            table_enable: bool, formula_enable: bool):
        try:
            cache_subdir = os.path.join(self._cache_dir, mode)
            os.makedirs(cache_subdir, exist_ok=True)

            if table_enable and formula_enable:
                func_suffix = '_a'
            elif table_enable:
                func_suffix = '_t'
            elif formula_enable:
                func_suffix = '_f'
            else:
                func_suffix = '_n'

            hash_id += func_suffix
            md_content = result.get('md_content', None)
            if md_content:
                cache_path = os.path.join(cache_subdir, f'{hash_id}.md')
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(md_content)

            content_list = result.get('content_list', None)
            if content_list:
                cache_path = os.path.join(cache_subdir, f'{hash_id}_content_list.json')
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(content_list, f, ensure_ascii=False, indent=4)

        except Exception as e:
            LOG.error(f'Failed to cache data for {hash_id}: {e}')

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
                raise HTTPException(status_code=400, detail=f'File Not Found: {file_path}: {e}')
        else:
            raise HTTPException(status_code=400, detail=f'Unsupported file type: {file_path.suffix}')

    def _convert_file_to_pdf(self, input_path, output_dir):
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f'The input file {input_path} does not exist.')

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
            raise Exception(f'LibreOffice convert failed: {process.stderr.decode()}')

    def _file_sha256(self, file_path: str) -> str:
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()


class MineruServer(ServerModule):
    def __init__(self,
                 cache_dir: str = None,
                 image_save_dir: str = None,
                 default_backend: str = 'pipeline',
                 default_lang: str = 'ch_server',
                 default_parse_method: str = 'auto',
                 default_formula_enable: bool = True,
                 default_table_enable: bool = True,
                 default_return_md: bool = False,
                 default_return_content_list: bool = True,
                 *args, **kwargs):
        mineru_server = MineruServerBase(
            cache_dir=cache_dir, image_save_dir=image_save_dir, default_backend=default_backend,
            default_lang=default_lang, default_parse_method=default_parse_method,
            default_formula_enable=default_formula_enable, default_table_enable=default_table_enable,
            default_return_md=default_return_md, default_return_content_list=default_return_content_list)
        super().__init__(mineru_server, *args, **kwargs)
