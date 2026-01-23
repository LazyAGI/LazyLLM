import os
import json
import subprocess
import platform
import shutil
import hashlib
import uuid
import tempfile
import atexit
import asyncio
import signal
import contextlib
from pathlib import Path
from typing import List, Optional, Union, Any, Dict
from lazyllm.thirdparty import fastapi

from lazyllm import LOG
from lazyllm import FastapiApp as app
from lazyllm.module import ServerModule

from lazyllm.thirdparty import mineru


def patch_mineru():
    from . import mineru_patches  # noqa: F401


mineru.register_patches(patch_mineru)

os.environ['TORCHDYNAMO_DISABLE'] = '1'

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


def calculate_file_hash(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

class LibreOfficeHelper:
    '''doc/docx/ppt/pptx → PDF via headless LibreOffice.'''

    DEFAULT_TIMEOUT = int(os.getenv('MINERU_LO_TIMEOUT', '1800'))  # seconds
    DEFAULT_MAX_RETRIES = int(os.getenv('MINERU_LO_MAX_RETRIES', '1'))

    SUPPORTED_SUFFIXES = {'.doc', '.docx', '.ppt', '.pptx'}
    FILTER_MAP = {
        '.doc': 'writer_pdf_Export',
        '.docx': 'writer_pdf_Export',
        '.ppt': 'impress_pdf_Export',
        '.pptx': 'impress_pdf_Export',
    }

    def __init__(
        self,
        default_async: Optional[bool] = None,
        default_concurrency: Optional[int] = None,
        default_timeout: Optional[int] = None,
        default_max_retries: Optional[int] = None,
        default_cmd: Optional[str] = None,
        default_cache: Optional[bool] = None,
        cache_manager: Optional[Any] = None,  # Injected MineruCacheManager
    ):
        self.available: bool = False
        self.cmd: Optional[str] = None
        self.version: Optional[str] = None
        self._detect()

        # Dependencies
        self.cache_manager = cache_manager

        # Load defaults (args > env > hardcoded)
        if default_async is not None:
            self.default_async = default_async
        else:
            self.default_async = bool(int(os.getenv('MINERU_LO_ASYNC', '1')))

        if default_concurrency is not None:
            self.default_concurrency = default_concurrency
        else:
            default_conc = max(1, min(8, (os.cpu_count() or 4) // 2))
            self.default_concurrency = int(os.getenv('MINERU_LO_CONCURRENCY', str(default_conc)))

        if default_timeout is not None:
            self.default_timeout = default_timeout
        else:
            self.default_timeout = int(os.getenv('MINERU_LO_TIMEOUT', str(self.DEFAULT_TIMEOUT)))

        if default_max_retries is not None:
            self.default_max_retries = default_max_retries
        else:
            self.default_max_retries = int(os.getenv('MINERU_LO_MAX_RETRIES', str(self.DEFAULT_MAX_RETRIES)))

        if default_cmd is not None:
            self.default_cmd = default_cmd
        else:
            self.default_cmd = os.getenv('MINERU_LO_CMD', None)

        if default_cache is not None:
            self.default_cache = default_cache
        else:
            self.default_cache = bool(int(os.getenv('MINERU_LO_CACHE', '1')))

        # Concurrency control
        self._sem = asyncio.Semaphore(self.default_concurrency)

    def _detect(self):
        if platform.system() != 'Linux':
            LOG.warning('[MINERU] Non-Linux: Office→PDF disabled; only PDF passthrough.')
            self.available = False
            return
        for name in ('libreoffice', 'soffice'):
            path = shutil.which(name)
            if not path:
                continue
            try:
                res = subprocess.run([path, '--version'], capture_output=True, text=True, timeout=5)
                if res.returncode == 0:
                    self.available = True
                    self.cmd = path
                    self.version = (res.stdout or res.stderr).strip().splitlines()[0]
                    LOG.info(f'[MINERU] LibreOffice: {self.version} ({self.cmd})')
                    break
            except Exception:
                continue
        if not self.available:
            LOG.warning('[MINERU] LibreOffice not found; only PDFs supported.')

    @classmethod
    def is_supported_suffix(cls, suffix: str) -> bool:
        return suffix.lower() in cls.SUPPORTED_SUFFIXES

    @classmethod
    def _guess_filter(cls, suffix: str) -> str:
        return cls.FILTER_MAP.get(suffix.lower(), 'writer_pdf_Export')

    @staticmethod
    def _kill_proc_group_sync(proc):
        if not proc:
            return
        pgid = proc.pid

        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        if proc.poll() is None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:

                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        else:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    @staticmethod
    async def _kill_proc_group_async(proc):
        if not proc:
            return
        pgid = proc.pid

        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        if proc.returncode is None:
            with contextlib.suppress(Exception):
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await proc.wait()
        else:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def convert_to_pdf(  # noqa: C901
        self,
        input_path: Path,
        output_dir: Path,
        *,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        cmd_override: Optional[str] = None,
        use_cache: Optional[bool] = None,
    ) -> Path:
        if not self.available:
            raise RuntimeError('LibreOffice unavailable')
        if not input_path.is_file():
            raise FileNotFoundError(f'Input not found: {input_path}')
        output_dir.mkdir(parents=True, exist_ok=True)

        timeout = int(timeout or self.default_timeout)
        max_retries = self.default_max_retries if max_retries is None else int(max_retries)
        soffice = cmd_override or self.default_cmd
        if not soffice:
            raise RuntimeError('No soffice command resolved')

        # Determine if we should use cache
        should_cache = use_cache if use_cache is not None else self.default_cache

        # Check Cache
        if should_cache and self.cache_manager:
            cached_pdf = self.cache_manager.get_conversion_cache(input_path)
            if cached_pdf:
                output_dir.mkdir(parents=True, exist_ok=True)
                out_path = output_dir / (input_path.stem + '.pdf')
                shutil.copy2(cached_pdf, out_path)
                return out_path

        lo_profile = tempfile.mkdtemp(prefix='lo_profile_')
        lo_profile_url = f'file://{lo_profile}'
        lo_filter = self._guess_filter(input_path.suffix)
        cmd = [
            soffice,
            '--headless', '--norestore', '--invisible', '--nodefault',
            '--nolockcheck', '--nologo',
            f'-env:UserInstallation={lo_profile_url}',
            '--convert-to', f'pdf:{lo_filter}', '--outdir', str(output_dir), str(input_path),
        ]
        env = os.environ.copy()
        env.setdefault('SAL_USE_VCLPLUGIN', 'headless')
        env.setdefault('LANG', env.get('LANG', 'C.UTF-8'))
        env.setdefault('LC_ALL', env.get('LC_ALL', 'C.UTF-8'))
        env.setdefault('HOME', env.get('HOME', '/tmp'))

        last_err = None
        for _ in range(max_retries + 1):
            proc = None
            try:
                # Use start_new_session to create a process group for easier cleanup
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                    text=True, start_new_session=True
                )
                try:
                    stdout, stderr = proc.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # Cleanup handled in finally
                    last_err = f'timeout after {timeout}s'
                    continue
                except Exception:
                    # Cleanup handled in finally
                    raise

                out_pdf = output_dir / (input_path.stem + '.pdf')
                # Consider success if a non-empty PDF exists (LO sometimes misreports rc)
                if out_pdf.exists() and out_pdf.stat().st_size > 0:
                    if should_cache and self.cache_manager:
                        self.cache_manager.save_conversion_cache(input_path, out_pdf)
                    return out_pdf
                last_err = (stdout or '') + (stderr or '')
            except Exception as e:
                last_err = str(e)
            finally:
                if proc:
                    self._kill_proc_group_sync(proc)
                shutil.rmtree(lo_profile, ignore_errors=True)
                lo_profile = tempfile.mkdtemp(prefix='lo_profile_')
                lo_profile_url = f'file://{lo_profile}'
                for i, token in enumerate(cmd):
                    if token.startswith('-env:UserInstallation='):
                        cmd[i] = f'-env:UserInstallation={lo_profile_url}'
                        break
        shutil.rmtree(lo_profile, ignore_errors=True)
        raise RuntimeError(f'LibreOffice convert failed: {input_path.name}: {last_err}')

    # --------------------------- ASYNC CONVERSION ---------------------------
    async def convert(
        self,
        input_path: Path,
        output_dir: Path,
        *,
        use_async: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ) -> Path:
        '''Unified async entry point for conversion with concurrency control.'''
        use_async = use_async if use_async is not None else self.default_async

        if use_async:
            return await self.convert_to_pdf_async(
                input_path, output_dir, use_cache=use_cache
            )
        else:
            async with self._sem:
                return await asyncio.to_thread(
                    self.convert_to_pdf,
                    input_path, output_dir, use_cache=use_cache
                )

    async def convert_to_pdf_async(  # noqa: C901
        self,
        input_path: Path,
        output_dir: Path,
        *,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        cmd_override: Optional[str] = None,
        use_cache: Optional[bool] = None,
    ) -> Path:
        if not self.available:
            raise RuntimeError('LibreOffice unavailable')

        async with self._sem:
            if not input_path.is_file():
                raise FileNotFoundError(f'Input not found: {input_path}')
            await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)

            timeout = int(timeout or self.default_timeout)
            max_retries = self.default_max_retries if max_retries is None else int(max_retries)
            soffice = cmd_override or self.default_cmd
            if not soffice:
                raise RuntimeError('No soffice command resolved')

            # Determine if we should use cache
            should_cache = use_cache if use_cache is not None else self.default_cache

            # Check Cache
            if should_cache and self.cache_manager:
                cached_pdf = await self.cache_manager.get_conversion_cache_async(input_path)
                if cached_pdf:
                    await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
                    out_path = output_dir / (input_path.stem + '.pdf')
                    await asyncio.to_thread(shutil.copy2, cached_pdf, out_path)
                    return out_path

            lo_profile = await asyncio.to_thread(tempfile.mkdtemp, prefix='lo_profile_')
            lo_profile_url = f'file://{lo_profile}'
            lo_filter = self._guess_filter(input_path.suffix)
            cmd = [
                soffice,
                '--headless', '--norestore', '--invisible', '--nodefault',
                '--nolockcheck', '--nologo',
                f'-env:UserInstallation={lo_profile_url}',
                '--convert-to', f'pdf:{lo_filter}', '--outdir', str(output_dir), str(input_path),
            ]
            env = os.environ.copy()
            env.setdefault('SAL_USE_VCLPLUGIN', 'headless')
            env.setdefault('LANG', env.get('LANG', 'C.UTF-8'))
            env.setdefault('LC_ALL', env.get('LC_ALL', 'C.UTF-8'))
            env.setdefault('HOME', env.get('HOME', '/tmp'))

            last_err = None
            for _ in range(max_retries + 1):
                proc = None
                try:
                    # Use a process group to ensure we can kill all subprocesses
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env=env,
                        start_new_session=True
                    )
                    try:
                        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                    except asyncio.TimeoutError:
                        # Cleanup handled in finally
                        last_err = f'timeout after {timeout}s'
                        continue
                    except Exception:
                        raise

                    out_pdf = output_dir / (input_path.stem + '.pdf')
                    if out_pdf.exists() and out_pdf.stat().st_size > 0:
                        if should_cache and self.cache_manager:
                            await self.cache_manager.save_conversion_cache_async(input_path, out_pdf)
                        return out_pdf
                    last_err = ((stdout_b or b'').decode(errors='ignore') + (stderr_b or b'').decode(errors='ignore'))
                except Exception as e:
                    last_err = str(e)
                finally:
                    if proc:
                        await self._kill_proc_group_async(proc)
                    await asyncio.to_thread(shutil.rmtree, lo_profile, ignore_errors=True)
                    lo_profile = await asyncio.to_thread(tempfile.mkdtemp, prefix='lo_profile_')
                    lo_profile_url = f'file://{lo_profile}'
                    for i, token in enumerate(cmd):
                        if token.startswith('-env:UserInstallation='):
                            cmd[i] = f'-env:UserInstallation={lo_profile_url}'
                            break
            await asyncio.to_thread(shutil.rmtree, lo_profile, ignore_errors=True)
            raise RuntimeError(f'LibreOffice convert failed: {input_path.name}: {last_err}')


class MineruCacheManager:
    def __init__(self, cache_dir: Optional[Union[str, Path]]):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.conversion_cache_dir = self.cache_dir / 'conversions'
            os.makedirs(self.conversion_cache_dir, exist_ok=True)
        else:
            self.conversion_cache_dir = None

    @property
    def is_enabled(self) -> bool:
        return self.cache_dir is not None

    def get_conversion_cache_dir(self) -> Optional[Path]:
        return self.conversion_cache_dir

    def get_parse_cache_dir(self, backend: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return self.cache_dir / backend

    def _get_func_suffix(self, table_enable: bool, formula_enable: bool) -> str:
        if table_enable and formula_enable:
            return '_a'
        elif table_enable:
            return '_t'
        elif formula_enable:
            return '_f'
        else:
            return '_n'

    def _get_cache_key(
        self,
        hash_id: str,
        table_enable: bool,
        formula_enable: bool,
        lang: str = None,
        parse_method: str = None
    ) -> str:
        suffix = self._get_func_suffix(table_enable, formula_enable)
        key_parts = [hash_id, suffix]

        if lang:
            key_parts.append(lang)
        if parse_method:
            key_parts.append(parse_method)

        return '_'.join(key_parts)

    async def check_parse_cache(  # noqa: C901
        self,
        files: List[str],
        results: Dict[str, dict],
        backend: str,
        return_md: bool,
        return_content_list: bool,
        table_enable: bool,
        formula_enable: bool,
        lang: str = None,
        parse_method: str = None
    ):
        if not self.cache_dir:
            return results, files

        func_suffix = self._get_func_suffix(table_enable, formula_enable)
        func_suffix_map = {
            '_a': ['_a'],
            '_t': ['_t', '_a'],
            '_f': ['_f', '_a'],
            '_n': ['_n', '_a', '_t', '_f']
        }
        func_suffix_list = func_suffix_map.get(func_suffix, [func_suffix])

        uncached_files = []
        cache_subdir = self.get_parse_cache_dir(backend)
        if not cache_subdir:
            # Should not happen if check_parse_cache is called after is_enabled check
            return results, files

        files_to_hash = [f for f in files if f not in results or not results[f]]
        if files_to_hash:
            hashes = await asyncio.gather(*[
                asyncio.to_thread(calculate_file_hash, f) for f in files_to_hash
            ])
            file_hashes = dict(zip(files_to_hash, hashes))
        else:
            file_hashes = {}

        for f in files:
            if f in results and results[f]:
                continue

            file_hash = file_hashes.get(f)
            if not file_hash:
                uncached_files.append(f)
                continue

            result = {}
            file_content_list_found = False
            file_md_found = False

            keys_to_check = []

            if lang or parse_method:
                for suffix in func_suffix_list:
                    key_parts = [file_hash, suffix]
                    if lang:
                        key_parts.append(lang)
                    if parse_method:
                        key_parts.append(parse_method)
                    keys_to_check.append('_'.join(key_parts))

            is_legacy_compatible = (
                lang in ['ch_server', 'ch', None] and parse_method in ['auto', None]
            )

            if is_legacy_compatible:
                for suffix in func_suffix_list:
                    keys_to_check.append(f'{file_hash}{suffix}')

            for cache_key in keys_to_check:
                if return_content_list and not file_content_list_found:
                    json_path = cache_subdir / f'{cache_key}_content_list.json'
                    if await asyncio.to_thread(os.path.isfile, json_path):
                        try:
                            result['content_list'] = await asyncio.to_thread(
                                lambda p: json.load(open(p, 'r', encoding='utf-8')), json_path
                            )
                            file_content_list_found = True
                        except Exception:
                            continue

                if return_md and not file_md_found:
                    md_path = cache_subdir / f'{cache_key}.md'
                    if await asyncio.to_thread(os.path.isfile, md_path):
                        try:
                            result['md_content'] = await asyncio.to_thread(
                                lambda p: open(p, 'r', encoding='utf-8').read(), md_path
                            )
                            file_md_found = True
                        except Exception:
                            continue

                if (not return_content_list or file_content_list_found) and \
                   (not return_md or file_md_found):
                    break

            results[f].update(result)

            needs_content_list = return_content_list
            needs_md = return_md
            has_content_list = file_content_list_found
            has_md = file_md_found

            if (needs_content_list and not has_content_list) or (needs_md and not has_md):
                uncached_files.append(f)

        LOG.debug(
            f'[MINERU] Cache check: {len(files) - len(uncached_files)}/{len(files)} files cached'
        )
        return results, uncached_files

    def get_conversion_cache(self, file_path: Union[str, Path]) -> Optional[Path]:
        '''(Sync) Get cached conversion result path if exists'''
        cache_subdir = self.get_conversion_cache_dir()
        if not cache_subdir:
            return None
        try:
            file_path = Path(file_path)
            file_hash = calculate_file_hash(file_path)
            cached_pdf_path = cache_subdir / f'{file_hash}.pdf'

            if cached_pdf_path.exists() and cached_pdf_path.stat().st_size > 0:
                LOG.debug(f'[MINERU] Found cached conversion: {file_hash}')
                return cached_pdf_path
        except Exception:
            pass
        return None

    async def get_conversion_cache_async(self, file_path: Union[str, Path]) -> Optional[Path]:
        '''(Async) Get cached conversion result path if exists'''
        return await asyncio.to_thread(self.get_conversion_cache, file_path)

    async def save_parse_result(
        self,
        hash_id: str,
        result: dict,
        mode: str,
        *,
        table_enable: bool,
        formula_enable: bool,
        lang: str = None,
        parse_method: str = None
    ) -> None:
        if not result or not self.cache_dir:
            return

        try:
            cache_subdir = self.get_parse_cache_dir(mode)
            if not cache_subdir:
                return
            cache_subdir.mkdir(parents=True, exist_ok=True)

            key = self._get_cache_key(hash_id, table_enable, formula_enable, lang, parse_method)

            tasks = []

            def _write_atomic(path: Path, content: str):
                temp_path = path.with_suffix(f'.tmp.{uuid.uuid4()}')
                try:
                    temp_path.write_text(content, encoding='utf-8')
                    temp_path.replace(path)
                except Exception:
                    if temp_path.exists():
                        temp_path.unlink()
                    raise

            if 'md_content' in result and result['md_content'].strip():
                md_path = cache_subdir / f'{key}.md'
                tasks.append(asyncio.to_thread(
                    _write_atomic, md_path, result['md_content']
                ))

            if 'content_list' in result and result['content_list']:
                json_path = cache_subdir / f'{key}_content_list.json'
                tasks.append(asyncio.to_thread(
                    _write_atomic, json_path,
                    json.dumps(result['content_list'], ensure_ascii=False, indent=2)
                ))

            if tasks:
                await asyncio.gather(*tasks)
                LOG.debug(f'[MINERU] Cached parse result: {key}')

        except Exception as e:
            LOG.warning(f'[MINERU] Cache write failed: {e}')

    def save_conversion_cache(self, file_path: Union[str, Path], converted_pdf_path: Path) -> None:
        '''(Sync) Save conversion result to cache'''
        cache_subdir = self.get_conversion_cache_dir()
        if not cache_subdir:
            return
        try:
            # cache_subdir created in init
            file_path = Path(file_path)
            file_hash = calculate_file_hash(file_path)
            cached_pdf_path = cache_subdir / f'{file_hash}.pdf'
            shutil.copy2(converted_pdf_path, cached_pdf_path)
            LOG.debug(f'[MINERU] Cached conversion: {file_hash}')
        except Exception as e:
            LOG.warning(f'[MINERU] Conversion cache write failed: {e}')

    async def save_conversion_cache_async(self, input_path: Path, converted_pdf_path: Path) -> None:
        '''(Async) Save conversion result to cache'''
        await asyncio.to_thread(self.save_conversion_cache, input_path, converted_pdf_path)

class MineruServerBase:
    def __init__(
        self,
        cache_manager: MineruCacheManager,
        image_save_dir: Optional[str] = None,
        default_backend: str = 'pipeline',
        default_lang: str = 'ch_server',
        default_parse_method: str = 'auto',
        default_formula_enable: bool = True,
        default_table_enable: bool = True,
        default_return_md: bool = False,
        default_return_content_list: bool = True,
        lo_helper: Optional[LibreOfficeHelper] = None
    ):
        if default_backend not in [
            'pipeline', 'vlm-vllm-engine', 'vlm-vllm-async-engine', 'vlm-transformers', 'hybrid-auto-engine'
        ]:
            raise ValueError('Invalid backend, \
                             only support pipeline, vlm-vllm-async-engine, vlm-transformers, hybrid-auto-engine')
        if default_lang not in ['ch', 'ch_server', 'ch_lite', 'en']:
            raise ValueError('Invalid language, \
                             only support ch, ch_server, ch_lite, en')
        self._default_backend = default_backend
        self._image_save_dir = os.path.join(image_save_dir, 'images') if image_save_dir else None
        self._default_lang = default_lang
        self._default_parse_method = default_parse_method
        self._default_formula_enable = default_formula_enable
        self._default_table_enable = default_table_enable
        self._default_return_md = default_return_md
        self._default_return_content_list = default_return_content_list
        self.cache_manager = cache_manager

        self._lo_helper = lo_helper or LibreOfficeHelper(
            default_async=True,
            default_concurrency=int(os.getenv('MINERU_LO_CONCURRENCY', '1')),
            default_cache=True,
            cache_manager=self.cache_manager
        )

        self._lo_available = self._lo_helper.available
        if self._lo_available:
            self._supported_office_types = sorted(list(LibreOfficeHelper.SUPPORTED_SUFFIXES))
        else:
            self._supported_office_types = []
        LOG.info(f'[MINERU] Supported office types: {self._supported_office_types}')

        self._middle_file_dir = tempfile.mkdtemp()
        atexit.register(lambda: shutil.rmtree(self._middle_file_dir, ignore_errors=True))

        if self._image_save_dir:
            os.makedirs(self._image_save_dir, exist_ok=True)

    @staticmethod
    def _move_images_to_output(source_dir: Path, target_dir: Path):
        for jpg_file in source_dir.glob('*.jpg'):
            try:
                shutil.move(str(jpg_file), str(target_dir / jpg_file.name))
            except Exception:
                with contextlib.suppress(Exception):
                    shutil.copy2(
                        str(jpg_file),
                        str(target_dir / jpg_file.name)
                    )

    async def _read_parse_result(
        self, suffix: str, pdf_name: str, parse_dir: str
    ) -> Optional[Union[str, dict]]:
        path = os.path.join(parse_dir, f'{pdf_name}{suffix}')
        if await asyncio.to_thread(os.path.exists, path):
            try:
                if suffix == '.md':
                    def _read_md():
                        with open(path, 'r', encoding='utf-8') as fp:
                            return fp.read()
                    return await asyncio.to_thread(_read_md)
                if suffix == '_content_list.json':
                    def _read_json():
                        with open(path, 'r', encoding='utf-8') as fp:
                            return json.load(fp)
                    return await asyncio.to_thread(_read_json)
            except (OSError, json.JSONDecodeError) as e:
                LOG.error(f'[MINERU] Failed to read parse result: {path}, error: {e}')
        return None

    # ------------------------------ Public API ------------------------------
    @app.post('/api/v1/pdf_parse')
    async def parse_pdf(  # noqa: C901
        self,
        files: List[str] = fastapi.Form([]),  # noqa B008
        upload_files: List[fastapi.UploadFile] = fastapi.File([]),  # noqa B008
        use_cache: bool = fastapi.Form(False),  # noqa B008
        lang: str = fastapi.Form('ch_server'),  # noqa B008
        backend: str = fastapi.Form(None),  # noqa B008
        parse_method: str = fastapi.Form('auto'),  # noqa B008
        formula_enable: bool = fastapi.Form(None),  # noqa B008
        table_enable: bool = fastapi.Form(None),  # noqa B008
        return_md: bool = fastapi.Form(None),  # noqa B008
        return_content_list: bool = fastapi.Form(None),  # noqa B008
    ):
        if files and upload_files:
            raise fastapi.HTTPException(
                status_code=400,
                detail='Either provide only \'files\' or only \'upload_files\'!'
            )
        for f in files:
            if not os.path.isfile(f):
                raise fastapi.HTTPException(status_code=400, detail=f'fastapi.File Not Found: {f}')

        if lang and lang not in ['ch', 'ch_server', 'ch_lite', 'en']:
            raise fastapi.HTTPException(status_code=400, detail='Invalid language')
        if backend and backend not in [
            'pipeline', 'vlm-vllm-engine', 'vlm-vllm-async-engine', 'vlm-transformers', 'hybrid-auto-engine'
        ]:
            raise fastapi.HTTPException(status_code=400, detail='Invalid backend, \
                             only support pipeline, vlm-vllm-async-engine, vlm-transformers, hybrid-auto-engine')

        # Create a unique temporary directory for this request
        unique_dir = await asyncio.to_thread(tempfile.mkdtemp, dir=self._middle_file_dir)
        unique_id = os.path.basename(unique_dir)
        req_id = unique_id[:8]  # Default ID

        if upload_files:
            files = await self._resolve_upload_files(upload_files, unique_dir)

        # Generate a request identifier based on input files
        # If multiple files, use the first file's name + count
        if files:
            first_file_name = Path(files[0]).name
            req_id = f'{first_file_name}'
            if len(files) > 1:
                req_id += f'+{len(files) - 1}'
        elif upload_files:
            first_file_name = upload_files[0].filename
            req_id = f'{first_file_name}'
            if len(upload_files) > 1:
                req_id += f'+{len(upload_files) - 1}'

        for f in files:
            suf = Path(f).suffix.lower()
            if suf not in self._supported_office_types + ['.pdf']:
                raise fastapi.HTTPException(status_code=400, detail=f'Unsupported file type: {suf}')

        backend = backend or self._default_backend
        lang = lang or self._default_lang
        parse_method = parse_method or self._default_parse_method
        formula_enable = self._default_formula_enable if formula_enable is None else formula_enable
        table_enable = self._default_table_enable if table_enable is None else table_enable
        if return_md is None:
            return_md = self._default_return_md
        if return_content_list is None:
            return_content_list = self._default_return_content_list

        lo_async = self._lo_helper.default_async
        lo_cache = self._lo_helper.default_cache

        effective_backend = backend
        if effective_backend == 'vlm-vllm-engine':
            LOG.info(f'[{req_id}] Auto-switching backend: vlm-vllm-engine → vlm-vllm-async-engine')
            effective_backend = 'vlm-vllm-async-engine'

        LOG.info(f'[{req_id}] Processing {len(files)} files. Backend: {effective_backend}, Lang: {lang}')

        try:
            results = {f: {} for f in files}

            # honor use_cache for parsed outputs
            files_to_process = files
            if use_cache and self.cache_manager.is_enabled:
                results, files_to_process = await self.cache_manager.check_parse_cache(
                    files, results, effective_backend, return_md, return_content_list,
                    table_enable, formula_enable, lang=lang, parse_method=parse_method
                )
                if not files_to_process:
                    hit_cnt = len([f for f in files if results[f]])
                    hit_rate = hit_cnt / len(files) * 100
                    LOG.info(
                        f'[{req_id}] CACHE HIT: {hit_cnt}/{len(files)} files ({hit_rate:.1f}%)'
                    )
                    results_list = [results[f] for f in files]
                    return fastapi.responses.JSONResponse(
                        status_code=200,
                        content={'result': results_list, 'unique_id': unique_id}
                    )
            elif use_cache and not self.cache_manager.is_enabled:
                LOG.warning(f'[{req_id}] CACHE_DIR not set; use_cache ignored.')

            # 1) Convert to PDFs with caching support
            pdf_paths: List[Path] = []
            out_dir = Path(unique_dir)
            conversion_cache_hits = 0

            for f in files_to_process:
                p = Path(f)
                if p.suffix.lower() in self._supported_office_types:
                    # Conversion with internal cache check by Helper
                    LOG.info(f'[{req_id}] Converting office file: {f}')
                    try:
                        # Concurrency controlled internally by Helper
                        converted_pdf = await self._lo_helper.convert(
                            p, out_dir, use_async=lo_async, use_cache=lo_cache
                        )
                    except Exception as e:
                        LOG.error(f'[{req_id}] Conversion failed: {e}')
                        converted_pdf = None

                    if converted_pdf and converted_pdf.exists():
                        pdf_paths.append(converted_pdf)
                    else:
                        raise fastapi.HTTPException(status_code=500, detail=f'Failed to convert {f}')

                elif p.suffix.lower() == '.pdf':
                    pdf_paths.append(p)
                else:
                    raise fastapi.HTTPException(
                        status_code=400, detail=f'Unsupported file type: {p.suffix}'
                    )

            if conversion_cache_hits > 0:
                LOG.info(
                    f'[{req_id}] Conversion cache hits: '
                    f'{conversion_cache_hits}/{len(files_to_process)} files'
                )

            LOG.info(f'[{req_id}] Starting parsing {len(pdf_paths)} PDFs with {effective_backend}...')
            pdf_file_names = [p.stem for p in pdf_paths]
            pdf_bytes_list = []
            for p in pdf_paths:
                try:
                    content = await asyncio.to_thread(mineru.cli.common.read_fn, p)
                    pdf_bytes_list.append(content)
                except Exception as e:
                    raise fastapi.HTTPException(status_code=400, detail=f'fastapi.File Not Found: {p}: {e}')
            lang_list = [lang] * len(pdf_bytes_list)
            params = dict(
                output_dir=unique_dir,
                pdf_file_names=pdf_file_names,
                pdf_bytes_list=pdf_bytes_list,
                p_lang_list=lang_list,
                backend=effective_backend,
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
            )
            if effective_backend == 'pipeline':  # pipeline backend is sync
                await asyncio.to_thread(mineru.cli.common.do_parse, **params)
            else:
                await mineru.cli.common.aio_do_parse(**params)
            LOG.info(f'[{req_id}] Parse completed for {len(pdf_paths)} files.')

            # 3) Collect outputs
            for src_path, pdf in zip(files_to_process, pdf_paths):
                pdf_name = pdf.stem
                if effective_backend.startswith('vlm'):
                    parse_dir = os.path.join(unique_dir, pdf_name, 'vlm')
                elif effective_backend.startswith('hybrid'):
                    parse_dir = os.path.join(unique_dir, pdf_name, f'hybrid_{parse_method}')
                else:
                    parse_dir = os.path.join(unique_dir, pdf_name, parse_method)

                if await asyncio.to_thread(os.path.exists, parse_dir):
                    hash_id = await asyncio.to_thread(calculate_file_hash, src_path)
                    md_content = await self._read_parse_result('.md', pdf_name, parse_dir)
                    content_list = await self._read_parse_result(
                        '_content_list.json', pdf_name, parse_dir
                    )
                    if return_md and md_content:
                        results[src_path]['md_content'] = md_content
                    if return_content_list and content_list:
                        results[src_path]['content_list'] = content_list
                    if self.cache_manager.is_enabled:
                        await self.cache_manager.save_parse_result(
                            hash_id, results[src_path], mode=effective_backend,
                            table_enable=table_enable, formula_enable=formula_enable,
                            lang=lang, parse_method=parse_method
                        )
                        LOG.info(f'[{req_id}] Cached result for {src_path}')
                    if self._image_save_dir:
                        source_dir = Path(f'{parse_dir}/images/')
                        target_dir = Path(self._image_save_dir)
                        await asyncio.to_thread(
                            self._move_images_to_output, source_dir, target_dir
                        )

            # merge cached results for already-cached inputs (if any)
            final_results = [results[f] for f in files]
            return fastapi.responses.JSONResponse(
                status_code=200,
                content={'result': final_results, 'unique_id': unique_id}
            )
        except Exception as e:
            LOG.error(f'[{req_id}] Parse Failed: {e}')
            return fastapi.responses.JSONResponse(
                status_code=500, content={'error': f'Failed to process file: {e}'}
            )
        finally:
            await asyncio.to_thread(shutil.rmtree, unique_dir, ignore_errors=True)

    async def _resolve_upload_files(
        self, upload_files: List[fastapi.File], unique_dir: str
    ) -> List[str]:
        if not upload_files:
            return []
        temp_upload_dir = os.path.join(self._middle_file_dir, f'{unique_dir}/upload')
        await asyncio.to_thread(os.makedirs, temp_upload_dir, exist_ok=True)
        file_paths: List[str] = []
        for upload_file in upload_files:
            temp_file_path = os.path.join(temp_upload_dir, upload_file.filename)

            def _save_sync(src_file, dest_path):
                with open(dest_path, 'wb') as buffer:
                    shutil.copyfileobj(src_file, buffer)

            await asyncio.to_thread(_save_sync, upload_file.file, temp_file_path)
            file_paths.append(temp_file_path)
        return file_paths


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
        cache_manager = MineruCacheManager(cache_dir) if cache_dir else None
        mineru_server = MineruServerBase(
            cache_manager=cache_manager, image_save_dir=image_save_dir, default_backend=default_backend,
            default_lang=default_lang, default_parse_method=default_parse_method,
            default_formula_enable=default_formula_enable, default_table_enable=default_table_enable,
            default_return_md=default_return_md, default_return_content_list=default_return_content_list)
        super().__init__(mineru_server, *args, **kwargs)
