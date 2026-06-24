from lazyllm.thirdparty import fsspec
from typing import Iterable, List, Optional, Union, Callable

from lazyllm.thirdparty import torch
import lazyllm
from lazyllm import LOG, config

from ....common import LazyLLMRegisterMetaClass
from ..doc_node import DocNode, RichDocNode
from lazyllm.module import ModuleBase
from lazyllm.module.module import module_cache, CacheNotFoundError
from pathlib import Path
import locale
import threading
import hashlib
from lazyllm.thirdparty import charset_normalizer

config.add('reader_cache', bool, True, 'READER_CACHE',
           description='Whether to enable reader cache (ModuleCache and OCR service use_cache).')

_READER_CACHE_SKIP_KEYS = frozenset({'use_cache', 'lazyllm_files', 'llm_chat_history'})


class LazyLLMReaderBase(ModuleBase, metaclass=LazyLLMRegisterMetaClass):
    post_action = None

    _encoding_cache = {}
    _cache_lock = threading.Lock()
    _cache_max_size = 1000

    def __init__(self, *args, return_trace: bool = True,
                 use_reader_cache: bool = config['reader_cache'], **kwargs):
        super().__init__(return_trace=return_trace)
        self._use_reader_cache = use_reader_cache

    @property
    def __reader_cache_hash__(self):
        cache_hash = f'Reader@{self.__class__.__name__}'
        if isinstance(self._use_reader_cache, str):
            cache_hash += f'@{self._use_reader_cache}'
        if hasattr(self, 'appendix_hash_key'):
            cache_hash += f'@{self.appendix_hash_key}'
        return cache_hash

    @staticmethod
    def _reader_cache_kwargs(kwargs: dict) -> dict:
        return {k: v for k, v in kwargs.items() if k not in _READER_CACHE_SKIP_KEYS}

    @staticmethod
    def _normalize_file_path(value) -> Optional[str]:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, str) and not value.startswith(('http://', 'https://')):
            return value
        return None

    @classmethod
    def _file_content_digest(cls, path: str) -> str:
        try:
            stat = Path(path).stat()
            return f'{stat.st_mtime}-{stat.st_size}'
        except OSError:
            return hashlib.md5(path.encode()).hexdigest()

    @classmethod
    def _reader_cache_file_token(cls, value):
        path = cls._normalize_file_path(value)
        if path is None:
            return value
        return f'file://{path}|{cls._file_content_digest(path)}'

    @classmethod
    def _reader_cache_args(cls, args: tuple) -> tuple:
        return tuple(cls._reader_cache_file_token(arg) for arg in args)

    @classmethod
    def _reader_cache_key_kwargs(cls, kwargs: dict) -> dict:
        return {k: cls._reader_cache_file_token(v) for k, v in cls._reader_cache_kwargs(kwargs).items()}

    def _reader_cache_read_enabled(self) -> bool:
        if not self._use_reader_cache:
            return False
        return 'R' in lazyllm.config['cache_mode']

    def _reader_cache_write_enabled(self) -> bool:
        if not self._reader_cache_read_enabled():
            return False
        return 'W' in lazyllm.config['cache_mode']

    def use_reader_cache(self, flag: Union[bool, str] = True):
        self._use_reader_cache = bool(flag) if isinstance(flag, bool) else flag or False
        return self

    def _call_impl(self, *args, **kw):
        cache_kw = self._reader_cache_key_kwargs(kw)
        cache_args = self._reader_cache_args(args)
        if self._reader_cache_read_enabled():
            try:
                return module_cache.get(self.__reader_cache_hash__, cache_args, cache_kw)
            except CacheNotFoundError:
                pass
        r = super()._call_impl(*args, **kw)
        if self._reader_cache_write_enabled():
            module_cache.set(self.__reader_cache_hash__, cache_args, cache_kw, r)
        return r

    def _lazy_load_data(self, *args, **load_kwargs) -> Iterable[DocNode]:
        raise NotImplementedError(f'{self.__class__.__name__} does not implement lazy_load_data method.')

    def _load_data(self, *args, **load_kwargs) -> List[DocNode]:
        return list(self._lazy_load_data(*args, **load_kwargs))

    @property
    def _active_use_cache(self) -> bool:
        return bool(self._use_reader_cache)

    def forward(self, *args, **kwargs) -> List[DocNode]:
        load_kwargs = {k: v for k, v in kwargs.items() if k not in _READER_CACHE_SKIP_KEYS}
        r = self._load_data(*args, **load_kwargs)
        r = [r] if isinstance(r, DocNode) else [] if r is None else r
        if r and self.post_action:
            r = [x for sub in [self.post_action(n) for n in r] for x in (sub if isinstance(sub, list) else [sub])]
        return r

    @classmethod
    def detect_encoding(cls, file_path: Union[str, Path], fs: Optional['fsspec.AbstractFileSystem'] = None,  # noqa: C901
                        sample_size: int = 10000, use_cache: bool = True,
                        enable_chardet: bool = True) -> str:
        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        fs = fs or get_default_fs()

        cache_key = str(file_path) if use_cache else None
        if cache_key:
            with cls._cache_lock:
                if cache_key in cls._encoding_cache:
                    cached_encoding = cls._encoding_cache[cache_key]
                    return cached_encoding

        try:
            with fs.open(file_path, 'rb') as f:
                raw_data = f.read(sample_size)
        except Exception as e:
            LOG.warning(f'Failed to read file {file_path}: {e}')
            return 'utf-8'

        if not raw_data:
            return 'utf-8'

        bom_encodings = [
            (b'\xef\xbb\xbf', 'utf-8-sig'),
            (b'\xff\xfe\x00\x00', 'utf-32-le'),
            (b'\x00\x00\xfe\xff', 'utf-32-be'),
            (b'\xff\xfe', 'utf-16-le'),
            (b'\xfe\xff', 'utf-16-be'),
        ]

        for bom, encoding in bom_encodings:
            if raw_data.startswith(bom):
                cls._cache_encoding(cache_key, encoding)
                return encoding

        has_high_bytes = any(b > 127 for b in raw_data[:1000])

        if has_high_bytes:
            chinese_encodings = ['gb18030', 'gbk', 'gb2312', 'big5']
            # Prefer UTF-8 when valid; otherwise fall back to Chinese encodings.
            # Do not require Chinese chars in the first N chars — CSV headers are often long ASCII.
            if cls._try_decode(raw_data, 'utf-8'):
                cls._cache_encoding(cache_key, 'utf-8')
                return 'utf-8'

            for encoding in chinese_encodings:
                if cls._try_decode(raw_data, encoding):
                    cls._cache_encoding(cache_key, encoding)
                    return encoding
        else:
            primary_encodings = ['utf-8', 'gb18030', 'gbk', 'gb2312', 'big5']
            for encoding in primary_encodings:
                if cls._try_decode(raw_data, encoding):
                    cls._cache_encoding(cache_key, encoding)
                    return encoding

        if cls._try_decode(raw_data, 'latin-1'):
            cls._cache_encoding(cache_key, 'latin-1')
            return 'latin-1'

        if enable_chardet:
            try:
                detected = charset_normalizer.from_path(file_path).best().encoding
                if detected:
                    cls._cache_encoding(cache_key, detected)
                    return detected
                else:
                    LOG.warning(f'Charset normalizer detection failed: {detected}')
            except Exception as e:
                LOG.warning(f'Charset normalizer detection failed: {e}')

        try:
            system_encoding = locale.getpreferredencoding(False)
            LOG.warning(f'Using system default encoding {system_encoding} for {file_path}')
            cls._cache_encoding(cache_key, system_encoding)
            return system_encoding
        except Exception:
            pass
        LOG.warning(f'Could not detect encoding for {file_path}, using utf-8 as fallback')
        cls._cache_encoding(cache_key, 'utf-8')
        return 'utf-8'

    @staticmethod
    def _try_decode(data: bytes, encoding: str) -> bool:
        try:
            data.decode(encoding)
            return True
        except (UnicodeDecodeError, LookupError):
            return False

    @classmethod
    def _cache_encoding(cls, cache_key: Optional[str], encoding: str) -> None:
        if cache_key is None:
            return

        with cls._cache_lock:
            if len(cls._encoding_cache) >= cls._cache_max_size:
                old_keys = list(cls._encoding_cache.keys())[:100]
                for key in old_keys:
                    del cls._encoding_cache[key]
                LOG.debug(f'Encoding cache cleaned: removed {len(old_keys)} entries')

            cls._encoding_cache[cache_key] = encoding

    @classmethod
    def clear_encoding_cache(cls) -> None:
        with cls._cache_lock:
            cls._encoding_cache.clear()

    @classmethod
    def get_encoding_cache_stats(cls) -> dict:
        with cls._cache_lock:
            return {
                'cache_size': len(cls._encoding_cache),
                'cache_max_size': cls._cache_max_size,
                'usage_ratio': len(cls._encoding_cache) / cls._cache_max_size if cls._cache_max_size > 0 else 0
            }


def get_default_fs():
    return fsspec.implementations.local.LocalFileSystem()

def is_default_fs(fs: 'fsspec.AbstractFileSystem') -> bool:
    return isinstance(fs, fsspec.implementations.local.LocalFileSystem) or not fs.auto_mkdir

def infer_torch_device() -> str:
    try:
        has_cuda = torch.cuda.is_available()
    except NameError:
        has_cuda = torch.cuda.is_available()
    except ImportError:
        return 'cpu'

    if has_cuda: return 'cuda'
    if torch.backends.mps.is_available(): return 'mps'
    return 'cpu'

config.add('auto_detect_encoding', bool, True, 'AUTO_DETECT_ENCODING',
           description='Whether auto detecting txt encoding')
config.add('enable_chardet', bool, True, 'ENABLE_CHARDET',
           description='Whether to use chardet when detect txt encoding')
config.add('use_encoding_cache', bool, True, 'USE_ENCODING_CACHE',
           description='Whether use cahce to accelerate txt encoding')


class TxtReader(LazyLLMReaderBase):
    def __init__(self, encoding: Optional[str] = None, return_trace: bool = True,
                 auto_detect_encoding: bool = config['auto_detect_encoding'],
                 enable_chardet: bool = config['enable_chardet'],
                 use_encoding_cache: bool = config['use_encoding_cache']) -> None:
        super().__init__(return_trace=return_trace)
        self._encoding = encoding
        self._auto_detect_encoding = auto_detect_encoding
        self._enable_chardet = enable_chardet
        self._use_encoding_cache = use_encoding_cache

    @property
    def appendix_hash_key(self):
        return f'{self._encoding}|{self._auto_detect_encoding}|{self._enable_chardet}'

    def _load_data(self, file: Path, fs: Optional['fsspec.AbstractFileSystem'] = None) -> List[DocNode]:
        if self._encoding:
            encoding = self._encoding
        elif self._auto_detect_encoding:
            encoding = self.detect_encoding(
                file, fs,
                use_cache=self._use_encoding_cache,
                enable_chardet=self._enable_chardet
            )
        else:
            encoding = 'utf-8'

        try:
            with (fs or get_default_fs()).open(file, mode='r', encoding=encoding) as f:
                content = f.read()
            return [DocNode(text=content)]
        except Exception:
            if not self._auto_detect_encoding and self._encoding:
                try:
                    detected_encoding = self.detect_encoding(
                        file, fs,
                        use_cache=self._use_encoding_cache,
                        enable_chardet=self._enable_chardet
                    )
                    with (fs or get_default_fs()).open(file, mode='r', encoding=detected_encoding) as f:
                        content = f.read()
                    return [DocNode(text=content)]
                except Exception as e:
                    LOG.error(f'Auto-detection also failed for {file}: {e}')
            elif self._auto_detect_encoding and self._enable_chardet:
                try:
                    detected = charset_normalizer.from_path(file).best()
                    if detected and detected.encoding and detected.encoding.lower() != encoding.lower():
                        with (fs or get_default_fs()).open(file, mode='r', encoding=detected.encoding) as f:
                            content = f.read()
                        return [DocNode(text=content)]
                except Exception as e2:
                    LOG.error(f'charset_normalizer also failed for {file}: {e2}')
            raise

class DefaultReader(TxtReader):
    def _load_data(self, file: Path, fs: Optional['fsspec.AbstractFileSystem'] = None) -> List[DocNode]:
        try:
            return super()._load_data(file, fs)
        except Exception as e:
            encoding_info = self._encoding if self._encoding else 'auto-detected encoding'
            LOG.error(f'Failed to read {file} with {encoding_info}: {e}. Skipping file.')
            return []

class _RichReader(LazyLLMReaderBase):
    def __init__(self, post_func: Optional[Callable] = None, split_doc: bool = True,
                 return_trace: bool = True, **kwargs):
        super().__init__(return_trace=return_trace, **kwargs)
        self._post_func = post_func
        self._split_doc = split_doc

    def forward(self, *args, **kwargs) -> List[DocNode]:
        nodes = super().forward(*args, **kwargs)
        if self._post_func:
            nodes = self._post_func(nodes)
            assert isinstance(nodes, list), f'Expected list, got {type(nodes)}, please check your post function'
            for n in nodes:
                assert isinstance(n, DocNode), f'Expected DocNode, got {type(n)}, \
                    please check your post function'
                if kwargs.get('extra_info'):
                    n.global_metadata.update(kwargs['extra_info'])
        if self._split_doc:
            return [RichDocNode(nodes, global_metadata=nodes[0].global_metadata if nodes else None)]
        else:
            if not nodes:
                return []
            texts = [b.text for b in nodes]
            return [DocNode(
                text='\n'.join(texts),
                metadata={'file_name': nodes[0].metadata.get('file_name', '')},
                global_metadata=nodes[0].global_metadata
            )]
