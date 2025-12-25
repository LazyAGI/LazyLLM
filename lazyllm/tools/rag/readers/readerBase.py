from lazyllm.thirdparty import fsspec
from typing import Iterable, List, Optional, Union

from lazyllm.thirdparty import torch
from lazyllm import LOG, config

from ....common import LazyLLMRegisterMetaClass
from ..doc_node import DocNode
from lazyllm.module import ModuleBase
from pathlib import Path
import locale
import threading
from lazyllm.thirdparty import charset_normalizer

class LazyLLMReaderBase(ModuleBase, metaclass=LazyLLMRegisterMetaClass):
    post_action = None

    _encoding_cache = {}
    _cache_lock = threading.Lock()
    _cache_max_size = 1000

    def __init__(self, *args, return_trace: bool = True, **kwargs):
        super().__init__(return_trace=return_trace)

    def _lazy_load_data(self, *args, **load_kwargs) -> Iterable[DocNode]:
        raise NotImplementedError(f'{self.__class__.__name__} does not implement lazy_load_data method.')

    def _load_data(self, *args, **load_kwargs) -> List[DocNode]:
        return list(self._lazy_load_data(*args, **load_kwargs))

    def forward(self, *args, **kwargs) -> List[DocNode]:
        r = self._load_data(*args, **kwargs)
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
            chinese_encodings = ['gbk', 'gb2312', 'big5']
            for encoding in chinese_encodings:
                if cls._try_decode(raw_data, encoding):
                    try:
                        decoded = raw_data.decode(encoding)
                        if any('\u4e00' <= c <= '\u9fff' for c in decoded[:100]):
                            cls._cache_encoding(cache_key, encoding)
                            return encoding
                    except Exception:
                        pass

            if cls._try_decode(raw_data, 'utf-8'):
                cls._cache_encoding(cache_key, 'utf-8')
                return 'utf-8'
        else:
            primary_encodings = ['utf-8', 'gbk', 'gb2312', 'big5']
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
