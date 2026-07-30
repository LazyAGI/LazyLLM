from lazyllm.thirdparty import fsspec
from typing import Iterable, List, Optional, Union, Callable

from lazyllm.thirdparty import torch
from lazyllm import LOG, config

from ....common import LazyLLMRegisterMetaClass
from ..doc_node import DocNode, RichDocNode
from lazyllm.module import ModuleBase
from . import reader_config_inject as _reader_config_inject  # noqa: F401
from pathlib import Path
import functools
import hashlib
import inspect
import json
import locale
import marshal
import threading
from lazyllm.thirdparty import charset_normalizer

_READER_CALL_SKIP_KEYS = frozenset({'use_cache', 'lazyllm_files', 'llm_chat_history'})


def _stable_cache_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {'bytes': hashlib.sha256(value).hexdigest()}
    if isinstance(value, (list, tuple)):
        return [_stable_cache_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _stable_cache_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    raise TypeError(f'Unsupported cache signature value: {type(value).__name__}')


def _callable_cache_signature(action):
    try:
        if isinstance(action, functools.partial):
            signature = {
                'type': 'partial',
                'func': _callable_cache_signature(action.func),
                'args': _stable_cache_value(action.args),
                'keywords': _stable_cache_value(action.keywords or {}),
            }
        elif inspect.ismethod(action):
            signature = {
                'type': 'method',
                'func': _callable_cache_signature(action.__func__),
                'owner': _callable_cache_signature(action.__self__),
            }
        elif inspect.isfunction(action):
            closure = tuple(cell.cell_contents for cell in (action.__closure__ or ()))
            signature = {
                'type': 'function',
                'module': action.__module__,
                'qualname': action.__qualname__,
                'code': hashlib.sha256(marshal.dumps(action.__code__)).hexdigest(),
                'defaults': _stable_cache_value(action.__defaults__ or ()),
                'kwdefaults': _stable_cache_value(action.__kwdefaults__ or {}),
                'closure': _stable_cache_value(closure),
            }
        elif inspect.isbuiltin(action):
            signature = {
                'type': 'builtin',
                'module': action.__module__,
                'qualname': action.__qualname__,
            }
        elif callable(action) and callable(getattr(action, 'sig_fields', None)):
            signature = {
                'type': f'{type(action).__module__}.{type(action).__qualname__}',
                'fields': _stable_cache_value(action.sig_fields()),
            }
        else:
            custom_hash = getattr(action, '__cache_hash__', None)
            if custom_hash is None:
                raise TypeError('Callable does not expose a stable cache signature')
            signature = {
                'type': f'{type(action).__module__}.{type(action).__qualname__}',
                'cache_hash': _stable_cache_value(custom_hash() if callable(custom_hash) else custom_hash),
            }
        serialized = json.dumps(signature, ensure_ascii=True, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(serialized.encode()).hexdigest()
    except (AttributeError, TypeError, ValueError):
        # An address-bearing repr may cause a safe cache miss, but cannot cause two
        # semantically different actions to share a cache entry.
        return f'unstable:{action!r}'


class LazyLLMReaderBase(ModuleBase, metaclass=LazyLLMRegisterMetaClass):
    post_action = None

    _encoding_cache = {}
    _cache_lock = threading.Lock()
    _cache_max_size = 1000

    def __init__(self, *args, return_trace: bool = True, **kwargs):
        super().__init__(return_trace=return_trace)
        self.use_cache(bool(config['reader_use_cache']))

    @property
    def __cache_hash__(self):
        cache_hash = super().__cache_hash__
        if self.post_action is not None:
            cache_hash += f'@post_action:{_callable_cache_signature(self.post_action)}'
        return cache_hash

    def _lazy_load_data(self, *args, **load_kwargs) -> Iterable[DocNode]:
        raise NotImplementedError(f'{self.__class__.__name__} does not implement lazy_load_data method.')

    def _load_data(self, *args, **load_kwargs) -> List[DocNode]:
        return list(self._lazy_load_data(*args, **load_kwargs))

    def forward(self, *args, **kwargs) -> List[DocNode]:
        load_kwargs = {k: v for k, v in kwargs.items() if k not in _READER_CALL_SKIP_KEYS}
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

config.add('reader_use_cache', bool, False, 'READER_USE_CACHE',
           description='Global ModuleBase reader content cache flag (OCR HTTP use_cache is separate).')
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
