import os
import json
import uuid
import hashlib
from typing import Callable, Optional, List, Union, Any

import lazyllm
from lazyllm.flow.flow import Pipeline

from ...common import LazyLLMRegisterMetaClass, package, Finalizer

def _is_number(s: str):
    try:
        int(s)
        return True
    except ValueError:
        if s == 'None' or len(s) == 0:
            return False
        else:
            raise ValueError('Invalid number: ' + s + '. You can enter an integer, None or an empyt string.')

def _is_chat_message(msg):
    return isinstance(msg, dict) and 'role' in msg and 'content' in msg

class LazyLLMFormatterBase(metaclass=LazyLLMRegisterMetaClass):
    def _load(self, msg: str):
        return msg

    def _parse_py_data_by_formatter(self, py_data):
        raise NotImplementedError('This data parse function is not implemented.')

    def format(self, msg):
        if _is_chat_message(msg): msg = msg['content']
        if isinstance(msg, str): msg = self._load(msg)
        return self._parse_py_data_by_formatter(msg)

    def __call__(self, *msg):
        return self.format(msg[0] if len(msg) == 1 else package(msg))

    def __or__(self, other):
        if not isinstance(other, LazyLLMFormatterBase):
            return NotImplemented
        return PipelineFormatter(other.__ror__(self))

    def __ror__(self, f: Callable) -> Pipeline:
        if isinstance(f, Pipeline):
            if not f._capture:
                _ = Finalizer(lambda: setattr(f, '_capture', True), lambda: setattr(f, '_capture', False))
            f._add(str(uuid.uuid4().hex) if len(f._item_names) else None, self)
            return f
        return Pipeline(f, self)


class PipelineFormatter(LazyLLMFormatterBase):
    def __init__(self, formatter: Pipeline):
        self._formatter = formatter

    def _parse_py_data_by_formatter(self, py_data):
        return self._formatter(py_data)

    def __or__(self, other):
        if isinstance(other, LazyLLMFormatterBase):
            if isinstance(other, PipelineFormatter): other = other._formatter
            return PipelineFormatter(self._formatter | other)
        return NotImplemented


class JsonLikeFormatter(LazyLLMFormatterBase):
    class _ListIdxes(tuple): pass
    class _DictKeys(tuple): pass

    def __init__(self, formatter: Optional[str] = None):
        if formatter and formatter.startswith('*['):
            self._return_package = True
            self._formatter = formatter.strip('*')
        else:
            self._return_package = False
            self._formatter = formatter

        if self._formatter:
            assert '*' not in self._formatter, '`*` can only be used before `[` in the beginning'
            self._formatter = self._formatter.strip().replace('{', '[{').replace('}', '}]')
            self._parse_formatter()
        else:
            self._slices = None

    def _parse_formatter(self):
        # Remove the surrounding brackets
        assert self._formatter.startswith('[') and self._formatter.endswith(']')
        slice_str = self._formatter.strip()[1:-1]
        dimensions = slice_str.split('][')
        slices = []

        for dim in dimensions:
            if '{' in dim:
                slices.append(__class__._DictKeys(d.strip() for d in dim[1:-1].split(',') if d.strip()))
            elif ':' in dim:
                assert ',' not in dim, '[a, b:c] is not supported'
                parts = dim.split(':')
                start = int(parts[0]) if _is_number(parts[0]) else None
                end = int(parts[1]) if len(parts) > 1 and _is_number(parts[1]) else None
                step = int(parts[2]) if len(parts) > 2 and _is_number(parts[2]) else None
                slices.append(slice(start, end, step))
            elif ',' in dim:
                slices.append(__class__._ListIdxes(d.strip() for d in dim.split(',') if d.strip()))
            else:
                slices.append(dim.strip())
        self._slices = slices

    def _parse_py_data_by_formatter(self, data, *, slices=None):  # noqa C901
        def _impl(data, slice):
            if isinstance(data, (tuple, list)) and isinstance(slice, str):
                return data[int(slice)]
            if isinstance(slice, __class__._ListIdxes):
                if isinstance(data, dict): return [data[k] for k in slice]
                elif isinstance(data, (tuple, list)): return type(data)(data[int(k)] for k in slice)
                else: raise RuntimeError('Only tuple/list/dict is supported for [a,b,c]')
            if isinstance(slice, __class__._DictKeys):
                assert isinstance(data, dict)
                if len(slice) == 1 and slice[0] == ':': return data
                return {k: data[k] for k in slice}
            return data[slice]

        if slices is None: slices = self._slices
        if not slices: return data
        curr_slice = slices[0]
        if isinstance(curr_slice, slice):
            if isinstance(data, dict):
                assert curr_slice.start is None and curr_slice.stop is None and curr_slice.step is None, (
                    'Only {:} and [:] is supported in dict slice')
                curr_slice = __class__._ListIdxes(data.keys())
            elif isinstance(data, (tuple, list)):
                return type(data)(self._parse_py_data_by_formatter(d, slices=slices[1:])
                                  for d in _impl(data, curr_slice))
        if isinstance(curr_slice, __class__._DictKeys):
            return {k: self._parse_py_data_by_formatter(v, slices=slices[1:])
                    for k, v in _impl(data, curr_slice).items()}
        elif isinstance(curr_slice, __class__._ListIdxes):
            tp = package if self._return_package else list if isinstance(data, dict) else type(data)
            return tp(self._parse_py_data_by_formatter(r, slices=slices[1:]) for r in _impl(data, curr_slice))
        else: return self._parse_py_data_by_formatter(_impl(data, curr_slice), slices=slices[1:])


class PythonFormatter(JsonLikeFormatter): pass


class EmptyFormatter(LazyLLMFormatterBase):
    def _parse_py_data_by_formatter(self, msg: str):
        return msg

class FunctionCallFormatter(LazyLLMFormatterBase):
    def format(self, msg):
        assert isinstance(msg, dict), 'FunctionCallFormatter only supports dict input.'
        return {k: msg[k] for k in ('role', 'content', 'tool_calls') if k in msg}

LAZYLLM_QUERY_PREFIX = '<lazyllm-query>'

def encode_query_with_filepaths(query: str = None, files: Union[str, List[str]] = None) -> str:
    query = query if query else ''
    if files:
        if isinstance(files, str): files = [files]
        assert isinstance(files, list), 'files must be a list.'
        assert all(isinstance(item, str) for item in files), 'All items in files must be strings'
        return LAZYLLM_QUERY_PREFIX + json.dumps({'query': query, 'files': files})
    else:
        return query

def decode_query_with_filepaths(query_files: str) -> Union[dict, str]:
    assert isinstance(query_files, str), 'query_files must be a str.'
    query_files = query_files.strip()
    if query_files.startswith(LAZYLLM_QUERY_PREFIX):
        try:
            obj = json.loads(query_files[len(LAZYLLM_QUERY_PREFIX):])
            return obj
        except json.JSONDecodeError as e:
            raise ValueError(f'JSON parsing failed: {e}')
    else:
        return query_files

def lazyllm_merge_query(*args: str) -> str:
    if len(args) == 1:
        return args[0]
    for item in args:
        assert isinstance(item, str), 'Merge object must be str!'
    querys = ''
    files = []
    for item in args:
        decode = decode_query_with_filepaths(item)
        if isinstance(decode, dict):
            querys += decode['query']
            files.extend(decode['files'])
        else:
            querys += decode
    return encode_query_with_filepaths(querys, files)

def _lazyllm_get_file_list(files: Any) -> list:
    if isinstance(files, str):
        decode = decode_query_with_filepaths(files)
        if isinstance(decode, str):
            return [decode]
        if isinstance(decode, dict):
            return decode['files']
    elif isinstance(files, dict) and set(files.keys()) == {'query', 'files'}:
        return files['files']
    elif isinstance(files, list) and all(isinstance(item, str) for item in files):
        return files
    else:
        raise TypeError(f'Not supported type: {type(files)}.')

class FileFormatter(LazyLLMFormatterBase):

    def __init__(self, formatter: str = 'decode'):
        self._mode = formatter.strip().lower()
        assert self._mode in ('decode', 'encode', 'merge')

    def _parse_py_data_by_formatter(self, py_data):
        if self._mode == 'merge':
            if isinstance(py_data, str):
                return py_data
            assert isinstance(py_data, package)
            return lazyllm_merge_query(*py_data)

        if isinstance(py_data, package):
            res = []
            for i_data in py_data:
                res.append(self._parse_py_data_by_formatter(i_data))
            return package(res)
        elif isinstance(py_data, (str, dict)):
            return self._decode_one_data(py_data)
        else:
            return py_data

    def _decode_one_data(self, py_data):
        if self._mode == 'decode':
            if isinstance(py_data, str):
                return decode_query_with_filepaths(py_data)
            else:
                return py_data
        else:
            if isinstance(py_data, dict) and 'query' in py_data and 'files' in py_data:
                return encode_query_with_filepaths(**py_data)
            else:
                return py_data


def file_content_hash(value):
    res = decode_query_with_filepaths(value)
    if isinstance(res, str):
        return hashlib.md5(value.encode()).hexdigest()
    query = res['query']
    file_path_list = res['files']

    hash_obj = hashlib.md5()
    hash_obj.update(query.encode('utf-8'))

    search_paths = [
        '',
        lazyllm.config['temp_dir'],
    ]
    for file_path in file_path_list:
        if os.path.isabs(file_path):
            full_path = file_path
        else:
            full_path = None
            for base_path in search_paths:
                candidate_path = os.path.join(base_path, file_path) if base_path else file_path
                if os.path.exists(candidate_path) and os.path.isfile(candidate_path):
                    full_path = candidate_path
                    break
            if full_path is None:
                full_path = file_path
        try:
            with open(full_path, 'rb') as f:
                while chunk := f.read(8192):
                    hash_obj.update(chunk)
        except OSError:
            lazyllm.LOG.debug(f'Error: File not found or cannot be read: {full_path}')
            hash_obj.update(full_path.encode('utf-8'))
    return int(hash_obj.hexdigest(), 16)


def proccess_path_recursively(value, process_func):
    if isinstance(value, str):
        if LAZYLLM_QUERY_PREFIX in value:
            res = decode_query_with_filepaths(value)
            if res['files']:
                replace_path = []
                for file_path in res['files']:
                    replace_path.append(process_func(file_path))
                res['files'] = replace_path
            return encode_query_with_filepaths(res['query'], res['files'])
        else:
            return value
    elif isinstance(value, (list, tuple)):
        process_items = []
        for item in value:
            process_items.append(proccess_path_recursively(item, process_func))
        if isinstance(value, tuple):
            return tuple(process_items)
        return process_items
    elif isinstance(value, dict):
        process_dict = {}
        for key, val in value.items():
            process_dict[key] = proccess_path_recursively(val, process_func)
        return process_dict
    elif isinstance(value, set):
        process_set = set()
        for item in value:
            process_set.add(proccess_path_recursively(item, process_func))
        return process_set
    else:
        return value

def path_relative_to_absolute(path):
    if os.path.isabs(path):
        return path
    absolute_path = os.path.join(lazyllm.config['temp_dir'], path)
    if os.path.exists(absolute_path):
        return os.path.abspath(absolute_path)
    else:
        return path

def path_absolute_to_relative(path):
    if not os.path.isabs(path):
        return path
    temp_dir_abs = os.path.abspath(lazyllm.config['temp_dir'])
    if path.startswith(temp_dir_abs):
        relative_path = path[len(temp_dir_abs):]
        if relative_path.startswith(os.sep):
            relative_path = relative_path[1:]
        return relative_path
    else:
        return path

def transform_path(value, mode='r2a'):
    assert mode in ('r2a', 'a2r')
    if mode == 'r2a':
        return proccess_path_recursively(value, path_relative_to_absolute)
    else:
        return proccess_path_recursively(value, path_absolute_to_relative)
