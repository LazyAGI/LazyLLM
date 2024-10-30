from ...common import LazyLLMRegisterMetaClass, package, \
    decode_query_with_filepaths, encode_query_with_filepaths
from typing import Optional

def is_number(s: str):
    try:
        int(s)
        return True
    except ValueError:
        if s == "None" or len(s) == 0:
            return False
        else:
            raise ValueError("Invalid number: " + s + ". You can enter an integer, None or an empyt string.")

class LazyLLMFormatterBase(metaclass=LazyLLMRegisterMetaClass):
    def _load(self, msg: str):
        return msg

    def _parse_py_data_by_formatter(self, py_data):
        raise NotImplementedError("This data parse function is not implemented.")

    def format(self, msg):
        if isinstance(msg, str): msg = self._load(msg)
        return self._parse_py_data_by_formatter(msg)

    def __call__(self, *msg):
        return self.format(msg[0] if len(msg) == 1 else package(msg))


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
        dimensions = slice_str.split("][")
        slices = []

        for dim in dimensions:
            if '{' in dim:
                slices.append(__class__._DictKeys(d.strip() for d in dim[1:-1].split(',') if d.strip()))
            elif ":" in dim:
                assert ',' not in dim, '[a, b:c] is not supported'
                parts = dim.split(":")
                start = int(parts[0]) if is_number(parts[0]) else None
                end = int(parts[1]) if len(parts) > 1 and is_number(parts[1]) else None
                step = int(parts[2]) if len(parts) > 2 and is_number(parts[2]) else None
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

class FileFormatter(LazyLLMFormatterBase):

    def __init__(self, formatter: str = 'decode'):
        self._mode = formatter.strip().lower()
        assert self._mode in ('decode', 'encode')

    def _parse_py_data_by_formatter(self, py_data):
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
