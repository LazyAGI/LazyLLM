from ...common import LazyLLMRegisterMetaClass

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

    def __call__(self, msg):
        return self.format(msg)


class JsonLikeFormatter(LazyLLMFormatterBase):
    def __init__(self, formatter: str = None):
        self._formatter = formatter
        if self._formatter:
            self._formatter = self._formatter.strip()
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
            if ":" in dim:
                assert ',' not in dim, '[a, b:c] is not supported'
                parts = dim.split(":")
                start = int(parts[0]) if is_number(parts[0]) else None
                end = int(parts[1]) if len(parts) > 1 and is_number(parts[1]) else None
                step = int(parts[2]) if len(parts) > 2 and is_number(parts[2]) else None
                slices.append(slice(start, end, step))
            elif ',' in dim:
                slices.append(tuple(d.strip() for d in dim.split(',') if d.strip()))
            else:
                slices.append(dim.strip())
        self._slices = slices

    def _parse_py_data_by_formatter(self, data, *, slices=None):
        def _impl(data, slice):
            if isinstance(data, (tuple, list)) and isinstance(slice, str):
                return data[int(slice)]
            if isinstance(slice, tuple):
                if isinstance(data, dict): return [data[k] for k in slice]
                elif isinstance(data, (tuple, list)): return [data[int(k)] for k in slice]
                else: raise RuntimeError('Only tuple/list/dict is supported for [a,b,c]')
            return data[slice]

        if slices is None: slices = self._slices
        if not slices: return data
        if isinstance(slices[0], slice): return [self._parse_py_data_by_formatter(d, slices=slices[1:])
                                                 for d in _impl(data, slices[0])]
        elif isinstance(slices[0], tuple):
            rs = _impl(data, slices[0])
            if isinstance(rs, dict):
                return {k: self._parse_py_data_by_formatter(v, slices=slices[1:]) for k, v in rs.items()}
            elif isinstance(rs, (tuple, list)):
                return [self._parse_py_data_by_formatter(r, slices=slices[1:]) for r in rs]
        else: return self._parse_py_data_by_formatter(_impl(data, slices[0]), slices=slices[1:])


class EmptyFormatter(LazyLLMFormatterBase):
    def _parse_py_data_by_formatter(self, msg: str):
        return msg
