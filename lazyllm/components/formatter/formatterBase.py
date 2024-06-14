from ...common import LazyLLMRegisterMetaClass

class LazyLLMFormatterBase(metaclass=LazyLLMRegisterMetaClass):
    def __init__(self, formatter: str = None):
        self._formatter = formatter
        if self._formatter:
            self._parse_formatter()
        else:
            self._slices = None

    def _parse_formatter(self):
        # Remove the surrounding brackets
        slice_str = self._formatter.strip()[1:-1]
        dimensions = slice_str.split(",")
        slices = []

        for dim in dimensions:
            if ":" in dim:
                parts = dim.split(":")
                start = int(parts[0]) if parts[0] else None
                end = int(parts[1]) if len(parts) > 1 and parts[1] else None
                step = int(parts[2]) if len(parts) > 2 and parts[2] else None
                slices.append(slice(start, end, step))
            else:
                slices.append(dim.strip())
        self._slices = slices

    def _str_to_python(self, msg: str):
        raise NotImplementedError("This str to python convert function is not implemented.")

    def _parse_py_data_by_formatter(self, py_data):
        raise NotImplementedError("This data parse function is not implemented.")

    def format(self, msg):
        if isinstance(msg, str):
            py_data = self._str_to_python(msg)
        else:
            py_data = msg
        res = self._parse_py_data_by_formatter(py_data)
        return res

class EmptyFormatter(LazyLLMFormatterBase):
    def format(self, msg):
        return msg
