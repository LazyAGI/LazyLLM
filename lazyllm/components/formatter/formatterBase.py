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
                start = int(parts[0]) if is_number(parts[0]) else None
                end = int(parts[1]) if len(parts) > 1 and is_number(parts[1]) else None
                step = int(parts[2]) if len(parts) > 2 and is_number(parts[2]) else None
                slices.append(slice(start, end, step))
            else:
                slices.append(dim.strip())
        self._slices = slices

    def _load(self, msg: str):
        raise NotImplementedError("This parse str function is not implemented.")

    def _parse_py_data_by_formatter(self, py_data):
        raise NotImplementedError("This data parse function is not implemented.")

    def format(self, msg):
        if isinstance(msg, str): msg = self._load(msg)
        return self._parse_py_data_by_formatter(msg)

    def __call__(self, msg):
        return self.format(msg)

class EmptyFormatter(LazyLLMFormatterBase):
    def format(self, msg):
        return msg
