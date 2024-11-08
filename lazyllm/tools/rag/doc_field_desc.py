from typing import Optional, Any

class DocFieldDesc:
    DTYPE_VARCHAR = 0

    def __init__(self, data_type: int, default_value: Optional[Any] = None,
                 max_length: Optional[int] = None):
        self.data_type = data_type
        self.default_value = default_value
        self.max_length = max_length
