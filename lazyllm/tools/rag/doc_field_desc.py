from typing import Optional

class DocFieldDesc:
    DTYPE_VARCHAR = 0

    def __init__(self, data_type: int = DTYPE_VARCHAR, max_length: Optional[int] = 65535):
        self.data_type = data_type
        self.max_length = max_length
