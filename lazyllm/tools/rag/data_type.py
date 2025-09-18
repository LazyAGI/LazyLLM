from enum import IntEnum

class DataType(IntEnum):
    """An enumeration."""
    VARCHAR = 0
    ARRAY = 1
    INT32 = 2
    FLOAT_VECTOR = 3
    SPARSE_FLOAT_VECTOR = 4
    BOOLEAN = 5
    FLOAT = 6
    INT64 = 7
    STRING = 8
