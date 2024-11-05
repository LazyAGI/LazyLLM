class DocFieldInfo:
    DTYPE_UNKNOWN = 0
    DTYPE_VARCHAR = 1

    def __init__(self, data_type: DTYPE_UNKNOWN):
        self.data_type = data_type
