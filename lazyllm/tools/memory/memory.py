from .suppliers import MemUMemory

class Memory():
    def __new__(cls) -> LazyLLMMemoryBase: