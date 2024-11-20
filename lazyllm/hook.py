from abc import ABC, abstractmethod


class LazyLLMHook(ABC):

    @abstractmethod
    def __init__(self, obj):
        pass

    @abstractmethod
    def pre_hook(self, *args, **kwargs):
        pass

    @abstractmethod
    def post_hook(self, output):
        pass

    @abstractmethod
    def report():
        pass
