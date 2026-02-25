from abc import ABC, abstractmethod


class PromptABC(ABC):

    @abstractmethod
    def build_prompt(self, *args, **kwargs) -> str:
        raise NotImplementedError

    def build_system_prompt(self) -> str:
        return ''
