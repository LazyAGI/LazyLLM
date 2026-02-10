'''Base class for prompt templates'''
from abc import ABC, abstractmethod


class PromptABC(ABC):
    '''Abstract base class for prompt templates.'''

    @abstractmethod
    def build_prompt(self, *args, **kwargs) -> str:
        '''Build the prompt string.'''
        raise NotImplementedError

    def build_system_prompt(self) -> str:
        '''Build the system prompt string. Override if needed.'''
        return ''
