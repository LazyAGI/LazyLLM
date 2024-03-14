from .core import LLMBase, register


class LazyLLMDataprocBase(LLMBase):
    pass


@register('dataproc')
def dummy():
    return 'trainset', 'evalset'