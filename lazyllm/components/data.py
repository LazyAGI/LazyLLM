from .core import ComponentBase, register


class LazyLLMDataprocBase(ComponentBase):
    pass


@register('dataproc')
def dummy():
    return 'trainset', 'evalset'
