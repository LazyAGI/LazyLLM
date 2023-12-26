from .core import LLMBase, register


class LazyLLMValidateBase(LLMBase):
    pass


@register(LazyLLMValidateBase)
def test1():
    pass

@register('Validate')
def test2():
    pass

@register('validate')
def test3():
    pass