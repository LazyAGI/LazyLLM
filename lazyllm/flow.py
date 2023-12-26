from lazyllm import LazyLLMRegisterMetaClass

class LazyLLMFlowBase(object, metaclass=LazyLLMRegisterMetaClass):
    pass


# input -> module1 -> module2 -> ... -> moduleN -> output
class Pileline(LazyLLMFlowBase):
    pass

#        /> module11 -> ... -> module1N -> out1 \
#  input -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
#        \> module31 -> ... -> module3N -> out3 /
class Parallel(LazyLLMFlowBase):
    pass