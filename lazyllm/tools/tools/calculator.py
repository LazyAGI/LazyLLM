from lazyllm.module import ModuleBase
from math import * # noqa. import math functions for expressions

class Calculator(ModuleBase):
    """
这是一个计算器应用，可以计算用户输入的表达式的值。


Examples:

    from lazyllm.tools.tools import Calculator
    calc = Calculator()
    """
    def __init__(self):
        super().__init__()

    def forward(self, exp: str, *args, **kwargs):
        """
计算用户输入的表达式的值。

Args:
    exp (str): 需要计算的表达式的值。必须符合 Python 计算表达式的语法。可使用 Python math 库中的数学函数。


Examples:

    from lazyllm.tools.tools import Calculator
    calc = Calculator()
    """
        return eval(exp)
