from lazyllm.module import ModuleBase
from math import * # noqa. import math functions for expressions

class Calculator(ModuleBase):
    """
This is a calculator application that can calculate the value of expressions entered by the user.


Examples:
    
    from lazyllm.tools.tools import Calculator
    calc = Calculator()
    """
    def __init__(self):
        super().__init__()

    def forward(self, exp: str, *args, **kwargs):
        """
Calculate the value of the user input expression.

Args:
    exp (str): The expression to be calculated. It must conform to the syntax for evaluating expressions in Python. Mathematical functions from the Python math library can be used.


Examples:
    
    from lazyllm.tools.tools import Calculator
    calc = Calculator()
    """
        return eval(exp)
