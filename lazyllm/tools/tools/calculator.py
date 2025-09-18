from lazyllm.module import ModuleBase
from math import * # noqa. import math functions for expressions

class Calculator(ModuleBase):
    """
Simple calculator module, inherits from ModuleBase.

Provides mathematical expression calculation functionality, supports basic arithmetic operations and math functions.


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
    *args: Variable positional arguments
    **kwargs: Variable keyword arguments 


Examples:
    
    from lazyllm.tools.tools import Calculator
    calc = Calculator()
    result1 = calc.forward("2 + 3 * 4")  
    print(f"2 + 3 * 4 = {result1}")
    """
        return eval(exp)
