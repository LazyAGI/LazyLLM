from lazyllm.module import ModuleBase
from math import * # noqa. import math functions for expressions

class Calculator(ModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, exp: str, *args, **kwargs):
        return eval(exp)
