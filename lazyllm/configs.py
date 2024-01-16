import os
from enum import Enum

class Mode(Enum):
    Display = 0,
    Normal = 1,
    Debug = 2,

if os.getenv('LAZYLLM_DISPLAY', False) in (True, 'TRUE', 'True', 1, 'ON', '1'):
    mode = Mode.Display
elif os.getenv('LAZYLLM_DEBUG', False) in (True, 'TRUE', 'True', 1, 'ON', '1'):
    mode = Mode.Debug
else:
    mode = Mode.Normal
