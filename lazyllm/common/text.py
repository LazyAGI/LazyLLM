from typing import Optional


class Color(object):
    red: str = '\033[31m'
    green: str = '\033[32m'
    yellow: str = '\033[33m'
    blue: str = '\033[34m'
    magenta: str = '\033[35m'
    cyan: str = '\033[36m'
    reset: str = '\033[0m'

def colored_text(text: str, color: Optional[str] = None):
    if not color: return text
    color = color if color.startswith('\033') else getattr(Color, color, Color.reset)
    return f'{color}{text}{Color.reset}'
