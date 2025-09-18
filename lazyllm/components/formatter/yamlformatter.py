import yaml
from .formatterbase import JsonLikeFormatter
import lazyllm


class YamlFormatter(JsonLikeFormatter):
    """A formatter for extracting structured information from YAML-formatted strings.

Inherits from JsonLikeFormatter. Uses the internal method to parse YAML strings into Python objects, and then applies JSON-like formatting rules to extract desired fields.

Suitable for handling nested YAML content with formatter-based field selection.


Examples:
    >>> from lazyllm.components.formatter import YamlFormatter
    >>> formatter = YamlFormatter("{name,age}")
    >>> msg = \"\"\" 
    ... name: Alice
    ... age: 30
    ... city: London
    ... \"\"\"
    >>> formatter(msg)
    {'name': 'Alice', 'age': 30}
    """
    def _load(self, msg: str):
        try:
            return yaml.load(msg, Loader=yaml.SafeLoader)
        except Exception as e:
            lazyllm.LOG.info(f'Error: {e}')
            return ''
