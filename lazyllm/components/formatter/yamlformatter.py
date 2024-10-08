import yaml
from .formatterbase import JsonLikeFormatter
import lazyllm


class YamlFormatter(JsonLikeFormatter):
    def _load(self, msg: str):
        try:
            return yaml.load(msg, Loader=yaml.SafeLoader)
        except Exception as e:
            lazyllm.LOG.info(f"Error: {e}")
            return ""
