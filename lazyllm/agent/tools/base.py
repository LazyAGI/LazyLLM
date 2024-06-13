from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

class BaseTool(ABC):
    name: str
    description: str
    parameters: Dict[str, Union[str, Dict, List, Any]]

    @abstractmethod
    def call(**kwages) -> Union[str, list, dict]:
        raise NotImplementedError