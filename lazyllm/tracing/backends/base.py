from abc import ABC, abstractmethod
from typing import Any, Dict


class TracingBackend(ABC):
    name = ''

    @abstractmethod
    def build_exporter(self):
        pass

    @abstractmethod
    def map_attributes(self, otel_attrs: Dict[str, Any]) -> Dict[str, Any]:
        '''Map generic OTel attributes to backend-specific OTel attributes.

        Called once at flush time (finish_span) to produce all attributes
        that should be written to the underlying OTel span.
        '''
        pass
