from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


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

    @abstractmethod
    def metadata_attributes(self, trace_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def observation_type_attributes(self, span_kind: str, semantic_type: Optional[str],
                                    trace_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def usage_attributes(self, usage: Dict[str, Any]) -> Dict[str, Any]:
        pass
