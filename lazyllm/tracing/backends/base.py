from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ..span import LazySpan


class TracingBackend(ABC):
    name = ''

    @abstractmethod
    def build_exporter(self):
        pass

    @abstractmethod
    def map_span_attributes(self, span: 'LazySpan') -> Dict[str, Any]:
        """Map a LazySpan to backend-specific OTel span attributes.

        Called once at flush time (finish_span) to produce all attributes
        that should be written to the underlying OTel span.
        """
        pass

    @abstractmethod
    def map_root_span_attributes(self, span: 'LazySpan') -> Dict[str, Any]:
        """Extra attributes only for root spans (trace-level metadata)."""
        pass
