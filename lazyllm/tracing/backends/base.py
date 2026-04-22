from abc import ABC, abstractmethod
from typing import Any, Dict, List

from lazyllm.tracing.consume.datamodel.raw import RawSpanRecord, RawTraceRecord


class TracingBackend(ABC):
    name = ''

    @abstractmethod
    def build_exporter(self):
        pass

    @abstractmethod
    def map_attributes(self, otel_attrs: Dict[str, Any]) -> Dict[str, Any]:
        pass


class ConsumeBackend(ABC):
    name: str

    @abstractmethod
    def fetch_trace(self, trace_id: str) -> RawTraceRecord:
        pass

    @abstractmethod
    def fetch_spans(self, trace_id: str) -> List[RawSpanRecord]:
        pass
