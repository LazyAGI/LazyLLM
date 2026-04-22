from abc import ABC, abstractmethod
from typing import Any, Dict, List

from lazyllm.tracing.consume.datamodel.raw import RawSpanRecord, RawTracePayload, RawTraceRecord


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
    def fetch_trace_payload(self, trace_id: str) -> RawTracePayload:
        pass

    def fetch_spans(self, trace_id: str) -> List[RawSpanRecord]:
        return self.fetch_trace_payload(trace_id).spans

    def fetch_trace(self, trace_id: str) -> RawTraceRecord:
        return self.fetch_trace_payload(trace_id).trace
