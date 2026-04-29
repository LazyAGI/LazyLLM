from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from lazyllm.tracing.datamodel.raw import RawSpanRecord, RawTracePayload, RawTraceRecord


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
    def fetch_trace_payload(
        self,
        trace_id: str,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> RawTracePayload:
        pass

    def fetch_spans(
        self,
        trace_id: str,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> List[RawSpanRecord]:
        return self.fetch_trace_payload(trace_id, timeout_seconds=timeout_seconds).spans

    def fetch_trace(
        self,
        trace_id: str,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> RawTraceRecord:
        return self.fetch_trace_payload(trace_id, timeout_seconds=timeout_seconds).trace
