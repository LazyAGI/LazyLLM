from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class TracingBackend(ABC):
    name = ''

    @abstractmethod
    def build_exporter(self):
        pass

    @abstractmethod
    def context_attributes(self, trace_ctx: Dict[str, Any], *, is_root_span: bool) -> Dict[str, Any]:
        pass

    @abstractmethod
    def input_attributes(self, args: tuple[Any, ...], kwargs: Dict[str, Any], *,
                         capture_payload: bool, is_root_span: bool) -> Dict[str, Any]:
        pass

    @abstractmethod
    def set_root_span_name(self, span: Any, span_name: str):
        pass

    @abstractmethod
    def output_attributes(self, text: str, *, is_root_span: bool) -> Dict[str, Any]:
        pass

    @abstractmethod
    def error_attributes(self, exc: Exception) -> Dict[str, Any]:
        pass

    @abstractmethod
    def metadata_attributes(self, trace_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def observation_type_attributes(self, span_kind: str, semantic_type: Optional[str],
                                    trace_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        pass
