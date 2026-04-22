from .datamodel.structured import StructuredTrace
from .datamodel.views import TraceDetailView


def assemble_detail_view(structured: StructuredTrace) -> TraceDetailView:
    raise NotImplementedError
