from .datamodel.structured import StructuredTrace
from .datamodel.views import TraceDetailView


def assemble_detail_view(structured: StructuredTrace) -> TraceDetailView:
    '''StructuredTrace → TraceDetailView；纯映射与白名单 raw_data（§8）。'''
    raise NotImplementedError
