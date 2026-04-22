from typing import List

from ..datamodel.raw import RawSpanRecord, RawTraceRecord
from ..datamodel.structured import StructuredTrace


def rebuild(trace: RawTraceRecord, spans: List[RawSpanRecord]) -> StructuredTrace:
    raise NotImplementedError
