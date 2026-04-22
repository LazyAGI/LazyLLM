from typing import List

from ..datamodel.raw import RawSpanRecord, RawTraceRecord
from ..datamodel.structured import StructuredTrace


def rebuild(trace: RawTraceRecord, spans: List[RawSpanRecord]) -> StructuredTrace:
    '''结构 + 时序 + 身份重建（§7）；零 I/O。'''
    raise NotImplementedError
