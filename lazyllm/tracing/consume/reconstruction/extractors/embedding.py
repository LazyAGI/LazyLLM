from typing import Any, Dict, Optional

from ...datamodel.raw import RawSpanRecord


def extract(span: RawSpanRecord) -> Optional[Dict[str, Any]]:
    '''P0: input_summary / model_name / dim（§7.5）。'''
    return None
