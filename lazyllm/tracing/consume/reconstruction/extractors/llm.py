from typing import Any, Dict, Optional

from ...datamodel.raw import RawSpanRecord


def extract(span: RawSpanRecord) -> Optional[Dict[str, Any]]:
    '''P0: prompt / answer / model_name / usage from lazyllm.* / gen_ai.*（§7.5）。'''
    return None
