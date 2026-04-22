from typing import Any, Dict, Optional

from ...datamodel.raw import RawSpanRecord


def extract(span: RawSpanRecord) -> Optional[Dict[str, Any]]:
    '''P0: query / topk / node_count / returned_node_ids / retrieve_scores（§7.5）。'''
    return None
