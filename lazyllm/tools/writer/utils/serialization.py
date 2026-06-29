from __future__ import annotations
import json
from typing import Any


def to_prompt_json(value: Any) -> str:
    def default(obj: Any) -> Any:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return str(obj)

    return json.dumps(value, ensure_ascii=False, indent=2, default=default)
