from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class ToolCallTurn:
    start: int
    stop: int
    result_indexes: Tuple[int, ...]
    current: bool


def describe_tool_turns(prior: List[Dict[str, Any]], current: List[Dict[str, Any]]) -> Tuple[ToolCallTurn, ...]:
    # Ranges address prior + current; stop is exclusive. A turn touching current
    # is protected as a whole, regardless of where its call/results were split.
    turns = []
    start = None
    pending = set()
    results = []
    for index, message in enumerate(prior + current):
        role = message.get('role')
        if role == 'assistant':
            start, pending, results = None, set(), []
            calls = message.get('tool_calls') or []
            if not isinstance(calls, list) or not calls:
                continue
            ids = [call.get('id') if isinstance(call, dict) else None for call in calls]
            if any(not isinstance(call_id, str) or not call_id for call_id in ids):
                continue
            if len(set(ids)) != len(ids):
                continue
            start, pending = index, set(ids)
        elif start is not None:
            call_id = message.get('tool_call_id')
            if role != 'tool' or not isinstance(call_id, str) or call_id not in pending:
                start, pending, results = None, set(), []
                continue
            pending.remove(call_id)
            results.append(index)
            if not pending:
                turns.append(ToolCallTurn(start, index + 1, tuple(results), index >= len(prior)))
                start, results = None, []
    return tuple(turns)
