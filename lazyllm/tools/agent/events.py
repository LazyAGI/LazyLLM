from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# --- Event type constants ---
TEXT_DELTA = 'agent.text.delta'
REASONING_DELTA = 'agent.reasoning.delta'
REASONING_FINISHED = 'agent.reasoning.finished'
TEXT_FINISHED = 'agent.text.finished'
TOOL_CALLS = 'agent.tool.calls'
TOOL_RESULTS = 'agent.tool.results'
PLAN_STARTED = 'agent.plan.started'
PLAN_FINISHED = 'agent.plan.finished'
AGENT_FINISHED = 'agent.finished'
AGENT_FAILED = 'agent.failed'


@dataclass
class AgentEvent:
    type: str
    agent: Optional[str] = None
    delta: Optional[str] = None
    text: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
