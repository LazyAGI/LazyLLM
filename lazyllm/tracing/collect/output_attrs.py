import contextvars
import json
import types
from typing import Any, Dict, Optional

from lazyllm.common import LOG
from lazyllm.common import globals as llm_globals

_RETRIEVER_MAX_TRACED_SCORES = 50

_retriever_output_attrs_var: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar('_lazyllm_retriever_output_attrs', default=None)
)
_reranker_output_attrs_var: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar('_lazyllm_reranker_output_attrs', default=None)
)
_llm_resolved_prompt_var: contextvars.ContextVar[Optional[Any]] = (
    contextvars.ContextVar('_lazyllm_llm_resolved_prompt', default=None)
)
_switch_matched_attrs_var: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar('_lazyllm_switch_matched_attrs', default=None)
)
_ifs_matched_attrs_var: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar('_lazyllm_ifs_matched_attrs', default=None)
)

_POST_PROCESS_INSTALLED = '__lazyllm_trace_post_process_installed__'
_POST_PROCESS_ORIG = '__lazyllm_trace_post_process_orig__'
_PROMPT_PROBE_INSTALLED = '__lazyllm_trace_prompt_probe_installed__'


def _mro_names(target: Any) -> set[str]:
    return {cls.__name__ for cls in type(target).__mro__}


def _is_retriever_target(target: Any) -> bool:
    names = _mro_names(target)
    return '_RetrieverBase' in names or any(n.endswith('Retriever') for n in names)


def _is_reranker_target(target: Any) -> bool:
    return 'Reranker' in _mro_names(target)


def _score_attrs(
    nodes: Any, score_attr: str, attr_key: str, *, max_scores: Optional[int] = None,
) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    if not nodes or not isinstance(nodes, list):
        return attrs
    attrs['lazyllm.output.doc_count'] = len(nodes)
    scores = []
    for node in nodes:
        score = getattr(node, score_attr, None)
        if score is None:
            continue
        try:
            scores.append(float(score))
        except (TypeError, ValueError) as exc:
            LOG.warning(f'Skipping non-numeric {score_attr} {score!r} on node: {exc}')
    if scores:
        if max_scores is not None:
            scores = scores[:max_scores]
        attrs[attr_key] = json.dumps(scores)
    return attrs


def _pop_context_attrs(var: contextvars.ContextVar) -> Dict[str, Any]:
    attrs = var.get()
    if attrs is not None:
        var.set(None)
        return attrs
    return {}


def record_retriever_output_attrs(nodes: Any) -> None:
    _retriever_output_attrs_var.set(
        _score_attrs(
            nodes,
            'similarity_score',
            'lazyllm.output.similarity_scores',
            max_scores=_RETRIEVER_MAX_TRACED_SCORES,
        )
    )


def record_reranker_output_attrs(nodes: Any) -> None:
    _reranker_output_attrs_var.set(
        _score_attrs(nodes, 'relevance_score', 'lazyllm.output.relevance_scores')
    )


def record_llm_resolved_prompt(prompt: Any) -> None:
    _llm_resolved_prompt_var.set(prompt)


def pop_llm_resolved_prompt() -> Any:
    return _pop_context_attrs(_llm_resolved_prompt_var) or None


def enter_switch_ifs_matched_scope(target: Any) -> Optional[contextvars.Token]:
    cls_name = type(target).__name__
    if cls_name == 'Switch':
        return _switch_matched_attrs_var.set({})
    if cls_name == 'IFS':
        return _ifs_matched_attrs_var.set({})
    return None


def exit_switch_ifs_matched_scope(token: Optional[contextvars.Token]):
    if token is not None:
        token.var.reset(token)


def push_switch_matched_attrs(matched: Dict[str, Any]) -> None:
    if _switch_matched_attrs_var.get() is not None:
        _switch_matched_attrs_var.set(matched)


def push_ifs_matched_attrs(matched: Dict[str, Any]) -> None:
    if _ifs_matched_attrs_var.get() is not None:
        _ifs_matched_attrs_var.set(matched)


def _matched_dict_from_frame(matched: Dict[str, Any]) -> Dict[str, Any]:
    return {f'lazyllm.matched.{k}': v for k, v in matched.items()}


def _peek_matched_attrs(var: contextvars.ContextVar) -> Dict[str, Any]:
    matched = var.get()
    if not matched:
        return {}
    return _matched_dict_from_frame(matched)


def _peek_switch_matched_attrs() -> Dict[str, Any]:
    return _peek_matched_attrs(_switch_matched_attrs_var)


def _peek_ifs_matched_attrs() -> Dict[str, Any]:
    return _peek_matched_attrs(_ifs_matched_attrs_var)


def _loop_output_attrs(target: Any) -> Dict[str, Any]:
    tid = getattr(target, 'id', None)
    if not callable(tid):
        return {}
    tr = llm_globals.get('trace')
    ai = tr.get('actual_iterations') if isinstance(tr, dict) else None
    count = ai.get(tid()) if isinstance(ai, dict) else None
    return {} if count is None else {'lazyllm.loop.actual_iterations': count}


def collect_trace_output_attrs(target: Any, output: Any) -> Dict[str, Any]:
    cls_name = type(target).__name__
    if _is_retriever_target(target):
        attrs = _pop_context_attrs(_retriever_output_attrs_var)
        return attrs or _score_attrs(
            output,
            'similarity_score',
            'lazyllm.output.similarity_scores',
            max_scores=_RETRIEVER_MAX_TRACED_SCORES,
        )
    if _is_reranker_target(target):
        attrs = _pop_context_attrs(_reranker_output_attrs_var)
        return attrs or _score_attrs(output, 'relevance_score', 'lazyllm.output.relevance_scores')
    if cls_name == 'Switch':
        return _peek_switch_matched_attrs()
    if cls_name == 'IFS':
        return _peek_ifs_matched_attrs()
    if cls_name == 'Loop':
        return _loop_output_attrs(target)
    return {}


def install_post_process_probe(obj: Any) -> None:
    if obj.__dict__.get(_POST_PROCESS_INSTALLED, False):
        return
    if not hasattr(obj, '_post_process') or not callable(obj._post_process):
        return
    if not (_is_retriever_target(obj) or _is_reranker_target(obj)):
        return

    orig = obj._post_process

    def _wrapper(self: Any, nodes: Any, *args: Any, **kwargs: Any):
        if _is_retriever_target(self):
            record_retriever_output_attrs(nodes)
        elif _is_reranker_target(self):
            record_reranker_output_attrs(nodes)
        return orig(nodes, *args, **kwargs)

    obj.__dict__[_POST_PROCESS_ORIG] = orig
    obj.__dict__[_POST_PROCESS_INSTALLED] = True
    obj._post_process = types.MethodType(_wrapper, obj)


def remove_post_process_probe(obj: Any) -> None:
    d = getattr(obj, '__dict__', None)
    if not d or not d.pop(_POST_PROCESS_INSTALLED, False):
        return
    d.pop(_POST_PROCESS_ORIG, None)
    d.pop('_post_process', None)


def install_prompt_probe(obj: Any) -> None:
    prompter = getattr(obj, '_prompt', None)
    if prompter is None:
        return
    pd = getattr(prompter, '__dict__', None)
    if pd is None or pd.get(_PROMPT_PROBE_INSTALLED, False):
        return
    orig = prompter.generate_prompt

    def _wrapper(*args, **kwargs):
        result = orig(*args, **kwargs)
        record_llm_resolved_prompt(result)
        return result

    prompter.generate_prompt = _wrapper
    pd[_PROMPT_PROBE_INSTALLED] = True


def remove_prompt_probe(obj: Any) -> None:
    prompter = getattr(obj, '_prompt', None)
    pd = getattr(prompter, '__dict__', None) if prompter is not None else None
    if not pd or not pd.pop(_PROMPT_PROBE_INSTALLED, False):
        return
    pd.pop('generate_prompt', None)
