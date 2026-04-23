import contextvars
import json
import types
from typing import Any, Dict, List, Optional

from lazyllm.common import LOG

_RETRIEVER_MAX_TRACED_SCORES = 50

_retriever_output_attrs_var: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar('_lazyllm_retriever_output_attrs', default=None)
)
_reranker_output_attrs_var: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar('_lazyllm_reranker_output_attrs', default=None)
)
_switch_matched_stack: contextvars.ContextVar[Optional[List[Dict[str, Any]]]] = (
    contextvars.ContextVar('_lazyllm_switch_matched_stack', default=None)
)
_ifs_matched_stack: contextvars.ContextVar[Optional[List[Dict[str, Any]]]] = (
    contextvars.ContextVar('_lazyllm_ifs_matched_stack', default=None)
)

_POST_PROCESS_INSTALLED = '__lazyllm_trace_post_process_installed__'
_POST_PROCESS_ORIG = '__lazyllm_trace_post_process_orig__'


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


def push_switch_matched_attrs(matched: Dict[str, Any]) -> None:
    current = _switch_matched_stack.get() or []
    _switch_matched_stack.set(current + [matched])


def push_ifs_matched_attrs(matched: Dict[str, Any]) -> None:
    current = _ifs_matched_stack.get() or []
    _ifs_matched_stack.set(current + [matched])


def _pop_switch_matched_attrs() -> Dict[str, Any]:
    stack = _switch_matched_stack.get()
    if not stack:
        return {}
    matched = stack[-1]
    _switch_matched_stack.set(stack[:-1] or None)
    return {f'lazyllm.matched.{k}': v for k, v in matched.items()}


def _pop_ifs_matched_attrs() -> Dict[str, Any]:
    stack = _ifs_matched_stack.get()
    if not stack:
        return {}
    matched = stack[-1]
    _ifs_matched_stack.set(stack[:-1] or None)
    return {f'lazyllm.matched.{k}': v for k, v in matched.items()}


def _loop_output_attrs(target: Any) -> Dict[str, Any]:
    count = getattr(target, '_trace_actual_iterations', None)
    if count is None:
        return {}
    try:
        setattr(target, '_trace_actual_iterations', None)
    except Exception:
        pass
    return {'lazyllm.loop.actual_iterations': count}


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
        return _pop_switch_matched_attrs()
    if cls_name == 'IFS':
        return _pop_ifs_matched_attrs()
    if cls_name == 'Loop':
        return _loop_output_attrs(target)
    return {}


def install_post_process_probe(obj: Any) -> None:
    if getattr(obj, _POST_PROCESS_INSTALLED, False):
        return
    if not hasattr(obj, '_post_process') or not callable(getattr(obj, '_post_process')):
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
