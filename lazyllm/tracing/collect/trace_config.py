import inspect
import json
import sys
import types
from enum import Enum
from typing import Any, Dict, FrozenSet, Optional, Set, Tuple

from lazyllm.common import LOG

from ..semantics import SemanticType


_TRACE_DEFAULT_EXCLUDE_KEYS: FrozenSet[str] = frozenset({
    'dict', 'weakref', 'slots', 'module_id', 'flow_id', 'hooks', 'identities',
    'lazyllm_after_registry_hook',
})

_TRACE_DEFAULT_EXCLUDE_SUBSTRINGS: Tuple[str, ...] = (
    'api_key', 'api_keys', 'secret', 'token', 'password', 'authorization',
    'credential', 'cookie', 'client', 'session', 'lock', 'store', 'headers',
    'sandbox', 'thread', 'executor', 'job',
)

def _private_attr_deny_type_names() -> FrozenSet[str]:
    return frozenset({
        'module', 'ModuleType', 'RLock', 'Lock', 'Semaphore', 'Event', 'Condition',
        'Thread', 'Process', 'Client', 'Session', 'HTTPSConnection', 'HTTPResponse',
    })


_TRACE_RAW_ATTR_EXCLUDE: FrozenSet[str] = frozenset({
    '__dict__', '__weakref__', '__slots__',
    '_module_id', '_flow_id', '_hooks', '_identities',
    '_otel_span', '_otel_span_cm', '_owns_lazy_trace',
    '_api_key', '__api_keys', '__headers', '_header', '_headers',
    '_client', '_session', '_sessions', '_lock', '_locks', '_store', '_pool',
    '_executor', '_thread', '_logger', '_log',
    '_docs', '_nodes', '_items', '_in_degree', '_out_degree', '_sorted_nodes',
    '_constants', '_formatter', '_prompt', '_reranker', '_embed', '_llm', '_impl',
    '_post_process_type', '_post_process_args',
})

_PARALLEL_LIKE_NAMES: FrozenSet[str] = frozenset({'Parallel', 'Diverter', 'Warp'})


def _safe_getattr(target: Any, attr: str, default: Any = None) -> Any:
    try:
        return getattr(target, attr, default)
    except Exception as exc:
        LOG.warning(f'Reading {attr!r} on {type(target).__name__} for trace metadata failed: {exc}')
        return default


def _normalize_private_trace_key(attr_name: str) -> Optional[str]:
    '''Strip leading ``_``; map name-mangled ``_Cls__x`` to ``x`` (plan step 3).'''
    if not attr_name.startswith('_'):
        return None
    if attr_name.startswith('__') and not attr_name.startswith('___'):
        return None
    if '__' in attr_name:
        idx = attr_name.rfind('__')
        if idx > 0:
            return attr_name[idx + 2:]
    return attr_name[1:]


def _target_trace_exclude(target: Any) -> Set[str]:
    keys: Set[str] = set()
    for attr in ('__trace_exclude__', '_trace_exclude'):
        raw = _safe_getattr(target, attr, None)
        if raw is None:
            continue
        if isinstance(raw, (str, bytes)):
            keys.add(str(raw))
            continue
        if isinstance(raw, dict):
            keys.update(str(k) for k in raw)
            continue
        try:
            keys.update(str(x) for x in raw)  # type: ignore[arg-type]
        except TypeError:
            LOG.warning(
                f'{type(target).__name__}.{attr} must be str/bytes/dict/iterable of keys; ignoring.'
            )
    return keys


def _is_sensitive_public_key(key: str) -> bool:
    lower = key.lower()
    return any(s in lower for s in _TRACE_DEFAULT_EXCLUDE_SUBSTRINGS)


def _should_collect_trace_value(key: str, value: Any, exclude: Set[str]) -> bool:
    if key in exclude or key in _TRACE_DEFAULT_EXCLUDE_KEYS:
        return False
    if _is_sensitive_public_key(key):
        return False
    if value is None:
        return False
    if callable(value) or inspect.isclass(value) or inspect.ismodule(value):
        return False
    if isinstance(value, types.ModuleType):
        return False
    tn = type(value).__name__
    if tn in _private_attr_deny_type_names():
        return False
    if isinstance(value, str) and value == '':
        return False
    if isinstance(value, (list, tuple)) and len(value) > 1024:
        return False
    if isinstance(value, dict) and len(value) > 256:
        return False
    try:
        json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return False
    return True


def _put_if_present(cfg: Dict[str, Any], key: str, value: Any) -> None:
    '''Omit only ``None`` and ``''``; keep ``False``, ``0``, and empty containers.'''
    if value is None or value == '':
        return
    cfg[key] = value


def _collect_private_trace_config(target: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if target is None:
        return out
    extra_excludes = _target_trace_exclude(target)
    d = _safe_getattr(target, '__dict__', None)
    if not isinstance(d, dict):
        return out
    for attr_name, value in d.items():
        if attr_name.startswith('__'):
            continue
        pub = _normalize_private_trace_key(attr_name)
        if pub is None:
            continue
        if attr_name in _TRACE_RAW_ATTR_EXCLUDE or pub in _TRACE_DEFAULT_EXCLUDE_KEYS:
            continue
        if pub in extra_excludes or attr_name in extra_excludes:
            continue
        if isinstance(value, Enum):
            value = value.value if getattr(value, 'value', None) is not None else str(value)
        if not _should_collect_trace_value(pub, value, extra_excludes):
            continue
        if isinstance(value, str) and len(value) > 8192:
            value = value[:8192] + '...<truncated>'
        out[pub] = value
    return out


def _looks_like_online_module(target: Any) -> bool:
    return any(cls.__name__ == 'LazyLLMOnlineBase' for cls in type(target).__mro__)


def _resolve_runtime_model_url(target: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    runtime: Dict[str, Any] = {}
    for key in ('model', 'model_name', 'embed_model_name', 'embed_name'):
        if key in kwargs and kwargs[key] not in (None, ''):
            runtime['model'] = kwargs[key]
            break
    for key in ('url', 'base_url', 'embed_url'):
        if key in kwargs and kwargs[key] not in (None, ''):
            runtime['base_url'] = kwargs[key]
            break
    if 'stream' in kwargs and kwargs['stream'] is not None:
        runtime['stream'] = kwargs['stream']
    return runtime


def _collect_llm_trace_config(target: Any, args: tuple, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for key, attrs in (
        ('model', ('_model_name', '_embed_model_name', 'base_model', '_base_model')),
        ('base_url', ('_base_url', '_embed_url', '_chat_url')),
        ('stream', ('_stream',)),
        ('skip_auth', ('_skip_auth',)),
        ('batch_size', ('_batch_size',)),
        ('num_worker', ('_num_worker',)),
        ('timeout', ('_timeout',)),
    ):
        for attr in attrs:
            v = _safe_getattr(target, attr, None)
            if v is not None and v != '':
                _put_if_present(cfg, key, v)
                break

    rt = _resolve_runtime_model_url(target, kwargs)
    if 'model' in rt:
        cfg['model'] = rt['model']
    if 'base_url' in rt:
        cfg['base_url'] = rt['base_url']
    if 'stream' in rt:
        cfg['stream'] = rt['stream']

    static_params = _safe_getattr(target, '_static_params', None)
    if isinstance(static_params, dict) and static_params:
        cfg['static_params'] = dict(static_params)

    cfg['class'] = type(target).__name__
    tv = _safe_getattr(target, 'type', None)
    if tv is not None and tv != '':
        cfg['type'] = tv
    series = _safe_getattr(target, 'series', None)
    if series:
        cfg['series'] = series
    return cfg


def _looks_like_retriever(target: Any) -> bool:
    return type(target).__name__ == 'Retriever'


def _collect_retriever_trace_config(target: Any) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    _put_if_present(cfg, 'group_name', _safe_getattr(target, '_group_name', None))
    _put_if_present(cfg, 'similarity', _safe_getattr(target, '_similarity', None))
    sco = _safe_getattr(target, '_similarity_cut_off', None)
    if sco is not None:
        cfg['similarity_cut_off'] = sco
    _put_if_present(cfg, 'index', _safe_getattr(target, '_index', None))
    _put_if_present(cfg, 'topk', _safe_getattr(target, '_topk', None))
    ek = _safe_getattr(target, '_embed_keys', None)
    if ek is not None:
        cfg['embed_keys'] = ek
    _put_if_present(cfg, 'target', _safe_getattr(target, '_target', None))
    _put_if_present(cfg, 'mode', _safe_getattr(target, '_mode', None))
    _put_if_present(cfg, 'output_format', _safe_getattr(target, '_output_format', None))
    _put_if_present(cfg, 'join', _safe_getattr(target, '_join', None))
    wt = _safe_getattr(target, '_weight', None)
    if wt is not None:
        cfg['weight'] = wt
    pri = _safe_getattr(target, '_priority', None)
    if pri is not None:
        cfg['priority'] = str(pri)
    return cfg


def _looks_like_reranker(target: Any) -> bool:
    return type(target).__name__ in ('Reranker', 'ModuleReranker')


def _collect_reranker_trace_config(target: Any) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    _put_if_present(cfg, 'name', _safe_getattr(target, '_name', None))
    _put_if_present(cfg, 'output_format', _safe_getattr(target, '_output_format', None))
    _put_if_present(cfg, 'join', _safe_getattr(target, '_join', None))
    extra = _safe_getattr(target, '_kwargs', None)
    if isinstance(extra, dict):
        for k, v in extra.items():
            _put_if_present(cfg, k, v)
    mn = _safe_getattr(target, '_model_name', None)
    _put_if_present(cfg, 'model', mn)
    return cfg


def _is_flow_target(target: Any) -> bool:
    return getattr(target, '_flow_id', None) is not None and hasattr(target, '_items')


def _flow_node_names(flow: Any) -> Dict[str, Any]:
    names = []
    items = _safe_getattr(flow, '_items', None) or []
    item_names = _safe_getattr(flow, '_item_names', None) or []
    pad = [None] * max(0, len(items) - len(item_names))
    zipped = zip((item_names + pad)[: len(items)], items)
    for alias, it in zipped:
        actual = getattr(it, '__name__', None) or type(it).__name__
        if alias and alias != actual:
            names.append(f'{alias} -> {actual}')
        else:
            names.append(actual)
    return {'node_count': len(items), 'node_names': names}


def _collect_flow_trace_config(target: Any) -> Dict[str, Any]:
    if not _is_flow_target(target):
        return {}
    cls_name = type(target).__name__
    nodes_map = _safe_getattr(target, '_nodes', None)
    if cls_name == 'Graph' and isinstance(nodes_map, dict) and nodes_map:
        start = getattr(type(target), 'start_node_name', '__start__')
        end = getattr(type(target), 'end_node_name', '__end__')
        user_nodes = [n for n in nodes_map if n not in (start, end)]
        edge_count = sum(len(getattr(n, 'outputs', []) or []) for n in nodes_map.values())
        return {
            'node_count': len(user_nodes),
            'node_names': user_nodes,
            'edge_count': edge_count,
        }

    cfg = dict(_flow_node_names(target))

    if cls_name in _PARALLEL_LIKE_NAMES:
        _put_if_present(cfg, 'scatter', _safe_getattr(target, '_scatter', None))
        _put_if_present(cfg, 'concurrent', _safe_getattr(target, '_concurrent', None))
        ppt = _safe_getattr(target, '_post_process_type', None)
        if ppt is not None and hasattr(ppt, 'name'):
            cfg['aggregation'] = ppt.name.lower()

    if hasattr(target, 'conds'):
        conds = getattr(target, 'conds', None)
        if conds is not None:
            cfg['conditions'] = [str(c) for c in conds]
        j = _safe_getattr(target, '_judge_on_full_input', None)
        if j is not None:
            cfg['judge_on_full_input'] = j

    if cls_name == 'Loop':
        lc = _safe_getattr(target, '_loop_count', None)
        if lc is not None:
            cfg['max_loop_count'] = lc if lc != sys.maxsize else 'unlimited'
        cfg['has_stop_condition'] = _safe_getattr(target, '_stop_condition', None) is not None
        j = _safe_getattr(target, '_judge_on_full_input', None)
        if j is not None:
            cfg['judge_on_full_input'] = j

    return cfg


def _merge_trace_entity_model_and_url_fields(merged: Dict[str, Any]) -> None:
    model = None
    for k in ('model', 'model_name', 'embed_model_name', 'base_model'):
        v = merged.get(k)
        if v is not None and v != '':
            model = v
            break
    if model is not None:
        merged['model'] = model
    for k in ('model_name', 'embed_model_name', 'base_model'):
        merged.pop(k, None)

    base_url = None
    for k in ('base_url', 'embed_url'):
        v = merged.get(k)
        if v is not None and v != '':
            base_url = v
            break
    if base_url is not None:
        merged['base_url'] = base_url
    merged.pop('embed_url', None)


def _merge_trace_entity_scalar_canonicals(merged: Dict[str, Any]) -> None:
    for canonical, sources in (
        ('stream', ('stream',)),
        ('skip_auth', ('skip_auth',)),
        ('batch_size', ('batch_size',)),
        ('num_worker', ('num_worker',)),
        ('timeout', ('timeout',)),
    ):
        for s in sources:
            if s in merged:
                v = merged.get(s)
                if v is not None and v != '':
                    merged[canonical] = v
                elif v is False or v == 0:
                    merged[canonical] = v
                break


def normalize_trace_entity_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(cfg)
    _merge_trace_entity_model_and_url_fields(merged)
    _merge_trace_entity_scalar_canonicals(merged)
    return merged


def enrich_trace_entity_identity(target: Any, cfg: Dict[str, Any]) -> None:
    if not cfg.get('class'):
        cfg['class'] = type(target).__name__
    if cfg.get('type') in (None, ''):
        tv = _safe_getattr(target, 'type', None)
        if tv is not None and tv != '':
            cfg['type'] = tv
    if cfg.get('series') in (None, ''):
        series = _safe_getattr(target, 'series', None)
        if series:
            cfg['series'] = series
    sp = _safe_getattr(target, '_static_params', None)
    if isinstance(sp, dict) and sp and not cfg.get('static_params'):
        cfg['static_params'] = dict(sp)


def collect_trace_config(
    target: Any,
    span_kind: str,
    args: tuple,
    kwargs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    kwargs = kwargs or {}
    cfg = _collect_private_trace_config(target)

    if _looks_like_online_module(target):
        for k, v in _collect_llm_trace_config(target, args, kwargs).items():
            cfg[k] = v
    elif _looks_like_retriever(target):
        cfg.update(_collect_retriever_trace_config(target))
    elif _looks_like_reranker(target):
        cfg.update(_collect_reranker_trace_config(target))
    elif span_kind == 'flow' or _is_flow_target(target):
        cfg.update(_collect_flow_trace_config(target))

    cfg = normalize_trace_entity_config(cfg)
    enrich_trace_entity_identity(target, cfg)
    return cfg


def _semantic_from_llmtype_enum(value: Any) -> Optional[str]:
    try:
        from lazyllm.components.utils.downloader.model_downloader import LLMType
    except Exception:
        return None
    try:
        if isinstance(value, LLMType):
            if value in (LLMType.EMBED, LLMType.MULTIMODAL_EMBED, LLMType.CROSS_MODAL_EMBED):
                return SemanticType.EMBEDDING
            if value == LLMType.RERANK:
                return SemanticType.RERANK
            return SemanticType.LLM
    except Exception:
        pass
    return None


def _semantic_from_type_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, Enum):
        value = value.value if getattr(value, 'value', None) is not None else str(value)
    s = str(value).upper()
    if 'EMBED' in s or s in ('EMBEDDING',):
        return SemanticType.EMBEDDING
    if 'RERANK' in s:
        return SemanticType.RERANK
    sem = _semantic_from_llmtype_enum(value)
    if sem:
        return sem
    if isinstance(value, str) and value.lower() in ('llm', 'chat'):
        return SemanticType.LLM
    return None


def _semantic_from_class_name(name: str, module: str) -> Optional[str]:
    if name in ('Retriever',):
        return SemanticType.RETRIEVER
    if name in ('Reranker', 'ModuleReranker'):
        return SemanticType.RERANK
    if 'Embed' in name and 'Multi' not in name and name not in ('OnlineMultiModalModule',):
        return SemanticType.EMBEDDING
    if name.endswith('Chat') or name == 'TrainableModule' or name == 'LLMBase':
        return SemanticType.LLM
    if name in ('LazyLLMAgentBase',):
        return SemanticType.AGENT
    if name in ('ToolManager',):
        return SemanticType.TOOL
    if 'agent' in module.lower() and 'Agent' in name:
        return SemanticType.AGENT
    if 'tools' in module.lower() and 'Tool' in name:
        return SemanticType.TOOL
    return None


def resolve_semantic_type_for_target(target: Any, span_kind: str) -> Optional[str]:
    if span_kind == 'flow':
        return SemanticType.WORKFLOW_CONTROL
    type_val = _safe_getattr(target, 'type', None)
    if type_val is None:
        type_val = _safe_getattr(target, '_type', None)
    sem = _semantic_from_type_value(type_val)
    if sem:
        return sem
    mod = getattr(type(target), '__module__', '') or ''
    sem = _semantic_from_class_name(type(target).__name__, mod)
    if sem:
        return sem
    return None
