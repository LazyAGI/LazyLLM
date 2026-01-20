import importlib.util

import re
import requests
from functools import lru_cache
from typing import Callable, List, Dict, Optional, Union, Any

import lazyllm
from lazyllm.thirdparty import spacy
from lazyllm import ModuleBase, LOG
from .doc_node import DocNode, MetadataMode
from .retriever import _PostProcess


class Reranker(ModuleBase, _PostProcess):
    registered_reranker = dict()

    def __new__(cls, name: str = 'ModuleReranker', *args, **kwargs):
        assert name in cls.registered_reranker, f'Reranker: {name} is not registered, please register first.'
        item = cls.registered_reranker[name]
        if isinstance(item, type) and issubclass(item, Reranker):
            return super(Reranker, cls).__new__(item)
        else:
            return super(Reranker, cls).__new__(cls)

    def __init__(self, name: str = 'ModuleReranker', target: Optional[str] = None,
                 output_format: Optional[str] = None, join: Union[bool, str] = False, **kwargs) -> None:
        super().__init__()
        self._name = name
        self._kwargs = kwargs
        lazyllm.deprecated(bool(target), '`target` parameter of reranker')
        _PostProcess.__init__(self, output_format, join)

    def forward(self, nodes: List[DocNode], query: str = '') -> List[DocNode]:
        results = self.registered_reranker[self._name](nodes, query=query, **self._kwargs)
        LOG.debug(f'Rerank use `{self._name}` and get nodes: {results}')
        return self._post_process(results)

    @classmethod
    def register_reranker(
        cls: 'Reranker', func: Optional[Callable] = None, batch: bool = False
    ):
        def decorator(f):
            if isinstance(f, type):
                cls.registered_reranker[f.__name__] = f
                return f
            else:
                def wrapper(nodes, **kwargs):
                    if batch:
                        return f(nodes, **kwargs)
                    else:
                        results = [f(node, **kwargs) for node in nodes]
                        return [result for result in results if result]

                cls.registered_reranker[f.__name__] = wrapper
                return wrapper

        return decorator(func) if func else decorator


@lru_cache(maxsize=None)
def get_nlp_and_matchers(language):
    nlp = spacy.blank(language)

    spec = importlib.util.find_spec('spacy.matcher')
    if spec is None:
        raise ImportError(
            'Please install spacy to use spacy module. '
            'You can install it with `pip install spacy==3.7.5`'
        )
    matcher_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matcher_module)

    required_matcher = matcher_module.PhraseMatcher(nlp.vocab)
    exclude_matcher = matcher_module.PhraseMatcher(nlp.vocab)
    return nlp, required_matcher, exclude_matcher


@Reranker.register_reranker
def KeywordFilter(node: DocNode, required_keys: Optional[List[str]] = None, exclude_keys: Optional[List[str]] = None,
                  language: str = 'en', **kwargs) -> Optional[DocNode]:
    assert required_keys or exclude_keys, 'One of required_keys or exclude_keys should be provided'
    nlp, required_matcher, exclude_matcher = get_nlp_and_matchers(language)
    if required_keys:
        required_matcher.add('RequiredKeywords', list(nlp.pipe(required_keys)))
    if exclude_keys:
        exclude_matcher.add('ExcludeKeywords', list(nlp.pipe(exclude_keys)))

    doc = nlp(node.get_text())
    if required_keys and not required_matcher(doc):
        return None
    if exclude_keys and exclude_matcher(doc):
        return None
    return node


@Reranker.register_reranker()
class ModuleReranker(Reranker):

    def __init__(self, name: str = 'ModuleReranker', model: Union[Callable, str] = None, target: Optional[str] = None,
                 output_format: Optional[str] = None, join: Union[bool, str] = False, **kwargs) -> None:
        super().__init__(name, target, output_format, join, **kwargs)
        assert model is not None, 'Reranker model must be specified as a model name or a callable.'
        if isinstance(model, str):
            self._reranker = lazyllm.TrainableModule(model)
        else:
            self._reranker = model

    def forward(self, nodes: List[DocNode], query: str = '') -> List[DocNode]:
        if not nodes:
            return self._post_process([])

        docs = [node.get_text(metadata_mode=MetadataMode.EMBED) for node in nodes]
        top_n = self._kwargs['topk'] if 'topk' in self._kwargs else len(docs)
        sorted_indices = self._reranker(query, documents=docs, top_n=top_n)
        results = []
        for index, relevance_score in sorted_indices:
            results.append(nodes[index].with_score(relevance_score))
        LOG.debug(f'Rerank use `{self._name}` and get nodes: {results}')
        return self._post_process(results)


@Reranker.register_reranker()
class UrlReranker(Reranker):
    """
    通用 HTTP 重排序器。

    通过将 query 与一批候选文本打包为 JSON 请求发送到远端 URL，
    解析返回的分数后对节点进行重排。

    远端服务期望的响应格式（默认）：
        List[{"index": int, "score": float}]
    其中 "index" 为该批次内文档的局部索引（从 0 开始），"score" 为相关性分数。
    """

    def __init__(
        self,
        name: str = "UrlReranker",
        url: Optional[str] = None,
        api_key: str = "api_key",
        batch_size: int = 64,
        truncate_text: bool = True,
        output_format: Optional[str] = None,
        join: Union[bool, str] = False,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            name: 重排序器名称。
            url: 远端重排序服务地址（必填）。
            api_key: 认证密钥（将置于 HTTP Bearer 头）。
            batch_size: 批大小（原 rerank_batch_size）。
            truncate_text: 是否在远端对文本进行截断。
            output_format, join, **kwargs: 继承自 Reranker 的可选参数。
            request_timeout: 请求超时时间，缺省为 DEFAULT_TIMEOUT。
        """
        super().__init__(name=name, output_format=output_format, join=join, **kwargs)
        if not url:
            raise ValueError("`url` 不能为空，请传入远端重排序服务地址。")

        self._url = url
        self._api_key = api_key
        self._batch_size = max(1, int(batch_size))
        self._truncate_text = bool(truncate_text)
        self._timeout = timeout

        self._headers: Dict[str, str] = self._build_headers()
        self._session = requests.Session()

    def _build_headers(self) -> Dict[str, str]:
        """构建 HTTP 头。"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    def _extract_top_k(self, total: int, **kwargs: Any) -> int:
        """从 kwargs 中解析 top_k/topk，默认取全部。"""
        top_k = kwargs.get("top_k", kwargs.get("topk", total))
        try:
            top_k = int(top_k)
        except Exception:
            top_k = total
        return max(0, min(top_k, total))

    def _get_format_content(self, nodes: List[DocNode], **kwargs: Any) -> List[str]:
        """
        生成待重排的文本列表。

        若提供 template（如: "标题:{title}\n正文:{text}"），将按节点 metadata 与 text 替换。
        支持的占位符来源：
          - {text}: 节点正文
          - {<metadata_key>}: 节点 metadata 中的键
        若占位符缺失对应值，则回退为空串。
        """
        template: Optional[str] = dict(kwargs).pop("template", None)
        if not template:
            return [n.get_text(metadata_mode=MetadataMode.EMBED) for n in nodes]

        placeholders = re.findall(r"{(\w+)}", template)

        formatted: List[str] = []
        for node in nodes:
            values = {
                key: node.text if key == 'text' else node.metadata.get(key, "") for key in placeholders
            }
            try:
                formatted.append(template.format(**values))
            except Exception as exc:
                LOG.warning("Template formatting failed; fallback to raw text: %s", exc)
                formatted.append(node.get_text(metadata_mode=MetadataMode.EMBED))
        return formatted

    def _encapsulated_data(self, query: str, texts: List[str], **kwargs: Any) -> Dict[str, Any]:
        """
        封装请求体。子类可重写。
        默认字段：
            {
                "query": "<用户查询>",
                "texts": ["doc1", "doc2", ...],
                "truncate": bool
            }
        """
        payload: Dict[str, Any] = {
            "query": query,
            "texts": list(texts),
            "truncate": self._truncate_text,
        }
        if kwargs:
            for k, v in kwargs.items():
                if k not in ("query", "texts", "truncate"):
                    payload[k] = v
        return payload

    def _parse_response(self, response: Any) -> List[float]:
        """
        解析远端返回为分数列表。子类可重写。

        期望输入：List[{"index": int, "score": float}]
        输出顺序：按 "index" 排序返回分数列表。
        """
        if not isinstance(response, list):
            LOG.warning("Response is not a list; attempting lenient parsing: %r", response)
            return []

        try:
            sorted_data = sorted(response, key=lambda x: x["index"])
            return [float(item["score"]) for item in sorted_data]
        except Exception as exc:
            LOG.error("Failed to parse response: %s; response=%r", exc, response)
            return []

    def forward(self, nodes: List[DocNode], query: str, **kwargs: Any) -> List[DocNode]:
        """
        对候选节点进行重排并返回 Top-K（若未指定则返回全部）。
        """
        if not nodes:
            return []

        texts = self._get_format_content(nodes, **kwargs)
        top_k = self._extract_top_k(len(texts), **kwargs)

        all_scores: List[float] = []
        for start in range(0, len(texts), self._batch_size):
            batch_texts = texts[start : start + self._batch_size]
            payload = self._encapsulated_data(query, batch_texts, **kwargs)

            try:
                resp = self._session.post(
                    self._url, json=payload, headers=self._headers, timeout=self._timeout
                )
                resp.raise_for_status()
                scores = self._parse_response(resp.json())
            except requests.RequestException as exc:
                LOG.error("HTTP request for reranking failed (this batch will be scored as 0): %s", exc)
                scores = []

            if len(scores) != len(batch_texts):
                LOG.warning(
                    "Returned scores count mismatches inputs: got=%d, expected=%d; padding with zeros.",
                    len(scores), len(batch_texts)
                )
                if len(scores) < len(batch_texts):
                    scores += [0.0] * (len(batch_texts) - len(scores))
                else:
                    scores = scores[: len(batch_texts)]

            all_scores.extend(scores)

        scored_nodes: List[DocNode] = [
            nodes[i].with_score(all_scores[i]) for i in range(len(nodes))
        ]

        scored_nodes.sort(key=lambda n: n.relevance_score, reverse=True)
        results = scored_nodes[:top_k] if top_k > 0 else scored_nodes
        LOG.debug(f"Rerank use `{self._name}` and get nodes: {results}")
        return self._post_process(results)


@Reranker.register_reranker()
class Qwen3Reranker(UrlReranker):
    """
    基于 Qwen3 样式 Prompt/响应协议的重排序器。
    请求体：
        {
            "query": "<拼装后的系统指令+用户查询>",
            "documents": ["<每个 doc 的拼装文本>", ...],
            ...          # 其他可选字段
        }
    响应体（期望）：
        {
            "results": [
                {"index": int, "relevance_score": float},
                ...
            ]
        }
    """

    _PROMPT_PREFIX = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        'Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n<|im_start|>user\n"
    )
    _PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    _QUERY_TEMPLATE = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
    _DOCUMENT_TEMPLATE = "<Document>: {doc}{suffix}"

    _LOCAL_TRUNCATE_MAX_CHARS = 16384
    _DEFAULT_TASK_DESCRIPTION = "Given a web search query, retrieve relevant passages that answer the query"

    def __init__(
        self,
        name: str = "Qwen3Reranker",
        url: Optional[str] = None,
        api_key: str = "api_key",
        batch_size: int = 64,
        truncate_text: bool = True,
        output_format: Optional[str] = None,
        join: Union[bool, str] = False,
        task_description: Optional[str] = None,
        request_timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            task_description: 任务描述，会被拼入 system/user 区块。
        """
        super().__init__(
            name=name,
            url=url,
            api_key=api_key,
            batch_size=batch_size,
            truncate_text=truncate_text,
            output_format=output_format,
            join=join,
            request_timeout=request_timeout,
            **kwargs,
        )
        self._task_description = task_description or self._DEFAULT_TASK_DESCRIPTION

    def _build_instruct(self, task_description: str, query: str) -> str:
        """拼装包含系统前缀与用户区块的 query 字符串。"""
        return self._QUERY_TEMPLATE.format(
            prefix=self._PROMPT_PREFIX, instruction=task_description, query=query
        )

    def _build_documents(self, texts: List[str]) -> List[str]:
        """
        将每条文本套入文档模板；若开启 truncate，则在这里进行**本地字符级截断**。
        - 截断阈值：_LOCAL_TRUNCATE_MAX_CHARS
        - 仅当 self._truncate_text 为 True 时生效
        """
        docs: List[str] = []

        def _truncate_if_needed(s: str) -> str:
            if not self._truncate_text:
                return s
            if len(s) <= self._LOCAL_TRUNCATE_MAX_CHARS:
                return s
            return s[: self._LOCAL_TRUNCATE_MAX_CHARS]

        for t in texts:
            t_norm = _truncate_if_needed(t or "")
            docs.append(self._DOCUMENT_TEMPLATE.format(doc=t_norm, suffix=self._PROMPT_SUFFIX))
        return docs

    def _encapsulated_data(self, query: str, texts: List[str], **kwargs: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "query": self._build_instruct(self._task_description, query),
            "documents": self._build_documents(texts),
        }
        if kwargs:
            for k, v in kwargs.items():
                if k not in ("query", "documents"):
                    payload[k] = v
        return payload

    def _parse_response(self, response: Any) -> List[float]:
        """
        期望输入：
            {"results": [{"index": int, "relevance_score": float}, ...]}
        """
        if not isinstance(response, dict) or "results" not in response:
            LOG.warning("response missing 'results' field: %r", response)
            return []

        results = response.get("results", [])
        try:
            results = sorted(results, key=lambda x: x["index"])
            return [float(item["relevance_score"]) for item in results]
        except Exception as exc:
            LOG.error("Failed to parse response: %s; response=%r", exc, response)
            return []


# User-defined similarity decorator
def register_reranker(func=None, batch=False):
    return Reranker.register_reranker(func, batch)
