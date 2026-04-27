import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urlparse

import requests

from lazyllm.common import LOG
from lazyllm.thirdparty import opentelemetry
from lazyllm.tracing.consume.datamodel.raw import RawSpanRecord, RawTracePayload, RawTraceRecord
from lazyllm.tracing.consume.errors import ConsumeBackendError, TraceNotFound
from lazyllm.tracing.semantics import SemanticType, is_valid_span_id

from ..base import ConsumeBackend, TracingBackend
from .config import (
    build_basic_auth_header,
    build_otlp_traces_endpoint,
    read_langfuse_connection,
)
from .semantics import SEMANTIC_TO_LANGFUSE_OBSERVATION_TYPE


class LangfuseBackend(TracingBackend):
    name = 'langfuse'

    _warned_semantic_types: set = set()

    def _copy_usage_attrs(self, attrs: Dict[str, Any], otel_attrs: Dict[str, Any]) -> None:
        for key in (
            'gen_ai.usage.input_tokens',
            'gen_ai.usage.output_tokens',
            'gen_ai.usage.total_tokens',
        ):
            if key in otel_attrs:
                attrs[key] = otel_attrs[key]

    def _copy_trace_attrs(self, attrs: Dict[str, Any], otel_attrs: Dict[str, Any]) -> None:
        trace_name = otel_attrs.get('lazyllm.trace.name')
        if trace_name:
            attrs['langfuse.trace.name'] = trace_name
        if 'session.id' in otel_attrs:
            attrs['session.id'] = otel_attrs['session.id']
        if 'user.id' in otel_attrs:
            attrs['user.id'] = otel_attrs['user.id']

        tags = otel_attrs.get('lazyllm.trace.tags')
        if tags:
            try:
                attrs['langfuse.trace.tags'] = json.dumps(tags, ensure_ascii=False)
            except TypeError as exc:
                LOG.warning(f'Failed to JSON-serialize lazyllm.trace.tags ({type(tags).__name__}): {exc}; '
                            f'falling back to string coercion.')
                attrs['langfuse.trace.tags'] = json.dumps(
                    [str(t) for t in tags] if isinstance(tags, (list, tuple)) else str(tags),
                    ensure_ascii=False,
                )

        if 'lazyllm.io.input' in otel_attrs:
            attrs['langfuse.trace.input'] = otel_attrs['lazyllm.io.input']
        if 'lazyllm.io.output' in otel_attrs:
            attrs['langfuse.trace.output'] = otel_attrs['lazyllm.io.output']

    def _copy_trace_metadata(self, attrs: Dict[str, Any], otel_attrs: Dict[str, Any]) -> None:
        prefix = 'lazyllm.trace.metadata.'
        for key, value in otel_attrs.items():
            if key.startswith(prefix):
                attrs[f'langfuse.trace.metadata.{key[len(prefix):]}'] = value

    def build_exporter(self):
        cfg = read_langfuse_connection()
        missing = [name for name, value in cfg.items() if not value]
        if missing:
            raise RuntimeError('Missing Langfuse tracing config: ' + ', '.join(missing))

        auth = build_basic_auth_header(cfg['public_key'], cfg['secret_key'])
        endpoint = build_otlp_traces_endpoint(cfg['host'])

        OTLPSpanExporter = opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter

        return OTLPSpanExporter(
            endpoint=endpoint,
            headers={'Authorization': auth},
        )

    def map_attributes(self, otel_attrs: Dict[str, Any]) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {}
        is_root_span = otel_attrs.get('lazyllm.span.is_root') is True

        semantic_type = otel_attrs.get('lazyllm.semantic_type')
        if semantic_type:
            langfuse_type = SEMANTIC_TO_LANGFUSE_OBSERVATION_TYPE.get(semantic_type)
            if langfuse_type is None:
                if semantic_type not in self._warned_semantic_types:
                    LOG.warning(
                        f'LangfuseBackend: unmapped semantic_type {semantic_type!r}; '
                        f'falling back to "span".'
                    )
                    self._warned_semantic_types.add(semantic_type)
                langfuse_type = 'span'
            attrs['langfuse.observation.type'] = langfuse_type

        model = otel_attrs.get(
            'gen_ai.request.model',
            otel_attrs.get('lazyllm.entity.config.model'),
        )
        if semantic_type == SemanticType.LLM and model:
            attrs['gen_ai.request.model'] = str(model)

        if 'lazyllm.io.input' in otel_attrs:
            attrs['langfuse.observation.input'] = otel_attrs['lazyllm.io.input']
        if 'lazyllm.io.output' in otel_attrs:
            attrs['langfuse.observation.output'] = otel_attrs['lazyllm.io.output']

        if 'lazyllm.error.message' in otel_attrs:
            attrs['langfuse.observation.status_message'] = str(otel_attrs['lazyllm.error.message'])

        self._copy_usage_attrs(attrs, otel_attrs)

        if is_root_span:
            self._copy_trace_attrs(attrs, otel_attrs)
            self._copy_trace_metadata(attrs, otel_attrs)

        return attrs


class LangfuseConsumeBackend(ConsumeBackend):
    name = 'langfuse'
    _CONNECT_TIMEOUT_S = 5.0
    _DEFAULT_READ_TIMEOUT_S = 30.0
    _MAX_HTTP_ATTEMPTS = 3
    _OBSERVATIONS_PAGE_LIMIT = 1000
    _RETRY_REQUEST = object()
    _RESPONSE_UNHANDLED = object()
    _PROMOTED_TRACE_FIELDS = frozenset({
        'id',
        'name',
        'timestamp',
        'tags',
        'userId',
        'sessionId',
        'input',
        'output',
        'metadata',
        'observations',
    })

    def _iso_to_epoch(self, value: Optional[str]) -> Optional[float]:
        if not value or not isinstance(value, str):
            return None
        text = value.replace('Z', '+00:00')
        try:
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            return None

    def _raw_span_from_obs(self, trace_id: str, obs: Dict[str, Any]) -> RawSpanRecord:
        if 'id' not in obs:
            raise ValueError('observation missing id')
        span_id = str(obs['id'])
        parent_raw = obs.get('parentObservationId')
        parent_span_id = str(parent_raw) if parent_raw is not None else None
        if parent_span_id is not None and not is_valid_span_id(parent_span_id):
            parent_span_id = None

        metadata = obs.get('metadata')
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            metadata = {'_': metadata}

        metadata_attrs = metadata.get('attributes')
        if isinstance(metadata_attrs, dict):
            attributes: Dict[str, Any] = dict(metadata_attrs)
        else:
            attributes = {
                key: value for key, value in metadata.items()
                if isinstance(key, str) and (
                    key.startswith('lazyllm.')
                    or key.startswith('gen_ai.')
                    or key.startswith('langfuse.')
                    or key in ('session.id', 'user.id')
                )
            }
        if obs.get('type') is not None:
            attributes.setdefault('langfuse.observation.type', obs['type'])
        if obs.get('model') is not None:
            attributes.setdefault('gen_ai.request.model', obs['model'])

        name = obs.get('name') or ''
        st = self._iso_to_epoch(obs.get('startTime'))
        if st is None:
            raise ValueError('observation missing startTime')

        return RawSpanRecord(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=name,
            start_time=st,
            end_time=self._iso_to_epoch(obs.get('endTime')),
            status='error' if obs.get('level') == 'ERROR' else 'ok',
            attributes=attributes,
            input=obs.get('input'),
            output=obs.get('output'),
            metadata=dict(metadata),
            error_message=obs.get('statusMessage'),
            raw=dict(obs),
        )

    def _require_connection(self) -> Tuple[str, Dict[str, str]]:
        cfg = read_langfuse_connection()
        missing = [k for k, v in cfg.items() if not v]
        if missing:
            raise ConsumeBackendError(
                'Missing Langfuse connection config (host / public_key / secret_key); '
                f'missing: {", ".join(missing)}'
            )
        host = str(cfg['host']).rstrip('/')
        auth = build_basic_auth_header(str(cfg['public_key']), str(cfg['secret_key']))
        headers = {'Authorization': auth, 'Accept': 'application/json'}
        return host, headers

    def _read_timeout_seconds(self, timeout_seconds: Optional[float]) -> float:
        if timeout_seconds is None:
            return self._DEFAULT_READ_TIMEOUT_S
        try:
            return max(1.0, float(timeout_seconds))
        except (TypeError, ValueError):
            return self._DEFAULT_READ_TIMEOUT_S

    def _handle_request_exception(
        self,
        exc: requests.RequestException,
        *,
        url: str,
        trace_id: Optional[str],
        attempt: int,
    ) -> None:
        if attempt + 1 < self._MAX_HTTP_ATTEMPTS:
            time.sleep(2**attempt)
            return
        netloc = urlparse(url).netloc or url
        msg = f'Langfuse HTTP request failed for host {netloc!r}'
        if trace_id is not None:
            msg += f', trace_id={trace_id!r}'
        raise ConsumeBackendError(msg) from exc

    def _handle_retryable_response(
        self,
        resp: requests.Response,
        url: str,
        attempt: int,
    ) -> object:
        final_attempt = attempt + 1 >= self._MAX_HTTP_ATTEMPTS
        if resp.status_code == 429:
            if final_attempt:
                raise ConsumeBackendError(
                    f'Langfuse rate limited (HTTP 429) for url ending {url[-48:]!r}'
                )
            raw = resp.headers.get('Retry-After')
            try:
                wait_s = float(raw) if raw is not None else 2**attempt
            except (TypeError, ValueError):
                wait_s = 2**attempt
            time.sleep(max(0.0, wait_s))
            return self._RETRY_REQUEST
        if 500 <= resp.status_code < 600:
            if final_attempt:
                raise ConsumeBackendError(
                    f'Langfuse server error HTTP {resp.status_code} for url ending {url[-48:]!r}'
                )
            time.sleep(2**attempt)
            return self._RETRY_REQUEST
        return self._RESPONSE_UNHANDLED

    def _handle_response_status(
        self,
        resp: requests.Response,
        url: str,
        *,
        trace_id: Optional[str],
        trace_not_found_raises: bool,
        observations_404_empty: bool,
        attempt: int,
    ) -> Any:
        if resp.status_code == 404:
            if trace_not_found_raises:
                raise TraceNotFound(trace_id or '')
            if observations_404_empty:
                return {
                    'data': [],
                    'meta': {
                        'page': 1,
                        'limit': self._OBSERVATIONS_PAGE_LIMIT,
                        'totalItems': 0,
                        'totalPages': 1,
                    },
                }
        if resp.status_code in (401, 403):
            raise ConsumeBackendError('authentication failed')
        retry = self._handle_retryable_response(resp, url, attempt)
        if retry is not self._RESPONSE_UNHANDLED:
            return retry
        if not resp.ok:
            raise ConsumeBackendError(f'Langfuse HTTP {resp.status_code} for url ending {url[-48:]!r}')
        return self._RESPONSE_UNHANDLED

    def _response_json(self, resp: requests.Response) -> Any:
        try:
            if not resp.content:
                return None
            return resp.json()
        except ValueError as exc:
            raise ConsumeBackendError('invalid JSON in Langfuse response') from exc

    def _request_json(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        *,
        trace_id: Optional[str] = None,
        trace_not_found_raises: bool = False,
        observations_404_empty: bool = False,
        timeout_seconds: Optional[float] = None,
    ) -> Any:
        timeouts = (self._CONNECT_TIMEOUT_S, self._read_timeout_seconds(timeout_seconds))
        for attempt in range(self._MAX_HTTP_ATTEMPTS):
            try:
                resp = requests.request(method, url, headers=headers, timeout=timeouts)
            except requests.RequestException as exc:
                self._handle_request_exception(exc, url=url, trace_id=trace_id, attempt=attempt)
                continue

            status_result = self._handle_response_status(
                resp,
                url,
                trace_id=trace_id,
                trace_not_found_raises=trace_not_found_raises,
                observations_404_empty=observations_404_empty,
                attempt=attempt,
            )
            if status_result is self._RETRY_REQUEST:
                continue
            if status_result is not self._RESPONSE_UNHANDLED:
                return status_result
            return self._response_json(resp)

        raise ConsumeBackendError('Langfuse HTTP request exhausted retries')

    def _fetch_trace_body(
        self,
        trace_id: str,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        host, headers = self._require_connection()
        url = f'{host}/api/public/traces/{trace_id}'
        body = self._request_json(
            'GET',
            url,
            headers,
            trace_id=trace_id,
            trace_not_found_raises=True,
            timeout_seconds=timeout_seconds,
        )
        if not isinstance(body, dict):
            raise ConsumeBackendError('Langfuse trace response is not a JSON object')
        return body

    def _raw_trace_from_body(self, trace_id: str, body: Dict[str, Any]) -> RawTraceRecord:
        metadata = body.get('metadata')
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            metadata = {'_': metadata}
        tags = body.get('tags')
        if not isinstance(tags, list):
            tags = []
        tid = body.get('id') or trace_id
        trace_raw = {
            key: value for key, value in body.items()
            if key not in self._PROMOTED_TRACE_FIELDS
        }
        try:
            return RawTraceRecord(
                trace_id=str(tid),
                name=body.get('name'),
                session_id=body.get('sessionId'),
                user_id=body.get('userId'),
                tags=[str(t) for t in tags],
                metadata=dict(metadata),
                input=body.get('input'),
                output=body.get('output'),
                start_time=self._iso_to_epoch(body.get('timestamp')),
                end_time=None,
                status=None,
                raw=trace_raw,
            )
        except ValueError as exc:
            raise ConsumeBackendError(f'invalid trace payload from Langfuse: {exc}') from exc

    def _raw_spans_from_observations(
        self,
        trace_id: str,
        observations: List[Any],
    ) -> List[RawSpanRecord]:
        out: List[RawSpanRecord] = []
        for row in observations:
            if not isinstance(row, dict):
                continue
            try:
                out.append(self._raw_span_from_obs(trace_id, row))
            except ValueError as exc:
                raise ConsumeBackendError(f'invalid observation payload: {exc}') from exc
        return out

    def fetch_trace_payload(
        self,
        trace_id: str,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> RawTracePayload:
        body = self._fetch_trace_body(trace_id, timeout_seconds=timeout_seconds)
        trace = self._raw_trace_from_body(trace_id, body)

        if 'observations' not in body:
            return RawTracePayload(
                trace=trace,
                spans=self.fetch_spans(trace.trace_id, timeout_seconds=timeout_seconds),
            )

        observations = body.get('observations')
        if not isinstance(observations, list):
            raise ConsumeBackendError('Langfuse trace observations field is not a JSON array')
        return RawTracePayload(
            trace=trace,
            spans=self._raw_spans_from_observations(trace.trace_id, observations),
        )

    def fetch_trace(
        self,
        trace_id: str,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> RawTraceRecord:
        return self._raw_trace_from_body(
            trace_id,
            self._fetch_trace_body(trace_id, timeout_seconds=timeout_seconds),
        )

    def fetch_spans(
        self,
        trace_id: str,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> List[RawSpanRecord]:
        host, headers = self._require_connection()

        out: List[RawSpanRecord] = []
        page = 1
        while True:
            q = urlencode(
                {
                    'traceId': trace_id,
                    'limit': self._OBSERVATIONS_PAGE_LIMIT,
                    'page': page,
                }
            )
            url = f'{host}/api/public/observations?{q}'
            body = self._request_json(
                'GET',
                url,
                headers,
                trace_id=trace_id,
                observations_404_empty=True,
                timeout_seconds=timeout_seconds,
            )
            if not isinstance(body, dict):
                raise ConsumeBackendError('Langfuse observations response is not a JSON object')
            rows = body.get('data')
            if not isinstance(rows, list):
                rows = []
            meta = body.get('meta') if isinstance(body.get('meta'), dict) else {}
            total_pages = int(meta.get('totalPages', 1) or 1)

            out.extend(self._raw_spans_from_observations(trace_id, rows))

            if not rows:
                break
            if page >= total_pages:
                break
            page += 1

        return out
