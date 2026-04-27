from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from ..parsing_service.base import (
    AddDocRequest as ParsingAddDocRequest,
    CancelTaskRequest as ParsingCancelTaskRequest,
    DeleteDocRequest as ParsingDeleteDocRequest,
    FileInfo as ParsingFileInfo,
    TransferParams as ParsingTransferParams,
    UpdateMetaRequest as ParsingUpdateMetaRequest,
)
from ..utils import BaseResponse
from .utils import normalize_api_base_url


class ParserClient:
    def __init__(self, parser_url: str):
        self._parser_url = normalize_api_base_url(parser_url)

    def _request(self, method: str, path: str, **kwargs):
        response = requests.request(method, f'{self._parser_url}{path}', timeout=8, **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(f'parser http error: {response.status_code} {response.text}')
        return response.json()

    def health(self):
        return BaseResponse.model_validate(self._request('GET', '/health'))

    def add_doc(self, task_id: str, kb_id: str, doc_id: str, file_path: str,
                metadata: Optional[Dict[str, Any]] = None, ng_names: Optional[List[str]] = None,
                task_type: Optional[str] = None,
                callback_url: Optional[str] = None, transfer_params: Optional[Dict[str, Any]] = None):
        req = ParsingAddDocRequest(
            task_id=task_id,
            ng_names=ng_names,
            task_type=task_type,
            kb_id=kb_id,
            callback_url=callback_url,
            feedback_url=callback_url,
            file_infos=[ParsingFileInfo(
                file_path=file_path,
                doc_id=doc_id,
                metadata=metadata or {},
                transfer_params=(
                    ParsingTransferParams.model_validate(transfer_params)
                    if transfer_params is not None else None
                ),
            )],
        )
        return BaseResponse.model_validate(self._request('POST', '/doc/add', json=req.model_dump(mode='json')))

    def update_meta(self, task_id: str, kb_id: str, doc_id: str,
                    metadata: Optional[Dict[str, Any]] = None, file_path: Optional[str] = None,
                    callback_url: Optional[str] = None):
        req = ParsingUpdateMetaRequest(
            task_id=task_id,
            kb_id=kb_id,
            callback_url=callback_url,
            feedback_url=callback_url,
            file_infos=[ParsingFileInfo(file_path=file_path, doc_id=doc_id, metadata=metadata or {})],
        )
        return BaseResponse.model_validate(
            self._request('POST', '/doc/meta/update', json=req.model_dump(mode='json'))
        )

    def delete_doc(self, task_id: str, kb_id: str, doc_id: str,
                   callback_url: Optional[str] = None,
                   node_group_ids_to_delete: Optional[List[str]] = None):
        req = ParsingDeleteDocRequest(
            task_id=task_id,
            kb_id=kb_id,
            doc_ids=[doc_id],
            callback_url=callback_url,
            feedback_url=callback_url,
            node_group_ids_to_delete=node_group_ids_to_delete,
        )
        return BaseResponse.model_validate(
            self._request('DELETE', '/doc/delete', json=req.model_dump(mode='json', exclude_none=True)))

    def cancel_task(self, task_id: str):
        req = ParsingCancelTaskRequest(task_id=task_id)
        return BaseResponse.model_validate(self._request('POST', '/doc/cancel', json=req.model_dump(mode='json')))

    def list_algorithms(self):
        return BaseResponse.model_validate(self._request('GET', '/v1/algo/list'))

    def get_algorithm_groups(self, algo_id: str):
        try:
            data = self._request('GET', f'/v1/algo/{algo_id}/groups')
            return BaseResponse.model_validate(data)
        except RuntimeError as exc:
            if '404' in str(exc):
                return BaseResponse(code=404, msg='algo not found', data=None)
            raise

    def list_doc_chunks(self, kb_id: str, doc_id: str, group: str, offset: int, page_size: int,
                        ng_names: Optional[List[str]] = None):
        params = {
            'kb_id': kb_id,
            'doc_id': doc_id,
            'group': group,
            'offset': offset,
            'page_size': page_size,
        }
        if ng_names:
            params['ng_names'] = ng_names
        data = self._request('GET', '/doc/chunks', params=params)
        return BaseResponse.model_validate(data)
