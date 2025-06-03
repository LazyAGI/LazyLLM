import os
import json
import uuid
import time
import requests

from pydantic import BaseModel
from urllib.parse import urljoin
from typing import Optional, List, Dict, Any, Union, Set

from lazyllm import warp, pipeline
from lazyllm.common import override
from lazyllm.tools.rag import (
    DocNode,
    IndexBase,
    DataType
)
from lazyllm.tools.rag.doc_node import (
    ImageDocNode,
    QADocNode
)
from lazyllm.tools.rag.global_metadata import (GlobalMetadataDesc, RAG_DOC_ID)
from lazyllm.tools.rag.store.store_base import DocStoreBase, LAZY_ROOT_NAME, LAZY_IMAGE_GROUP, BUILDIN_GLOBAL_META_DESC
from lazyllm.tools.rag.store.utils import upload_data_to_s3, download_data_from_s3

INSERT_BATCH_SIZE = 3000
INSERT_MAX_RETRIES = 20

def _fibonacci_backoff(max_retries: int = INSERT_MAX_RETRIES):
    a, b = 1, 1
    for _ in range(max_retries):
        yield a
        a, b = b, a + b


class Segment(BaseModel):
    segment_id: str
    knowledge_base_id: Optional[str] = "__default__"
    document_id: str
    group_name: str
    content: str
    meta: Dict[str, Any]
    excluded_embed_metadata_keys: Optional[List[str]] = []
    excluded_llm_metadata_keys: Optional[List[str]] = []
    global_meta: Optional[Dict[str, Any]] = {}
    parent: str
    children: List[str] = []
    embedding_state: Optional[List[str]] = []
    answer: Optional[str] = ""
    image_keys: Optional[List[str]] = []


class SenseCoreStore(DocStoreBase):
    def __init__(self, kb_id: str = "__default__", uri: str = "",
                 global_metadata_desc: Dict[str, GlobalMetadataDesc] = None, **kwargs):
        super().__init__(kb_id=kb_id, uri=uri)
        if global_metadata_desc:
            self._global_metadata_desc = global_metadata_desc | BUILDIN_GLOBAL_META_DESC
        else:
            self._global_metadata_desc = BUILDIN_GLOBAL_META_DESC
        self._s3_config = kwargs.get("s3_config")

    @override
    def _connect_store(self, uri: str) -> bool:
        # TODO get the url for testing connection
        return True

    def _serialize_node(self, node: DocNode) -> Dict:
        """ serialize node to dict """

        if node._group == LAZY_ROOT_NAME:
            content = json.dumps(node._content)
            obj_key = f"lazyllm/lazyllm_root/{node._uid}.json"
            upload_data_to_s3(
                content,
                bucket_name=self._s3_config["bucket_name"],
                object_key=obj_key,
                aws_access_key_id=self._s3_config["access_key"],
                aws_secret_access_key=self._s3_config["secret_access_key"],
                use_minio=self._s3_config["use_minio"],
                endpoint_url=self._s3_config["endpoint_url"],
            )
            node._content = obj_key
        elif node._group == LAZY_IMAGE_GROUP:
            image_path = node._image_path
            image_file_name = os.path.basename(image_path)
            obj_key = f"lazyllm/images/{image_file_name}"
            with open(image_path, "rb") as f:
                upload_data_to_s3(
                    f.read(),
                    bucket_name=self._s3_config["bucket_name"],
                    object_key=f"lazyllm/lazyllm_images/{image_file_name}",
                    aws_access_key_id=self._s3_config["access_key"],
                    aws_secret_access_key=obj_key,
                    use_minio=self._s3_config["use_minio"],
                    endpoint_url=self._s3_config["endpoint_url"],
                )
                node._image_path = obj_key
        segment = Segment(
            segment_id=node._uid,
            document_id=node.metadata.get(RAG_DOC_ID),
            group_name=node._group,
            content=node._content if isinstance(node._content, str) else json.dumps(node._content),
            meta=node.metadata,
            excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
            excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
            global_meta=node.global_metadata,
            parent=node.parent._uid if node.parent else "",
            children=node.children.keys() if node.children else [],
            embedding_state=node._embedding_state,
        )

        if isinstance(node, ImageDocNode):
            segment.image_keys = [node._image_path]
        elif isinstance(node, QADocNode):
            segment.answer = node._answer

        return segment.model_dump()

    def _deserialize_node(self, segment: Dict) -> DocNode:
        """ deserialize node from dict """
        if len(segment.get("answer", "")):
            node = QADocNode(
                query=segment["content"],
                answer=segment["answer"],
                uid=segment["segment_id"],
                group=segment["group_name"],
                metadata=segment["meta"],
                global_metadata=segment["global_meta"],
                parent=segment["parent"],
            )
        elif len(segment.get("image_keys", [])):
            node = ImageDocNode(
                image_path=segment["image_keys"][0],
                uid=segment["segment_id"],
                group=segment["group_name"],
                metadata=segment["meta"],
                global_metadata=segment["global_meta"],
                parent=segment["parent"],
            )
        else:
            node = DocNode(
                uid=segment["segment_id"],
                content=segment["content"],
                group=segment["group_name"],
                metadata=segment["meta"],
                global_metadata=segment["global_meta"],
                parent=segment["parent"],
            )
        node.excluded_llm_metadata_keys = segment["excluded_embed_metadata_keys"]
        node.excluded_embed_metadata_keys = segment["excluded_llm_metadata_keys"]
        if node._group == LAZY_ROOT_NAME:
            obj_key = node._content
            node._content = download_data_from_s3(
                bucket_name=self._s3_config["bucket_name"],
                object_key=obj_key,
                aws_access_key_id=self._s3_config["access_key"],
                aws_secret_access_key=self._s3_config["secret_access_key"],
                use_minio=self._s3_config["use_minio"],
                endpoint_url=self._s3_config["endpoint_url"],
                encoding="utf-8"
            )
        return node

    def _create_filters_str(self, filters: Dict[str, Union[str, int, List, Set]]) -> str:
        ret_str = ""
        for name, candidates in filters.items():
            desc = self._global_metadata_desc.get(name)
            if not desc:
                raise ValueError(f'cannot find desc of field [{name}]')

            key = name
            if (not isinstance(candidates, List)) and (not isinstance(candidates, Set)):
                candidates = list(candidates)
            if desc.data_type == DataType.ARRAY:
                # https://github.com/milvus-io/milvus/discussions/35279
                # `array_contains_any` requires milvus >= 2.4.3 and is not supported in local(aka lite) mode.
                ret_str += f'array_contains_any({key}, {candidates}) and '
            else:
                ret_str += f'{key} in {candidates} and '

        if len(ret_str) > 0:
            return ret_str[:-5]  # truncate the last ' and '
        return ret_str

    @override
    def update_nodes(self, nodes: List[DocNode]):
        """ update nodes to the store """
        batched_nodes = [nodes[i:i + INSERT_BATCH_SIZE] for i in range(0, len(nodes), INSERT_BATCH_SIZE)]

        with pipeline() as insert_ppl:
            insert_ppl.get_ids = warp(self._upload_nodes_and_insert).aslist
            insert_ppl.check_status = warp(self._check_insert_task_status)

        insert_ppl(batched_nodes)
        return

    def _upload_nodes_and_insert(self, segments: List[DocNode]) -> str:
        job_id = str(uuid.uuid4())
        segments = [self._serialize_node(node) for node in segments]
        obj_key = f"lazyllm/segments/{job_id}.jsonl"

        upload_data_to_s3(
            data=segments,
            bucket_name=self._s3_config["bucket_name"],
            object_key=obj_key,
            aws_access_key_id=self._s3_config["access_key"],
            aws_secret_access_key=self._s3_config["secret_access_key"],
            use_minio=self._s3_config["use_minio"],
            endpoint_url=self._s3_config["endpoint_url"],
        )

        url = urljoin(self._uri, "v1/writerSegmentJob:submit")
        params = {"writer_segment_job_id": job_id}
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload = {
            "dataset_id": self._kb_id,
            "file_key": obj_key,
        }

        response = requests.post(url, params=params, headers=headers, json=payload)
        response.raise_for_status()

        return job_id

    def _check_insert_job_status(self, job_id: str) -> None:
        """ check if the insert task is finished """
        url = urljoin(self._uri, f"v1/writerSegmentJobs/{job_id}")
        headers = {
            "Accept": "application/json",
        }
        for wait_time in _fibonacci_backoff():
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            status = response.json()["state"]
            if status == 2:
                return
            elif status == 3:
                raise Exception(f"Insert task {job_id} failed")
            else:
                time.sleep(wait_time)
        raise Exception(f"Insert task {job_id} failed after seconds")

    @override
    def remove_nodes(
        self,
        doc_ids: Optional[List[str]] = None,
        uids: Optional[List[str]] = None
    ) -> None:
        """ remove nodes from the store by doc_ids or uids """
        url = urljoin(self._uri, "v1/segments:bulkDelete")
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/json",
        }
        if doc_ids:
            payload = {
                "dataset_id": self._kb_id,
                "document_ids": doc_ids,
            }
        else:
            payload = {
                "dataset_id": self._kb_id,
                "segment_ids": uids,
            }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return

    @override
    def get_nodes(self, group_name: Optional[str] = None, uids: Optional[List[str]] = None) -> List[DocNode]:
        """ get nodes from the store """
        if not (uids or group_name):
            raise ValueError("group_name or uids must be provided")

        url = urljoin(self._uri, "v1/segments:scroll")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload = {
            "dataset_id": self._kb_id,
        }
        if group_name:
            payload['group'] = group_name
        if uids:
            payload['segment_ids'] = uids

        segments = []
        while True:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            if not len(data.get('segments', [])):
                break
            segments.extend(data['segments'])
            next_page_token = data.get('next_page_token')
            if not next_page_token:
                break
            payload['page_token'] = next_page_token

        return [self._deserialize_node(segment) for segment in segments]

    @override
    def query(
        self,
        query: str,
        group_name: str,
        topk: int = 10,
        embed_keys: Optional[List[str]] = None,
        filters: Optional[Dict[str, Union[str, int, List, Set]]] = None,
        **kwargs
    ) -> List[DocNode]:
        """ search nodes from the store """

        url = urljoin(self._uri, "v1/segments:hybrid")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        filter_str = self._create_filters_str(filters) if filters else None
        nodes = []
        for embed_key in embed_keys:
            payload = {
                "query": query,
                "top_k": topk,
                "filters": filter_str,
                "group": group_name,
                "embedding_model": embed_key,
            }
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            nodes.extend([self._deserialize_node(node) for node in response.json()['segments']])
        return nodes

    @override
    def register_index(self, type: str, index: IndexBase) -> None:
        """ register index to the store (for store that support hook only)"""
        raise NotImplementedError(
            "register_index is not supported for SenseCoreStore."
            "Please use register_index for store that support hook"
        )

    @override
    def get_index(self, type: Optional[str] = None) -> Optional[IndexBase]:
        """ get registered index from the store """
        raise NotImplementedError('get_index is not supported for SenseCoreStore.')

    @override
    def update_doc_meta(self, doc_id: str, metadata: dict) -> None:
        """ update doc meta """
        # TODO 性能优化
        nodes = self.get_nodes(uids=[doc_id])
        for node in nodes:
            node.metadata.update(metadata)
        self.update_nodes(nodes)
        return

    @override
    def is_group_active(self, name: str) -> bool:
        """ check if a group has nodes (active) """
        url = urljoin(self._uri, "/v1/segments:scroll")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload = {
            "dataset_id": self._kb_id,
            "group": name,
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        if len(data.get("segments", [])):
            return True
        return False
