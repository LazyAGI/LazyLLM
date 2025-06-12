import os
import json
import uuid
import time
import requests

from pydantic import BaseModel
from urllib.parse import urljoin
from typing import Optional, List, Dict, Any, Union, Set

from lazyllm import warp, pipeline, LOG, config
from lazyllm.common import override
from lazyllm.tools.rag.index_base import IndexBase
from lazyllm.tools.rag.data_type import DataType
from lazyllm.tools.rag.doc_node import (
    ImageDocNode,
    QADocNode,
    DocNode
)
from lazyllm.tools.rag.global_metadata import (GlobalMetadataDesc, RAG_DOC_ID)

from .store_base import DocStoreBase, LAZY_ROOT_NAME, BUILDIN_GLOBAL_META_DESC
from .utils import upload_data_to_s3, download_data_from_s3, fibonacci_backoff, create_file_path

INSERT_BATCH_SIZE = 3000


class Segment(BaseModel):
    segment_id: str
    dataset_id: Optional[str] = "__default__"
    document_id: str
    group: str
    content: Optional[str] = ""
    meta: str
    global_meta: str
    excluded_embed_metadata_keys: Optional[List[str]] = []
    excluded_llm_metadata_keys: Optional[List[str]] = []
    parent: str
    children: Dict[str, Any] = {}
    embedding_state: Optional[List[str]] = []
    answer: Optional[str] = ""
    image_keys: Optional[List[str]] = []
    image_map: Optional[Dict[str, str]] = {}
    number: Optional[int] = 0


class SenseCoreStore(DocStoreBase):
    def __init__(self, group_embed_keys: Dict[str, Set[str]],
                 global_metadata_desc: Dict[str, GlobalMetadataDesc] = None,
                 kb_id: str = "__default__", uri: str = "", **kwargs):
        if global_metadata_desc:
            self._global_metadata_desc = global_metadata_desc | BUILDIN_GLOBAL_META_DESC
        else:
            self._global_metadata_desc = BUILDIN_GLOBAL_META_DESC
        self._group_embed_keys = group_embed_keys
        self._s3_config = kwargs.get("s3_config")
        super().__init__(kb_id=kb_id, uri=uri)

    @override
    def _connect_store(self, uri: str) -> bool:
        # TODO get the url for testing connection
        self._check_s3()
        return True

    def _check_s3(self):
        obj_key = "lazyllm/warmup.txt"
        upload_data_to_s3(
            "warmup",
            bucket_name=self._s3_config["bucket_name"],
            object_key=obj_key,
            aws_access_key_id=self._s3_config["access_key"],
            aws_secret_access_key=self._s3_config["secret_access_key"],
            use_minio=self._s3_config["use_minio"],
            endpoint_url=self._s3_config["endpoint_url"],
        )
        return

    def _serialize_node(self, node: DocNode) -> Dict:
        """ serialize node to dict """

        segment = Segment(
            segment_id=node._uid,
            dataset_id=node.global_metadata.get("kb_id", None) or self._kb_id,
            document_id=node.global_metadata.get(RAG_DOC_ID),
            group=node._group,
            meta=json.dumps(node.metadata),
            excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
            excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
            global_meta=json.dumps(node.global_metadata),
            parent=node.parent._uid if node.parent else "",
            children={group: {"ids": [n._uid for n in c_l]} for group, c_l in node.children.items()},
            embedding_state=node._embedding_state,
            number=node.metadata.get("store_num", 0)
        )

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
            segment.content = obj_key
        else:
            segment.content = node._content

        if isinstance(node, ImageDocNode):
            image_path = node._image_path
            image_file_name = os.path.basename(image_path)
            obj_key = f"lazyllm/images/{image_file_name}"
            with open(image_path, "rb") as f:
                upload_data_to_s3(
                    f.read(),
                    bucket_name=self._s3_config["bucket_name"],
                    object_key=obj_key,
                    aws_access_key_id=self._s3_config["access_key"],
                    aws_secret_access_key=self._s3_config["secret_access_key"],
                    use_minio=self._s3_config["use_minio"],
                    endpoint_url=self._s3_config["endpoint_url"],
                )
                segment.image_keys = [obj_key]
        elif isinstance(node, QADocNode):
            segment.answer = node._answer
        elif node.__class__.__name__ == 'MixDocNode':
            image_paths = node._image_path
            content = node._content
            for image_path in image_paths:
                image_file_name = os.path.basename(image_path)
                obj_key = f"lazyllm/images/{image_file_name}"
                try:
                    prefix = config['process_path_prefix']
                except Exception as e:
                    LOG.info(f"No process_path_prefix found in config {e}")
                    prefix = ""
                with open(create_file_path(path=image_path, prefix=prefix), "rb") as f:
                    upload_data_to_s3(
                        f.read(),
                        bucket_name=self._s3_config["bucket_name"],
                        object_key=obj_key,
                        aws_access_key_id=self._s3_config["access_key"],
                        aws_secret_access_key=self._s3_config["secret_access_key"],
                        use_minio=self._s3_config["use_minio"],
                        endpoint_url=self._s3_config["endpoint_url"],
                    )
                    segment.image_keys.append(obj_key)
                    if isinstance(content, str):
                        content = content.replace(image_path, obj_key)
            segment.content = content
        return segment.model_dump()

    def _deserialize_node(self, segment: Dict) -> DocNode:
        """ deserialize node from dict """
        if len(segment.get("answer", "")):
            node = QADocNode(
                query=segment["content"],
                answer=segment["answer"],
                uid=segment["segment_id"],
                group=segment["group"],
                metadata=json.loads(segment["meta"]),
                global_metadata=json.loads(segment["global_meta"]),
                parent=segment["parent"],
            )
        elif len(segment.get("image_keys", [])):
            node = ImageDocNode(
                image_path=segment["image_keys"][0],
                uid=segment["segment_id"],
                group=segment["group"],
                metadata=json.loads(segment["meta"]),
                global_metadata=json.loads(segment["global_meta"]),
                parent=segment["parent"],
            )
        else:
            node = DocNode(
                uid=segment["segment_id"],
                content=segment["content"],
                group=segment["group"],
                metadata=json.loads(segment["meta"]),
                global_metadata=json.loads(segment["global_meta"]),
                parent=segment["parent"],
            )
        node.excluded_llm_metadata_keys = segment["excluded_embed_metadata_keys"]
        node.excluded_embed_metadata_keys = segment["excluded_llm_metadata_keys"]
        if segment["children"]:
            children = {group: item["ids"] for group, item in segment["children"].items()}
        else:
            children = {}
        node.children = children

        if node._group == LAZY_ROOT_NAME:
            obj_key = node._content
            content = download_data_from_s3(
                bucket_name=self._s3_config["bucket_name"],
                object_key=obj_key,
                aws_access_key_id=self._s3_config["access_key"],
                aws_secret_access_key=self._s3_config["secret_access_key"],
                use_minio=self._s3_config["use_minio"],
                endpoint_url=self._s3_config["endpoint_url"],
                encoding="utf-8"
            )
            node._content = json.loads(content)
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
        cnt = 1
        for node in nodes:
            node._metadata["store_num"] = cnt
            cnt += 1

        with pipeline() as insert_ppl:
            insert_ppl.get_ids = warp(self._upload_nodes_and_insert).aslist
            insert_ppl.check_status = warp(self._check_insert_job_status)

        batched_nodes = [nodes[i:i + INSERT_BATCH_SIZE] for i in range(0, len(nodes), INSERT_BATCH_SIZE)]
        insert_ppl(batched_nodes)
        return

    def _upload_nodes_and_insert(self, segments: List[DocNode]) -> str:
        job_id = str(uuid.uuid4())
        groups = set()
        for node in segments:
            groups.add(node._group)
        groups = list(groups)

        segments = [self._serialize_node(node) for node in segments]
        dataset_id = None
        for segment in segments:
            dataset_id = segment.get("dataset_id", None)
            break
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

        try:
            url = urljoin(self._uri, "v1/writerSegmentJob:submit")
            params = {"writer_segment_job_id": job_id}
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            payload = {
                "dataset_id": dataset_id or self._kb_id,
                "file_key": obj_key,
                "groups": groups
            }

            response = requests.post(url, params=params, headers=headers, json=payload)
            response.raise_for_status()
            LOG.info(f"SenseCore Store: insert task {job_id} submitted")
        except Exception as e:
            LOG.error(f"SenseCore Store: insert task {job_id} failed: {e}")
            raise e
        return job_id

    def _check_insert_job_status(self, job_id: str) -> None:
        """ check if the insert task is finished """
        url = urljoin(self._uri, f"v1/writerSegmentJobs/{job_id}")
        headers = {
            "Accept": "application/json",
        }
        for wait_time in fibonacci_backoff(max_retries=15):
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            status = response.json()["state"]
            if status == 2:
                LOG.info(f"SenseCore Store: insert task {job_id} finished")
                return
            elif status == 3:
                raise Exception(f"Insert task {job_id} failed")
            else:
                time.sleep(wait_time)
        raise Exception(f"Insert task {job_id} failed after seconds")

    @override
    def remove_nodes(
        self,
        group_name: Optional[str] = None,
        dataset_id: Optional[str] = None,
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
                "dataset_id": dataset_id or self._kb_id,
                "document_ids": doc_ids,
            }
        else:
            payload = {
                "dataset_id": dataset_id or self._kb_id,
                "segment_ids": uids,
            }
        if group_name:
            payload["group"] = group_name
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return

    @override
    def get_nodes(
        self,
        group_name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        uids: Optional[List[str]] = None,
        doc_ids: Optional[Set] = None
    ) -> List[DocNode]:
        """ get nodes from the store """
        if not (uids or group_name):
            raise ValueError("group_name or uids must be provided")

        url = urljoin(self._uri, "v1/segments:scroll")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload = {
            "dataset_id": dataset_id or self._kb_id,
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
        if doc_ids:
            segments = [segment for segment in segments if segment['document_id'] in doc_ids]
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
        dataset_ids = []
        if filters:
            for name, candidates in filters.items():
                desc = self._global_metadata_desc.get(name)
                if not desc:
                    raise ValueError(f'cannot find desc of field [{name}]')
                key = name
                if key == "kb_id":
                    if (not isinstance(candidates, List)) and (not isinstance(candidates, Set)):
                        candidates = list(candidates)
                    dataset_ids = candidates

        if dataset_ids:
            hybrid_search_datasets = [{"dataset_id": dataset_id} for dataset_id in dataset_ids]
        else:
            hybrid_search_datasets = [{"dataset_id": self._kb_id}]
        nodes = []
        for embed_key in embed_keys:
            payload = {
                "query": query,
                "hybrid_search_datasets": hybrid_search_datasets,
                "hybrid_search_type": 2,
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
        dataset_id = metadata.get("kb_id", None)
        for group in self.activated_groups():
            nodes = self.get_nodes(group_name=group, dataset_id=dataset_id, doc_ids=[doc_id])
            for node in nodes:
                node.metadata.update(metadata)
        self.update_nodes(nodes)
        return

    @override
    def activate_group(self, group_names: Union[str, List[str]]) -> bool:
        if isinstance(group_names, str): group_names = [group_names]
        active_groups = []
        for group_name in group_names:
            if group_name.isupper():
                LOG.error(f"Group name {group_name} should be lowercase (`_` is allowed)")
                continue
            active_groups.append(group_name)
        self._activated_groups.update(active_groups)

    @override
    def is_group_active(self, name: str) -> bool:
        """ check if a group has nodes (active) """
        try:
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
        except Exception as e:
            LOG.error(f"is_group_active error for group {name}: {str(e)}")

        return False
