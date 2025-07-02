import os
import json
import copy
import uuid
import time
import requests

from pydantic import BaseModel, Field
from urllib.parse import urljoin
from typing import Optional, List, Dict, Any, Union, Set

from .store_base import StoreBase, LAZY_ROOT_NAME, BUILDIN_GLOBAL_META_DESC, IMAGE_PATTERN, INSERT_BATCH_SIZE
from .utils import upload_data_to_s3, download_data_from_s3, fibonacci_backoff, create_file_path

from ..index_base import IndexBase
from ..data_type import DataType
from ..doc_node import ImageDocNode, QADocNode, DocNode
from ..global_metadata import GlobalMetadataDesc, RAG_DOC_ID, RAG_DOC_KB_ID

from lazyllm import warp, pipeline, LOG, config
from lazyllm.common import override
from lazyllm.thirdparty import boto3


class Segment(BaseModel):
    segment_id: str
    dataset_id: Optional[str] = "__default__"
    document_id: str
    group: str
    content: Optional[str] = ""
    meta: str
    global_meta: str
    excluded_embed_metadata_keys: Optional[List[str]] = Field(default_factory=list)
    excluded_llm_metadata_keys: Optional[List[str]] = Field(default_factory=list)
    parent: Optional[str] = ""
    children: Dict[str, Any] = Field(default_factory=dict)
    embedding_state: Optional[List[str]] = Field(default_factory=list)
    answer: Optional[str] = ""
    image_keys: Optional[List[str]] = Field(default_factory=list)
    number: Optional[int] = 0


class SenseCoreStore(StoreBase):
    def __init__(self, group_embed_keys: Dict[str, Set[str]],
                 global_metadata_desc: Dict[str, GlobalMetadataDesc] = None,
                 kb_id: str = "__default__", uri: str = "", **kwargs):
        self._uri = uri
        self._kb_id = kb_id
        self._group_embed_keys = group_embed_keys
        self._s3_config = kwargs.get("s3_config")
        self._image_url_config = kwargs.get("image_url_config")
        self._activated_groups = set()
        if global_metadata_desc:
            self._global_metadata_desc = global_metadata_desc | BUILDIN_GLOBAL_META_DESC
        else:
            self._global_metadata_desc = BUILDIN_GLOBAL_META_DESC

        if self._connect_store(uri):
            LOG.info(f"Connected to doc store {self._uri}")
        else:
            raise ConnectionError(f"Failed to connect to doc store {self._uri}")

    @override
    def _connect_store(self, uri: str) -> bool:
        # TODO get the url for testing connection
        self._check_s3()
        return True

    def _check_s3(self):
        obj_key = "lazyllm/warmup.txt"
        upload_data_to_s3("warmup", bucket_name=self._s3_config["bucket_name"], object_key=obj_key,
                          aws_access_key_id=self._s3_config["access_key"],
                          aws_secret_access_key=self._s3_config["secret_access_key"],
                          use_minio=self._s3_config["use_minio"], endpoint_url=self._s3_config["endpoint_url"])
        return

    def _serialize_node(self, node: DocNode) -> Dict:  # noqa: C901
        """ serialize node to dict """
        segment = Segment(segment_id=node._uid, dataset_id=node.global_metadata.get(RAG_DOC_KB_ID, None) or self._kb_id,
                          document_id=node.global_metadata.get(RAG_DOC_ID), group=node._group,
                          meta=json.dumps(node._metadata, ensure_ascii=False),
                          excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
                          excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
                          global_meta=json.dumps(node.global_metadata, ensure_ascii=False),
                          children={group: {"ids": [n._uid for n in c_l]} for group, c_l in node.children.items()},
                          embedding_state=node._embedding_state, number=node._metadata.get("store_num", 0))
        if node.parent:
            if isinstance(node.parent, DocNode):
                segment.parent = node.parent._uid
            elif isinstance(node.parent, str):
                segment.parent = node.parent

        if node._group == LAZY_ROOT_NAME:
            # content is root, process image key
            content = json.dumps(node._content, ensure_ascii=False)
            # image extract
            matches = IMAGE_PATTERN.findall(content)
            for title, image_path in matches:
                if image_path.startswith("lazyllm"):
                    continue
                image_file_name = os.path.basename(image_path)
                obj_key = f"lazyllm/images/{image_file_name}"
                try:
                    prefix = config['image_path_prefix']
                except Exception:
                    prefix = os.getenv("RAG_IMAGE_PATH_PREFIX", "")
                file_path = create_file_path(path=image_path, prefix=prefix)
                try:
                    with open(file_path, "rb") as f:
                        upload_data_to_s3(f.read(), bucket_name=self._s3_config["bucket_name"], object_key=obj_key,
                                          aws_access_key_id=self._s3_config["access_key"],
                                          aws_secret_access_key=self._s3_config["secret_access_key"],
                                          use_minio=self._s3_config["use_minio"],
                                          endpoint_url=self._s3_config["endpoint_url"])
                        content = content.replace(image_path, obj_key)
                except FileNotFoundError:
                    LOG.error(f"Cannot find image path: {image_path} (local path {file_path}), skip...")
                except Exception as e:
                    LOG.error(f"Error when uploading `{image_path}` {e!r}")
            node._content = json.loads(content)
            # image extract
            matches = IMAGE_PATTERN.findall(content)
            for title, image_path in matches:
                segment.image_keys.append(image_path)

            # upload content
            obj_key = f"lazyllm/lazyllm_root/{node._uid}.json"
            upload_data_to_s3(content.encode('utf-8'), bucket_name=self._s3_config["bucket_name"], object_key=obj_key,
                              aws_access_key_id=self._s3_config["access_key"],
                              aws_secret_access_key=self._s3_config["secret_access_key"],
                              use_minio=self._s3_config["use_minio"], endpoint_url=self._s3_config["endpoint_url"])
            segment.content = obj_key
        else:
            segment.content = node._content

            content = json.dumps(node._content, ensure_ascii=False)
            # image extract
            matches = IMAGE_PATTERN.findall(content)
            for title, image_path in matches:
                segment.image_keys.append(image_path)

        if isinstance(node, ImageDocNode):
            image_path = node._image_path
            image_file_name = os.path.basename(image_path)
            obj_key = f"lazyllm/images/{image_file_name}"
            with open(image_path, "rb") as f:
                upload_data_to_s3(f.read(), bucket_name=self._s3_config["bucket_name"], object_key=obj_key,
                                  aws_access_key_id=self._s3_config["access_key"],
                                  aws_secret_access_key=self._s3_config["secret_access_key"],
                                  use_minio=self._s3_config["use_minio"], endpoint_url=self._s3_config["endpoint_url"])
                segment.image_keys = [obj_key]
        elif isinstance(node, QADocNode):
            answer = node._answer
            # image extract
            matches = IMAGE_PATTERN.findall(answer)
            for title, image_path in matches:
                if image_path.startswith("lazyllm"):
                    continue
                image_file_name = os.path.basename(image_path)
                obj_key = f"lazyllm/images/{image_file_name}"
                try:
                    prefix = config['image_path_prefix']
                except Exception:
                    prefix = os.getenv("RAG_IMAGE_PATH_PREFIX", "")
                file_path = create_file_path(path=image_path, prefix=prefix)
                try:
                    with open(file_path, "rb") as f:
                        upload_data_to_s3(f.read(), bucket_name=self._s3_config["bucket_name"], object_key=obj_key,
                                          aws_access_key_id=self._s3_config["access_key"],
                                          aws_secret_access_key=self._s3_config["secret_access_key"],
                                          use_minio=self._s3_config["use_minio"],
                                          endpoint_url=self._s3_config["endpoint_url"])
                        answer = answer.replace(image_path, obj_key)
                except FileNotFoundError:
                    LOG.error(f"Cannot find image path: {image_path} (local path {file_path}), skip...")
                except Exception as e:
                    LOG.error(f"Error when uploading `{image_path}` {e!r}")
            node._answer = answer

            matches = IMAGE_PATTERN.findall(node._answer)
            for title, image_path in matches:
                segment.image_keys.append(image_path)

            segment.answer = node._answer
        return segment.model_dump()

    def _deserialize_node(self, segment: Dict) -> DocNode:
        """ deserialize node from dict """
        if len(segment.get("answer", "")):
            node = QADocNode(query=segment["content"], answer=segment["answer"], uid=segment["segment_id"],
                             group=segment["group"], metadata=json.loads(segment["meta"]),
                             global_metadata=json.loads(segment["global_meta"]), parent=segment["parent"])
        else:
            node = DocNode(uid=segment["segment_id"], content=segment["content"], group=segment["group"],
                           metadata=json.loads(segment["meta"]), global_metadata=json.loads(segment["global_meta"]),
                           parent=segment["parent"])
        node.excluded_llm_metadata_keys = segment["excluded_embed_metadata_keys"]
        node.excluded_embed_metadata_keys = segment["excluded_llm_metadata_keys"]
        if segment["children"]:
            children = {group: item["ids"] for group, item in segment["children"].items()}
        else:
            children = {}
        node.children = children
        if node._group == LAZY_ROOT_NAME and node._content.startswith("lazyllm/lazyllm_root/"):
            obj_key = node._content
            content = download_data_from_s3(bucket_name=self._s3_config["bucket_name"], object_key=obj_key,
                                            aws_access_key_id=self._s3_config["access_key"],
                                            aws_secret_access_key=self._s3_config["secret_access_key"],
                                            use_minio=self._s3_config["use_minio"],
                                            endpoint_url=self._s3_config["endpoint_url"], encoding="utf-8")
            node._content = json.loads(content)
        if segment.get("metadata", {}) is not None:
            node = node.with_sim_score(score=segment.get("metadata", {}).get("score", 0))
        return node

    def _create_filters_str(self, filters: Dict[str, Union[str, int, List, Set]]) -> str:
        ret_str = ""
        for name, candidates in filters.items():
            desc = self._global_metadata_desc.get(name)
            if not desc:
                raise ValueError(f'cannot find desc of field [{name}]')
            key = name
            if isinstance(candidates, str):
                candidates = [candidates]
            if (not isinstance(candidates, List)) and (not isinstance(candidates, Set)):
                candidates = list(candidates)
            if desc.data_type == DataType.ARRAY:
                ret_str += f'array_contains_any({key}, {candidates}) and '
            else:
                ret_str += f'{key} in {candidates} and '

        if len(ret_str) > 0:
            return ret_str[:-5]  # truncate the last ' and '
        return ret_str

    @override
    def update_nodes(self, nodes: List[DocNode]):
        """ update nodes to the store """
        filtered_nodes = []
        for node in nodes:
            if isinstance(node, QADocNode):
                kb_id = node.global_metadata.get(RAG_DOC_KB_ID)
                source_file = node.metadata["source_file_name"]
                source_chunk = node.metadata["source_chunk"]
                target_nodes = self.query(query=source_chunk, group_name="block", topk=1, embed_keys=["bge_m3_dense"],
                                          filters={"kb_id": [kb_id], "file_name": [source_file]})
                if not len(target_nodes):
                    LOG.warning(f"cannot find file for qa node: source_file {source_file}, chunk {source_chunk}")
                    continue
            filtered_nodes.append(node)
        if not filtered_nodes:
            LOG.warning("no nodes to update")
            return
        group_cnt = {}
        for node in filtered_nodes:
            if node._group not in group_cnt:
                group_cnt[node._group] = 1
            node._metadata["store_num"] = group_cnt[node._group]
            group_cnt[node._group] += 1

        with pipeline() as insert_ppl:
            insert_ppl.get_ids = warp(self._upload_nodes_and_insert).aslist
            insert_ppl.check_status = warp(self._check_insert_job_status)

        batched_nodes = [
            filtered_nodes[i:i + INSERT_BATCH_SIZE] for i in range(0, len(filtered_nodes), INSERT_BATCH_SIZE)]
        insert_ppl(batched_nodes)
        return

    def _upload_nodes_and_insert(self, segments: List[DocNode]) -> str:
        try:
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

            upload_data_to_s3(data=segments, bucket_name=self._s3_config["bucket_name"], object_key=obj_key,
                              aws_access_key_id=self._s3_config["access_key"],
                              aws_secret_access_key=self._s3_config["secret_access_key"],
                              use_minio=self._s3_config["use_minio"], endpoint_url=self._s3_config["endpoint_url"])
            url = urljoin(self._uri, "v1/writerSegmentJob:submit")
            params = {"writer_segment_job_id": job_id}
            headers = {"Accept": "application/json", "Content-Type": "application/json"}
            payload = {"dataset_id": dataset_id or self._kb_id, "file_key": obj_key, "groups": groups}

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
        headers = {"Accept": "application/json"}
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
    def remove_nodes(self, group_name: Optional[str] = None, dataset_id: Optional[str] = None,
                     doc_ids: Optional[List[str]] = None, uids: Optional[List[str]] = None) -> None:
        """ remove nodes from the store by doc_ids or uids """
        try:
            url = urljoin(self._uri, "v1/segments:bulkDelete")
            headers = {"Accept": "*/*", "Content-Type": "application/json"}
            if doc_ids:
                payload = {"dataset_id": dataset_id or self._kb_id, "document_ids": doc_ids}
            else:
                payload = {"dataset_id": dataset_id or self._kb_id, "segment_ids": uids}
            if group_name:
                payload["group"] = group_name
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
        except Exception as e:
            LOG.error(f"SenseCore Store: remove task failed: {e}")
            raise e
        return

    @override
    def get_nodes(self, group_name: Optional[str] = None, uids: Optional[List[str]] = None,  # noqa: C901
                  doc_ids: Optional[Set] = None, dataset_id: Optional[str] = None,
                  display: bool = False) -> List[DocNode]:
        """ get nodes from the store """
        if not (uids or group_name):
            raise ValueError("group_name or uids must be provided")
        if doc_ids and len(doc_ids) > 1:
            raise ValueError("[Sensecore Store] - get_nodes: doc_ids must be a single value")
        doc_id = doc_ids[0] if doc_ids else None
        dataset_id = dataset_id or self._kb_id

        if doc_id and not uids:
            url = urljoin(self._uri, f"v1/datasets/{dataset_id}/documents/{doc_id}/segments:search")
        else:
            url = urljoin(self._uri, "v1/segments:scroll")

        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        payload = {"dataset_id": dataset_id}
        if group_name:
            payload["group"] = group_name
        if doc_id:
            payload["document_id"] = doc_id
        if uids:
            payload["segment_ids"] = uids
        else:
            payload["page_size"] = 100
        segments = []
        while True:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                LOG.warning(f"SenseCore Store: get node task failed: {response.text}")
                break
            data = response.json()
            batch = data.get("segments", [])
            if not batch:
                break
            segments.extend(batch)
            next_page_token = data.get('next_page_token')
            if not next_page_token:
                break
            payload['page_token'] = next_page_token
        if doc_ids:
            segments = [segment for segment in segments if segment['document_id'] in doc_ids]
        if display:
            segments = self._apply_display(segments)
        return [self._deserialize_node(s) for s in segments]

    def _apply_display(self, segments: List[dict]) -> List[dict]:
        out = []
        for s in segments:
            if not s.get('is_active', True):
                continue
            if s.get('display_content'):
                s['content'] = s['display_content']
            out.append(s)
        return out

    def _multi_modal_process(self, query: str, images: List[str]):
        urls = []
        s3 = boto3.client('s3', aws_access_key_id=self._image_url_config["access_key"],
                          aws_secret_access_key=self._image_url_config["secret_access_key"],
                          endpoint_url=self._image_url_config["endpoint_url"])
        for image in images:
            query = query + "<image>\n"
            url = s3.generate_presigned_url(ClientMethod='get_object',
                                            Params={'Bucket': self._image_url_config["bucket_name"], 'Key': image},
                                            ExpiresIn=3600)
            urls.append(url)
        return query, urls

    @override
    def query(self, query: str, group_name: str, topk: int = 10, embed_keys: Optional[List[str]] = None,  # noqa: C901
              filters: Optional[Dict[str, Union[str, int, List, Set]]] = None, **kwargs) -> List[DocNode]:
        """ search nodes from the store """
        try:
            if not embed_keys:
                raise ValueError("[Sensecore Store] Query: embed_keys must be provided")
            url = urljoin(self._uri, "v1/segments:hybrid")
            headers = {"Accept": "application/json", "Content-Type": "application/json"}

            original_filters = copy.deepcopy(filters)
            if group_name == 'qa':
                filters = {"kb_id": filters.get("kb_id", [])}
            filter_str = self._create_filters_str(filters) if filters else None

            dataset_ids = []
            if filters:
                for name, candidates in filters.items():
                    desc = self._global_metadata_desc.get(name)
                    if not desc:
                        raise ValueError(f'cannot find desc of field [{name}]')
                    key = name
                    if key == "kb_id":
                        if isinstance(candidates, str):
                            candidates = [candidates]
                        if (not isinstance(candidates, List)) and (not isinstance(candidates, Set)):
                            candidates = list(candidates)
                        dataset_ids = candidates

            if dataset_ids:
                hybrid_search_datasets = [{"dataset_id": dataset_id} for dataset_id in dataset_ids]
            else:
                LOG.error(f"SenseCore Store: no dataset_id provided, please check your filters: {filters}")
                return []

            images = kwargs.get("images", [])
            if images:
                query, images = self._multi_modal_process(query, images)

            nodes = []
            for embed_key in embed_keys:
                payload = {"query": query, "hybrid_search_datasets": hybrid_search_datasets, "hybrid_search_type": 2,
                           "top_k": topk, "filters": filter_str, "group": group_name, "embedding_model": embed_key,
                           "images": images}
                LOG.info(f"[Sensecore Store]: query request body: {payload}.")
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                segments = response.json()['segments']
                segments = [s for s in segments if s['is_active']]
                for s in segments:
                    if len(s.get('display_content', '')):
                        s['content'] = s['display_content']
                if group_name == 'qa':
                    for segment in segments:
                        node = self._deserialize_node(segment)
                        source_file = node.metadata.get("source_file_name", "")
                        if not source_file:
                            continue
                        source_chunk = node.metadata.get("source_chunk", "")
                        original_filters["file_name"] = [source_file]
                        target_nodes = self.query(query=source_chunk, group_name="block", topk=1,
                                                  embed_keys=["bge_m3_dense"], filters=original_filters)
                        if len(target_nodes):
                            node.global_metadata.update(target_nodes[0].global_metadata)
                            node.metadata.update(target_nodes[0].metadata)
                            nodes.append(node)
                else:
                    nodes.extend([self._deserialize_node(node) for node in segments])
            return nodes
        except Exception as e:
            LOG.error(f"SenseCore Store: query task failed: {e}")
            raise e

    @override
    def register_index(self, type: str, index: IndexBase) -> None:
        """ register index to the store (for store that support hook only)"""
        raise NotImplementedError("register_index is not supported for SenseCoreStore."
                                  "Please use register_index for store that support hook")

    @override
    def get_index(self, type: Optional[str] = None) -> Optional[IndexBase]:
        """ get registered index from the store """
        raise NotImplementedError('get_index is not supported for SenseCoreStore.')

    @override
    def update_doc_meta(self, doc_id: str, metadata: dict) -> None:
        """ update doc meta """
        # TODO 性能优化
        dataset_id = metadata.get(RAG_DOC_KB_ID, None)
        nodes: List[DocNode] = []
        for group in self.activated_groups():
            group_nodes = self.get_nodes(group_name=group, dataset_id=dataset_id, doc_ids=[doc_id])
            nodes.extend(group_nodes)

        for node in nodes:
            node.global_metadata.update(metadata)
        self.update_nodes(nodes)
        return

    @override
    def all_groups(self) -> List[str]:
        """ get all node groups for Document """
        return list(self._activated_groups)

    @override
    def activate_group(self, group_names: Union[str, List[str]]):
        if isinstance(group_names, str): group_names = [group_names]
        active_groups = []
        for group_name in group_names:
            if group_name.isupper():
                LOG.error(f"Group name {group_name} should be lowercase (`_` is allowed)")
                continue
            active_groups.append(group_name)
        self._activated_groups.update(active_groups)

    @override
    def activated_groups(self):
        return list(self._activated_groups)

    @override
    def is_group_active(self, name: str) -> bool:
        """ check if a group has nodes (active) """
        try:
            url = urljoin(self._uri, "/v1/segments:scroll")
            headers = {"Accept": "application/json", "Content-Type": "application/json"}
            payload = {"dataset_id": self._kb_id, "group": name}

            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            if len(data.get("segments", [])):
                return True
        except Exception as e:
            LOG.error(f"is_group_active error for group {name}: {str(e)}")
        return False

    @override
    def clear_cache(self, group_names: Optional[List[str]] = None) -> None:
        """ clear cache for a group """
        raise NotImplementedError("clear_cache is not supported for SenseCoreStore.")
