import os
import json
import uuid
import time
import requests

from pydantic import BaseModel, Field
from urllib.parse import urljoin
from typing import Optional, List, Dict, Any, Union, Set

from ..store_base import (LazyLLMStoreBase, StoreCapability, LAZY_ROOT_NAME, IMAGE_PATTERN, INSERT_BATCH_SIZE,
                          DEFAULT_KB_ID, SegmentType)
from ..utils import upload_data_to_s3, download_data_from_s3, fibonacci_backoff, create_file_path, presign_obj_from_s3

from ...data_type import DataType
from ...global_metadata import GlobalMetadataDesc, RAG_DOC_ID, RAG_KB_ID

from lazyllm import warp, pipeline, LOG, config, package
from lazyllm.common import override
from lazyllm.thirdparty import boto3

PRESIGN_EXPIRE_TIME = 3600 * 24 * 7


class Segment(BaseModel):
    segment_id: str
    dataset_id: Optional[str] = '__default__'
    document_id: str
    group: str
    content: Optional[str] = ''
    meta: str
    global_meta: str
    excluded_embed_metadata_keys: Optional[List[str]] = Field(default_factory=list)
    excluded_llm_metadata_keys: Optional[List[str]] = Field(default_factory=list)
    parent: Optional[str] = ''
    children: Optional[Dict[str, Any]] = Field(default_factory=dict)
    embedding_state: Optional[List[str]] = Field(default_factory=list)
    answer: Optional[str] = ''
    image_keys: Optional[List[str]] = Field(default_factory=list)
    number: Optional[int] = 0
    copy_source: Optional[Dict[str, str]] = Field(default_factory=dict)


class SenseCoreStore(LazyLLMStoreBase):
    capability = StoreCapability.ALL
    need_embedding = False
    supports_index_registration = False

    def __init__(self, uri: str = '', **kwargs):
        self._uri = uri
        self._s3_config = kwargs.get('s3_config')
        self._image_url_config = kwargs.get('image_url_config')
        self._uploaded_image_keys = set()
        self._path_prefix = kwargs.get('path_prefix')
        if not self._path_prefix:
            try:
                self._path_prefix = config['image_path_prefix']
            except Exception:
                self._path_prefix = os.getenv('RAG_IMAGE_PATH_PREFIX', '')

    @property
    def dir(self):
        return None

    @override
    def connect(self, global_metadata_desc: Optional[Dict[str, GlobalMetadataDesc]] = None, **kwargs) -> None:
        self._check_s3()
        self._global_metadata_desc = global_metadata_desc or {}
        LOG.info(f'[SenseCore Store - connect] connected to {self._uri}')

    def _check_s3(self):
        obj_key = 'lazyllm/warmup.txt'
        upload_data_to_s3('warmup', bucket_name=self._s3_config['bucket_name'], object_key=obj_key,
                          aws_access_key_id=self._s3_config['access_key'],
                          aws_secret_access_key=self._s3_config['secret_access_key'],
                          use_minio=self._s3_config['use_minio'], endpoint_url=self._s3_config['endpoint_url'])
        LOG.info(f'[SenseCore Store - check_s3] uploaded warmup.txt to {self._s3_config["bucket_name"]}')
        return

    def _upload_image_if_needed(self, file_path: str, obj_key: str):
        if obj_key in self._uploaded_image_keys:
            return

        with open(file_path, 'rb') as f:
            upload_data_to_s3(
                f.read(),
                bucket_name=self._s3_config['bucket_name'],
                object_key=obj_key,
                aws_access_key_id=self._s3_config['access_key'],
                aws_secret_access_key=self._s3_config['secret_access_key'],
                use_minio=self._s3_config['use_minio'],
                endpoint_url=self._s3_config['endpoint_url']
            )

        self._uploaded_image_keys.add(obj_key)

    def _serialize_data(self, data: dict) -> Dict:  # noqa: C901
        data = dict(data)
        content = json.dumps(data.get('content', ''), ensure_ascii=False)
        matches = IMAGE_PATTERN.findall(content)
        doc_id = data.get('doc_id', '')
        kb_id = data.get(RAG_KB_ID, DEFAULT_KB_ID)
        for _, image_path in matches:
            if image_path.startswith('lazyllm'):
                continue
            image_file_name = os.path.basename(image_path)
            obj_key = f'lazyllm/images/{kb_id}/{doc_id}/{image_file_name}'
            file_path = create_file_path(path=image_path, prefix=self._path_prefix)
            try:
                self._upload_image_if_needed(file_path, obj_key)
                content = content.replace(image_path, obj_key)
            except FileNotFoundError:
                LOG.error(f'Cannot find image path: {image_path} (local path {file_path}), skip...')
            except Exception as e:
                LOG.error(f'Error when uploading `{image_path}` {e!r}')
        data['content'] = json.loads(content)

        # special requirement: item called `table_image_map` in metadata, need to upload to s3
        if data.get('meta', {}).get('table_image_map', {}):
            for k, md_info in data['meta']['table_image_map'].items():
                matches = IMAGE_PATTERN.findall(md_info)
                if not matches:
                    continue
                image_path = matches[0][1]
                if image_path.startswith('lazyllm'):
                    continue
                image_name = os.path.basename(image_path)
                obj_key = f'lazyllm/images/{kb_id}/{doc_id}/{image_name}'
                file_path = create_file_path(path=image_path, prefix=self._path_prefix)
                try:
                    self._upload_image_if_needed(file_path, obj_key)
                    md_info = md_info.replace(image_path, obj_key)
                    data['meta']['table_image_map'][k] = md_info
                except FileNotFoundError:
                    LOG.error(f'Cannot find image: {image_path} (local path {file_path}, obj key {obj_key}), skip...')
                except Exception as e:
                    LOG.error(f'Error when uploading `{image_path}` (local path {file_path}, obj key {obj_key}) {e!r}')

        if data.get('group') == LAZY_ROOT_NAME:
            obj_key = f"lazyllm/lazyllm_root/{kb_id}/{doc_id}/{data.get('uid')}.json"
            upload_data_to_s3(content.encode('utf-8'), bucket_name=self._s3_config['bucket_name'], object_key=obj_key,
                              aws_access_key_id=self._s3_config['access_key'],
                              aws_secret_access_key=self._s3_config['secret_access_key'],
                              use_minio=self._s3_config['use_minio'], endpoint_url=self._s3_config['endpoint_url'])
            data['content'] = obj_key

        segment = Segment(segment_id=data.get('uid', ''), dataset_id=data.get(RAG_KB_ID, ''),
                          document_id=data.get('doc_id', ''), group=data.get('group', ''),
                          content=data.get('content', ''), meta=json.dumps(data.get('meta', {}), ensure_ascii=False),
                          excluded_embed_metadata_keys=data.get('excluded_embed_metadata_keys', []),
                          excluded_llm_metadata_keys=data.get('excluded_llm_metadata_keys', []),
                          parent=data.get('parent', ''),
                          global_meta=json.dumps(data.get('global_meta', {}), ensure_ascii=False),
                          answer=data.get('answer', ''), number=data.get('number', 0))
        if len(data.get('copy_source', {})):
            segment.copy_source = {
                'dataset_id': data.get('copy_source', {}).get(RAG_KB_ID, DEFAULT_KB_ID),
                'document_id': data.get('copy_source', {}).get(RAG_DOC_ID, ''),
                'segment_id': data.get('copy_source', {}).get('uid', '')
            }
        # image extract
        if isinstance(segment.content, str):
            target = segment.content
        else:
            target = json.dumps(segment.content)
        matches = IMAGE_PATTERN.findall(target)
        for _, image_path in matches:
            segment.image_keys.append(image_path)

        if data.get('type') == SegmentType.IMAGE.value and data.get('image_keys'):
            image_path = data.get('image_keys', [])[0]
            image_file_name = os.path.basename(image_path)
            obj_key = f'lazyllm/images/{kb_id}/{doc_id}/{image_file_name}'
            try:
                self._upload_image_if_needed(image_path, obj_key)
                segment.image_keys = [obj_key]
            except FileNotFoundError:
                LOG.error(f'Cannot find image path: {image_path} (local path {image_path}), skip...')
            except Exception as e:
                LOG.error(f'Error when uploading `{image_path}` {e!r}')
        elif data.get('type') == SegmentType.QA.value and data.get('answer'):
            answer = data.get('answer')
            matches = IMAGE_PATTERN.findall(answer)
            for _, image_path in matches:
                if image_path.startswith('lazyllm'):
                    continue
                image_file_name = os.path.basename(image_path)
                obj_key = f'lazyllm/images/{kb_id}/{doc_id}/{image_file_name}'
                file_path = create_file_path(path=image_path, prefix=self._path_prefix)
                try:
                    self._upload_image_if_needed(file_path, obj_key)
                    answer = answer.replace(image_path, obj_key)
                except FileNotFoundError:
                    LOG.error(f'Cannot find image path: {image_path} (local path {file_path}), skip...')
                except Exception as e:
                    LOG.error(f'Error when uploading `{image_path}` {e!r}')
            data['answer'] = answer
            matches = IMAGE_PATTERN.findall(data['answer'])
            for _, image_path in matches:
                segment.image_keys.append(image_path)
            segment.answer = data['answer']
        return segment.model_dump()

    def _deserialize_data(self, segment: Dict, display: bool = False) -> Dict:
        data = {
            'uid': segment.get('segment_id', ''),
            'doc_id': segment.get('document_id', ''),
            'group': segment.get('group', ''),
            'content': segment.get('content', '') if not display else segment.get('display_content'),
            'meta': json.loads(segment.get('meta', '{}')),
            'global_meta': json.loads(segment.get('global_meta', '{}')),
            'number': segment.get('number', 0),
            'kb_id': segment.get('dataset_id', ''),
            'excluded_embed_metadata_keys': segment.get('excluded_embed_metadata_keys', []),
            'excluded_llm_metadata_keys': segment.get('excluded_llm_metadata_keys', []),
            'parent': segment.get('parent', ''),
            'answer': segment.get('answer', ''),
            'image_keys': segment.get('image_keys', []),
        }
        if len(data.get('answer', '')):
            data['type'] = SegmentType.QA.value
        else:
            data['type'] = SegmentType.TEXT.value
        if data.get('group') == LAZY_ROOT_NAME and data.get('content').startswith('lazyllm/lazyllm_root/'):
            obj_key = data.get('content')
            content = download_data_from_s3(bucket_name=self._s3_config['bucket_name'], object_key=obj_key,
                                            aws_access_key_id=self._s3_config['access_key'],
                                            aws_secret_access_key=self._s3_config['secret_access_key'],
                                            use_minio=self._s3_config['use_minio'],
                                            endpoint_url=self._s3_config['endpoint_url'], encoding='utf-8')
            data['content'] = json.loads(content)

        if display and data.get('meta', {}).get('table_image_map', {}):
            for k, v in data['meta']['table_image_map'].items():
                matches = IMAGE_PATTERN.findall(v)
                if not matches:
                    continue
                image_path = matches[0][1]
                if not image_path.startswith('lazyllm'):
                    LOG.warning(f'[SenseCore Store]: table_image value must start with lazyllm, value: {image_path}')
                    continue
                url = presign_obj_from_s3(
                    bucket_name=self._s3_config['bucket_name'],
                    object_key=image_path,
                    aws_access_key_id=self._s3_config['access_key'],
                    aws_secret_access_key=self._s3_config['secret_access_key'],
                    endpoint_url=self._s3_config.get('external_endpoint_url', self._s3_config['endpoint_url']),
                    region_name=self._s3_config.get('region_name', 'us-east-1'),
                    client_method='get_object',
                    expires_in=PRESIGN_EXPIRE_TIME,
                )
                data['meta']['table_image_map'][k] = v.replace(image_path, url)
        return data

    def _create_filters_str(self, filters: Dict[str, Union[str, int, List, Set]]) -> str:
        ret_str = ''
        for name, candidates in filters.items():
            desc = self._global_metadata_desc.get(name)
            if not desc:
                raise ValueError(f'cannot find desc of field [{name}]')
            key = name
            if isinstance(candidates, str):
                candidates = [candidates]
            if (not isinstance(candidates, list)) and (not isinstance(candidates, set)):
                candidates = list(candidates)
            if desc.data_type == DataType.ARRAY:
                ret_str += f'array_contains_any({key}, {candidates}) and '
            else:
                ret_str += f'{key} in {candidates} and '

        if len(ret_str) > 0:
            return ret_str[:-5]  # truncate the last ' and '
        return ret_str

    def _upload_data_and_insert(self, data: List[dict], job_type: str = 'insert') -> str:
        try:
            job_id = str(uuid.uuid4())
            groups = set()
            for item in data:
                groups.add(item.get('group'))
            groups = list(groups)
            data = [self._serialize_data(item) for item in data]
            dataset_id = None
            for item in data:
                dataset_id = item.get('dataset_id', None)
                break
            if not dataset_id:
                raise ValueError('dataset_id is required in SenseCoreStore')

            obj_key = f'lazyllm/segments/{job_id}.jsonl'

            upload_data_to_s3(data=data, bucket_name=self._s3_config['bucket_name'], object_key=obj_key,
                              aws_access_key_id=self._s3_config['access_key'],
                              aws_secret_access_key=self._s3_config['secret_access_key'],
                              use_minio=self._s3_config['use_minio'], endpoint_url=self._s3_config['endpoint_url'])
            url = urljoin(self._uri, 'v1/writerSegmentJob:submit')
            params = {'writer_segment_job_id': job_id}
            headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
            payload = {'dataset_id': dataset_id or self._kb_id, 'file_key': obj_key,
                       'groups': groups, 'job_type': job_type}

            response = requests.post(url, params=params, headers=headers, json=payload)
            response.raise_for_status()
            LOG.info(f'SenseCore Store: insert task {job_id} submitted, payload:{payload}')
        except Exception as e:
            LOG.error(f'SenseCore Store: insert task {job_id} failed: {e}')
            raise e
        return job_id

    def _check_insert_job_status(self, job_id: str) -> None:
        url = urljoin(self._uri, f'v1/writerSegmentJobs/{job_id}')
        headers = {'Accept': 'application/json'}
        check_start_time = time.time()
        flag = False
        for wait_time in fibonacci_backoff(max_retries=16):
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            status = response.json()['state']
            if status == 2:
                flag = True
                break
            elif status == 3:
                break
            else:
                time.sleep(wait_time)
        check_end_time = time.time()
        if not flag:
            LOG.error(f'SenseCore Store: insert task {job_id} failed after {check_end_time - check_start_time}s')
            raise Exception(f'Insert task {job_id} failed after {check_end_time - check_start_time}s')
        LOG.info(f'SenseCore Store: insert task {job_id} finished after {check_end_time - check_start_time}s')
        return

    def _get_group_name(self, collection_name: str) -> str:
        return collection_name.split('_')[-1] if 'lazyllm_root' not in collection_name else 'lazyllm_root'

    @override
    def upsert(self, collection_name: str, data: List[dict]) -> bool:
        if not data: return True
        try:
            upsert_start_time = time.time()
            job_type = 'insert' if not len(data[0].get('copy_source', {})) else 'copy'
            with pipeline() as insert_ppl:
                insert_ppl.get_ids = warp(self._upload_data_and_insert).aslist
                insert_ppl.check_status = warp(self._check_insert_job_status)

            batched_data = [
                package(data[i:i + INSERT_BATCH_SIZE], job_type) for i in range(0, len(data), INSERT_BATCH_SIZE)
            ]
            insert_ppl(batched_data)
            upsert_end_time = time.time()
            LOG.info(f'[SenseCore Store - upsert] Upsert done! collection_name:{collection_name}, '
                     f'Time:{upsert_end_time - upsert_start_time}s')
            return True
        except Exception as e:
            LOG.error(f'[SenseCore Store - upsert] insert task failed: {e}')
            return False

    @override
    def delete(self, collection_name: str, criteria: dict, **kwargs) -> bool:
        try:
            url = urljoin(self._uri, 'v1/segments:bulkDelete')
            headers = {'Accept': '*/*', 'Content-Type': 'application/json'}
            doc_ids = criteria.get(RAG_DOC_ID)
            if doc_ids:
                payload = {'dataset_id': criteria.get(RAG_KB_ID), 'document_ids': doc_ids}
            else:
                payload = {'dataset_id': criteria.get(RAG_KB_ID), 'segment_ids': criteria.get('uid')}
            if collection_name:
                payload['group'] = self._get_group_name(collection_name)
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
        except Exception as e:
            LOG.error(f'[SenseCore Store - delete] task col: {collection_name}\ncriteria: {criteria}\n{e}')
            return True
        return True

    @override
    def get(self, collection_name: str, criteria: dict, **kwargs) -> List[dict]:  # noqa: C901
        try:
            uids = criteria.get('uid')
            doc_ids = criteria.get(RAG_DOC_ID)
            kb_id = criteria.get(RAG_KB_ID, DEFAULT_KB_ID)
            if not (uids or collection_name):
                raise ValueError('group or uids must be provided')
            if doc_ids and len(doc_ids) > 1:
                raise ValueError('[Sensecore Store - get]: doc_ids must be a single value')
            doc_id = doc_ids[0] if doc_ids else None
            if doc_id and not uids:
                url = urljoin(self._uri, f'v1/datasets/{kb_id}/documents/{doc_id}/segments:search')
            else:
                url = urljoin(self._uri, 'v1/segments:scroll')
            headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
            payload = {'dataset_id': kb_id}
            if collection_name:
                payload['group'] = self._get_group_name(collection_name)
            if doc_id:
                payload['document_id'] = doc_id
            if uids:
                payload['segment_ids'] = uids
            else:
                payload['page_size'] = 1000
            # TODO(chenjiahao): sensecore store itself does not support parent/number index yet
            if criteria.get('parent'):
                payload['parent'] = criteria.get('parent')
            if criteria.get('number'):
                payload['number'] = criteria.get('number')
            segments = []
            while True:
                response = requests.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    LOG.warning(f'SenseCore Store: get task failed: url {url}, data: {payload}, e:{response.text}')
                    break
                data = response.json()
                batch = data.get('segments', [])
                if not batch:
                    break
                segments.extend(batch)
                next_page_token = data.get('next_page_token')
                if not next_page_token:
                    break
                payload['page_token'] = next_page_token
            if doc_ids:
                segments = [segment for segment in segments if segment['document_id'] in doc_ids]
            return [self._deserialize_data(s, display=kwargs.get('display', False)) for s in segments]
        except Exception as e:
            LOG.error(f'[SenseCore Store - get]:task failed: {e}')
            return []

    def _multi_modal_process(self, query: str, images: List[str]):
        urls = []
        s3 = boto3.client('s3', aws_access_key_id=self._image_url_config['access_key'],
                          aws_secret_access_key=self._image_url_config['secret_access_key'],
                          endpoint_url=self._image_url_config['endpoint_url'])
        for image in images:
            query = query + '<image>\n'
            url = s3.generate_presigned_url(ClientMethod='get_object',
                                            Params={'Bucket': self._image_url_config['bucket_name'], 'Key': image},
                                            ExpiresIn=3600)
            urls.append(url)
        return query, urls

    @override
    def search(self, collection_name: str, query: Union[str, dict, List[float]], topk: int,  # noqa: C901
               filters: Optional[Dict[str, Union[str, int, List, Set]]] = None,
               embed_key: Optional[str] = None, **kwargs) -> List[dict]:
        try:
            if not embed_key:
                raise ValueError('[Sensecore Store] Query: embed_key must be provided')
            url = urljoin(self._uri, 'v1/segments:hybrid')
            headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}

            filter_str = self._create_filters_str(filters) if filters else None
            dataset_ids = []
            if filters:
                for name, candidates in filters.items():
                    desc = self._global_metadata_desc.get(name)
                    if not desc:
                        raise ValueError(f'cannot find desc of field [{name}]')
                    key = name
                    if key == RAG_KB_ID:
                        if isinstance(candidates, str):
                            candidates = [candidates]
                        if (not isinstance(candidates, list)) and (not isinstance(candidates, set)):
                            candidates = list(candidates)
                        dataset_ids = candidates
                        break
            if dataset_ids:
                hybrid_search_datasets = [{'dataset_id': dataset_id} for dataset_id in dataset_ids]
            else:
                LOG.error(f'SenseCore Store: no dataset_id provided, please check your filters: {filters}')
                return []

            images = kwargs.get('images', [])
            if images:
                query, images = self._multi_modal_process(query, images)
            payload = {'query': query, 'hybrid_search_datasets': hybrid_search_datasets, 'hybrid_search_type': 2,
                       'top_k': topk, 'filters': filter_str, 'group': self._get_group_name(collection_name),
                       'embedding_model': embed_key, 'images': images}
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            segments = response.json()['segments']
            segments = [s for s in segments if s.get('is_active', True)]
            return [self._deserialize_data(s, display=True) for s in segments]
        except Exception as e:
            LOG.error(f'SenseCore Store: query task failed: {e}')
            raise e
