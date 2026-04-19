# Copyright (c) 2026 LazyAGI. All rights reserved.
import os
from typing import Any, Dict, List, Optional, Tuple

from lazyllm import config, thirdparty

from ..base import LazyLLMFSBase, CloudFSBufferedFile

config.add('s3_access_key', str, None, 'AWS_ACCESS_KEY_ID', description='S3 access key (boto3 official).')
config.add('s3_secret_key', str, None, 'AWS_SECRET_ACCESS_KEY', description='S3 secret key (boto3 official).')
config.add('s3_endpoint_url', str, None, 'CLOUDFS_S3_ENDPOINT_URL', description='S3-compatible endpoint URL (optional).')
config.add('s3_region_name', str, None, 'AWS_DEFAULT_REGION', description='S3 region (boto3 official).')

try:
    from botocore.exceptions import ClientError
except ImportError:
    ClientError = Exception  # type: ignore[misc, assignment]


class S3FS(LazyLLMFSBase):
    _fs_protocol_key = 's3'

    def __init__(self, token: str = '', base_url: Optional[str] = None,
                 access_key: Optional[str] = None,
                 secret_key: Optional[str] = None,
                 endpoint_url: Optional[str] = None,
                 region_name: Optional[str] = None,
                 auth: str = 'static',
                 **storage_options):
        if auth == 'dynamic':
            self._access_key = ''
            self._secret_key = ''
            self._endpoint_url = endpoint_url or None
            self._region_name = region_name or None
            self._s3_client = None
            super().__init__(token='', base_url=base_url, auth='dynamic', **storage_options)
            return
        _ak = config['s3_access_key'] or os.environ.get('AWS_ACCESS_KEY_ID')
        access_key = access_key or token or _ak or ''
        secret_key = secret_key or config['s3_secret_key'] or os.environ.get('AWS_SECRET_ACCESS_KEY') or ''
        endpoint_url = endpoint_url or config['s3_endpoint_url'] or ''
        region_name = region_name or config['s3_region_name'] or os.environ.get('AWS_DEFAULT_REGION') or ''
        self._access_key = access_key
        self._secret_key = secret_key
        self._endpoint_url = endpoint_url or None
        self._region_name = region_name or None
        self._s3_client = None
        super().__init__(token=token or access_key or '', base_url=base_url, **storage_options)

    def _setup_auth(self) -> None:
        boto3 = thirdparty.boto3
        session = boto3.session.Session()
        kwargs: Dict[str, Any] = {}
        if self._access_key:
            kwargs['aws_access_key_id'] = self._access_key
        if self._secret_key:
            kwargs['aws_secret_access_key'] = self._secret_key
        if self._endpoint_url:
            kwargs['endpoint_url'] = self._endpoint_url
        if self._region_name:
            kwargs['region_name'] = self._region_name
        self._s3_client = session.client('s3', **kwargs)

    def _parse_s3_path(self, path: str) -> Tuple[str, str]:
        parts = self._parse_path(path)
        if not parts:
            return '', ''
        bucket = parts[0]
        key = '/'.join(parts[1:]) if len(parts) > 1 else ''
        return bucket, key

    def ls(self, path: str, detail: bool = True, **kwargs) -> List:
        bucket, prefix = self._parse_s3_path(path)
        if not bucket:
            return self._list_buckets(detail)
        return self._list_objects(bucket, prefix, detail)

    def info(self, path: str, **kwargs) -> Dict[str, Any]:
        bucket, key = self._parse_s3_path(path)
        if not bucket:
            return self._entry('/', ftype='directory')
        if not key:
            try:
                self._s3_client.head_bucket(Bucket=bucket)
                return self._entry(f'/{bucket}', ftype='directory')
            except ClientError as e:
                if e.response.get('Error', {}).get('Code') in ('404', 'NoSuchBucket'):
                    raise FileNotFoundError(path) from e
                raise
        try:
            resp = self._s3_client.head_object(Bucket=bucket, Key=key)
            mtime = resp['LastModified'].timestamp() if resp.get('LastModified') else None
            return self._entry(
                name=path, size=resp.get('ContentLength', 0),
                ftype='file', mtime=mtime,
                etag=resp.get('ETag', '').strip('"'),
                content_type=resp.get('ContentType', ''),
            )
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') in ('404', 'NoSuchKey'):
                return self._entry(path, ftype='directory')
            raise

    def _open(self, path: str, mode: str = 'rb',
              block_size: Optional[int] = None,
              autocommit: bool = True,
              cache_options: Optional[Dict] = None,
              **kwargs) -> CloudFSBufferedFile:
        return CloudFSBufferedFile(
            self, path, mode=mode,
            block_size=block_size or self.blocksize,
            autocommit=autocommit, cache_options=cache_options,
        )

    def mkdir(self, path: str, create_parents: bool = True, **kwargs) -> None:
        bucket, key = self._parse_s3_path(path)
        if not bucket:
            raise ValueError('path must include at least a bucket name')
        if not key:
            self._s3_client.create_bucket(Bucket=bucket)
            return
        prefix = key.rstrip('/') + '/'
        self._s3_client.put_object(Bucket=bucket, Key=prefix, Body=b'')

    def rmdir(self, path: str) -> None:
        bucket, key = self._parse_s3_path(path)
        if not bucket:
            return
        if not key:
            self._s3_client.delete_bucket(Bucket=bucket)
            return
        prefix = key.rstrip('/') + '/'
        self._s3_client.delete_object(Bucket=bucket, Key=prefix)

    def rm_file(self, path: str) -> None:
        bucket, key = self._parse_s3_path(path)
        if not bucket or not key:
            raise ValueError(f'invalid S3 path: {path!r}')
        self._s3_client.delete_object(Bucket=bucket, Key=key)

    def copy(self, path1: str, path2: str, recursive: bool = False, **kwargs) -> None:
        src_bucket, src_key = self._parse_s3_path(path1)
        dst_bucket, dst_key = self._parse_s3_path(path2)
        if not src_bucket or not dst_bucket:
            raise ValueError(f'S3 path must include a bucket: {path1!r}, {path2!r}')
        if src_key and not src_key.endswith('/'):
            if not dst_key:
                raise ValueError(f'Invalid S3 destination path: {path2!r}')
            self._s3_client.copy_object(CopySource={'Bucket': src_bucket, 'Key': src_key},
                                        Bucket=dst_bucket, Key=dst_key)
            return
        if not recursive:
            raise ValueError(f'Cannot copy directory {path1} without recursive=True')
        src_prefix = src_key.rstrip('/') + '/' if src_key else ''
        dst_prefix = dst_key.rstrip('/') + '/' if dst_key else ''
        paginator = self._s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=src_bucket, Prefix=src_prefix):
            for obj in page.get('Contents', []):
                rel = obj['Key'][len(src_prefix):]
                self._s3_client.copy_object(CopySource={'Bucket': src_bucket, 'Key': obj['Key']},
                                            Bucket=dst_bucket, Key=dst_prefix + rel)

    def move(self, path1: str, path2: str, recursive: bool = False, **kwargs) -> None:
        src_bucket, src_key = self._parse_s3_path(path1)
        dst_bucket, dst_key = self._parse_s3_path(path2)
        if not src_bucket or not dst_bucket:
            raise ValueError(f'S3 path must include a bucket: {path1!r}, {path2!r}')
        if src_key and not src_key.endswith('/'):
            if not dst_key:
                raise ValueError(f'Invalid S3 destination path: {path2!r}')
            self._s3_client.copy_object(CopySource={'Bucket': src_bucket, 'Key': src_key},
                                        Bucket=dst_bucket, Key=dst_key)
            self._s3_client.delete_object(Bucket=src_bucket, Key=src_key)
            return
        if not recursive:
            raise ValueError(f'Cannot move directory {path1} without recursive=True')
        src_prefix = src_key.rstrip('/') + '/' if src_key else ''
        dst_prefix = dst_key.rstrip('/') + '/' if dst_key else ''
        paginator = self._s3_client.get_paginator('list_objects_v2')
        to_delete: List[str] = []
        for page in paginator.paginate(Bucket=src_bucket, Prefix=src_prefix):
            for obj in page.get('Contents', []):
                rel = obj['Key'][len(src_prefix):]
                self._s3_client.copy_object(CopySource={'Bucket': src_bucket, 'Key': obj['Key']},
                                            Bucket=dst_bucket, Key=dst_prefix + rel)
                to_delete.append(obj['Key'])
        for key in to_delete:
            self._s3_client.delete_object(Bucket=src_bucket, Key=key)

    def _download_range(self, path: str, start: int, end: int) -> bytes:
        bucket, key = self._parse_s3_path(path)
        if not bucket or not key:
            raise FileNotFoundError(path)
        range_header = f'bytes={start}-{end - 1}'
        resp = self._s3_client.get_object(Bucket=bucket, Key=key, Range=range_header)
        return resp['Body'].read()

    def _upload_data(self, path: str, data: bytes) -> None:
        bucket, key = self._parse_s3_path(path)
        if not bucket or not key:
            raise ValueError(f'invalid S3 path: {path!r}')
        self._s3_client.put_object(Bucket=bucket, Key=key, Body=data)

    def _platform_supports_webhook(self) -> bool:
        return True

    def _register_webhook(self, webhook_url: str, events: List[str], path: str) -> Dict[str, Any]:
        bucket, _ = self._parse_s3_path(path)
        if not bucket:
            raise ValueError('path must include a bucket name')
        s3_events = [
            's3:ObjectCreated:*',
            's3:ObjectRemoved:*',
        ]
        config: Dict[str, Any] = {
            'TopicConfigurations': [{
                'TopicArn': webhook_url,
                'Events': s3_events,
            }],
        }
        self._s3_client.put_bucket_notification_configuration(
            Bucket=bucket,
            NotificationConfiguration=config,
        )
        return {'bucket': bucket, 'events': s3_events, 'webhook_url': webhook_url}

    def _list_buckets(self, detail: bool) -> List:
        resp = self._s3_client.list_buckets()
        buckets = resp.get('Buckets', [])
        if detail:
            return [
                self._entry(
                    name=f'/{b["Name"]}', ftype='directory',
                    mtime=b['CreationDate'].timestamp() if b.get('CreationDate') else None,
                )
                for b in buckets
            ]
        return [f'/{b["Name"]}' for b in buckets]

    def _list_objects(self, bucket: str, prefix: str, detail: bool) -> List:
        kwargs: Dict[str, Any] = {'Bucket': bucket, 'Delimiter': '/'}
        if prefix:
            kwargs['Prefix'] = prefix.rstrip('/') + '/'
        results = []
        while True:
            resp = self._s3_client.list_objects_v2(**kwargs)
            for cp in resp.get('CommonPrefixes', []):
                raw = cp['Prefix'].rstrip('/')
                name = f'/{bucket}/{raw}'
                if detail:
                    results.append(self._entry(name=name, ftype='directory'))
                else:
                    results.append(name)
            for obj in resp.get('Contents', []):
                key = obj['Key']
                name = f'/{bucket}/{key}'
                mtime = obj['LastModified'].timestamp() if obj.get('LastModified') else None
                if detail:
                    results.append(self._entry(
                        name=name, size=obj.get('Size', 0),
                        ftype='file', mtime=mtime,
                        etag=obj.get('ETag', '').strip('"'),
                    ))
                else:
                    results.append(name)
            if resp.get('IsTruncated'):
                kwargs['ContinuationToken'] = resp['NextContinuationToken']
            else:
                break
        return results
